import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import html as htmllib
import io
import logging
import os
import random
import re
from time import monotonic
from typing import Any, Literal, Optional
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
import httpx
from openai import AsyncOpenAI
import redis.asyncio as redis
from redis.exceptions import RedisError
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500
GREETING_PREFIXES = ("hi", "hello", "hey", "yo", "sup", "what's up", "whats up", "good morning", "good afternoon", "good evening")
MENTION_TOKEN_REGEX = re.compile(r"<@!?\d+>")
LEADING_MENTION_REGEX = re.compile(r"^(?:\s*<@!?\d+>\s*)+")
META_IMAGE_REGEX = re.compile(
    r"<meta[^>]+(?:property|name)=[\"'](?:og:image|twitter:image)[\"'][^>]+content=[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)


CONFIG_FILENAME = os.getenv("CONFIG_FILE", "config.yaml")


def get_config(filename: Optional[str] = None) -> dict[str, Any]:
    with open(filename or CONFIG_FILENAME, encoding="utf-8") as file:
        return yaml.safe_load(file)


def clamp_01(value: float) -> float:
    return max(0.0, min(1.0, value))


def is_greeting_message(content: str) -> bool:
    normalized = " ".join(content.lower().split())
    return any(normalized.startswith(prefix) for prefix in GREETING_PREFIXES)


config = get_config()
curr_model = next(iter(config["models"]))

msg_nodes = {}
last_task_time = 0
next_response_time_by_channel = {}
cooldown_lock = asyncio.Lock()
redis_client = None
redis_client_url = None
redis_client_lock = asyncio.Lock()
afk_scheduler_task = None


def get_bot_identity() -> str:
    if discord_bot.user is not None:
        return str(discord_bot.user.id)
    if config.get("client_id"):
        return str(config["client_id"])
    return "unknown-bot"


async def get_redis_client(curr_config: dict[str, Any]):
    global redis_client, redis_client_url

    redis_url = str(curr_config.get("redis_url", "") or "").strip()
    if not redis_url:
        return None

    async with redis_client_lock:
        if redis_client is None or redis_client_url != redis_url:
            redis_client = redis.from_url(redis_url, decode_responses=True)
            redis_client_url = redis_url
            await redis_client.ping()

    return redis_client


def now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def build_afk_followup_delays(curr_config: dict[str, Any]) -> list[float]:
    base_pairs = [
        (
            max(float(curr_config.get("afk_first_followup_seconds", 600) or 0), 0),
            max(float(curr_config.get("afk_first_followup_jitter_seconds", 0) or 0), 0),
        ),
        (
            max(float(curr_config.get("afk_second_followup_seconds", 3600) or 0), 0),
            max(float(curr_config.get("afk_second_followup_jitter_seconds", 0) or 0), 0),
        ),
    ]

    delays = []
    for base_delay, jitter in base_pairs:
        if base_delay <= 0:
            continue
        if jitter > 0:
            sampled_delay = max(0.0, base_delay + random.uniform(-jitter, jitter))
        else:
            sampled_delay = base_delay
        delays.append(sampled_delay)

    return delays


def message_looks_open_ended(content: str) -> bool:
    normalized = " ".join(content.lower().split())
    if not normalized:
        return False
    if "?" in normalized:
        return True
    prefixes = (
        "anyone",
        "any thoughts",
        "what do you think",
        "what should we",
        "what's on",
        "whats on",
        "ideas",
    )
    return any(normalized.startswith(prefix) for prefix in prefixes)


def apply_generated_mention_policy(content: str, curr_config: dict[str, Any]) -> str:
    if content == "":
        return content

    cleaned = LEADING_MENTION_REGEX.sub("", content).lstrip()
    mentions_mode = str(curr_config.get("generated_user_mentions_mode", "question_only") or "question_only").lower().strip()
    if mentions_mode not in {"always", "question_only", "never"}:
        mentions_mode = "question_only"

    if mentions_mode == "never":
        cleaned = MENTION_TOKEN_REGEX.sub("", cleaned)
    elif mentions_mode == "question_only" and "?" not in cleaned:
        cleaned = MENTION_TOKEN_REGEX.sub("", cleaned)

    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def build_system_prompt_for_model(curr_config: dict[str, Any], accept_usernames: bool) -> Optional[str]:
    if not (system_prompt := curr_config.get("system_prompt")):
        return None

    now = datetime.now().astimezone()
    system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
    if accept_usernames:
        system_prompt += "\n\nUser identifiers are Discord IDs. Prefer normal feed-style replies; only use '<@ID>' when directly addressing a specific person."
    return system_prompt


def normalize_text_for_dedupe(value: str) -> str:
    normalized = value.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"<@!?\d+>", "<@user>", normalized)
    normalized = re.sub(r"[^a-z0-9<@> ?!.,'-]", "", normalized)
    return normalized


def sanitize_proactive_message(raw_text: str, max_chars: int, target_user_id: Optional[int]) -> str:
    text = (raw_text or "").strip()
    text = re.sub(r"@everyone|@here|<@&\d+>", "", text)
    if target_user_id is None:
        text = re.sub(r"<@!?\d+>", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    if max_chars > 0:
        text = text[:max_chars].rstrip()
    return text


async def fetch_image_asset_from_url(url: str) -> Optional[tuple[bytes, str]]:
    try:
        response = await httpx_client.get(url, follow_redirects=True, timeout=20)
    except Exception:
        return None

    content_type = (response.headers.get("content-type") or "").lower()
    if content_type.startswith("image/"):
        return response.content, content_type

    if "text/html" in content_type:
        match = META_IMAGE_REGEX.search(response.text or "")
        if match:
            candidate_url = htmllib.unescape(match.group(1)).strip()
            if candidate_url:
                candidate_url = urljoin(str(response.url), candidate_url)
                try:
                    image_response = await httpx_client.get(candidate_url, follow_redirects=True, timeout=20)
                except Exception:
                    return None

                image_content_type = (image_response.headers.get("content-type") or "").lower()
                if image_content_type.startswith("image/"):
                    return image_response.content, image_content_type

    return None


async def maybe_send_curated_gif_reply(
    trigger_msg: discord.Message,
    reply_target: discord.Message,
    curr_config: dict[str, Any],
    redis_client_instance,
    reference_text: str,
) -> None:
    if trigger_msg.channel.type == discord.ChannelType.private:
        return
    if not curr_config.get("gif_replies_enabled", False):
        return

    gif_urls = [str(url).strip() for url in (curr_config.get("gif_reply_urls") or []) if str(url).strip()]
    if gif_urls == []:
        return

    gif_reply_chance = clamp_01(float(curr_config.get("gif_reply_chance", 0.1) or 0.1))
    if random.random() > gif_reply_chance:
        return

    keyword_filters = [str(word).lower().strip() for word in (curr_config.get("gif_reply_keyword_filters") or []) if str(word).strip()]
    if keyword_filters:
        searchable = f"{trigger_msg.content}\n{reference_text}".lower()
        if not any(keyword in searchable for keyword in keyword_filters):
            return

    gif_reply_cooldown_seconds = max(int(float(curr_config.get("gif_reply_cooldown_seconds", 180) or 180)), 0)
    gif_reply_max_per_hour_per_channel = max(int(float(curr_config.get("gif_reply_max_per_hour_per_channel", 4) or 4)), 0)
    now = now_ts()
    channel_id = trigger_msg.channel.id

    if redis_client_instance is not None:
        last_gif_ts_key = f"llmcord:gif:last_ts:{channel_id}"
        if gif_reply_cooldown_seconds > 0:
            last_gif_ts = float(await redis_client_instance.get(last_gif_ts_key) or 0)
            if now - last_gif_ts < gif_reply_cooldown_seconds:
                return

        hour_bucket = int(now // 3600)
        per_hour_key = f"llmcord:gif:hour_count:{channel_id}:{hour_bucket}"
        if gif_reply_max_per_hour_per_channel > 0:
            hour_count = int(await redis_client_instance.get(per_hour_key) or 0)
            if hour_count >= gif_reply_max_per_hour_per_channel:
                return

    gif_payload = None
    randomized_urls = gif_urls[:]
    random.shuffle(randomized_urls)
    for candidate_url in randomized_urls:
        gif_payload = await fetch_image_asset_from_url(candidate_url)
        if gif_payload is not None:
            break

    if gif_payload is None:
        return

    image_bytes, image_content_type = gif_payload
    if "gif" in image_content_type:
        extension = "gif"
    elif "webp" in image_content_type:
        extension = "webp"
    elif "png" in image_content_type:
        extension = "png"
    elif "jpeg" in image_content_type or "jpg" in image_content_type:
        extension = "jpg"
    else:
        extension = "img"

    try:
        file = discord.File(io.BytesIO(image_bytes), filename=f"reaction.{extension}")
        await reply_target.reply(file=file, silent=True, mention_author=False)
    except (discord.Forbidden, discord.NotFound, discord.HTTPException):
        return

    if redis_client_instance is not None:
        await redis_client_instance.set(f"llmcord:gif:last_ts:{channel_id}", now, ex=max(gif_reply_cooldown_seconds, 3600))
        hour_bucket = int(now // 3600)
        per_hour_key = f"llmcord:gif:hour_count:{channel_id}:{hour_bucket}"
        hour_count = await redis_client_instance.incr(per_hour_key)
        if hour_count == 1:
            await redis_client_instance.expire(per_hour_key, 2 * 3600)


def in_quiet_hours(curr_config: dict[str, Any]) -> bool:
    if not curr_config.get("quiet_hours_enabled", False):
        return False

    tz_name = str(curr_config.get("quiet_hours_timezone", "UTC") or "UTC")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc

    start_hour = int(curr_config.get("quiet_hours_start_hour", 23) or 23) % 24
    end_hour = int(curr_config.get("quiet_hours_end_hour", 8) or 8) % 24
    hour = datetime.now(timezone.utc).astimezone(tz).hour

    if start_hour == end_hour:
        return False
    if start_hour < end_hour:
        return start_hour <= hour < end_hour
    return hour >= start_hour or hour < end_hour


def get_timezone(curr_config: dict[str, Any]):
    tz_name = str(curr_config.get("quiet_hours_timezone", "UTC") or "UTC")
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return timezone.utc


def today_key(curr_config: dict[str, Any]) -> str:
    return datetime.now(timezone.utc).astimezone(get_timezone(curr_config)).strftime("%Y-%m-%d")


async def generate_proactive_starter_text(curr_config: dict[str, Any], target_user_id: Optional[int], max_chars: int) -> str:
    provider_slash_model = curr_model
    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)
    provider_config = curr_config["providers"][provider]
    openai_client = AsyncOpenAI(base_url=provider_config["base_url"], api_key=provider_config.get("api_key", "sk-no-key-required"))

    model_parameters = curr_config["models"].get(provider_slash_model, None)
    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    accept_usernames = any(provider_slash_model.lower().startswith(x) for x in PROVIDERS_SUPPORTING_USERNAMES)
    system_prompt = build_system_prompt_for_model(curr_config, accept_usernames)

    mention_instruction = (
        f"Start with '<@{target_user_id}>' and ask that person directly."
        if target_user_id is not None
        else "Do not include any user mentions."
    )
    user_instruction = (
        "Write exactly one short proactive Discord message to re-start casual chat naturally. "
        f"Keep it under {max_chars} characters, one sentence, conversational, and not repetitive. "
        "No lists, no markdown, no hashtags, no @everyone, no @here, no role mentions. "
        f"{mention_instruction}"
    )

    messages: list[dict[str, str]] = [dict(role="user", content=user_instruction)]
    if system_prompt:
        messages.append(dict(role="system", content=system_prompt))

    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=messages[::-1],
            stream=False,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
        )
    except Exception:
        logging.exception("Error generating proactive starter text")
        return ""

    content = ""
    if response.choices:
        message_content = response.choices[0].message.content
        if isinstance(message_content, str):
            content = message_content
        elif isinstance(message_content, list):
            content = "".join(
                part.get("text", "")
                for part in message_content
                if isinstance(part, dict)
            )

    content = sanitize_proactive_message(content, max_chars, target_user_id)
    if target_user_id is not None and content != "" and "<@" not in content:
        content = f"<@{target_user_id}> {content}".strip()
        content = sanitize_proactive_message(content, max_chars, target_user_id)
    return content


async def generate_afk_followup_text(curr_config: dict[str, Any], max_chars: int) -> str:
    provider_slash_model = curr_model
    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)
    provider_config = curr_config["providers"][provider]
    openai_client = AsyncOpenAI(base_url=provider_config["base_url"], api_key=provider_config.get("api_key", "sk-no-key-required"))

    model_parameters = curr_config["models"].get(provider_slash_model, None)
    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    accept_usernames = any(provider_slash_model.lower().startswith(x) for x in PROVIDERS_SUPPORTING_USERNAMES)
    system_prompt = build_system_prompt_for_model(curr_config, accept_usernames)

    user_instruction = (
        "Write exactly one short, natural follow-up Discord reply for a conversation that has gone quiet. "
        f"Keep it under {max_chars} characters, one sentence, low-pressure, and conversational. "
        "Do not include user mentions, no lists, no markdown, no hashtags, and no @everyone/@here."
    )

    messages: list[dict[str, str]] = [dict(role="user", content=user_instruction)]
    if system_prompt:
        messages.append(dict(role="system", content=system_prompt))

    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=messages[::-1],
            stream=False,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
        )
    except Exception:
        logging.exception("Error generating AFK follow-up text")
        return ""

    content = ""
    if response.choices:
        message_content = response.choices[0].message.content
        if isinstance(message_content, str):
            content = message_content
        elif isinstance(message_content, list):
            content = "".join(
                part.get("text", "")
                for part in message_content
                if isinstance(part, dict)
            )

    content = sanitize_proactive_message(content, max_chars=max_chars, target_user_id=None)
    return content


async def maybe_send_proactive_starter(curr_config: dict[str, Any], redis_client_instance, channel_id: int) -> None:
    if not curr_config.get("proactive_starters_enabled", False):
        return
    if in_quiet_hours(curr_config) and curr_config.get("proactive_respect_quiet_hours", True):
        return

    now = now_ts()
    human_idle_seconds = max(float(curr_config.get("proactive_idle_human_seconds", 1800) or 0), 0)
    channel_idle_seconds = max(float(curr_config.get("proactive_idle_channel_seconds", 900) or 0), 0)
    chance = clamp_01(float(curr_config.get("proactive_starter_chance", 0.35) or 0.35))
    max_daily = max(int(float(curr_config.get("proactive_max_per_day_per_channel", 6) or 6)), 0)
    claim_ttl_seconds = max(int(float(curr_config.get("proactive_claim_ttl_seconds", 45) or 45)), 10)
    mention_enabled = bool(curr_config.get("proactive_mention_enabled", False))
    mention_chance = clamp_01(float(curr_config.get("proactive_mention_chance", 0.5) or 0.5))
    mention_recent_user_seconds = max(float(curr_config.get("proactive_mention_recent_user_seconds", 172800) or 0), 0)
    mention_max_per_user_per_day = max(int(float(curr_config.get("proactive_mention_max_per_user_per_day", 1) or 1)), 0)

    last_human_ts = float(await redis_client_instance.get(f"llmcord:afk:last_human_ts:{channel_id}") or 0)
    last_any_msg_ts = float(await redis_client_instance.get(f"llmcord:channel:last_message_ts:{channel_id}") or 0)
    if human_idle_seconds > 0 and now - last_human_ts < human_idle_seconds:
        return
    if channel_idle_seconds > 0 and now - last_any_msg_ts < channel_idle_seconds:
        return
    if await redis_client_instance.get(f"llmcord:active_responder:{channel_id}"):
        return
    if random.random() > chance:
        return

    day_key = today_key(curr_config)
    daily_key = f"llmcord:proactive:daily:{channel_id}:{day_key}"
    if max_daily > 0 and int(await redis_client_instance.get(daily_key) or 0) >= max_daily:
        return

    claim_key = f"llmcord:proactive:claim:{channel_id}"
    claimed = await redis_client_instance.set(claim_key, get_bot_identity(), ex=claim_ttl_seconds, nx=True)
    if not claimed:
        return

    channel = discord_bot.get_channel(channel_id)
    if channel is None:
        try:
            channel = await discord_bot.fetch_channel(channel_id)
        except (discord.NotFound, discord.HTTPException):
            channel = None
    if channel is None:
        return

    target_user_id = None
    if mention_enabled and mention_max_per_user_per_day > 0 and random.random() <= mention_chance:
        recent_humans_key = f"llmcord:channel:recent_humans:{channel_id}"
        min_recent_ts = now - mention_recent_user_seconds
        candidate_user_ids = await redis_client_instance.zrevrangebyscore(recent_humans_key, max="+inf", min=min_recent_ts, start=0, num=25)
        random.shuffle(candidate_user_ids)

        for candidate_user_id_str in candidate_user_ids:
            try:
                candidate_user_id = int(candidate_user_id_str)
            except ValueError:
                continue
            if discord_bot.user and candidate_user_id == discord_bot.user.id:
                continue

            mention_daily_key = f"llmcord:proactive:mention_user_daily:{channel_id}:{candidate_user_id}:{day_key}"
            mention_count = int(await redis_client_instance.get(mention_daily_key) or 0)
            if mention_count >= mention_max_per_user_per_day:
                continue

            target_user_id = candidate_user_id
            break

    retries = max(int(float(curr_config.get("proactive_generation_retries", 3) or 3)), 1)
    dedupe_window_seconds = max(int(float(curr_config.get("proactive_dedupe_window_seconds", 86400) or 86400)), 60)
    proactive_max_chars = max(int(float(curr_config.get("proactive_generated_max_chars", 180) or 180)), 40)
    proactive_hashes_key = f"llmcord:proactive:text_hashes:{channel_id}"

    await redis_client_instance.zremrangebyscore(proactive_hashes_key, min=0, max=now - dedupe_window_seconds)

    starter_text = None
    starter_text_hash = None
    for _ in range(retries):
        candidate_text = await generate_proactive_starter_text(curr_config, target_user_id, proactive_max_chars)
        if candidate_text == "":
            continue

        normalized = normalize_text_for_dedupe(candidate_text)
        if normalized == "":
            continue

        text_hash = sha256(normalized.encode("utf-8")).hexdigest()
        if await redis_client_instance.zscore(proactive_hashes_key, text_hash) is not None:
            continue

        starter_text = candidate_text
        starter_text_hash = text_hash
        break

    if starter_text is None:
        fallback_text = str(curr_config.get("proactive_fallback_starter", "quick check-in: if you're around, what's the vibe tonight?") or "")
        starter_text = sanitize_proactive_message(fallback_text, proactive_max_chars, target_user_id)
        if target_user_id is not None and "<@" not in starter_text:
            starter_text = f"<@{target_user_id}> {starter_text}".strip()
        if starter_text == "":
            return

    try:
        await channel.send(
            starter_text,
            silent=True,
            allowed_mentions=discord.AllowedMentions(users=True, roles=False, everyone=False),
        )
    except (discord.Forbidden, discord.NotFound, discord.HTTPException):
        return

    if starter_text_hash is not None:
        await redis_client_instance.zadd(proactive_hashes_key, {starter_text_hash: now})
        await redis_client_instance.expire(proactive_hashes_key, dedupe_window_seconds + 24 * 3600)

    ttl_seconds = 3 * 24 * 3600
    daily_count = await redis_client_instance.incr(daily_key)
    if daily_count == 1:
        await redis_client_instance.expire(daily_key, ttl_seconds)
    await redis_client_instance.set(f"llmcord:channel:last_message_ts:{channel_id}", now, ex=7 * 24 * 3600)
    if target_user_id is not None and mention_max_per_user_per_day > 0:
        mention_daily_key = f"llmcord:proactive:mention_user_daily:{channel_id}:{target_user_id}:{day_key}"
        mention_daily_count = await redis_client_instance.incr(mention_daily_key)
        if mention_daily_count == 1:
            await redis_client_instance.expire(mention_daily_key, ttl_seconds)


async def maybe_schedule_afk_followup(new_msg: discord.Message, curr_config: dict[str, Any], redis_client_instance) -> None:
    if new_msg.channel.type == discord.ChannelType.private or new_msg.author.bot:
        return
    if not curr_config.get("afk_followup_enabled", False):
        return
    if redis_client_instance is None:
        return
    if curr_config.get("afk_open_question_only", True) and not message_looks_open_ended(new_msg.content):
        return

    delays = build_afk_followup_delays(curr_config)
    if delays == []:
        return

    max_followups = max(int(float(curr_config.get("afk_max_followups_per_message", 2) or 2)), 1)
    max_followups = min(max_followups, len(delays))
    if max_followups <= 0:
        return

    source_human_ts = now_ts()
    channel_id = new_msg.channel.id
    source_message_id = new_msg.id
    item_id = f"{channel_id}:{source_message_id}"
    seed_ttl_seconds = int(max(delays[:max_followups]) + 24 * 3600)
    seed_key = f"llmcord:afk:seed:{item_id}"
    schedule_zset_key = "llmcord:afk:schedule"
    item_hash_key = f"llmcord:afk:item:{item_id}"
    channel_last_human_ts_key = f"llmcord:afk:last_human_ts:{channel_id}"

    await redis_client_instance.set(channel_last_human_ts_key, source_human_ts, ex=7 * 24 * 3600)
    seeded = await redis_client_instance.set(seed_key, get_bot_identity(), ex=seed_ttl_seconds, nx=True)
    if not seeded:
        return

    await redis_client_instance.hset(
        item_hash_key,
        mapping=dict(
            channel_id=str(channel_id),
            source_message_id=str(source_message_id),
            source_human_ts=str(source_human_ts),
            followups_sent="0",
            next_followup_index="0",
            max_followups=str(max_followups),
            delays_csv=",".join(str(delay) for delay in delays),
            created_by=get_bot_identity(),
        ),
    )
    await redis_client_instance.expire(item_hash_key, seed_ttl_seconds)
    await redis_client_instance.zadd(schedule_zset_key, {item_id: source_human_ts + delays[0]})


async def process_afk_followup_item(item_id: str, curr_config: dict[str, Any], redis_client_instance) -> None:
    item_hash_key = f"llmcord:afk:item:{item_id}"
    claim_key = f"llmcord:afk:claim:{item_id}"
    schedule_zset_key = "llmcord:afk:schedule"

    claimed = await redis_client_instance.set(claim_key, get_bot_identity(), ex=45, nx=True)
    if not claimed:
        return

    item = await redis_client_instance.hgetall(item_hash_key)
    if item == {}:
        await redis_client_instance.zrem(schedule_zset_key, item_id)
        return

    channel_id = int(item["channel_id"])
    source_message_id = int(item["source_message_id"])
    source_human_ts = float(item["source_human_ts"])
    followups_sent = int(item.get("followups_sent", "0"))
    next_followup_index = int(item.get("next_followup_index", "0"))
    max_followups = int(item.get("max_followups", "1"))
    delays = [float(part) for part in item.get("delays_csv", "").split(",") if part]
    if delays == [] or next_followup_index >= len(delays) or followups_sent >= max_followups:
        await redis_client_instance.zrem(schedule_zset_key, item_id)
        await redis_client_instance.delete(item_hash_key)
        return

    if in_quiet_hours(curr_config):
        await redis_client_instance.zadd(schedule_zset_key, {item_id: now_ts() + 300})
        return

    if curr_config.get("afk_cancel_on_any_human_message", True):
        channel_last_human_ts_key = f"llmcord:afk:last_human_ts:{channel_id}"
        latest_human_ts = float(await redis_client_instance.get(channel_last_human_ts_key) or 0)
        if latest_human_ts > source_human_ts:
            await redis_client_instance.zrem(schedule_zset_key, item_id)
            await redis_client_instance.delete(item_hash_key)
            return

    afk_followup_chance = clamp_01(float(curr_config.get("afk_followup_chance", 0.5) or 0.5))
    if random.random() > afk_followup_chance:
        next_followup_index += 1
        if next_followup_index >= min(max_followups, len(delays)):
            await redis_client_instance.zrem(schedule_zset_key, item_id)
            await redis_client_instance.delete(item_hash_key)
        else:
            await redis_client_instance.hset(item_hash_key, mapping=dict(next_followup_index=str(next_followup_index)))
            await redis_client_instance.zadd(schedule_zset_key, {item_id: source_human_ts + delays[next_followup_index]})
        return

    channel = discord_bot.get_channel(channel_id)
    if channel is None:
        try:
            channel = await discord_bot.fetch_channel(channel_id)
        except (discord.NotFound, discord.HTTPException):
            channel = None
    if channel is None:
        await redis_client_instance.zadd(schedule_zset_key, {item_id: now_ts() + 120})
        return

    try:
        source_msg = await channel.fetch_message(source_message_id)
    except (discord.NotFound, discord.HTTPException):
        source_msg = None
    if source_msg is None:
        await redis_client_instance.zrem(schedule_zset_key, item_id)
        await redis_client_instance.delete(item_hash_key)
        return

    afk_followup_retries = max(int(float(curr_config.get("afk_followup_generation_retries", 2) or 2)), 1)
    afk_followup_max_chars = max(int(float(curr_config.get("afk_followup_max_chars", 140) or 140)), 40)
    followup_text = ""
    for _ in range(afk_followup_retries):
        followup_text = await generate_afk_followup_text(curr_config, max_chars=afk_followup_max_chars)
        if followup_text != "":
            break
    if followup_text == "":
        followup_text = str(
            curr_config.get("afk_followup_fallback_text", "no rush. drop your thoughts when you're back.") or ""
        ).strip()[:afk_followup_max_chars]
    if followup_text == "":
        await redis_client_instance.zrem(schedule_zset_key, item_id)
        await redis_client_instance.delete(item_hash_key)
        return

    try:
        await source_msg.reply(followup_text, silent=True)
    except (discord.NotFound, discord.HTTPException):
        await redis_client_instance.zadd(schedule_zset_key, {item_id: now_ts() + 120})
        return

    followups_sent += 1
    next_followup_index += 1
    if followups_sent >= max_followups or next_followup_index >= len(delays):
        await redis_client_instance.zrem(schedule_zset_key, item_id)
        await redis_client_instance.delete(item_hash_key)
    else:
        await redis_client_instance.hset(
            item_hash_key,
            mapping=dict(followups_sent=str(followups_sent), next_followup_index=str(next_followup_index)),
        )
        await redis_client_instance.zadd(schedule_zset_key, {item_id: source_human_ts + delays[next_followup_index]})


async def afk_followup_scheduler_loop() -> None:
    while True:
        try:
            curr_config = await asyncio.to_thread(get_config)
            poll_seconds = max(float(curr_config.get("afk_scheduler_poll_seconds", 5) or 5), 1)
            redis_client_instance = await get_redis_client(curr_config)

            if redis_client_instance is None or not curr_config.get("afk_followup_enabled", False):
                await asyncio.sleep(poll_seconds)
                continue

            due_item_ids = await redis_client_instance.zrangebyscore("llmcord:afk:schedule", min=0, max=now_ts(), start=0, num=10)
            for item_id in due_item_ids:
                await process_afk_followup_item(item_id, curr_config, redis_client_instance)

            if curr_config.get("proactive_starters_enabled", False):
                configured_channel_ids = set(curr_config.get("proactive_channel_ids") or [])
                if configured_channel_ids == set():
                    observed_channel_ids = {int(x) for x in await redis_client_instance.smembers("llmcord:observed_channels")}
                    candidate_channel_ids = observed_channel_ids
                else:
                    candidate_channel_ids = configured_channel_ids

                for channel_id in sorted(candidate_channel_ids):
                    await maybe_send_proactive_starter(curr_config, redis_client_instance, int(channel_id))

            await asyncio.sleep(poll_seconds + random.uniform(0, 0.5))
        except asyncio.CancelledError:
            return
        except Exception:
            logging.exception("Error in AFK follow-up scheduler")
            await asyncio.sleep(3)

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
activity = discord.CustomActivity(name=(config.get("status_message") or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()


@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if user_is_admin := interaction.user.id in config["permissions"]["users"]["admin_ids"]:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config

    if curr_str == "":
        config = await asyncio.to_thread(get_config)

    choices = [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()]

    return choices[:25]


@discord_bot.event
async def on_ready() -> None:
    global afk_scheduler_task

    if client_id := config.get("client_id"):
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")

    await discord_bot.tree.sync()

    if afk_scheduler_task is None or afk_scheduler_task.done():
        afk_scheduler_task = asyncio.create_task(afk_followup_scheduler_loop())


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time, next_response_time_by_channel

    is_dm = new_msg.channel.type == discord.ChannelType.private

    config = await asyncio.to_thread(get_config)

    if config.get("debug_log_all_messages", False):
        is_mentioned = bool(discord_bot.user and discord_bot.user in new_msg.mentions)
        logging.info(
            "Event seen: author_id=%s author_is_bot=%s channel_id=%s is_dm=%s mentioned=%s",
            new_msg.author.id,
            new_msg.author.bot,
            new_msg.channel.id,
            is_dm,
            is_mentioned,
        )

    # Ignore only this bot's own messages so different bot users can still collaborate.
    if discord_bot.user and new_msg.author.id == discord_bot.user.id:
        return

    has_direct_mention = bool(discord_bot.user and discord_bot.user in new_msg.mentions)
    treat_everyone_as_directed = bool(config.get("treat_everyone_as_directed", False))
    has_everyone_directive = bool(getattr(new_msg, "mention_everyone", False) and treat_everyone_as_directed)

    async def get_reply_parent_message() -> Optional[discord.Message]:
        if not (parent_msg_id := getattr(new_msg.reference, "message_id", None)):
            return None

        try:
            return new_msg.reference.cached_message or await new_msg.channel.fetch_message(parent_msg_id)
        except (discord.NotFound, discord.HTTPException):
            return None

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    parent_msg = await get_reply_parent_message() if not is_dm else None
    is_direct_reply = bool(parent_msg and discord_bot.user and parent_msg.author.id == discord_bot.user.id)
    is_reply_to_other_bot = bool(parent_msg and parent_msg.author.bot and discord_bot.user and parent_msg.author.id != discord_bot.user.id)
    is_directed_message = has_direct_mention or is_direct_reply or has_everyone_directive
    should_process = True

    strict_reply_targeting = bool(config.get("strict_reply_targeting", True))
    if strict_reply_targeting and is_reply_to_other_bot and not (has_direct_mention or has_everyone_directive):
        return

    direct_mention_retry_enabled = bool(config.get("direct_mention_retry_enabled", True))
    remaining_direct_wait_seconds = max(float(config.get("direct_mention_max_wait_seconds", 7200) or 0), 0) if is_directed_message else 0

    if not is_dm:
        autonomous_bot_only_mode = config.get("autonomous_bot_only_mode", False)
        autonomous_channel_ids = set(config.get("autonomous_channel_ids", []))
        in_autonomous_scope = autonomous_bot_only_mode and (not autonomous_channel_ids or any(id in autonomous_channel_ids for id in channel_ids))

        if in_autonomous_scope:
            # Explicit @mentions/replies should always trigger in autonomous channels.
            if has_direct_mention or is_direct_reply:
                should_process = True
            else:
                # Otherwise, use per-bot probabilistic participation to avoid dogpiles.
                group_response_chance = clamp_01(float(config.get("group_response_chance", 1.0) or 1.0))
                greeting_response_chance = config.get("greeting_response_chance")
                response_priority_weight = max(float(config.get("response_priority_weight", 1.0) or 1.0), 0.0)

                effective_chance = group_response_chance
                if greeting_response_chance is not None and is_greeting_message(new_msg.content):
                    effective_chance = clamp_01(float(greeting_response_chance or 0.0))

                effective_chance = clamp_01(effective_chance * response_priority_weight)
                should_process = random.random() < effective_chance
        else:
            should_process = has_direct_mention or is_direct_reply

    allow_dms = config.get("allow_dms", True)

    permissions = config["permissions"]

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    redis_client_instance = None
    if not is_dm:
        try:
            redis_client_instance = await get_redis_client(config)
        except RedisError:
            logging.exception("Redis unavailable, continuing without distributed coordination")
            redis_client_instance = None

        if redis_client_instance is not None:
            channel_id = new_msg.channel.id
            now = now_ts()
            await redis_client_instance.sadd("llmcord:observed_channels", str(channel_id))
            await redis_client_instance.set(f"llmcord:channel:last_message_ts:{channel_id}", now, ex=7 * 24 * 3600)

            if not new_msg.author.bot:
                await redis_client_instance.set(f"llmcord:afk:last_human_ts:{channel_id}", now, ex=7 * 24 * 3600)
                recent_humans_key = f"llmcord:channel:recent_humans:{channel_id}"
                await redis_client_instance.zadd(recent_humans_key, {str(new_msg.author.id): now})
                await redis_client_instance.expire(recent_humans_key, 14 * 24 * 3600)
                await maybe_schedule_afk_followup(new_msg, config, redis_client_instance)

        if not should_process:
            return

    reaction_delay_base_seconds = max(float(config.get("reaction_delay_base_seconds", 0) or 0), 0)
    reaction_delay_jitter_seconds = max(float(config.get("reaction_delay_jitter_seconds", 0) or 0), 0)
    if not is_dm and (reaction_delay_base_seconds > 0 or reaction_delay_jitter_seconds > 0):
        await asyncio.sleep(reaction_delay_base_seconds + random.uniform(0, reaction_delay_jitter_seconds))

    global_channel_cooldown_seconds = max(float(config.get("global_channel_cooldown_seconds", 0) or 0), 0)
    global_channel_arbitration_jitter_seconds = max(float(config.get("global_channel_arbitration_jitter_seconds", 0) or 0), 0)
    if not is_dm and (global_channel_cooldown_seconds > 0 or global_channel_arbitration_jitter_seconds > 0):
        while True:
            if global_channel_arbitration_jitter_seconds > 0:
                await asyncio.sleep(random.uniform(0, global_channel_arbitration_jitter_seconds))

            try:
                recent_channel_msgs = [m async for m in new_msg.channel.history(limit=25)]
            except (discord.NotFound, discord.HTTPException):
                recent_channel_msgs = []

            latest_channel_msg = recent_channel_msgs[0] if recent_channel_msgs else None
            # If another bot has already replied after this trigger message, back off.
            if latest_channel_msg and latest_channel_msg.id != new_msg.id and latest_channel_msg.author.bot and latest_channel_msg.created_at > new_msg.created_at:
                return

            wait_seconds = 0.0
            if global_channel_cooldown_seconds > 0:
                latest_bot_msg = next((m for m in recent_channel_msgs if m.author.bot and m.id != new_msg.id), None)
                if latest_bot_msg:
                    age_seconds = (discord.utils.utcnow() - latest_bot_msg.created_at).total_seconds()
                    if age_seconds < global_channel_cooldown_seconds:
                        wait_seconds = global_channel_cooldown_seconds - age_seconds

            if wait_seconds <= 0:
                break

            if is_directed_message and direct_mention_retry_enabled and remaining_direct_wait_seconds > 0:
                sleep_for = min(wait_seconds, remaining_direct_wait_seconds)
                await asyncio.sleep(max(sleep_for, 0))
                remaining_direct_wait_seconds = max(0.0, remaining_direct_wait_seconds - sleep_for)
                if remaining_direct_wait_seconds <= 0:
                    return
                continue

            return

    response_cooldown_seconds = max(float(config.get("response_cooldown_seconds", 0) or 0), 0)
    response_cooldown_jitter_seconds = max(float(config.get("response_cooldown_jitter_seconds", 0) or 0), 0)
    if response_cooldown_seconds > 0 or response_cooldown_jitter_seconds > 0:
        channel_id = new_msg.channel.id
        while True:
            curr_time = monotonic()
            async with cooldown_lock:
                next_available_time = next_response_time_by_channel.get(channel_id, 0)
                if curr_time >= next_available_time:
                    next_response_time_by_channel[channel_id] = curr_time + response_cooldown_seconds + random.uniform(0, response_cooldown_jitter_seconds)
                    break
                wait_seconds = next_available_time - curr_time

            if is_directed_message and direct_mention_retry_enabled and remaining_direct_wait_seconds > 0:
                sleep_for = min(wait_seconds, remaining_direct_wait_seconds)
                await asyncio.sleep(max(sleep_for, 0))
                remaining_direct_wait_seconds = max(0.0, remaining_direct_wait_seconds - sleep_for)
                if remaining_direct_wait_seconds <= 0:
                    return
                continue

            return

    redis_claim_state = None
    if not is_dm:
        if redis_client_instance is not None:
            channel_id = new_msg.channel.id
            source_message_id = new_msg.id
            bot_identity = get_bot_identity()

            floor_lock_ttl_seconds = max(int(float(config.get("floor_lock_ttl_seconds", 45) or 45)), 5)
            active_responder_ttl_seconds = max(int(float(config.get("active_responder_ttl_seconds", 90) or 90)), 10)
            source_message_window_seconds = max(int(float(config.get("source_message_window_seconds", 180) or 180)), 30)
            max_responses_per_source_message = max(int(float(config.get("max_responses_per_source_message", 2) or 2)), 1)
            followup_response_chance = clamp_01(float(config.get("followup_response_chance", 0.15) or 0.15))
            floor_lock_key = f"llmcord:floor:{channel_id}:{source_message_id}"
            source_count_key = f"llmcord:source_count:{channel_id}:{source_message_id}"
            active_responder_key = f"llmcord:active_responder:{channel_id}"

            claimed_floor = await redis_client_instance.set(floor_lock_key, bot_identity, ex=floor_lock_ttl_seconds, nx=True)
            if not claimed_floor:
                return

            response_index = await redis_client_instance.incr(source_count_key)
            if response_index == 1:
                await redis_client_instance.expire(source_count_key, source_message_window_seconds)

            if response_index > max_responses_per_source_message and not is_directed_message:
                await redis_client_instance.decr(source_count_key)
                return

            if response_index > 1 and not is_directed_message and random.random() >= followup_response_chance:
                await redis_client_instance.decr(source_count_key)
                return

            claimed_active = await redis_client_instance.set(active_responder_key, bot_identity, ex=active_responder_ttl_seconds, nx=True)
            if not claimed_active:
                if not (is_directed_message and direct_mention_retry_enabled):
                    await redis_client_instance.decr(source_count_key)
                    return
                active_responder_key = None

            redis_claim_state = dict(
                client=redis_client_instance,
                source_count_key=source_count_key,
                active_responder_key=active_responder_key,
                bot_identity=bot_identity,
            )

    provider_slash_model = curr_model
    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)

    provider_config = config["providers"][provider]

    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    model_parameters = config["models"].get(provider_slash_model, None)

    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(provider_slash_model.lower().startswith(x) for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)
    max_response_chars = max(int(config.get("max_response_chars", 0) or 0), 0)

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [component.content for component in curr_msg.components if component.type == discord.ComponentType.text_display]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference == None
                        and discord_bot.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id != None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    if system_prompt := build_system_prompt_for_model(config, accept_usernames):
        messages.append(dict(role="system", content=system_prompt))

    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []

    openai_kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)

    if use_plain_responses := config.get("use_plain_responses", False):
        max_message_length = 4000
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        embed = discord.Embed.from_dict(dict(fields=[dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)]))

    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        reply_kwargs.setdefault("mention_author", False)
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)

        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()

    try:
        async with new_msg.channel.typing():
            async for chunk in await openai_client.chat.completions.create(**openai_kwargs):
                if finish_reason != None:
                    break

                if not (choice := chunk.choices[0] if chunk.choices else None):
                    continue

                finish_reason = choice.finish_reason

                prev_content = curr_content or ""
                curr_content = choice.delta.content or ""

                new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                if response_contents == [] and new_content == "":
                    continue

                # Hard-cap assistant output size when configured.
                if max_response_chars > 0:
                    total_chars = sum(len(content) for content in response_contents)
                    remaining_chars = max_response_chars - total_chars
                    if remaining_chars <= 0:
                        finish_reason = "length"
                        break
                    if len(new_content) > remaining_chars:
                        new_content = new_content[:remaining_chars]
                        finish_reason = "length"

                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")

                response_contents[-1] += new_content

                if not use_plain_responses:
                    time_delta = datetime.now().timestamp() - last_task_time

                    ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                    msg_split_incoming = finish_reason == None and len(response_contents[-1] + curr_content) > max_message_length
                    is_final_edit = finish_reason != None or msg_split_incoming
                    is_good_finish = finish_reason != None and finish_reason.lower() in ("stop", "end_turn")

                    if start_next_msg or ready_to_edit or is_final_edit:
                        visible_content = apply_generated_mention_policy(response_contents[-1], config)
                        embed.description = visible_content if is_final_edit else (visible_content + STREAMING_INDICATOR)
                        embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                        if start_next_msg:
                            await reply_helper(embed=embed, silent=True)
                        else:
                            await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                            await response_msgs[-1].edit(embed=embed)

                        last_task_time = datetime.now().timestamp()

            if use_plain_responses:
                for content in response_contents:
                    visible_content = apply_generated_mention_policy(content, config)
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=visible_content)))

    except Exception:
        logging.exception("Error while generating response")
    finally:
        if redis_claim_state is not None:
            redis_client_instance = redis_claim_state["client"]
            active_responder_key = redis_claim_state["active_responder_key"]
            source_count_key = redis_claim_state["source_count_key"]
            bot_identity = redis_claim_state["bot_identity"]
            # Release the typing floor only if we still own it.
            if active_responder_key is not None:
                try:
                    if await redis_client_instance.get(active_responder_key) == bot_identity:
                        await redis_client_instance.delete(active_responder_key)
                except RedisError:
                    logging.exception("Failed to release Redis active responder key")
            # If generation produced no messages, roll back this reserved response slot.
            if response_msgs == []:
                try:
                    await redis_client_instance.decr(source_count_key)
                except RedisError:
                    logging.exception("Failed to roll back Redis source response count")

    final_text = apply_generated_mention_policy("".join(response_contents), config)
    if response_msgs:
        await maybe_send_curated_gif_reply(
            trigger_msg=new_msg,
            reply_target=response_msgs[-1],
            curr_config=config,
            redis_client_instance=redis_client_instance,
            reference_text=final_text,
        )

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = final_text
        msg_nodes[response_msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def main() -> None:
    await discord_bot.start(config["bot_token"])


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
