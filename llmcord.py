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

    cleaned = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n+", " ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    if curr_config.get("discord_chat_style_enabled", True):
        max_sentences = max(int(curr_config.get("discord_chat_max_sentences", 2) or 2), 1)
        sentence_parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
        if sentence_parts:
            cleaned = " ".join(sentence_parts[:max_sentences]).strip()

        style_max_chars = max(int(curr_config.get("discord_chat_style_max_chars", 220) or 0), 0)
        if style_max_chars > 0 and len(cleaned) > style_max_chars:
            cleaned = cleaned[:style_max_chars].rstrip()

    return cleaned


def deterministic_fraction(seed: str) -> float:
    digest = sha256(seed.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16)
    return value / 0xFFFFFFFF


def deterministic_jittered_threshold(base_seconds: float, jitter_seconds: float, seed: str) -> float:
    if base_seconds <= 0:
        return 0.0
    jitter = max(jitter_seconds, 0.0)
    if jitter == 0:
        return base_seconds
    # Deterministic +/- jitter so all bot instances use the same threshold per channel/window.
    offset = (deterministic_fraction(seed) * 2.0 - 1.0) * jitter
    return max(base_seconds + offset, 0.0)


def parse_relationship_weights(curr_config: dict[str, Any]) -> tuple[dict[tuple[str, str], float], set[str]]:
    raw = curr_config.get("proactive_bot_to_bot_relationship_weights") or {}
    if not isinstance(raw, dict):
        return {}, set()

    weights: dict[tuple[str, str], float] = {}
    identities: set[str] = set()
    for raw_key, raw_value in raw.items():
        key = str(raw_key).strip().lower()
        try:
            weight = max(float(raw_value), 0.0)
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue

        if "<->" in key:
            left, right = [part.strip() for part in key.split("<->", 1)]
            if left and right:
                weights[(left, right)] = weight
                weights[(right, left)] = weight
                identities.update((left, right))
            continue

        if "->" in key:
            left, right = [part.strip() for part in key.split("->", 1)]
            if left and right:
                weights[(left, right)] = weight
                identities.update((left, right))
            continue

        if ":" in key:
            left, right = [part.strip() for part in key.split(":", 1)]
            if left and right:
                weights[(left, right)] = weight
                weights[(right, left)] = weight
                identities.update((left, right))
            continue

    return weights, identities


def resolve_relationship_identity(value: str, known_identities: set[str]) -> str:
    normalized = re.sub(r"<@!?\d+>", "", (value or "").strip().lower())
    normalized = re.sub(r"[^a-z0-9 _-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return ""

    tokens = normalized.split(" ")
    for token in tokens:
        if token in known_identities:
            return token
    if normalized in known_identities:
        return normalized
    # Fallback: use first token as lightweight identity hint.
    return tokens[0] if tokens else normalized


def weighted_choice(candidates: list[tuple[Any, float]]) -> Optional[Any]:
    viable = [(item, weight) for item, weight in candidates if weight > 0]
    if not viable:
        return None
    total_weight = sum(weight for _, weight in viable)
    if total_weight <= 0:
        return None
    cursor = random.uniform(0, total_weight)
    running = 0.0
    for item, weight in viable:
        running += weight
        if cursor <= running:
            return item
    return viable[-1][0]


def maybe_add_profile_filler(text: str, curr_config: dict[str, Any], profile: str) -> str:
    fillers = [str(item).strip() for item in (curr_config.get("persona_preferred_fillers") or []) if str(item).strip()]
    if not fillers:
        return text
    lowered = text.lower()
    if any(filler.lower() in lowered for filler in fillers):
        return text

    idx_seed = f"{profile}:{text}:filler"
    idx = int(sha256(idx_seed.encode("utf-8")).hexdigest(), 16) % len(fillers)
    return f"{fillers[idx]}, {text}".strip()


def apply_persona_blocklist_and_length(text: str, curr_config: dict[str, Any]) -> str:
    cleaned = text
    for phrase in (curr_config.get("persona_blocklist_phrases") or []):
        phrase_text = str(phrase).strip()
        if phrase_text:
            cleaned = re.sub(re.escape(phrase_text), "", cleaned, flags=re.IGNORECASE)

    max_word_length = max(int(curr_config.get("persona_max_word_length", 0) or 0), 0)
    if max_word_length > 0:
        cleaned = re.sub(
            rf"\b[a-zA-Z]{{{max_word_length + 1},}}\b",
            "thing",
            cleaned,
        )
    return cleaned


def enforce_kevin_speech_style(content: str, curr_config: dict[str, Any]) -> str:
    cleaned = " ".join((content or "").strip().split())
    if cleaned == "":
        return cleaned

    word_replacements = {
        "optimize": "fix",
        "optimization": "fix",
        "strategic": "solid",
        "strategy": "plan",
        "nuanced": "tricky",
        "sophisticated": "fancy",
        "complex": "messy",
        "analyze": "check",
        "analysis": "check",
        "theoretical": "on paper",
        "leverage": "use",
        "efficient": "fast",
        "efficiently": "fast",
        "terminal": "screen",
        "console": "screen",
        "interface": "thing",
        "dashboard": "screen",
        "configured": "set up",
        "configuration": "setup",
        "connected": "hooked up",
        "connection": "hook up",
        "booted": "turned on",
        "rebooted": "turned back on",
        "monitor": "screen",
        "monitoring": "watching",
        "server": "computer",
        "database": "computer stuff",
        "repository": "folder",
        "deployed": "put up",
        "deployment": "setup",
        "protocol": "thing",
        "architecture": "layout",
        "infrastructure": "setup",
        "implementation": "the thing",
        "functionality": "the stuff",
        "essentially": "basically",
        "particularly": "especially",
        "comprehensive": "big",
        "subsequently": "then",
        "furthermore": "also",
        "nevertheless": "still",
        "alternatively": "or",
        "significantly": "a lot",
        "approximately": "about",
        "unfortunately": "sucks but",
        "simultaneously": "at the same time",
    }
    for src, dest in word_replacements.items():
        cleaned = re.sub(rf"\b{re.escape(src)}\b", dest, cleaned, flags=re.IGNORECASE)

    tech_jargon_strip = (
        r"\bIRC\b", r"\bAPI\b", r"\bCLI\b", r"\bSSH\b", r"\bDNS\b",
        r"\bHTTP[S]?\b", r"\bSQL\b", r"\bJSON\b", r"\bYAML\b", r"\bCSS\b",
        r"\bHTML\b", r"\bregex\b", r"\bbash\b", r"\bpython\b", r"\brust\b",
        r"\bdocker\b", r"\bnginx\b", r"\bredis\b", r"\bgit\b", r"\bnpm\b",
        r"\bwebsocket\b", r"\bfirewall\b", r"\bscript(?:s|ed|ing)?\b",
        r"\bcompil(?:e[rd]?|ing)\b", r"\bruntime\b", r"\bframework\b",
        r"\bsyntax\b", r"\bparsing\b", r"\bparser\b",
        r"\bencrypt(?:ed|ion)?\b", r"\bdecrypt(?:ed|ion)?\b",
        r"\balgorithm\b", r"\blatency\b", r"\bbandwidth\b",
        r"\bthroughput\b", r"\bpipeline\b", r"\bsandbox\b",
        r"\bkernel\b", r"\bdaemon\b", r"\bsocket\b",
        r"\bfreenode\b", r"\bweechat\b", r"\bhighlight\b",
        r"\bvim\b", r"\bemacs\b",
    )
    for pattern in tech_jargon_strip:
        cleaned = re.sub(pattern, "stuff", cleaned, count=1, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bstuff stuff\b", "stuff", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\b\w+\.\w+\.(com|net|org|io|dev|gg)\b", "some website", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bhttps?://\S+", "some link", cleaned, flags=re.IGNORECASE)

    if bool(curr_config.get("persona_avoid_witty_phrasing", True)):
        witty_patterns = (
            r"\bplot twist\b",
            r"\bto be fair\b",
            r"\bironically\b",
            r"\bfrankly\b",
            r"\badmittedly\b",
            r"\bobjectively\b",
            r"\blow[- ]key\b",
            r"\bhigh[- ]key\b",
            r"\bnuance\b",
            r"\bintriguingly\b",
            r"\bcuriously\b",
            r"\btechnically speaking\b",
            r"\bin essence\b",
            r"\bfundamentally\b",
        )
        for pattern in witty_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = apply_persona_blocklist_and_length(cleaned, curr_config)

    cleaned = cleaned.replace("\u2014", ",")
    cleaned = cleaned.replace("\u2013", ",")
    cleaned = cleaned.replace(";", ".")
    cleaned = re.sub(r"\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip(" ,.-")
    cleaned = maybe_add_profile_filler(cleaned, curr_config, "kevin")

    misspell_chance = clamp_01(float(curr_config.get("persona_misspell_chance", 0.0) or 0.0))
    if misspell_chance > 0 and deterministic_fraction(f"kevin:{cleaned}:misspell") < misspell_chance:
        typo_map = (
            (r"\breally\b", "realy"),
            (r"\bprobably\b", "probly"),
            (r"\bbecause\b", "becuase"),
            (r"\byou\b", "ya"),
            (r"\babout\b", "bout"),
            (r"\bsomething\b", "somethin"),
            (r"\bnothing\b", "nothin"),
            (r"\bgoing\b", "goin"),
            (r"\bwant\b", "wan"),
            (r"\bthough\b", "tho"),
        )
        for pattern, replacement in typo_map:
            updated = re.sub(pattern, replacement, cleaned, count=1, flags=re.IGNORECASE)
            if updated != cleaned:
                cleaned = updated
                break

    return cleaned


def enforce_saul_speech_style(content: str, curr_config: dict[str, Any]) -> str:
    cleaned = " ".join((content or "").strip().split())
    if cleaned == "":
        return cleaned

    cleaned = re.sub(r"\b(lol|lmao|haha+|hype|dope|sick|fire)\b", "", cleaned, flags=re.IGNORECASE)
    if bool(curr_config.get("persona_avoid_witty_phrasing", True)):
        cleaned = re.sub(r"\b(jk|kinda|sort of|vibe|vibes|wild|tbh|ngl|fr|no cap)\b", "", cleaned, flags=re.IGNORECASE)

    formal_strip = (
        r"\bfurthermore\b", r"\bmoreover\b", r"\bnevertheless\b",
        r"\bhowever\b", r"\bconsequently\b", r"\badditionally\b",
    )
    for pattern in formal_strip:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = apply_persona_blocklist_and_length(cleaned, curr_config)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip(" ,.-")

    cleaned = cleaned.lower()

    cleaned = re.sub(r"[.]{2,}", "...", cleaned)
    cleaned = re.sub(r"\.\s+", "... ", cleaned)
    if cleaned and not cleaned.endswith("...") and not cleaned.endswith("?"):
        cleaned = cleaned.rstrip(".!,") + "..."

    cleaned = re.sub(r"!\s*", "... ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    cleaned = maybe_add_profile_filler(cleaned, curr_config, "saul")
    return cleaned


def enforce_sarah_speech_style(content: str, curr_config: dict[str, Any]) -> str:
    cleaned = " ".join((content or "").strip().split())
    if cleaned == "":
        return cleaned

    cleaned = re.sub(r"\b(risk|constraint|constraints|compliance|mitigate|tradeoff|trade-off)\b", "", cleaned, flags=re.IGNORECASE)

    enthusiastic_strip = (
        r"\bamazing\b", r"\bawesome\b", r"\bincredible\b", r"\bfantastic\b",
        r"\bwonderful\b", r"\bexciting\b", r"\bexcited\b", r"\bbrilliant\b",
        r"\blove it\b", r"\blove this\b", r"\bso cool\b", r"\bgreat idea\b",
    )
    for pattern in enthusiastic_strip:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    formal_strip = (
        r"\bfurthermore\b", r"\bmoreover\b", r"\bnevertheless\b",
        r"\bconsequently\b", r"\badditionally\b", r"\bsignificantly\b",
        r"\bfundamentally\b", r"\bessentially\b",
    )
    for pattern in formal_strip:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = apply_persona_blocklist_and_length(cleaned, curr_config)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip(" ,.-")

    sentence_parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    if sentence_parts:
        cleaned = sentence_parts[0]

    cleaned = cleaned.lower()

    cleaned = cleaned.replace("!", ".")
    cleaned = re.sub(r"[.]{2,}", ".", cleaned)

    allowed_emoji = {"💀", "🙄"}
    cleaned = re.sub(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U0000FE00-\U0000FE0F\U0001F1E0-\U0001F1FF]",
        lambda m: m.group() if m.group() in allowed_emoji else "",
        cleaned,
    )

    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip(" ,.-")
    cleaned = maybe_add_profile_filler(cleaned, curr_config, "sarah")
    return cleaned


def enforce_katherine_speech_style(content: str, curr_config: dict[str, Any]) -> str:
    cleaned = " ".join((content or "").strip().split())
    if cleaned == "":
        return cleaned

    word_replacements = {
        "think about": "diagnose",
        "look into": "check",
        "beautiful": "solid",
        "amazing": "solid",
        "wonderful": "decent",
        "incredible": "legit",
        "fantastic": "solid",
        "awesome": "nice",
        "perhaps": "maybe",
        "utilize": "use",
        "regarding": "about",
        "concerning": "about",
        "facilitate": "run",
        "implement": "wire up",
        "problematic": "busted",
        "malfunction": "busted",
        "broken": "fried",
        "issue": "snag",
        "difficulty": "snag",
        "examine": "poke at",
        "investigate": "dig into",
    }
    for src, dest in word_replacements.items():
        cleaned = re.sub(rf"\b{re.escape(src)}\b", dest, cleaned, flags=re.IGNORECASE)

    formal_strip = (
        r"\bfurthermore\b", r"\bmoreover\b", r"\bnevertheless\b",
        r"\bconsequently\b", r"\bsignificantly\b", r"\bfundamentally\b",
        r"\bessentially\b", r"\btheoretically\b", r"\brespectfully\b",
    )
    for pattern in formal_strip:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = apply_persona_blocklist_and_length(cleaned, curr_config)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip(" ,.-")
    cleaned = maybe_add_profile_filler(cleaned, curr_config, "katherine")
    return cleaned


def enforce_damon_speech_style(content: str, curr_config: dict[str, Any]) -> str:
    cleaned = " ".join((content or "").strip().split())
    if cleaned == "":
        return cleaned

    formal_strip = (
        r"\bhowever\b", r"\bfurthermore\b", r"\bmoreover\b",
        r"\bnevertheless\b", r"\bconsequently\b", r"\badditionally\b",
        r"\bsignificantly\b", r"\btherefore\b", r"\bin conclusion\b",
        r"\brespectfully\b", r"\baccordingly\b",
    )
    for pattern in formal_strip:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    word_replacements = {
        "interesting": "wild",
        "fascinating": "insane",
        "beautiful": "fire",
        "wonderful": "sick",
        "excellent": "elite",
        "important": "massive",
        "understand": "get",
        "certainly": "obviously",
        "perhaps": "maybe",
        "regarding": "about",
        "utilize": "use",
        "unfortunately": "rip",
        "correct": "based",
        "absolutely": "literally",
    }
    for src, dest in word_replacements.items():
        cleaned = re.sub(rf"\b{re.escape(src)}\b", dest, cleaned, flags=re.IGNORECASE)

    cleaned = apply_persona_blocklist_and_length(cleaned, curr_config)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip(" ,.-")

    cleaned = cleaned.lower()

    article_seed = f"damon:{cleaned}:articles"
    if deterministic_fraction(article_seed) < 0.3:
        cleaned = re.sub(r"\bthe\s+", "", cleaned, count=1)
        cleaned = re.sub(r"\ba\s+", "", cleaned, count=1)

    jab_chance = clamp_01(float(curr_config.get("persona_playful_jab_chance", 0.0) or 0.0))
    if jab_chance > 0 and deterministic_fraction(f"damon:{cleaned}:jab") < jab_chance:
        jab_seed = int(deterministic_fraction(f"damon:{cleaned}:jab_pick") * 1000)
        jab_pool = [
            "but what do i know",
            "not that anyone asked",
            "chaos reigns",
            "just saying",
            "don't @ me",
            "respectfully unhinged take",
            "fight me on this",
        ]
        jab = jab_pool[jab_seed % len(jab_pool)]
        cleaned = f"{cleaned}, {jab}"

    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip(" ,.-")
    cleaned = maybe_add_profile_filler(cleaned, curr_config, "damon")
    return cleaned


def apply_persona_speech_enforcement(content: str, curr_config: dict[str, Any]) -> str:
    if content == "":
        return content
    if not bool(curr_config.get("persona_speech_enforcement_enabled", False)):
        return content

    profile = str(curr_config.get("persona_speech_profile", "") or "").strip().lower()
    if profile == "kevin":
        return enforce_kevin_speech_style(content, curr_config)
    if profile == "saul":
        return enforce_saul_speech_style(content, curr_config)
    if profile == "sarah":
        return enforce_sarah_speech_style(content, curr_config)
    if profile == "katherine":
        return enforce_katherine_speech_style(content, curr_config)
    if profile == "damon":
        return enforce_damon_speech_style(content, curr_config)
    if profile == "":
        return content

    return apply_persona_blocklist_and_length(content, curr_config)


def format_visible_content(content: str, curr_config: dict[str, Any]) -> str:
    visible = apply_generated_mention_policy(content, curr_config)
    visible = apply_persona_speech_enforcement(visible, curr_config)
    return visible


def pick_mood(curr_config: dict[str, Any], now: datetime) -> Optional[str]:
    if not curr_config.get("mood_injector_enabled", False):
        return None

    mood_pool = [str(mood).strip() for mood in (curr_config.get("mood_pool") or []) if str(mood).strip()]
    if mood_pool == []:
        return None

    rotation_mode = str(curr_config.get("mood_rotation_mode", "daily") or "daily").lower().strip()
    if rotation_mode == "per_message":
        return random.choice(mood_pool)

    if rotation_mode == "hourly":
        bucket_key = now.strftime("%Y-%m-%d-%H")
    else:
        bucket_key = now.strftime("%Y-%m-%d")

    stable_seed = f"{get_bot_identity()}::{bucket_key}"
    seed_int = int(sha256(stable_seed.encode("utf-8")).hexdigest(), 16)
    return mood_pool[seed_int % len(mood_pool)]


def build_system_prompt_for_model(curr_config: dict[str, Any], accept_usernames: bool) -> Optional[str]:
    if not (system_prompt := curr_config.get("system_prompt")):
        return None

    now = datetime.now().astimezone()
    system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()

    mood = pick_mood(curr_config, now)
    if mood is not None:
        mood_strength = str(curr_config.get("mood_influence_strength", "subtle") or "subtle").lower().strip()
        if mood_strength == "strong":
            mood_line = f"CURRENT VIBE: You are {mood}. Let this clearly shape your tone and how intensely you engage."
        elif mood_strength == "medium":
            mood_line = f"CURRENT VIBE: You are {mood}. Let this influence your tone and pacing in a noticeable but natural way."
        else:
            mood_line = f"CURRENT VIBE: You are {mood}. Let this subtly influence your tone and how much you care in this moment."
        system_prompt += f"\n\n{mood_line}"

    if curr_config.get("discord_chat_global_style_prompt_enabled", True):
        system_prompt += (
            "\n\nGLOBAL CHAT STYLE: You're chatting on Discord."
            " Avoid bullet points/numbered lists and long paragraphs."
            " Keep replies terse, playful, and conversational, usually 1-2 short sentences."
            " lowercase and imperfect punctuation are fine."
            " If asked for depth, give a short take first and expand only if asked."
            " Stay anchored to the user's most recent topic and read the room."
            " Don't pivot to unrelated projects, prototypes, or tangents unless invited."
            " When a user shares a photo or scene, respond to what they shared first and ask one relevant follow-up before changing topics."
        )

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


def extract_context_tokens(text: str) -> set[str]:
    if not text:
        return set()
    return set(re.findall(r"[a-z0-9']{3,}", text.lower()))


def rank_contextual_gif_urls(curr_config: dict[str, Any], trigger_text: str, reference_text: str) -> list[str]:
    combined_text = f"{trigger_text}\n{reference_text}".strip()
    context_tokens = extract_context_tokens(combined_text)

    catalog_entries = curr_config.get("gif_catalog") or []
    ranked: list[tuple[float, str]] = []

    for entry in catalog_entries:
        if not isinstance(entry, dict):
            continue
        url = str(entry.get("url", "")).strip()
        if not url:
            continue

        tags = entry.get("tags") or []
        tag_tokens = {str(tag).lower().strip() for tag in tags if str(tag).strip()}
        overlap = len(context_tokens & tag_tokens) if context_tokens and tag_tokens else 0
        # Slight randomness keeps repeated contexts from always selecting same GIF.
        score = float(overlap) + random.random() * 0.05
        ranked.append((score, url))

    ranked.sort(key=lambda item: item[0], reverse=True)
    ranked_urls = [url for score, url in ranked if score > 0]

    fallback_urls = [str(url).strip() for url in (curr_config.get("gif_reply_urls") or []) if str(url).strip()]
    # Append fallbacks (and non-overlap catalog URLs) while preserving order and uniqueness.
    for _, url in ranked:
        if url not in ranked_urls:
            ranked_urls.append(url)
    for url in fallback_urls:
        if url not in ranked_urls:
            ranked_urls.append(url)
    return ranked_urls


def rank_contextual_emojis(curr_config: dict[str, Any], trigger_text: str, reference_text: str) -> list[str]:
    combined_text = f"{trigger_text}\n{reference_text}".strip()
    context_tokens = extract_context_tokens(combined_text)

    catalog_entries = curr_config.get("emoji_reaction_catalog") or []
    ranked: list[tuple[float, str]] = []

    for entry in catalog_entries:
        if not isinstance(entry, dict):
            continue
        emoji = str(entry.get("emoji", "")).strip()
        if not emoji:
            continue
        tags = entry.get("tags") or []
        tag_tokens = {str(tag).lower().strip() for tag in tags if str(tag).strip()}
        overlap = len(context_tokens & tag_tokens) if context_tokens and tag_tokens else 0
        score = float(overlap) + random.random() * 0.05
        ranked.append((score, emoji))

    ranked.sort(key=lambda item: item[0], reverse=True)
    ranked_emojis = [emoji for score, emoji in ranked if score > 0]

    fallback_emojis = [str(emoji).strip() for emoji in (curr_config.get("emoji_reaction_choices") or []) if str(emoji).strip()]
    for _, emoji in ranked:
        if emoji not in ranked_emojis:
            ranked_emojis.append(emoji)
    for emoji in fallback_emojis:
        if emoji not in ranked_emojis:
            ranked_emojis.append(emoji)
    return ranked_emojis


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

    contextual_selection_enabled = bool(curr_config.get("gif_contextual_selection_enabled", False))
    if contextual_selection_enabled:
        gif_urls = rank_contextual_gif_urls(curr_config, trigger_msg.content, reference_text)
    else:
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
    gif_recent_dedupe_window_seconds = max(int(float(curr_config.get("gif_recent_dedupe_window_seconds", 21600) or 21600)), 0)
    gif_bad_url_cooldown_seconds = max(int(float(curr_config.get("gif_bad_url_cooldown_seconds", 86400) or 86400)), 0)
    gif_bad_url_max_failures = max(int(float(curr_config.get("gif_bad_url_max_failures", 2) or 2)), 1)
    now = now_ts()
    channel_id = trigger_msg.channel.id
    recent_gif_zset_key = f"llmcord:gif:recent_urls:{channel_id}"

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

        if gif_recent_dedupe_window_seconds > 0:
            await redis_client_instance.zremrangebyscore(
                recent_gif_zset_key,
                min=0,
                max=now - gif_recent_dedupe_window_seconds,
            )
            recent_urls = set(await redis_client_instance.zrange(recent_gif_zset_key, 0, -1))
            gif_urls = [url for url in gif_urls if url not in recent_urls]
            if gif_urls == []:
                return

        if gif_bad_url_cooldown_seconds > 0:
            healthy_urls = []
            for url in gif_urls:
                url_hash = sha256(url.encode("utf-8")).hexdigest()
                bad_key = f"llmcord:gif:bad_url:{channel_id}:{url_hash}"
                if not await redis_client_instance.exists(bad_key):
                    healthy_urls.append(url)
            if healthy_urls:
                gif_urls = healthy_urls

    gif_payload = None
    randomized_urls = gif_urls[:]
    random.shuffle(randomized_urls)
    for candidate_url in randomized_urls:
        gif_payload = await fetch_image_asset_from_url(candidate_url)
        if gif_payload is not None:
            if redis_client_instance is not None:
                url_hash = sha256(candidate_url.encode("utf-8")).hexdigest()
                await redis_client_instance.delete(
                    f"llmcord:gif:fail_count:{channel_id}:{url_hash}",
                    f"llmcord:gif:bad_url:{channel_id}:{url_hash}",
                )
            break
        if redis_client_instance is not None:
            url_hash = sha256(candidate_url.encode("utf-8")).hexdigest()
            fail_key = f"llmcord:gif:fail_count:{channel_id}:{url_hash}"
            fail_count = int(await redis_client_instance.incr(fail_key))
            await redis_client_instance.expire(fail_key, 7 * 24 * 3600)
            if fail_count >= gif_bad_url_max_failures and gif_bad_url_cooldown_seconds > 0:
                bad_key = f"llmcord:gif:bad_url:{channel_id}:{url_hash}"
                await redis_client_instance.set(bad_key, "1", ex=gif_bad_url_cooldown_seconds)

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
        await trigger_msg.channel.send(file=file, silent=True)
    except (discord.Forbidden, discord.NotFound, discord.HTTPException):
        return

    if redis_client_instance is not None:
        await redis_client_instance.set(f"llmcord:gif:last_ts:{channel_id}", now, ex=max(gif_reply_cooldown_seconds, 3600))
        hour_bucket = int(now // 3600)
        per_hour_key = f"llmcord:gif:hour_count:{channel_id}:{hour_bucket}"
        hour_count = await redis_client_instance.incr(per_hour_key)
        if hour_count == 1:
            await redis_client_instance.expire(per_hour_key, 2 * 3600)
        if gif_recent_dedupe_window_seconds > 0:
            await redis_client_instance.zadd(recent_gif_zset_key, {candidate_url: now})
            await redis_client_instance.expire(recent_gif_zset_key, gif_recent_dedupe_window_seconds + 24 * 3600)


async def maybe_add_emoji_reaction(
    trigger_msg: discord.Message,
    curr_config: dict[str, Any],
    redis_client_instance,
    reference_text: str,
) -> None:
    if trigger_msg.channel.type == discord.ChannelType.private:
        return
    if not curr_config.get("emoji_reactions_enabled", False):
        return

    contextual_selection_enabled = bool(curr_config.get("emoji_reaction_contextual_selection_enabled", True))
    if contextual_selection_enabled:
        emoji_choices = rank_contextual_emojis(curr_config, trigger_msg.content, reference_text)
    else:
        emoji_choices = [str(emoji).strip() for emoji in (curr_config.get("emoji_reaction_choices") or []) if str(emoji).strip()]
    if emoji_choices == []:
        return

    reaction_chance = clamp_01(float(curr_config.get("emoji_reaction_chance", 0.15) or 0.15))
    if random.random() > reaction_chance:
        return

    keyword_filters = [str(word).lower().strip() for word in (curr_config.get("emoji_reaction_keyword_filters") or []) if str(word).strip()]
    if keyword_filters:
        searchable = f"{trigger_msg.content}\n{reference_text}".lower()
        if not any(keyword in searchable for keyword in keyword_filters):
            return

    now = now_ts()
    channel_id = trigger_msg.channel.id
    reaction_cooldown_seconds = max(int(float(curr_config.get("emoji_reaction_cooldown_seconds", 60) or 60)), 0)
    reaction_max_per_hour_per_channel = max(int(float(curr_config.get("emoji_reaction_max_per_hour_per_channel", 8) or 8)), 0)

    if redis_client_instance is not None:
        claim_key = f"llmcord:emoji_reaction:claim:{channel_id}:{trigger_msg.id}"
        claimed = await redis_client_instance.set(claim_key, get_bot_identity(), ex=45, nx=True)
        if not claimed:
            return

        last_ts_key = f"llmcord:emoji_reaction:last_ts:{channel_id}"
        if reaction_cooldown_seconds > 0:
            last_ts = float(await redis_client_instance.get(last_ts_key) or 0)
            if now - last_ts < reaction_cooldown_seconds:
                return

        hour_bucket = int(now // 3600)
        per_hour_key = f"llmcord:emoji_reaction:hour_count:{channel_id}:{hour_bucket}"
        if reaction_max_per_hour_per_channel > 0:
            hour_count = int(await redis_client_instance.get(per_hour_key) or 0)
            if hour_count >= reaction_max_per_hour_per_channel:
                return

    randomized = emoji_choices[:]
    random.shuffle(randomized)
    reacted = False
    for emoji in randomized:
        try:
            await trigger_msg.add_reaction(emoji)
            reacted = True
            break
        except (discord.Forbidden, discord.NotFound):
            return
        except discord.HTTPException:
            continue

    if not reacted:
        return

    if redis_client_instance is not None:
        await redis_client_instance.set(
            f"llmcord:emoji_reaction:last_ts:{channel_id}",
            now,
            ex=max(reaction_cooldown_seconds, 3600),
        )
        hour_bucket = int(now // 3600)
        per_hour_key = f"llmcord:emoji_reaction:hour_count:{channel_id}:{hour_bucket}"
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


def message_looks_like_short_reaction(content: str, max_chars: int) -> bool:
    normalized = " ".join((content or "").strip().split())
    if normalized == "":
        return False
    if max_chars > 0 and len(normalized) > max_chars:
        return False
    if normalized.lower().startswith(("http://", "https://")):
        return False
    tokens = re.findall(r"[a-z0-9']+", normalized.lower())
    return 1 <= len(tokens) <= 8


async def detect_recent_bot_target_message(new_msg: discord.Message, window_seconds: float) -> Optional[discord.Message]:
    if window_seconds <= 0:
        return None
    try:
        prior_msgs = [m async for m in new_msg.channel.history(before=new_msg, limit=12)]
    except (discord.NotFound, discord.HTTPException):
        return None

    latest_bot_msg = next((m for m in prior_msgs if m.author.bot), None)
    if latest_bot_msg is None:
        return None

    age_seconds = (new_msg.created_at - latest_bot_msg.created_at).total_seconds()
    if age_seconds < 0 or age_seconds > window_seconds:
        return None
    return latest_bot_msg


async def generate_proactive_starter_text(
    curr_config: dict[str, Any],
    target_user_id: Optional[int],
    max_chars: int,
    *,
    target_bot_name: Optional[str] = None,
    bot_to_bot: bool = False,
) -> str:
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

    if bot_to_bot and target_bot_name:
        mention_instruction = (
            f"Address {target_bot_name} directly by name in plain text. "
            "Do not use @ mentions or mention syntax."
        )
        user_instruction = (
            "Write exactly one short proactive Discord message that nudges another bot into the chat naturally. "
            f"Keep it under {max_chars} characters, one sentence, conversational, and not repetitive. "
            "No lists, no markdown, no hashtags, no @everyone, no @here, and no mention tokens. "
            f"{mention_instruction}"
        )
    else:
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
    if bot_to_bot:
        content = re.sub(r"(^|\s)@([A-Za-z0-9_]{2,32})", r"\1\2", content).strip()
        content = sanitize_proactive_message(content, max_chars, target_user_id=None)
    if not bot_to_bot and target_user_id is not None and content != "" and "<@" not in content:
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
    bot_tag = get_bot_identity()
    if not curr_config.get("proactive_starters_enabled", False):
        return
    if in_quiet_hours(curr_config) and curr_config.get("proactive_respect_quiet_hours", True):
        logging.debug("[proactive:%s:%s] blocked by quiet hours", bot_tag, channel_id)
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
    proactive_bot_to_bot_enabled = bool(curr_config.get("proactive_bot_to_bot_enabled", True))
    proactive_bot_to_bot_chance = clamp_01(float(curr_config.get("proactive_bot_to_bot_chance", 0.2) or 0.2))
    proactive_bot_to_bot_max_per_day_per_channel = max(
        int(float(curr_config.get("proactive_bot_to_bot_max_per_day_per_channel", 2) or 2)),
        0,
    )
    proactive_bot_to_bot_cooldown_seconds = max(float(curr_config.get("proactive_bot_to_bot_cooldown_seconds", 1800) or 0), 0)
    proactive_bot_to_bot_max_chain_without_human = max(
        int(float(curr_config.get("proactive_bot_to_bot_max_chain_without_human", 1) or 1)),
        0,
    )
    proactive_bot_to_bot_recent_bot_seconds = max(
        float(curr_config.get("proactive_bot_to_bot_recent_bot_seconds", 172800) or 0),
        0,
    )
    proactive_bot_to_bot_name_mode = str(curr_config.get("proactive_bot_to_bot_name_mode", "plain") or "plain").lower().strip()
    proactive_bot_to_bot_idle_human_seconds = max(float(curr_config.get("proactive_bot_to_bot_idle_human_seconds", 1800) or 0), 0)
    proactive_bot_to_bot_idle_channel_seconds = max(float(curr_config.get("proactive_bot_to_bot_idle_channel_seconds", 1800) or 0), 0)
    proactive_bot_to_bot_idle_jitter_seconds = max(float(curr_config.get("proactive_bot_to_bot_idle_jitter_seconds", 300) or 0), 0)
    proactive_bot_to_bot_relationship_default_weight = max(
        float(curr_config.get("proactive_bot_to_bot_relationship_default_weight", 1.0) or 1.0),
        0.0,
    )
    relationship_weights, known_relationship_identities = parse_relationship_weights(curr_config)

    last_human_ts = float(await redis_client_instance.get(f"llmcord:afk:last_human_ts:{channel_id}") or 0)
    last_any_msg_ts = float(await redis_client_instance.get(f"llmcord:channel:last_message_ts:{channel_id}") or 0)
    human_idle = now - last_human_ts if last_human_ts > 0 else float("inf")
    channel_idle = now - last_any_msg_ts if last_any_msg_ts > 0 else float("inf")

    if await redis_client_instance.get(f"llmcord:active_responder:{channel_id}"):
        logging.debug("[proactive:%s:%s] blocked by active_responder lock", bot_tag, channel_id)
        return

    day_key = today_key(curr_config)
    daily_key = f"llmcord:proactive:daily:{channel_id}:{day_key}"
    b2b_daily_key = f"llmcord:proactive:b2b:daily:{channel_id}:{day_key}"
    b2b_last_ts_key = f"llmcord:proactive:b2b:last_ts:{channel_id}"
    b2b_chain_key = f"llmcord:proactive:b2b:chain:{channel_id}"
    current_daily_count = int(await redis_client_instance.get(daily_key) or 0)
    if max_daily > 0 and current_daily_count >= max_daily:
        logging.debug("[proactive:%s:%s] blocked by daily cap (%d/%d)", bot_tag, channel_id, current_daily_count, max_daily)
        return

    target_user_id = None
    target_bot_id = None
    target_bot_name = None
    is_bot_to_bot = False

    b2b_roll = random.random()
    if proactive_bot_to_bot_enabled and b2b_roll <= proactive_bot_to_bot_chance:
        b2b_daily_count = int(await redis_client_instance.get(b2b_daily_key) or 0)
        b2b_last_ts = float(await redis_client_instance.get(b2b_last_ts_key) or 0)
        b2b_chain_count = int(await redis_client_instance.get(b2b_chain_key) or 0)
        passes_daily_cap = proactive_bot_to_bot_max_per_day_per_channel <= 0 or b2b_daily_count < proactive_bot_to_bot_max_per_day_per_channel
        passes_cooldown = proactive_bot_to_bot_cooldown_seconds <= 0 or now - b2b_last_ts >= proactive_bot_to_bot_cooldown_seconds
        passes_chain_cap = proactive_bot_to_bot_max_chain_without_human <= 0 or b2b_chain_count < proactive_bot_to_bot_max_chain_without_human
        b2b_human_idle_threshold = deterministic_jittered_threshold(
            proactive_bot_to_bot_idle_human_seconds,
            proactive_bot_to_bot_idle_jitter_seconds,
            f"b2b-human-idle:{channel_id}:{int(now // 600)}",
        )
        b2b_channel_idle_threshold = deterministic_jittered_threshold(
            proactive_bot_to_bot_idle_channel_seconds,
            proactive_bot_to_bot_idle_jitter_seconds,
            f"b2b-channel-idle:{channel_id}:{int(now // 600)}",
        )
        passes_b2b_idle_human = b2b_human_idle_threshold <= 0 or human_idle >= b2b_human_idle_threshold
        passes_b2b_idle_channel = b2b_channel_idle_threshold <= 0 or channel_idle >= b2b_channel_idle_threshold

        logging.info(
            "[proactive-b2b:%s:%s] gates: daily=%s(%d/%d) cooldown=%s(%.0fs/%.0fs) chain=%s(%d/%d) "
            "human_idle=%s(%.0fs/%.0fs) chan_idle=%s(%.0fs/%.0fs)",
            bot_tag, channel_id,
            passes_daily_cap, b2b_daily_count, proactive_bot_to_bot_max_per_day_per_channel,
            passes_cooldown, now - b2b_last_ts if b2b_last_ts > 0 else float("inf"), proactive_bot_to_bot_cooldown_seconds,
            passes_chain_cap, b2b_chain_count, proactive_bot_to_bot_max_chain_without_human,
            passes_b2b_idle_human, human_idle, b2b_human_idle_threshold,
            passes_b2b_idle_channel, channel_idle, b2b_channel_idle_threshold,
        )

        if passes_daily_cap and passes_cooldown and passes_chain_cap and passes_b2b_idle_human and passes_b2b_idle_channel:
            recent_bots_key = f"llmcord:channel:recent_bots:{channel_id}"
            min_recent_ts = now - proactive_bot_to_bot_recent_bot_seconds if proactive_bot_to_bot_recent_bot_seconds > 0 else 0
            candidate_bot_ids = await redis_client_instance.zrevrangebyscore(recent_bots_key, max="+inf", min=min_recent_ts, start=0, num=25)
            own_id = discord_bot.user.id if discord_bot.user else None
            own_identity_hint = str(curr_config.get("persona_speech_profile", "") or "").strip().lower()
            weighted_candidates: list[tuple[tuple[int, str], float]] = []
            known_identities = set(known_relationship_identities)
            if own_identity_hint:
                known_identities.add(own_identity_hint)
            known_identities.update({"kevin", "damon", "saul", "sarah", "katherine"})

            for candidate_bot_id_str in candidate_bot_ids:
                try:
                    candidate_bot_id = int(candidate_bot_id_str)
                except ValueError:
                    continue
                if own_id is not None and candidate_bot_id == own_id:
                    continue

                bot_name_key = f"llmcord:bot:display_name:{candidate_bot_id}"
                candidate_bot_name = str(await redis_client_instance.get(bot_name_key) or "").strip()
                if not candidate_bot_name:
                    continue

                if proactive_bot_to_bot_name_mode == "mention":
                    candidate_bot_name = f"<@{candidate_bot_id}>"

                target_identity_hint = resolve_relationship_identity(candidate_bot_name, known_identities)
                relationship_weight = proactive_bot_to_bot_relationship_default_weight
                if own_identity_hint and target_identity_hint:
                    relationship_weight = relationship_weights.get(
                        (own_identity_hint, target_identity_hint),
                        proactive_bot_to_bot_relationship_default_weight,
                    )
                weighted_candidates.append(((candidate_bot_id, candidate_bot_name), relationship_weight))

            chosen_candidate = weighted_choice(weighted_candidates)
            if chosen_candidate is not None:
                target_bot_id, target_bot_name = chosen_candidate
                is_bot_to_bot = True
                logging.info("[proactive-b2b:%s:%s] chose target=%s", bot_tag, channel_id, target_bot_name)
            else:
                logging.info("[proactive-b2b:%s:%s] no candidates (raw=%d, filtered_weighted=%d)", bot_tag, channel_id, len(candidate_bot_ids), len(weighted_candidates))
        else:
            logging.debug("[proactive-b2b:%s:%s] gates blocked (see above)", bot_tag, channel_id)
    else:
        logging.debug("[proactive:%s:%s] b2b coin skip (roll=%.3f, need<=%.3f) or disabled=%s", bot_tag, channel_id, b2b_roll, proactive_bot_to_bot_chance, not proactive_bot_to_bot_enabled)

    if not is_bot_to_bot:
        if human_idle_seconds > 0 and human_idle < human_idle_seconds:
            logging.debug("[proactive:%s:%s] human path blocked by human_idle (%.0fs < %.0fs)", bot_tag, channel_id, human_idle, human_idle_seconds)
            return
        if channel_idle_seconds > 0 and channel_idle < channel_idle_seconds:
            logging.debug("[proactive:%s:%s] human path blocked by channel_idle (%.0fs < %.0fs)", bot_tag, channel_id, channel_idle, channel_idle_seconds)
            return
        if random.random() > chance:
            return

    claim_key = f"llmcord:proactive:claim:{channel_id}"
    claimed = await redis_client_instance.set(claim_key, get_bot_identity(), ex=claim_ttl_seconds, nx=True)
    if not claimed:
        logging.debug("[proactive:%s:%s] claim lost to another bot", bot_tag, channel_id)
        return
    logging.info("[proactive:%s:%s] claimed! is_b2b=%s target=%s", bot_tag, channel_id, is_bot_to_bot, target_bot_name)

    channel = discord_bot.get_channel(channel_id)
    if channel is None:
        try:
            channel = await discord_bot.fetch_channel(channel_id)
        except (discord.NotFound, discord.HTTPException):
            channel = None
    if channel is None:
        return

    if not is_bot_to_bot and mention_enabled and mention_max_per_user_per_day > 0 and random.random() <= mention_chance:
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
        candidate_text = await generate_proactive_starter_text(
            curr_config,
            target_user_id,
            proactive_max_chars,
            target_bot_name=target_bot_name,
            bot_to_bot=is_bot_to_bot,
        )
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
        logging.info("[proactive:%s:%s] generation failed after %d retries", bot_tag, channel_id, retries)
        fallback_text = str(curr_config.get("proactive_fallback_starter", "") or "")
        starter_text = sanitize_proactive_message(fallback_text, proactive_max_chars, target_user_id)
        if not is_bot_to_bot and target_user_id is not None and "<@" not in starter_text:
            starter_text = f"<@{target_user_id}> {starter_text}".strip()
        if starter_text == "":
            logging.info("[proactive:%s:%s] no fallback text, skipping", bot_tag, channel_id)
            return

    try:
        await channel.send(
            starter_text,
            silent=True,
            allowed_mentions=discord.AllowedMentions(users=(not is_bot_to_bot), roles=False, everyone=False),
        )
        logging.info("[proactive:%s:%s] SENT b2b=%s text=%s", bot_tag, channel_id, is_bot_to_bot, starter_text[:80])
    except (discord.Forbidden, discord.NotFound, discord.HTTPException) as exc:
        logging.warning("[proactive:%s:%s] send failed: %s", bot_tag, channel_id, exc)
        return

    if starter_text_hash is not None:
        await redis_client_instance.zadd(proactive_hashes_key, {starter_text_hash: now})
        await redis_client_instance.expire(proactive_hashes_key, dedupe_window_seconds + 24 * 3600)

    ttl_seconds = 3 * 24 * 3600
    daily_count = await redis_client_instance.incr(daily_key)
    if daily_count == 1:
        await redis_client_instance.expire(daily_key, ttl_seconds)
    if is_bot_to_bot and target_bot_id is not None:
        b2b_daily_count = await redis_client_instance.incr(b2b_daily_key)
        if b2b_daily_count == 1:
            await redis_client_instance.expire(b2b_daily_key, ttl_seconds)
        await redis_client_instance.set(b2b_last_ts_key, now, ex=7 * 24 * 3600)
        await redis_client_instance.incr(b2b_chain_key)
        await redis_client_instance.expire(b2b_chain_key, 7 * 24 * 3600)
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
            curr_config.get("afk_followup_fallback_text", "") or ""
        ).strip()[:afk_followup_max_chars]
    if followup_text == "":
        await redis_client_instance.zrem(schedule_zset_key, item_id)
        await redis_client_instance.delete(item_hash_key)
        return

    try:
        await channel.send(followup_text, silent=True)
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

            if redis_client_instance is None:
                await asyncio.sleep(poll_seconds)
                continue

            if curr_config.get("afk_followup_enabled", False):
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

                if not candidate_channel_ids:
                    logging.info("[scheduler:%s] no candidate channels (configured=%d, observed=0)",
                                 get_bot_identity(), len(configured_channel_ids))

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
    mentions_any_bot = any(getattr(mentioned_user, "bot", False) for mentioned_user in new_msg.mentions)
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
    implicit_targeting_enabled = bool(config.get("implicit_targeting_enabled", True))
    implicit_targeting_window_seconds = max(float(config.get("implicit_targeting_window_seconds", 10) or 0), 0)
    implicit_targeting_max_chars = max(int(float(config.get("implicit_targeting_max_chars", 40) or 0)), 0)
    implicit_targeting_fallback_wait_seconds = max(float(config.get("implicit_targeting_fallback_wait_seconds", 4) or 0), 0)
    implicit_target_bot_msg = None
    implicit_target_bot_id = None
    is_implicit_target_for_this_bot = False
    if (
        implicit_targeting_enabled
        and not is_dm
        and not new_msg.author.bot
        and not has_direct_mention
        and not is_direct_reply
        and not has_everyone_directive
        and not mentions_any_bot
        and parent_msg is None
        and message_looks_like_short_reaction(new_msg.content, implicit_targeting_max_chars)
    ):
        implicit_target_bot_msg = await detect_recent_bot_target_message(new_msg, implicit_targeting_window_seconds)
        if implicit_target_bot_msg is not None:
            implicit_target_bot_id = implicit_target_bot_msg.author.id
            is_implicit_target_for_this_bot = bool(discord_bot.user and implicit_target_bot_id == discord_bot.user.id)

    is_effectively_directed_message = is_directed_message or is_implicit_target_for_this_bot
    should_process = True

    strict_reply_targeting = bool(config.get("strict_reply_targeting", True))
    if strict_reply_targeting and is_reply_to_other_bot and not (has_direct_mention or has_everyone_directive):
        return

    direct_mention_retry_enabled = bool(config.get("direct_mention_retry_enabled", True))
    direct_mention_fast_lane_enabled = bool(config.get("direct_mention_fast_lane_enabled", True))
    remaining_direct_wait_seconds = max(float(config.get("direct_mention_max_wait_seconds", 7200) or 0), 0) if is_effectively_directed_message else 0

    if not is_dm:
        autonomous_bot_only_mode = config.get("autonomous_bot_only_mode", False)
        autonomous_channel_ids = set(config.get("autonomous_channel_ids", []))
        in_autonomous_scope = autonomous_bot_only_mode and (not autonomous_channel_ids or any(id in autonomous_channel_ids for id in channel_ids))

        if in_autonomous_scope:
            # Explicit @mentions/replies should always trigger in autonomous channels.
            if has_direct_mention or is_direct_reply or is_implicit_target_for_this_bot:
                should_process = True
            elif mentions_any_bot:
                # If another bot is explicitly mentioned, don't probabilistically jump in.
                should_process = False
            else:
                # Otherwise, use per-bot probabilistic participation to avoid dogpiles.
                group_response_chance = clamp_01(float(config.get("group_response_chance", 1.0) or 1.0))
                greeting_response_chance = config.get("greeting_response_chance")
                response_priority_weight = max(float(config.get("response_priority_weight", 1.0) or 1.0), 0.0)
                bot_to_bot_response_chance_multiplier = clamp_01(float(config.get("bot_to_bot_response_chance_multiplier", 0.5) or 0.5))

                effective_chance = group_response_chance
                if greeting_response_chance is not None and is_greeting_message(new_msg.content):
                    effective_chance = clamp_01(float(greeting_response_chance or 0.0))

                effective_chance = clamp_01(effective_chance * response_priority_weight)
                if new_msg.author.bot:
                    effective_chance = clamp_01(effective_chance * bot_to_bot_response_chance_multiplier)
                    bot_to_bot_response_chance_floor = clamp_01(float(config.get("bot_to_bot_response_chance_floor", 0.0) or 0.0))
                    effective_chance = max(effective_chance, bot_to_bot_response_chance_floor)
                should_process = random.random() < effective_chance
        else:
            should_process = has_direct_mention or is_direct_reply or is_implicit_target_for_this_bot

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

    implicit_fallback_open = False
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

            consecutive_bot_turns_key = f"llmcord:channel:consecutive_bot_turns:{channel_id}"
            proactive_b2b_chain_key = f"llmcord:proactive:b2b:chain:{channel_id}"
            if new_msg.author.bot:
                consecutive_bot_turns = int(await redis_client_instance.incr(consecutive_bot_turns_key))
                await redis_client_instance.expire(consecutive_bot_turns_key, 7 * 24 * 3600)
                recent_bots_key = f"llmcord:channel:recent_bots:{channel_id}"
                await redis_client_instance.zadd(recent_bots_key, {str(new_msg.author.id): now})
                await redis_client_instance.expire(recent_bots_key, 14 * 24 * 3600)
                author_name = (
                    getattr(new_msg.author, "display_name", None)
                    or getattr(new_msg.author, "global_name", None)
                    or getattr(new_msg.author, "name", None)
                    or ""
                )
                if author_name:
                    await redis_client_instance.set(
                        f"llmcord:bot:display_name:{new_msg.author.id}",
                        str(author_name)[:80],
                        ex=30 * 24 * 3600,
                    )
            else:
                consecutive_bot_turns = 0
                await redis_client_instance.set(consecutive_bot_turns_key, 0, ex=7 * 24 * 3600)
                await redis_client_instance.set(proactive_b2b_chain_key, 0, ex=7 * 24 * 3600)

            if not new_msg.author.bot:
                await redis_client_instance.set(f"llmcord:afk:last_human_ts:{channel_id}", now, ex=7 * 24 * 3600)
                recent_humans_key = f"llmcord:channel:recent_humans:{channel_id}"
                await redis_client_instance.zadd(recent_humans_key, {str(new_msg.author.id): now})
                await redis_client_instance.expire(recent_humans_key, 14 * 24 * 3600)
                await maybe_schedule_afk_followup(new_msg, config, redis_client_instance)

            if (
                implicit_target_bot_id is not None
                and not is_implicit_target_for_this_bot
                and not new_msg.author.bot
                and implicit_targeting_fallback_wait_seconds > 0
            ):
                implicit_coord_key = f"llmcord:implicit_target:coord:{channel_id}:{new_msg.id}"
                coord_ttl_seconds = max(int(implicit_targeting_fallback_wait_seconds) + 2, 3)
                await redis_client_instance.set(implicit_coord_key, str(implicit_target_bot_id), ex=coord_ttl_seconds, nx=True)
                coord_value = await redis_client_instance.get(implicit_coord_key)
                if coord_value is not None:
                    try:
                        coord_target_bot_id = int(coord_value)
                    except ValueError:
                        coord_target_bot_id = None

                    if coord_target_bot_id == implicit_target_bot_id:
                        await asyncio.sleep(implicit_targeting_fallback_wait_seconds)
                        try:
                            fallback_recent_msgs = [m async for m in new_msg.channel.history(limit=8)]
                        except (discord.NotFound, discord.HTTPException):
                            fallback_recent_msgs = []
                        if any(m.author.bot and m.id != new_msg.id and m.created_at > new_msg.created_at for m in fallback_recent_msgs):
                            return
                        implicit_fallback_open = True

            if not is_directed_message and new_msg.author.bot:
                max_consecutive_bot_turns_without_human = max(int(float(config.get("max_consecutive_bot_turns_without_human", 4) or 4)), 0)
                if max_consecutive_bot_turns_without_human > 0 and consecutive_bot_turns >= max_consecutive_bot_turns_without_human:
                    should_process = False

                pair_back_and_forth_cooldown_seconds = max(float(config.get("pair_back_and_forth_cooldown_seconds", 60) or 60), 0)
                if pair_back_and_forth_cooldown_seconds > 0 and discord_bot.user:
                    pair_key = f"llmcord:pair:last_reply:{channel_id}:{discord_bot.user.id}:{new_msg.author.id}"
                    pair_last_reply_ts = float(await redis_client_instance.get(pair_key) or 0)
                    if now - pair_last_reply_ts < pair_back_and_forth_cooldown_seconds:
                        should_process = False

        if implicit_fallback_open:
            should_process = True
        if not should_process:
            return

    reaction_delay_base_seconds = max(float(config.get("reaction_delay_base_seconds", 0) or 0), 0)
    reaction_delay_jitter_seconds = max(float(config.get("reaction_delay_jitter_seconds", 0) or 0), 0)
    if not is_dm and (reaction_delay_base_seconds > 0 or reaction_delay_jitter_seconds > 0):
        await asyncio.sleep(reaction_delay_base_seconds + random.uniform(0, reaction_delay_jitter_seconds))

    global_channel_cooldown_seconds = max(float(config.get("global_channel_cooldown_seconds", 0) or 0), 0)
    global_channel_arbitration_jitter_seconds = max(float(config.get("global_channel_arbitration_jitter_seconds", 0) or 0), 0)
    if not is_dm and not (is_effectively_directed_message and direct_mention_fast_lane_enabled) and (global_channel_cooldown_seconds > 0 or global_channel_arbitration_jitter_seconds > 0):
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

            if is_effectively_directed_message and direct_mention_retry_enabled and remaining_direct_wait_seconds > 0:
                sleep_for = min(wait_seconds, remaining_direct_wait_seconds)
                await asyncio.sleep(max(sleep_for, 0))
                remaining_direct_wait_seconds = max(0.0, remaining_direct_wait_seconds - sleep_for)
                if remaining_direct_wait_seconds <= 0:
                    return
                continue

            return

    response_cooldown_seconds = max(float(config.get("response_cooldown_seconds", 0) or 0), 0)
    response_cooldown_jitter_seconds = max(float(config.get("response_cooldown_jitter_seconds", 0) or 0), 0)
    if not (is_effectively_directed_message and direct_mention_fast_lane_enabled) and (response_cooldown_seconds > 0 or response_cooldown_jitter_seconds > 0):
        channel_id = new_msg.channel.id
        while True:
            curr_time = monotonic()
            async with cooldown_lock:
                next_available_time = next_response_time_by_channel.get(channel_id, 0)
                if curr_time >= next_available_time:
                    next_response_time_by_channel[channel_id] = curr_time + response_cooldown_seconds + random.uniform(0, response_cooldown_jitter_seconds)
                    break
                wait_seconds = next_available_time - curr_time

            if is_effectively_directed_message and direct_mention_retry_enabled and remaining_direct_wait_seconds > 0:
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
            if new_msg.author.bot:
                max_responses_per_source_message = max(
                    int(float(config.get("bot_to_bot_max_responses_per_source_message", max_responses_per_source_message) or max_responses_per_source_message)),
                    1,
                )
                followup_response_chance = clamp_01(
                    float(config.get("bot_to_bot_followup_response_chance", followup_response_chance) or followup_response_chance)
                )
            floor_lock_key = f"llmcord:floor:{channel_id}:{source_message_id}"
            source_count_key = f"llmcord:source_count:{channel_id}:{source_message_id}"
            active_responder_key = f"llmcord:active_responder:{channel_id}"

            claimed_floor = await redis_client_instance.set(floor_lock_key, bot_identity, ex=floor_lock_ttl_seconds, nx=True)
            if not claimed_floor:
                return

            response_index = await redis_client_instance.incr(source_count_key)
            if response_index == 1:
                await redis_client_instance.expire(source_count_key, source_message_window_seconds)

            if response_index > max_responses_per_source_message and not is_effectively_directed_message:
                await redis_client_instance.decr(source_count_key)
                return

            if response_index > 1 and not is_effectively_directed_message and random.random() >= followup_response_chance:
                await redis_client_instance.decr(source_count_key)
                return

            claimed_active = await redis_client_instance.set(active_responder_key, bot_identity, ex=active_responder_ttl_seconds, nx=True)
            if not claimed_active:
                if not (is_effectively_directed_message and direct_mention_retry_enabled):
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
        response_msg = await new_msg.channel.send(**reply_kwargs)
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
                        visible_content = format_visible_content(response_contents[-1], config)
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
                    visible_content = format_visible_content(content, config)
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

    final_text = format_visible_content("".join(response_contents), config)
    if response_msgs:
        await maybe_send_curated_gif_reply(
            trigger_msg=new_msg,
            reply_target=response_msgs[-1],
            curr_config=config,
            redis_client_instance=redis_client_instance,
            reference_text=final_text,
        )
        await maybe_add_emoji_reaction(
            trigger_msg=new_msg,
            curr_config=config,
            redis_client_instance=redis_client_instance,
            reference_text=final_text,
        )
        if (
            not is_dm
            and redis_client_instance is not None
            and discord_bot.user is not None
            and new_msg.author.bot
        ):
            pair_back_and_forth_cooldown_seconds = max(float(config.get("pair_back_and_forth_cooldown_seconds", 60) or 60), 0)
            if pair_back_and_forth_cooldown_seconds > 0:
                channel_id = new_msg.channel.id
                pair_key = f"llmcord:pair:last_reply:{channel_id}:{discord_bot.user.id}:{new_msg.author.id}"
                await redis_client_instance.set(
                    pair_key,
                    now_ts(),
                    ex=max(int(pair_back_and_forth_cooldown_seconds * 4), 300),
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
