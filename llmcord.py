import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import logging
import os
import random
from time import monotonic
from typing import Any, Literal, Optional

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

intents = discord.Intents.default()
intents.message_content = True
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
    if client_id := config.get("client_id"):
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")

    await discord_bot.tree.sync()


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

    async def is_reply_to_this_bot() -> bool:
        if not (parent_msg_id := getattr(new_msg.reference, "message_id", None)):
            return False

        try:
            parent_msg = new_msg.reference.cached_message or await new_msg.channel.fetch_message(parent_msg_id)
        except (discord.NotFound, discord.HTTPException):
            return False

        return bool(discord_bot.user and parent_msg.author.id == discord_bot.user.id)

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    is_direct_reply = await is_reply_to_this_bot() if not is_dm else False

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

        if not should_process:
            return


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

    reaction_delay_base_seconds = max(float(config.get("reaction_delay_base_seconds", 0) or 0), 0)
    reaction_delay_jitter_seconds = max(float(config.get("reaction_delay_jitter_seconds", 0) or 0), 0)
    if not is_dm and (reaction_delay_base_seconds > 0 or reaction_delay_jitter_seconds > 0):
        await asyncio.sleep(reaction_delay_base_seconds + random.uniform(0, reaction_delay_jitter_seconds))

    global_channel_cooldown_seconds = max(float(config.get("global_channel_cooldown_seconds", 0) or 0), 0)
    global_channel_arbitration_jitter_seconds = max(float(config.get("global_channel_arbitration_jitter_seconds", 0) or 0), 0)
    if not is_dm and (global_channel_cooldown_seconds > 0 or global_channel_arbitration_jitter_seconds > 0):
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

        if global_channel_cooldown_seconds > 0:
            latest_bot_msg = next((m for m in recent_channel_msgs if m.author.bot and m.id != new_msg.id), None)
            if latest_bot_msg and (discord.utils.utcnow() - latest_bot_msg.created_at).total_seconds() < global_channel_cooldown_seconds:
                return

    response_cooldown_seconds = max(float(config.get("response_cooldown_seconds", 0) or 0), 0)
    response_cooldown_jitter_seconds = max(float(config.get("response_cooldown_jitter_seconds", 0) or 0), 0)
    if response_cooldown_seconds > 0 or response_cooldown_jitter_seconds > 0:
        channel_id = new_msg.channel.id
        curr_time = monotonic()
        async with cooldown_lock:
            if curr_time < next_response_time_by_channel.get(channel_id, 0):
                return
            next_response_time_by_channel[channel_id] = curr_time + response_cooldown_seconds + random.uniform(0, response_cooldown_jitter_seconds)

    redis_claim_state = None
    if not is_dm:
        try:
            redis_client_instance = await get_redis_client(config)
        except RedisError:
            logging.exception("Redis unavailable, continuing without distributed coordination")
            redis_client_instance = None

        if redis_client_instance is not None:
            channel_id = new_msg.channel.id
            source_message_id = new_msg.id
            bot_identity = get_bot_identity()

            floor_lock_ttl_seconds = max(int(float(config.get("floor_lock_ttl_seconds", 45) or 45)), 5)
            active_responder_ttl_seconds = max(int(float(config.get("active_responder_ttl_seconds", 90) or 90)), 10)
            source_message_window_seconds = max(int(float(config.get("source_message_window_seconds", 180) or 180)), 30)
            max_responses_per_source_message = max(int(float(config.get("max_responses_per_source_message", 2) or 2)), 1)
            followup_response_chance = clamp_01(float(config.get("followup_response_chance", 0.15) or 0.15))
            is_directed_message = has_direct_mention or is_direct_reply

            floor_lock_key = f"llmcord:floor:{channel_id}:{source_message_id}"
            source_count_key = f"llmcord:source_count:{channel_id}:{source_message_id}"
            active_responder_key = f"llmcord:active_responder:{channel_id}"

            claimed_floor = await redis_client_instance.set(floor_lock_key, bot_identity, ex=floor_lock_ttl_seconds, nx=True)
            if not claimed_floor:
                return

            response_index = await redis_client_instance.incr(source_count_key)
            if response_index == 1:
                await redis_client_instance.expire(source_count_key, source_message_window_seconds)

            if response_index > max_responses_per_source_message:
                await redis_client_instance.decr(source_count_key)
                return

            if response_index > 1 and not is_directed_message and random.random() >= followup_response_chance:
                await redis_client_instance.decr(source_count_key)
                return

            claimed_active = await redis_client_instance.set(active_responder_key, bot_identity, ex=active_responder_ttl_seconds, nx=True)
            if not claimed_active:
                await redis_client_instance.decr(source_count_key)
                return

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

    if system_prompt := config.get("system_prompt"):
        now = datetime.now().astimezone()

        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        if accept_usernames:
            system_prompt += "\n\nUser's names are their Discord IDs and should be typed as '<@ID>'."

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
                        embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                        embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                        if start_next_msg:
                            await reply_helper(embed=embed, silent=True)
                        else:
                            await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                            await response_msgs[-1].edit(embed=embed)

                        last_task_time = datetime.now().timestamp()

            if use_plain_responses:
                for content in response_contents:
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))

    except Exception:
        logging.exception("Error while generating response")
    finally:
        if redis_claim_state is not None:
            redis_client_instance = redis_claim_state["client"]
            active_responder_key = redis_claim_state["active_responder_key"]
            source_count_key = redis_claim_state["source_count_key"]
            bot_identity = redis_claim_state["bot_identity"]
            # Release the typing floor only if we still own it.
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

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
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
