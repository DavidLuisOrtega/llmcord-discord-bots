<h1 align="center">
  llmcord
</h1>

<h3 align="center"><i>
  Talk to LLMs with your friends!
</i></h3>

<p align="center">
  <img src="https://github.com/user-attachments/assets/7791cc6b-6755-484f-a9e3-0707765b081f" alt="">
</p>

llmcord transforms Discord into a collaborative LLM frontend. It works with practically any LLM, remote or locally hosted.

## Features

### Reply-based chat system:
Just @ the bot to start a conversation and reply to continue. Build conversations with reply chains!

You can:
- Branch conversations endlessly
- Continue other people's conversations
- @ the bot while replying to ANY message to include it in the conversation

Additionally:
- When DMing the bot, conversations continue automatically (no reply required). To start a fresh conversation, just @ the bot. You can still reply to continue from anywhere.
- You can branch conversations into [threads](https://support.discord.com/hc/en-us/articles/4403205878423-Threads-FAQ). Just create a thread from any message and @ the bot inside to continue.
- Back-to-back messages from the same user are automatically chained together. Just reply to the latest one and the bot will see all of them.

---

### Model switching with `/model`:
![image](https://github.com/user-attachments/assets/568e2f5c-bf32-4b77-ab57-198d9120f3d2)

llmcord supports remote models from:
- [OpenAI API](https://platform.openai.com/docs/models)
- [xAI API](https://docs.x.ai/docs/models)
- [Google Gemini API](https://ai.google.dev/gemini-api/docs/models)
- [Mistral API](https://docs.mistral.ai/getting-started/models/models_overview)
- [Groq API](https://console.groq.com/docs/models)
- [OpenRouter API](https://openrouter.ai/models)

Or run local models with:
- [Ollama](https://ollama.com)
- [LM Studio](https://lmstudio.ai)
- [vLLM](https://github.com/vllm-project/vllm)

...Or use any other OpenAI compatible API server.

---

### And more:
- Supports image attachments when using a vision model (like gpt-5, grok-4, claude-4, etc.)
- Supports text file attachments (.txt, .py, .c, etc.)
- Customizable personality (aka system prompt)
- User identity aware (OpenAI API and xAI API only)
- Streamed responses (turns green when complete, automatically splits into separate messages when too long)
- Hot reloading config (you can change settings without restarting the bot)
- Displays helpful warnings when appropriate (like "⚠️ Only using last 25 messages" when the customizable message limit is exceeded)
- Caches message data in a size-managed (no memory leaks) and mutex-protected (no race conditions) global dictionary to maximize efficiency and minimize Discord API calls
- Fully asynchronous
- 1 Python file, ~300 lines of code

## Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/jakobdylanc/llmcord
   cd llmcord
   ```

2. Create a copy of "config-example.yaml" named "config.yaml" and set it up:

### Discord settings:

| Setting | Description |
| --- | --- |
| **bot_token** | Create a new Discord bot at [discord.com/developers/applications](https://discord.com/developers/applications) and generate a token under the "Bot" tab. Also enable "MESSAGE CONTENT INTENT". |
| **client_id** | Found under the "OAuth2" tab of the Discord bot you just made. |
| **status_message** | Set a custom message that displays on the bot's Discord profile.<br /><br />**Max 128 characters.** |
| **max_text** | The maximum amount of text allowed in a single message, including text from file attachments.<br /><br />Default: `100,000` |
| **max_images** | The maximum number of image attachments allowed in a single message.<br /><br />Default: `5`<br /><br />**Only applicable when using a vision model.** |
| **max_messages** | The maximum number of messages allowed in a reply chain. When exceeded, the oldest messages are dropped.<br /><br />Default: `25` |
| **max_response_chars** | Hard cap for each bot reply length. Set to `0` to disable. Useful for shorter multi-bot turns.<br /><br />Default: `0` |
| **response_cooldown_seconds** | Minimum delay between this bot's responses in the same channel/thread. Useful to reduce runaway bot-to-bot loops.<br /><br />Default: `0` (disabled) |
| **response_cooldown_jitter_seconds** | Adds random delay on top of `response_cooldown_seconds` (uniform range from `0` to this value) so bot turns are less synchronized.<br /><br />Default: `0` |
| **global_channel_cooldown_seconds** | Shared channel-level cooldown based on recent bot messages in channel history. Helps prevent many bots posting back-to-back at once.<br /><br />Default: `0` (disabled) |
| **global_channel_arbitration_jitter_seconds** | Small random wait before checking channel history to decide which bot wins a turn when several are triggered together.<br /><br />Default: `0` |
| **reaction_delay_base_seconds** | Base post-probability reaction delay before this bot attempts to claim the floor. Use per persona to simulate different social latencies.<br /><br />Default: `0` |
| **reaction_delay_jitter_seconds** | Random delay added to `reaction_delay_base_seconds` (uniform `0..N`) for natural staggered timing.<br /><br />Default: `0` |
| **group_response_chance** | In autonomous mode, probability (0-1) this bot responds to a non-directed message (not explicitly @mentioned and not a direct reply).<br /><br />Default: `1.0` |
| **greeting_response_chance** | Optional override probability (0-1) for greeting-like messages (`hi`, `hello`, etc.) in autonomous mode.<br /><br />Default: `1.0` |
| **response_priority_weight** | Multiplier applied to autonomous participation chance. Values over `1` make this bot more likely to join; values under `1` make it quieter.<br /><br />Default: `1.0` |
| **followup_response_chance** | After one bot has already claimed a message, probability (0-1) this bot can become the follow-up responder for that same source message.<br /><br />Default: `0.15` |
| **max_responses_per_source_message** | Hard cap on total bot responses for one source message when Redis coordination is enabled. Set to `2` for "usually 1-2 replies".<br /><br />Default: `2` |
| **source_message_window_seconds** | TTL window for tracking response count per source message in Redis.<br /><br />Default: `180` |
| **floor_lock_ttl_seconds** | TTL for the per-channel+message floor claim lock in Redis.<br /><br />Default: `45` |
| **active_responder_ttl_seconds** | TTL for the per-channel active responder lock while a bot is generating/typing.<br /><br />Default: `90` |
| **autonomous_bot_only_mode** | When `true`, in configured autonomous channels the bot can react to other bot-authored messages without requiring mentions/replies. It still ignores its own messages.<br /><br />Default: `false` |
| **autonomous_channel_ids** | Channel/category IDs where autonomous mode is active. Leave empty to apply autonomous mode to all channels this bot can access.<br /><br />Default: `[]` |
| **redis_url** | Optional Redis connection URL (`redis://...`). When set, enables distributed floor lock + active responder coordination across multiple bot containers.<br /><br />Default: `""` (disabled) |
| **afk_followup_enabled** | Enables delayed follow-up nudges for open-ended human messages when the conversation goes quiet.<br /><br />Default: `false` |
| **afk_open_question_only** | When `true`, only schedule AFK follow-ups for messages that look open-ended (question-like).<br /><br />Default: `true` |
| **afk_first_followup_seconds** | Delay before first AFK follow-up after a human source message.<br /><br />Default: `600` |
| **afk_first_followup_jitter_seconds** | Random +/- jitter applied to the first AFK follow-up delay for each source message.<br /><br />Default: `0` |
| **afk_second_followup_seconds** | Delay before second AFK follow-up after the same source message.<br /><br />Default: `3600` |
| **afk_second_followup_jitter_seconds** | Random +/- jitter applied to the second AFK follow-up delay for each source message.<br /><br />Default: `0` |
| **afk_max_followups_per_message** | Maximum AFK follow-ups per source human message.<br /><br />Default: `2` |
| **afk_followup_chance** | Probability (0-1) each AFK follow-up attempt actually sends (adds natural randomness).<br /><br />Default: `0.5` |
| **afk_cancel_on_any_human_message** | Cancel pending AFK follow-ups if any newer human message arrives in that channel.<br /><br />Default: `true` |
| **afk_scheduler_poll_seconds** | How often each bot checks Redis for due AFK follow-up jobs.<br /><br />Default: `5` |
| **quiet_hours_enabled** | Suppresses AFK follow-up sending during quiet hours (jobs are deferred, not dropped).<br /><br />Default: `false` |
| **quiet_hours_timezone** | IANA timezone for quiet-hour checks (example: `America/New_York`).<br /><br />Default: `UTC` |
| **quiet_hours_start_hour** | Quiet-hours start hour (0-23, local to `quiet_hours_timezone`).<br /><br />Default: `23` |
| **quiet_hours_end_hour** | Quiet-hours end hour (0-23, local to `quiet_hours_timezone`).<br /><br />Default: `8` |
| **proactive_starters_enabled** | Enables bot-initiated conversation starters when a channel has been idle.<br /><br />Default: `false` |
| **proactive_respect_quiet_hours** | When `true`, proactive starters are suppressed during configured quiet hours.<br /><br />Default: `true` |
| **proactive_idle_human_seconds** | Minimum time since the last human message before a proactive starter is allowed.<br /><br />Default: `1800` |
| **proactive_idle_channel_seconds** | Minimum time since any channel message before a proactive starter is allowed.<br /><br />Default: `900` |
| **proactive_starter_chance** | Probability (0-1) a due proactive check actually sends a starter message.<br /><br />Default: `0.35` |
| **proactive_max_per_day_per_channel** | Daily cap for proactive starters per channel.<br /><br />Default: `6` |
| **proactive_claim_ttl_seconds** | Distributed lock TTL for proactive starter claims per channel.<br /><br />Default: `45` |
| **proactive_channel_ids** | Optional list of channel IDs eligible for proactive starters. Empty means use observed channels.<br /><br />Default: `[]` |
| **proactive_mention_enabled** | When `true`, proactive starters may @mention a recently active human in-channel.<br /><br />Default: `false` |
| **proactive_mention_chance** | Probability (0-1) a proactive starter attempt uses a targeted @mention format.<br /><br />Default: `0.5` |
| **proactive_mention_recent_user_seconds** | How far back to consider users as "recently active" for proactive @mentions.<br /><br />Default: `172800` (48 hours) |
| **proactive_mention_max_per_user_per_day** | Per-user daily cap for proactive in-channel @mentions.<br /><br />Default: `1` |
| **proactive_starter_templates** | Candidate short opener lines used for proactive bot-initiated messages.<br /><br />Default: three generic templates |
| **proactive_mention_templates** | Candidate templates for targeted proactive starters. Use `{mention}` placeholder to insert a user ping.<br /><br />Default: two templates |
| **afk_followup_templates** | Candidate short follow-up lines used when AFK reminders trigger.<br /><br />Default: two generic templates |
| **generated_user_mentions_mode** | Mention policy for generated replies: `always`, `question_only`, or `never`. `question_only` strips user mentions unless the message is phrased as a question.<br /><br />Default: `question_only` |
| **use_plain_responses** | When set to `true` the bot will use plaintext responses instead of embeds. Plaintext responses have a shorter character limit so the bot's messages may split more often.<br /><br />Default: `false`<br /><br />**Also disables streamed responses and warning messages.** |
| **allow_dms** | Set to `false` to disable direct message access.<br /><br />Default: `true` |
| **permissions** | Configure access permissions for `users`, `roles` and `channels`, each with a list of `allowed_ids` and `blocked_ids`.<br /><br />Control which `users` are admins with `admin_ids`. Admins can change the model with `/model` and DM the bot even if `allow_dms` is `false`.<br /><br />**Leave `allowed_ids` empty to allow ALL in that category.**<br /><br />**Role and channel permissions do not affect DMs.**<br /><br />**You can use [category](https://support.discord.com/hc/en-us/articles/115001580171-Channel-Categories-101) IDs to control channel permissions in groups.** |

### LLM settings:

| Setting | Description |
| --- | --- |
| **providers** | Add the LLM providers you want to use, each with a `base_url` and optional `api_key` entry. Popular providers (`openai`, `openrouter`, `ollama`, etc.) are already included.<br /><br />**Only supports OpenAI compatible APIs.**<br /><br />**Some providers may need `extra_headers` / `extra_query` / `extra_body` entries for extra HTTP data. See the included `azure-openai` provider for an example.** |
| **models** | Add the models you want to use in `<provider>/<model>: <parameters>` format (examples are included). When you run `/model` these models will show up as autocomplete suggestions.<br /><br />**Refer to each provider's documentation for supported parameters.**<br /><br />**The first model in your `models` list will be the default model at startup.**<br /><br />**Some vision models may need `:vision` added to the end of their name to enable image support.** |
| **system_prompt** | Write anything you want to customize the bot's behavior!<br /><br />**Leave blank for no system prompt.**<br /><br />**You can use the `{date}` and `{time}` tags in your system prompt to insert the current date and time, based on your host computer's time zone.** |

3. Run the bot:

   **No Docker:**
   ```bash
   python -m pip install -U -r requirements.txt
   python llmcord.py
   ```

   To use a different config file:
   ```bash
   CONFIG_FILE=config_somebot.yaml python llmcord.py
   ```

   **With Docker:**
   ```bash
   docker compose up
   ```

## Five-bot private channel setup

Use one config per bot account and run all five processes at once.

- `config_advocate.yaml`
- `config_architect.yaml`
- `config_coder.yaml`
- `config_creative.yaml`
- `config_security.yaml`

Recommended safeguards for bot-only channels:
- Set `permissions.channels.allowed_ids` to your private channel ID in each config.
- Keep `response_cooldown_seconds` enabled (example: `2.0`).
- Set `redis_url` so all bot containers coordinate through distributed floor locking.
- Set `autonomous_bot_only_mode: true` and populate `autonomous_channel_ids` with your private channel ID.
- Keep role-specific `system_prompt` text so each bot adds different value.

Local run (5 terminals):
```bash
CONFIG_FILE=config_advocate.yaml python llmcord.py
CONFIG_FILE=config_architect.yaml python llmcord.py
CONFIG_FILE=config_coder.yaml python llmcord.py
CONFIG_FILE=config_creative.yaml python llmcord.py
CONFIG_FILE=config_security.yaml python llmcord.py
```

Docker run (all 5 at once):
```bash
docker compose up --build
```

## Notes

- If you're having issues, try my suggestions [here](https://github.com/jakobdylanc/llmcord/issues/19)

- Only models from OpenAI API and xAI API are "user identity aware" because only they support the "name" parameter in the message object. Hopefully more providers support this in the future.

- PRs are welcome :)

## Star History

<a href="https://star-history.com/#jakobdylanc/llmcord&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date" />
  </picture>
</a>
