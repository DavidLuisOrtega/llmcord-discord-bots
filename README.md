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
| **bot_to_bot_response_chance_multiplier** | Multiplier applied when the triggering message author is another bot, reducing bot-to-bot chain momentum.<br /><br />Default: `0.5` |
| **bot_to_bot_response_chance_floor** | Minimum response probability used for bot-authored trigger messages in autonomous mode. Useful for increasing bot-to-bot conversation frequency without changing human-message response rates.<br /><br />Default: `0.0` |
| **max_consecutive_bot_turns_without_human** | Maximum consecutive bot-authored turns allowed in a channel before bots pause until a human posts.<br /><br />Default: `4` |
| **pair_back_and_forth_cooldown_seconds** | Cooldown to discourage immediate A↔B back-and-forth loops between the same bot pair.<br /><br />Default: `60` |
| **followup_response_chance** | After one bot has already claimed a message, probability (0-1) this bot can become the follow-up responder for that same source message.<br /><br />Default: `0.15` |
| **max_responses_per_source_message** | Hard cap on total bot responses for one source message when Redis coordination is enabled. Set to `2` for "usually 1-2 replies".<br /><br />Default: `2` |
| **bot_to_bot_followup_response_chance** | Follow-up probability override used when the source message author is another bot. Higher values increase bot-to-bot chatter while keeping source-message caps.<br /><br />Default: `0.3` |
| **bot_to_bot_max_responses_per_source_message** | Source-message response cap override used for bot-authored source messages. Lets bot-to-bot exchanges run slightly longer than human-triggered turns.<br /><br />Default: `3` |
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
| **afk_followup_generation_retries** | Number of model-generation attempts for AFK follow-up text before fallback text is used.<br /><br />Default: `2` |
| **afk_followup_max_chars** | Maximum characters for generated AFK follow-up text.<br /><br />Default: `140` |
| **afk_followup_fallback_text** | Optional fallback AFK follow-up text when generation fails. Leave empty to skip the follow-up instead of sending scripted text.<br /><br />Default: `""` |
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
| **proactive_generation_retries** | Number of generation attempts for proactive openers before fallback text is used.<br /><br />Default: `3` |
| **proactive_dedupe_window_seconds** | Duplicate suppression window for proactive starter text in a channel.<br /><br />Default: `86400` |
| **proactive_generated_max_chars** | Max characters for generated proactive starters.<br /><br />Default: `180` |
| **proactive_fallback_starter** | Optional fallback proactive message if generation fails or all attempts are duplicates. Leave empty to skip sending instead of posting scripted text.<br /><br />Default: `""` |
| **proactive_mention_enabled** | When `true`, proactive starters may @mention a recently active human in-channel.<br /><br />Default: `false` |
| **proactive_mention_chance** | Probability (0-1) a proactive starter attempt uses a targeted @mention format.<br /><br />Default: `0.5` |
| **proactive_mention_recent_user_seconds** | How far back to consider users as "recently active" for proactive @mentions.<br /><br />Default: `172800` (48 hours) |
| **proactive_mention_max_per_user_per_day** | Per-user daily cap for proactive in-channel @mentions.<br /><br />Default: `1` |
| **proactive_bot_to_bot_enabled** | Enables proactive starters where one bot addresses another bot in-channel.<br /><br />Default: `true` |
| **proactive_bot_to_bot_chance** | Probability (0-1) that an eligible proactive attempt is bot-to-bot instead of human-targeted.<br /><br />Default: `0.2` |
| **proactive_bot_to_bot_max_per_day_per_channel** | Daily cap for bot-to-bot proactive starters per channel.<br /><br />Default: `2` |
| **proactive_bot_to_bot_cooldown_seconds** | Minimum spacing between bot-to-bot proactive starters in the same channel.<br /><br />Default: `1800` |
| **proactive_bot_to_bot_max_chain_without_human** | Maximum consecutive bot-to-bot proactive starters allowed before a human message resets the chain.<br /><br />Default: `1` |
| **proactive_bot_to_bot_recent_bot_seconds** | Time window used to select recently active bot candidates for bot-to-bot starters.<br /><br />Default: `172800` |
| **proactive_bot_to_bot_name_mode** | Name style for bot-to-bot starters. Use `plain` to avoid mentions/pings.<br /><br />Default: `plain` |
| **implicit_targeting_enabled** | Enables implicit recipient detection for short untagged human follow-ups (for example: `booo`, `lol`, `nah`).<br /><br />Default: `true` |
| **implicit_targeting_window_seconds** | Max age of the most recent bot message to infer as the likely target.<br /><br />Default: `10` |
| **implicit_targeting_max_chars** | Max message length eligible for implicit-target heuristics.<br /><br />Default: `40` |
| **implicit_targeting_fallback_wait_seconds** | Time to give inferred target bot first chance before allowing fallback responses.<br /><br />Default: `4` |
| **generated_user_mentions_mode** | Mention policy for generated replies: `always`, `question_only`, or `never`. `question_only` strips user mentions unless the message is phrased as a question.<br /><br />Default: `question_only` |
| **strict_reply_targeting** | When `true`, replies to another bot's message are ignored unless this bot is explicitly @mentioned. Prevents cross-bot hijacking of direct replies.<br /><br />Default: `true` |
| **direct_mention_retry_enabled** | When `true`, direct @mentions/replies are retried after cooldowns instead of being dropped immediately.<br /><br />Default: `true` |
| **direct_mention_fast_lane_enabled** | When `true`, direct @mentions/replies bypass normal cooldown gates for faster response handling.<br /><br />Default: `true` |
| **direct_mention_max_wait_seconds** | Maximum total wait time for a directed message to clear retryable gates before giving up.<br /><br />Default: `180` |
| **treat_everyone_as_directed** | When `true`, `@everyone` messages are treated like directed prompts for trigger logic while still respecting anti-dogpile pacing/caps.<br /><br />Default: `false` |
| **gif_replies_enabled** | Enables optional curated GIF replies in-channel after normal bot text responses.<br /><br />Default: `false` |
| **gif_reply_chance** | Probability (0-1) of sending a GIF reply when GIF mode is enabled.<br /><br />Default: `0.1` |
| **gif_reply_cooldown_seconds** | Minimum delay between GIF replies in a channel.<br /><br />Default: `180` |
| **gif_reply_max_per_hour_per_channel** | Maximum GIF replies allowed per channel per hour.<br /><br />Default: `4` |
| **gif_recent_dedupe_window_seconds** | Time window for suppressing recently used GIF URLs in a channel.<br /><br />Default: `21600` |
| **gif_bad_url_cooldown_seconds** | Quarantine duration for GIF URLs that repeatedly fail fetch/resolve checks in a channel.<br /><br />Default: `86400` |
| **gif_bad_url_max_failures** | Number of failed fetch attempts before a GIF URL is temporarily quarantined.<br /><br />Default: `2` |
| **gif_contextual_selection_enabled** | When `true`, ranks GIF candidates by overlap between message context and configured GIF tags.<br /><br />Default: `true` |
| **gif_reply_keyword_filters** | Optional list of lowercase keywords; when set, GIF replies only trigger if a keyword appears in the human prompt or bot response text.<br /><br />Default: `[]` |
| **gif_reply_urls** | Curated list of GIF or GIF-page URLs. The bot fetches and uploads the resolved image so Discord renders media instead of bare links. Leave empty to disable GIF output even if enabled.<br /><br />Default: example list |
| **gif_catalog** | Optional tagged GIF catalog (`url` + `tags`) used for context-aware GIF selection. Falls back to `gif_reply_urls` when no tag match is found.<br /><br />Default: example tagged entries |
| **emoji_reactions_enabled** | Enables optional emoji reactions (e.g. 👍 🔥 😂) on incoming messages.<br /><br />Default: `false` |
| **emoji_reaction_chance** | Probability (0-1) of adding an emoji reaction when reaction mode is enabled.<br /><br />Default: `0.15` |
| **emoji_reaction_cooldown_seconds** | Minimum delay between emoji reactions in a channel.<br /><br />Default: `60` |
| **emoji_reaction_max_per_hour_per_channel** | Maximum emoji reactions allowed per channel per hour.<br /><br />Default: `8` |
| **emoji_reaction_contextual_selection_enabled** | When `true`, ranks candidate reaction emojis using message context and catalog tags before fallback choices.<br /><br />Default: `true` |
| **emoji_reaction_keyword_filters** | Optional lowercase keyword gate for emoji reactions based on human prompt or generated reply text.<br /><br />Default: `[]` |
| **emoji_reaction_choices** | Candidate emoji list used for reactions; one is chosen randomly per reaction.<br /><br />Default: `["👍","🔥","😂","👏","💯"]` |
| **emoji_reaction_catalog** | Optional tagged emoji catalog (`emoji` + `tags`) for context-aware reaction selection. Falls back to `emoji_reaction_choices` if no tag matches.<br /><br />Default: example tagged entries |
| **mood_injector_enabled** | Enables optional mood injection into the system prompt to add human-like variability.<br /><br />Default: `false` |
| **mood_rotation_mode** | Mood rotation cadence: `daily`, `hourly`, or `per_message`.<br /><br />Default: `daily` |
| **mood_influence_strength** | How strongly mood affects tone: `subtle`, `medium`, `strong`.<br /><br />Default: `subtle` |
| **mood_pool** | List of mood strings available for injection when mood mode is enabled.<br /><br />Default: included example list |
| **discord_chat_global_style_prompt_enabled** | Appends a global Discord-style behavior rule to the system prompt (short, playful, no list-heavy formatting).<br /><br />Default: `true` |
| **discord_chat_style_enabled** | Enables output post-processing to flatten bullets/lists and keep replies chat-like.<br /><br />Default: `true` |
| **discord_chat_max_sentences** | Maximum sentence count kept after chat-style post-processing.<br /><br />Default: `2` |
| **discord_chat_style_max_chars** | Character cap applied by chat-style post-processing (set `0` to disable this cap).<br /><br />Default: `220` |
| **persona_speech_enforcement_enabled** | Enables optional post-generation speech enforcement for persona-specific voice shaping. Useful when model defaults sound too polished for a character.<br /><br />Default: `false` |
| **persona_speech_profile** | Selects the speech profile used by enforcement. Built-in profiles: `kevin`, `saul`, `sarah`, `katherine`, `damon`.<br /><br />Default: `""` |
| **persona_avoid_witty_phrasing** | When speech enforcement is enabled, strips common witty/snarky transition phrases to keep wording more literal.<br /><br />Default: `true` |
| **persona_max_word_length** | Replaces overly long words in enforced persona mode. Set to `0` to disable.<br /><br />Default: `0` |
| **persona_blocklist_phrases** | Optional list of phrases removed from generated replies in enforced persona mode.<br /><br />Default: `[]` |
| **persona_preferred_fillers** | Optional list of simple fillers (for example: `honestly`, `my gut`) that can be prefixed when absent, to reinforce persona texture.<br /><br />Default: `[]` |
| **persona_misspell_chance** | Kevin-only option. Chance (0-1) to inject a light intentional typo so Kevin sounds less polished.<br /><br />Default: `0.0` |
| **persona_playful_jab_chance** | Damon-only option. Chance (0-1) to append a playful jab/roast sentence.<br /><br />Default: `0.0` |
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

- `config_kevin.yaml`
- `config_saul.yaml`
- `config_katherine.yaml`
- `config_damon.yaml`
- `config_sarah.yaml`

Recommended safeguards for bot-only channels:
- Set `permissions.channels.allowed_ids` to your private channel ID in each config.
- Keep `response_cooldown_seconds` enabled (example: `2.0`).
- Set `redis_url` so all bot containers coordinate through distributed floor locking.
- Set `autonomous_bot_only_mode: true` and populate `autonomous_channel_ids` with your private channel ID.
- Keep role-specific `system_prompt` text so each bot adds different value.

Local run (5 terminals):
```bash
CONFIG_FILE=config_kevin.yaml python llmcord.py
CONFIG_FILE=config_saul.yaml python llmcord.py
CONFIG_FILE=config_katherine.yaml python llmcord.py
CONFIG_FILE=config_damon.yaml python llmcord.py
CONFIG_FILE=config_sarah.yaml python llmcord.py
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
