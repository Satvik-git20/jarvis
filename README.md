# JARVIS

A voice-first assistant that runs on free AI tiers, and exposes itself to
[opencode](https://opencode.ai) as a set of tools.

Nothing here costs money and nothing needs a credit card. Every provider is a
no-card free tier or a local model, and the router fails over between them
automatically so hitting one provider's rate limit does not end your request.

## What is connected

**Brains** (chat, tool-calling, reasoning) - tried in this order:

| Provider | Free allowance | Card |
|---|---|---|
| Ollama (local) | unlimited, offline | no |
| OpenRouter `:free` | 20 rpm / 50 rpd (1000 rpd after $10 lifetime) | no |
| Groq | 30 rpm / 1000 rpd / 200k tpd | no |
| Cerebras | 30 rpm / 14.4k rpd / 1M tpd | no |
| Google Gemini | free on Flash; limits per project | no |

**Voice** - `faster-whisper` local on the GPU with Groq
`whisper-large-v3-turbo` as fallback; Edge TTS for replies with Kokoro-82M as
the fully offline option.

**Search** - Tavily, then Jina Reader, then `ddgs`.

**Also** - Pollinations (image gen), Jina + Ollama (embeddings), RapidOCR
(OCR), Open-Meteo (weather), Wikipedia and Hacker News (knowledge).

## Setup

```powershell
uv sync                 # core, a few seconds
uv sync --extra search --extra memory
uv sync --extra audio --extra stt-cuda --extra tts --extra wake
Copy-Item .env.example .env    # then fill in whatever keys you have
uv run python scripts/doctor.py   # health-check every integration
```

The system works with zero keys configured. Keyless services (local Ollama,
ddgs, Jina Reader keyless, Open-Meteo, Wikipedia, HN, local Whisper) come up on
their own; adding keys only widens the failover chain.

### Keys

Put keys in `.env`, which is gitignored. For OpenRouter you do not need to
copy anything: if `.env` has no `OPENROUTER_API_KEY`, JARVIS falls back to the
credential opencode already stores in `~/.local/share/opencode/auth.json`.
Your own `.env` always wins.

`jarvis doctor` lists every key you are missing with the exact signup URL:

```
To switch these on  (all free, no card)
  groq          GROQ_API_KEY      https://console.groq.com/keys
  cerebras      CEREBRAS_API_KEY  https://cloud.cerebras.ai
  gemini        GEMINI_API_KEY    https://aistudio.google.com/apikey
  tavily        TAVILY_API_KEY    https://app.tavily.com
  jina_embed    JINA_API_KEY      https://jina.ai/embeddings
  pollinations  POLLINATIONS_KEY  https://enter.pollinations.ai
```

None are required. Adding them only widens the failover chain.

### Privacy

Google's free tier states that prompt content is used to improve their
products. Set `JARVIS_EXCLUDE_PRIVACY_UNSAFE=1` to remove Gemini from the
chain entirely.

### Durable storage

JARVIS stores conversations, message history, provider usage, cooldowns, and
future conversation summaries in `.jarvis/jarvis.db`. SQLite runs in WAL mode
with transactional migrations. The daemon is the only process that writes this
file: the voice loop, `jarvis talk`, and the Chrome panel are HTTP clients of
the daemon, so there is exactly one writer per database.

Existing installations are backward compatible: on first startup JARVIS
imports `.jarvis/sessions.json` and `.jarvis/budget.json` into SQLite exactly
once. The legacy files are left untouched as a recovery copy; after import,
all new conversation and budget writes go only to `jarvis.db`.

The schema currently contains `conversations`, `messages`, `summaries`, and
`provider_logs`. Migration versions are recorded in the database and every
migration is applied atomically. Back up `.jarvis/jarvis.db` before moving the
data directory or performing a manual downgrade.

**Migration path.** `v0` (racy `sessions.json` + `budget.json`) becomes `v1`
(`jarvis.db`) on first open: the legacy files are imported inside one
transaction, guarded by the `legacy_imports` table so it happens exactly once,
and are then left untouched as a recovery copy. Every later schema change is a
new `Migration(n, ...)` applied atomically and recorded in `schema_migrations`;
a failed migration rolls back its partial schema and does not record a version,
so retrying is always safe.

**Rollback.** Stop JARVIS, delete `.jarvis/jarvis.db`, and start again: the
untouched legacy JSON re-imports and you are back at `v0`. To keep current data
while downgrading the code, copy `jarvis.db` aside first — old code cannot read
it, but nothing is lost when you upgrade again.

**Concurrency.** The daemon is the sole writer of `jarvis.db`; it opens the
database with WAL plus `BEGIN IMMEDIATE` transactions and a 10s busy timeout,
so the rare second writer (a migration run, `jarvis doctor`, a second daemon
starting up) queues instead of failing. Every other entry point — voice CLI,
`jarvis talk`, the Chrome panel, opencode — goes through the daemon's HTTP API
(`/ask`, `/status`, `/session/...`), and none of them opens the database. The
daemon must be running for `jarvis run` and `jarvis talk` to get an answer;
if it is down they say so instead of writing a second copy of the state.

On Windows the daemon runs uvicorn on the *selector* event loop rather than
the default proactor loop: the proactor accept path permanently stops
accepting after a single failed `AcceptEx` (a client that dies in the accept
queue), leaving the daemon reachable-looking but unresponsive. The selector
loop keeps its read handler armed and heals instead — see
`uvicorn_loop_setting()` in `jarvis/daemon.py`.

## Voice

```powershell
uv run python scripts/test_voice.py all        # check each stage on its own
uv run python scripts/test_voice.py calibrate  # measure your room's noise
uv run python -m jarvis run                    # start the loop
```

Press **Enter** to talk. Type a question instead at any time, and
`/quit`, `/status`, `/devices`, `/nospeak` work as commands.
The loop answers through the daemon, so start it first in another terminal —
the voice session, the Chrome panel, and opencode then share one context.

```powershell
uv run python -m jarvis run --wake             # also listen for "hey jarvis"
uv run python -m jarvis run --cpu              # whisper on cpu, leave the gpu free
uv run python -m jarvis run --tts-local       # force offline speech
```

**How a turn works:** Silero VAD decides when you have stopped talking ->
faster-whisper on the GPU transcribes -> the router picks a brain -> the answer
is spoken. Measured on a GTX 1650: transcription ~650 ms for an 11 s utterance,
and the whole loop is usable in real time.

**No calibration is needed.** The gate tracks the room's noise level at runtime
and accepts speech at 3x that, so it works in a quiet office and next to a fan
without tuning. A fixed threshold could not: measured here, the floor was 0.0000
while someone talked it read 0.14, and a value tuned during the noisy moment
would have rejected ordinary quiet speech outright.

**About the wake word:** it does not work on Windows + Python 3.12, and the code
says so rather than pretending otherwise. openWakeWord's ONNX path returns
`0.0000` for every model on real human speech that contains its own activation,
while random feature vectors score 0.95 from the same session — the graph runs,
the feature pipeline feeding it does not. Reproduced on 0.5.1 and 0.6.0. Their
docs say tflite is the preferred backend on x86, and `tflite-runtime` ships no
Windows wheel for 3.12, so only the broken path installs.

`jarvis test-wake` prints the probe scores. Push-to-talk is the default input
method and needs none of this.

## Chrome side panel

Chat with JARVIS from a Chrome side panel — the same daemon, sessions, and
budget as the voice loop and opencode.

```powershell
uv run python -m jarvis daemon            # must be running
```

Then `chrome://extensions` → **Developer mode** → **Load unpacked** → select
`chrome-extension/`. Click the JARVIS toolbar icon, open **Settings**, and paste
`JARVIS_DAEMON_TOKEN` from `.env`.

The extension only ever talks to `127.0.0.1:8765`. The daemon answers CORS for
`chrome-extension://` origins only: a web page can reach the loopback port but
cannot read a response, and every route still requires the token regardless.
The token is stored in that browser profile's local storage and never synced.
See `chrome-extension/README.md` for details.

## Autostart

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_autostart.ps1
```

Registers a hidden scheduled task so the daemon is up at logon, plus Start Menu
shortcuts for the voice loop and the daemon. Undo with the same script and
`-Remove`.

## Checking for leaked keys

```powershell
uv run python scripts/check_secrets.py
```

Reports which files hold a live credential, whether git tracks any of them, and
whether `.env` is ignored — without printing a single secret value. Run it
before pushing anywhere public.

## Running

```powershell
uv run python -m jarvis daemon     # HTTP API on 127.0.0.1:8765 for opencode
uv run python -m jarvis run        # voice/text loop
uv run python -m jarvis doctor     # health-check every integration
uv run python -m jarvis talk "what is 2+2"
```

The daemon binds to loopback only and requires the `JARVIS_DAEMON_TOKEN`
header, because any web page your browser loads can reach `127.0.0.1`. It is an
API, not a website: `/` returns 404, `/health` returns JSON, and `/docs` is a
Swagger page.

Start the daemon first: `jarvis run` and `jarvis talk` are HTTP clients of it,
and will print a startup hint (or a clear error) when it is not running.

## opencode integration

`~/.config/opencode/plugins/jarvis.ts` is loaded automatically and registers
five tools with opencode:

- `jarvis_ask` - route a question through JARVIS's own free-AI brain
- `jarvis_search` - live web results via the search cascade
- `jarvis_remember` - read/write long-term memory (sqlite-vec)
- `jarvis_image` - generate an image with Pollinations
- `jarvis_status` - provider health and remaining daily budget

Because conversation state lives in the daemon, a `jarvis_ask` from opencode
and a spoken "Hey Jarvis" share the same context and the same rate-limit
budget.

## Hardware notes

This was built against a GTX 1650 (4 GB VRAM) and 16 GB RAM. Whisper and
Ollama cannot both hold the GPU, so the default is Whisper on CUDA with a
cloud brain. `--local-brain` swaps that: Ollama takes the GPU and Whisper
drops to CPU int8.
