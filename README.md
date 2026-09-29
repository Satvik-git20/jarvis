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

### Privacy

Google's free tier states that prompt content is used to improve their
products. Set `JARVIS_EXCLUDE_PRIVACY_UNSAFE=1` to remove Gemini from the
chain entirely.

## Voice

```powershell
uv run python scripts/test_voice.py all        # check each stage on its own
uv run python scripts/test_voice.py calibrate  # measure your room's noise
uv run python -m jarvis run                    # start the loop
```

Press **Enter** to talk. Type a question instead at any time, and
`/quit`, `/status`, `/devices`, `/nospeak` work as commands.

```powershell
uv run python -m jarvis run --wake             # also listen for "hey jarvis"
uv run python -m jarvis run --cpu              # whisper on cpu, leave the gpu free
uv run python -m jarvis run --tts-local       # force offline speech
```

**How a turn works:** Silero VAD decides when you have stopped talking ->
faster-whisper on the GPU transcribes -> the router picks a brain -> the answer
is spoken. Measured on a GTX 1650: transcription ~650 ms for an 11 s utterance,
and the whole loop is usable in real time.

**If it never stops listening**, the room is loud enough that the VAD reads it
as speech. Run `test_voice.py calibrate` and put the suggested value in `.env`
as `JARVIS_RMS_FLOOR`. `--vad-threshold 0.6` is the other lever.

**About the wake word:** `hey_jarvis` is a real pretrained model, but it was
trained on human speech, so text-to-speech will not trigger it. Verify it with
your own voice:

```powershell
uv run python -m jarvis test-wake
```

Push-to-talk is the default because it always works.

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
