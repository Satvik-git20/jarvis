"""Entry point.

    python -m jarvis            start the voice/text loop
    python -m jarvis.daemon     start the HTTP API for opencode
    python -m jarvis doctor     health-check every integration
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .config import get_settings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="jarvis", description=__doc__)
    ap.add_argument("--local-brain", action="store_true",
                    help="prefer the local Ollama model over cloud providers")
    sub = ap.add_subparsers(dest="command")

    serve = sub.add_parser("daemon", help="run the HTTP API for the opencode plugin")
    serve.add_argument("--port", type=int, default=None)

    sub.add_parser("doctor", help="health-check every integration")
    sub.add_parser("devices", help="list microphones and speakers")
    sub.add_parser("test-voice", help="test the mic, stt and speakers in turn")

    wake = sub.add_parser("test-wake", help="record 3s and score the wake word")
    wake.add_argument("--threshold", type=float, default=0.5)

    talk = sub.add_parser("talk", help="one-shot text question, prints the answer")
    talk.add_argument("prompt", nargs="+")
    talk.add_argument("--local-brain", action="store_true")

    run = sub.add_parser("run", help="start the voice conversation loop")
    run.add_argument("--wake", action="store_true", help="listen for 'hey jarvis'")
    run.add_argument("--no-speak", action="store_true", help="do not speak replies")
    run.add_argument("--local-brain", action="store_true")
    run.add_argument("--whisper", default="small", help="tiny|base|small|medium")
    run.add_argument("--cpu", action="store_true", help="run whisper on the cpu")
    run.add_argument("--tts-local", action="store_true", help="prefer offline kokoro")
    run.add_argument("--voice", default=None, help="edge-tts voice name")
    run.add_argument("--mic", default=None, help="input device index")
    run.add_argument("--speaker", default=None, help="output device index")
    run.add_argument("--silence-ms", type=int, default=700)
    run.add_argument("--rms-floor", type=float, default=None,
                     help="reject audio quieter than this; see: test-voice --calibrate")
    run.add_argument("--vad-threshold", type=float, default=None,
                     help="speech probability required to count as speech (0-1)")
    run.add_argument("--verbose", action="store_true")

    args = ap.parse_args(argv)

    if args.command == "doctor":
        from scripts.doctor import main as doctor_main

        # Pass an empty argv: `doctor` itself is not one of the doctor's own
        # options, and re-parsing sys.argv would reject it.
        return asyncio.run(doctor_main([]))

    if args.command == "daemon":
        from .daemon import serve

        serve()
        return 0

    if args.command == "devices":
        from scripts.test_voice import show_devices

        show_devices()
        return 0

    if args.command == "test-voice":
        from scripts.test_voice import run_checks

        return asyncio.run(run_checks())

    if args.command == "test-wake":
        from scripts.test_voice import run_wake_check

        return asyncio.run(run_wake_check(threshold=args.threshold))

    if args.command == "talk":
        return asyncio.run(_talk(" ".join(args.prompt), args.local_brain))

    from .ui.cli import run_cli

    # `run` options live on the subparser, so with no subcommand they do not
    # exist on the namespace at all. Read them through a defaulting getter
    # rather than assuming `args.wake` is there.
    opt = lambda name, default=None: getattr(args, name, default)  # noqa: E731

    voice = {
        "wake": opt("wake", False), "speak": not opt("no_speak", False),
        "whisper_model": opt("whisper", get_settings().whisper_model),
        "prefer_gpu": not opt("cpu", False),
        "tts_voice": opt("voice"), "tts_local": opt("tts_local", False),
        "input_device": _maybe_int(opt("mic")),
        "output_device": _maybe_int(opt("speaker")),
        "silence_ms": opt("silence_ms", 700),
        "rms_floor": opt("rms_floor"), "vad_threshold": opt("vad_threshold"),
        "verbose": opt("verbose", False),
    }
    return asyncio.run(run_cli(local_brain=args.local_brain, **voice))


def _maybe_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


async def _talk(prompt: str, local_brain: bool) -> int:
    from .config import get_settings
    from .core.budget import BudgetLedger
    from .core.providers.base import Message
    from .core.router import AllProvidersExhausted, Router

    s = get_settings()
    s.ensure_data_dir()
    router = Router(BudgetLedger(), s)
    try:
        result = await router.complete(
            [Message("user", prompt)],
            prefer="ollama" if local_brain else None,
        )
    except AllProvidersExhausted as exc:
        print(f"no provider could answer: {exc}", file=sys.stderr)
        return 1
    print(f"[{result.provider}/{result.model}  {result.latency_ms}ms]\n{result.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
