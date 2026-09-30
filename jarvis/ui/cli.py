"""The JARVIS conversation loop.

Three ways in, one way out:

    press Enter      push-to-talk, the default and the most reliable
    say "hey jarvis" opt-in wake word, once verified with your own voice
    type a question  always available, useful when the room is noisy

All three share the same session and the same provider budget as the opencode
tools, because they all read the same daemon-equivalent state held in-process.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import time
from dataclasses import dataclass

from ..config import Settings, get_settings
from ..core.budget import BudgetLedger
from ..core.conversation import DEFAULT_SYSTEM, SessionStore
from ..core.providers.base import Message
from ..core.router import AllProvidersExhausted, Router
from ..voice import tts as tts_mod
from ..voice.audio import TARGET_SR, Recorder, list_inputs, list_outputs
from ..voice.stt import Transcriber
from ..voice.wake import WakeWord

log = logging.getLogger("jarvis.cli")

GREEN, RED, YELLOW, DIM, CYAN, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[36m", "\033[1m", "\033[0m"
)


@dataclass
class Config:
    wake: bool = False
    speak: bool = True
    local_brain: bool = False
    input_device: int | str | None = None
    output_device: int | str | None = None
    whisper_model: str = "small"
    prefer_gpu: bool = True
    tts_voice: str | None = None
    tts_local: bool = False
    silence_ms: int = 700
    rms_floor: float | None = None
    vad_threshold: float | None = None
    session_id: str | None = None
    quiet: bool = False
    verbose: bool = False


class Jarvis:
    def __init__(self, cfg: Config, settings: Settings | None = None):
        self.cfg = cfg
        self.settings = settings or get_settings()
        self.settings.ensure_data_dir()
        data = self.settings.data_dir
        self.router = Router(BudgetLedger(data / "budget.json"), self.settings)
        self.sessions = SessionStore(data / "sessions.json")
        self.stt = Transcriber(self.settings, model=cfg.whisper_model,
                               prefer_gpu=cfg.prefer_gpu)
        self.speaker = tts_mod.Speaker(self.settings, voice=cfg.tts_voice,
                                       prefer_local=cfg.tts_local,
                                       playback_device=cfg.output_device)
        self.recorder = Recorder(
            cfg.input_device,
            silence_ms=cfg.silence_ms,
            threshold=self.settings.vad_threshold if cfg.vad_threshold is None
            else cfg.vad_threshold,
            rms_floor=self.settings.rms_floor if cfg.rms_floor is None
            else cfg.rms_floor,
        )
        self.wake = WakeWord() if cfg.wake else None
        self._stop = False

    # --- helpers ---------------------------------------------------------

    def say(self, text: str, *, colour: str = "") -> None:
        if not self.cfg.quiet:
            print(f"{colour}{text}{RESET}", flush=True)

    def status_line(self) -> str:
        ready = self.router.ready()
        dev = "gpu" if self.stt.prefer_gpu else "cpu"
        return (f"{DIM}providers: {','.join(ready) or 'none'} · "
                f"stt: whisper-{self.cfg.whisper_model}/{dev} · "
                f"tts: {'kokoro' if self.cfg.tts_local else 'edge'}"
                f"{RESET}")

    async def ask(self, prompt: str) -> str:
        sid = self.sessions.get_or_create(self.cfg.session_id)
        msgs = self.sessions.history(sid, system=DEFAULT_SYSTEM)
        msgs.append(Message("user", prompt))
        try:
            result = await self.router.complete(
                msgs, prefer="ollama" if self.cfg.local_brain else None, timeout=300.0
            )
        except AllProvidersExhausted as exc:
            detail = "; ".join(f"{n}: {w}" for n, w in exc.attempts)
            return f"I could not reach any AI provider. {detail}"
        self.sessions.append(sid, prompt, result.text)
        self.say(f"    {DIM}[{result.provider}/{result.model} {result.latency_ms}ms]{RESET}",
                 colour=DIM)
        return result.text

    async def respond(self, text: str) -> None:
        answer = await self.ask(text)
        self.say(f"{GREEN}JARVIS{RESET} {answer}")
        if self.cfg.speak and answer:
            speech = await self.speaker.synthesize(answer)
            if speech.audio.size:
                self.say(f"    {DIM}(speaking, {speech.engine}, "
                         f"{speech.duration:.1f}s){RESET}", colour=DIM)
                await asyncio.to_thread(self.speaker.play, speech)
            else:
                self.say(f"    {RED}(no tts audio){RESET}", colour=RED)

    # --- input modes -----------------------------------------------------

    def _check_mic(self) -> bool:
        try:
            inputs = list_inputs()
        except Exception as exc:
            self.say(f"no audio input available: {exc}", colour=RED)
            return False
        if not inputs:
            self.say("no microphone found", colour=RED)
            return False
        chosen = self.recorder.device
        dev = next((d for d in inputs if d.index == chosen), inputs[0] if chosen is None
                   else None)
        name = dev.name if dev else str(chosen)
        self.say(f"    {DIM}mic: {name}{RESET}", colour=DIM)
        return True

    def listen_once(self) -> str | None:
        """Block until one utterance ends, then transcribe it."""
        self.say(f"{CYAN}listening...{RESET}")
        try:
            for utterance in self.recorder.utterances(timeout=30.0):
                if utterance.duration < 0.25:
                    continue
                tr = self.stt.transcribe(utterance.audio, TARGET_SR)
                self.say(f"    {DIM}heard ({tr.engine}, {tr.latency_ms}ms): "
                         f"{tr.text!r}{RESET}", colour=DIM)
                if tr.ok:
                    return tr.text
                return None
        except KeyboardInterrupt:
            return None
        self.say(f"    {DIM}heard nothing{RESET}", colour=DIM)
        return None

    async def wake_loop_once(self, seconds: float = 30.0) -> bool:
        """Listen for the wake word, then capture the request that follows."""
        deadline = time.time() + seconds
        self.say(f"{CYAN}waiting for \"hey jarvis\"...{RESET}")
        got = False
        try:
            for utterance in self.recorder.utterances(timeout=seconds):
                if self.wake and self.wake.detect(utterance.audio):
                    self.say(f"{GREEN}wake word detected{RESET}")
                    got = True
                    break
                if time.time() > deadline:
                    break
        except KeyboardInterrupt:
            return False
        return got

    # --- main loop -------------------------------------------------------

    async def run(self) -> int:
        self.say(f"{BOLD}JARVIS{RESET} {DIM}v0.1.0 · /quit to exit · "
                 f"Enter=push-to-talk{', say the wake word' if self.cfg.wake else ''}{RESET}")
        self.say(self.status_line(), colour=DIM)
        if self.cfg.speak:
            hint = self.speaker.kokoro_setup_hint()
            if hint and not self.cfg.tts_local:
                self.say(f"    {YELLOW}offline TTS not installed (edge-tts in use){RESET}",
                         colour=YELLOW)
        if self.cfg.wake and self.wake and not self.wake.available():
            self.say(f"    {RED}wake word unavailable: {self.wake.import_error}{RESET}",
                     colour=RED)
            self.wake = None
        elif self.cfg.wake and self.wake:
            healthy, detail = self.wake.selftest()
            if not healthy:
                self.say(f"    {RED}wake word model is dead on this machine{RESET}")
                self.say(f"      {DIM}{detail}{RESET}", colour=DIM)
                self.say(f"      {DIM}falling back to push-to-talk{RESET}", colour=DIM)
                self.cfg.wake = False
                self.wake = None
            else:
                self.say(f"    {GREEN}wake word ready{RESET} {DIM}({detail}){RESET}")

        self.say(f"    {DIM}loading speech model (first run downloads ~500 MB){RESET}",
                 colour=DIM)
        self.stt.warmup()
        self.say(f"    {DIM}stt on {self.stt.device}{RESET}", colour=DIM)

        mic_ok = self._check_mic()

        while not self._stop:
            try:
                if self.cfg.wake and self.wake and mic_ok:
                    woke = await self.wake_loop_once(30.0)
                    if not woke:
                        continue
                    heard = self.listen_once()
                    if heard:
                        await self.respond(heard)
                    continue

                if not mic_ok:
                    line = await asyncio.to_thread(input, "you> ")
                else:
                    self.say(f"{CYAN}press Enter to speak, or type /quit{RESET}")
                    line = await asyncio.to_thread(input, "> ")
                line = (line or "").strip()
                if not line:
                    if not mic_ok:
                        continue
                    heard = self.listen_once()
                    if heard:
                        await self.respond(heard)
                    continue
                if line in ("/quit", "/exit", "/q"):
                    break
                if line == "/status":
                    self.say(self.status_line(), colour=DIM)
                    continue
                if line == "/devices":
                    self._show_devices()
                    continue
                if line.startswith("/say "):
                    if self.cfg.speak:
                        speech = await self.speaker.synthesize(line[5:])
                        if speech.audio.size:
                            await asyncio.to_thread(self.speaker.play, speech)
                    else:
                        self.say(line[5:])
                    continue
                if line == "/nospeak":
                    self.cfg.speak = False
                    self.say("speech output off", colour=DIM)
                    continue
                if line == "/speak":
                    self.cfg.speak = True
                    self.say("speech output on", colour=DIM)
                    continue
                await self.respond(line)
            except KeyboardInterrupt:
                self.say("")
                break
            except EOFError:
                break
        return 0

    def _show_devices(self) -> None:
        for kind, fn in (("input", list_inputs), ("output", list_outputs)):
            try:
                self.say(f"{BOLD}{kind} devices{RESET}")
                for d in fn():
                    mark = f"{GREEN}*{RESET}" if d.is_default else " "
                    self.say(f" {mark} [{d.index}] {d.name} "
                             f"({d.channels}ch @{d.samplerate}Hz)")
            except Exception as exc:
                self.say(f"  {kind}: {exc}", colour=RED)


async def run_cli(*, local_brain: bool = False, **kwargs) -> int:
    logging.basicConfig(
        level=logging.INFO if kwargs.get("verbose") else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _use_utf8_console()
    cfg = Config(local_brain=local_brain, **kwargs)
    with contextlib.suppress(KeyboardInterrupt):
        return await Jarvis(cfg).run()
    return 0


def _use_utf8_console() -> None:
    """Make the Windows console print the separators in the status line.

    The source is valid UTF-8, but a legacy console codepage (437, 1252)
    renders the separators as replacement characters. Reconfiguring is better
    than stripping them, so any future non-ASCII output is also fixed.
    """
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # py3.7+


def main_sync(**kwargs) -> int:
    return asyncio.run(run_cli(**kwargs))


if __name__ == "__main__":
    sys.exit(main_sync())
