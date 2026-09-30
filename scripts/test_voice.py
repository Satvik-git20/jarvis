"""Check the voice hardware and models one at a time.

Each stage is tested on its own so a failure points at the actual problem
instead of "the voice thing does not work".

    uv run python scripts/test_voice.py devices
    uv run python scripts/test_voice.py all
    uv run python scripts/test_voice.py wake

The wake check needs your own voice. Text-to-speech will not trigger a keyword
spotter trained on human speech, so a synthetic test proves nothing either way.
"""

from __future__ import annotations

import asyncio
import sys
import time

import numpy as np

from jarvis.voice.audio import TARGET_SR, Recorder, list_inputs, list_outputs

GREEN, RED, YELLOW, DIM, CYAN, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[36m", "\033[0m"
)


def show_devices() -> int:
    print(f"\n{DIM}input devices{RESET}")
    try:
        for d in list_inputs():
            mark = f"{GREEN}*{RESET}" if d.is_default else " "
            print(f" {mark} [{d.index}] {d.name} ({d.channels}ch @{d.samplerate}Hz)")
    except Exception as exc:
        print(f"  {RED}failed: {exc}{RESET}")
        return 1
    print(f"\n{DIM}output devices{RESET}")
    try:
        for d in list_outputs():
            mark = f"{GREEN}*{RESET}" if d.is_default else " "
            print(f" {mark} [{d.index}] {d.name} ({d.channels}ch @{d.samplerate}Hz)")
    except Exception as exc:
        print(f"  {RED}failed: {exc}{RESET}")
        return 1
    print()
    return 0


def _brief(exc: Exception, limit: int = 160) -> str:
    """Exception text without the multi-megabyte arrays some libraries embed."""
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


def calibrate(seconds: float = 6.0) -> int:
    """Measure the room's noise floor while the user stays quiet.

    The VAD threshold alone is not portable between rooms. This measures
    ambient level and speech-flag rate with nobody talking, then suggests an
    energy floor that sits above the noise.
    """
    import time

    import sounddevice as sd
    import torch
    from silero_vad import load_silero_vad

    print(f"\n{CYAN}calibration{RESET}")
    print(f"  {DIM}stay quiet for {seconds:.0f}s while this measures the room{RESET}")
    time.sleep(1.0)

    vad = load_silero_vad(onnx=True)
    levels: list[float] = []
    flagged = 0
    total = 0
    t0 = time.time()
    with sd.InputStream(samplerate=TARGET_SR, blocksize=512, device=None,
                        channels=1, dtype="float32") as stream:
        while time.time() - t0 < seconds:
            block, _ = stream.read(512)
            x = np.asarray(block, dtype=np.float32).reshape(-1)
            rms = float(np.sqrt(np.mean(np.square(x))))
            levels.append(rms)
            total += 1
            if rms >= 0.015 and float(vad(torch.from_numpy(x.copy()), TARGET_SR)) >= 0.5:
                flagged += 1

    if not levels:
        print(f"  {RED}no audio captured{RESET}")
        return 1
    lv = np.array(levels)
    noise_p90 = float(np.percentile(lv, 90))
    rate = 100.0 * flagged / max(1, total)
    print(f"  ambient rms: median {np.median(lv):.4f}  p90 {noise_p90:.4f}  "
          f"max {lv.max():.4f}")
    print(f"  frames a speech detector would call speech: {rate:.0f}%")

    if rate < 5.0:
        print(f"  {GREEN}room is quiet enough{RESET} - defaults are fine "
              f"(rms_floor 0.015, threshold 0.5)")
        return 0
    floor = round(max(0.015, noise_p90 * 1.8), 4)
    print(f"  {YELLOW}noisy room{RESET}")
    print(f"    suggested rms_floor: {floor}")
    print(f"    add to .env:  JARVIS_RMS_FLOOR={floor}")
    print(f"    {DIM}if it still cuts you off, raise --vad-threshold to 0.6{RESET}")
    return 0


def check_mic() -> int:
    print(f"\n{CYAN}microphone{RESET}")
    try:
        rec = Recorder()
        print(f"  {DIM}listening for up to 6s - say anything{RESET}")
        got = None
        for utt in rec.utterances(timeout=6.0):
            got = utt
            break
    except Exception as exc:
        print(f"  {RED}capture failed: {_brief(exc)}{RESET}")
        return 1
    if got is None:
        print(f"  {YELLOW}no speech detected (endpointing may be too strict){RESET}")
        return 1
    peak = float(np.max(np.abs(got.audio))) if got.audio.size else 0.0
    print(f"  {GREEN}captured{RESET} {got.duration:.1f}s, peak {peak:.2f}, "
          f"reason {got.reason}")
    return 0


def check_stt() -> int:
    print(f"\n{CYAN}speech to text{RESET}")
    from jarvis.voice.stt import Transcriber

    tr = Transcriber(model="small")
    t = time.perf_counter()
    tr.warmup()
    if tr._model is None:
        print(f"  {RED}whisper failed to load{RESET}")
        return 1
    print(f"  {DIM}loaded on {tr.device} in {time.perf_counter() - t:.1f}s{RESET}")

    print(f"  {DIM}speak a short sentence now (max 8s){RESET}")
    got = None
    for utt in Recorder().utterances(timeout=15.0):
        got = utt
        break
    if got is None:
        print(f"  {YELLOW}nothing captured; skipping transcription{RESET}")
        return 1
    res = tr.transcribe(got.audio, TARGET_SR)
    if not res.ok:
        print(f"  {YELLOW}no speech recognised (check your mic){RESET}")
        return 1
    print(f"  {GREEN}transcribed{RESET} ({res.engine}, {res.latency_ms}ms)")
    print(f"    \"{res.text}\"")
    return 0


async def check_tts() -> int:
    print(f"\n{CYAN}text to speech{RESET}")
    from jarvis.voice.tts import Speaker

    sp = Speaker()
    hint = sp.kokoro_setup_hint()
    if hint:
        print(f"  {YELLOW}kokoro (offline) not installed{RESET}")
        print(f"    {DIM}{hint.splitlines()[0]}{RESET}")
    else:
        print(f"  {DIM}kokoro installed{RESET}")

    print(f"  {DIM}speaking a test phrase on your speakers{RESET}")
    try:
        speech = await sp.speak("Jarvis online. Audio output working.")
    except Exception as exc:
        print(f"  {RED}tts failed: {_brief(exc)}{RESET}")
        return 1
    if not speech.audio.size:
        print(f"  {RED}no audio produced{RESET}")
        return 1
    print(f"  {GREEN}played{RESET} engine={speech.engine} "
          f"{speech.duration:.1f}s in {speech.latency_ms}ms")
    if speech.engine == "edge":
        print(f"    {DIM}note: edge-tts is a Microsoft endpoint with no formal ToS;"
              f" use --tts-local to force the offline engine{RESET}")
    return 0


async def check_brain() -> int:
    print(f"\n{CYAN}ai brain{RESET}")
    from jarvis.config import get_settings
    from jarvis.core.budget import BudgetLedger
    from jarvis.core.providers.base import Message
    from jarvis.core.router import AllProvidersExhausted, Router

    s = get_settings()
    s.ensure_data_dir()
    router = Router(BudgetLedger(), s)
    print(f"  ready: {', '.join(router.ready()) or 'none'}")
    try:
        res = await router.complete([Message("user", "Say OK.")], max_tokens=16)
    except AllProvidersExhausted as exc:
        print(f"  {RED}no provider answered: {_brief(exc)}{RESET}")
        return 1
    print(f"  {GREEN}answered{RESET} via {res.provider}/{res.model} "
          f"in {res.latency_ms}ms: {res.text.strip()[:60]!r}")
    return 0


async def run_wake_check(threshold: float = 0.5) -> int:
    """Probe whether the wake-word model works at all, then score live audio."""
    print(f"\n{CYAN}wake word{RESET}")
    from jarvis.voice.wake import WakeWord, setup_hint

    w = WakeWord(threshold=threshold)
    if not w.available():
        print(f"  {RED}openwakeword unavailable: {w.import_error}{RESET}")
        print(f"  {DIM}{setup_hint()}{RESET}")
        return 1

    healthy, detail = w.selftest()
    print(f"  model probe: {detail}")
    if not healthy:
        print(f"  {RED}model produces no activations on this machine{RESET}")
        print(f"    {DIM}{detail}{RESET}")
        print(f"  {YELLOW}push-to-talk still works and is the default{RESET}")
        return 1
    print(f"  {GREEN}model produces activations{RESET}")

    print("  say \"hey jarvis\" within 4 seconds")
    audio = np.zeros(0, dtype=np.float32)
    for utt in Recorder().utterances(timeout=12.0):
        audio = utt.audio
        break
    if audio.size == 0:
        print(f"  {RED}nothing captured; cannot score{RESET}")
        return 1
    peak = w.peak_score(audio)
    hit = w.detect(audio)
    print(f"  captured {len(audio) / TARGET_SR:.1f}s, peak score {peak:.3f} "
          f"(threshold {threshold})")
    if hit:
        print(f"  {GREEN}detected{RESET} at {hit.at:.1f}s, score {hit.score:.3f}")
        return 0
    print(f"  {YELLOW}not detected{RESET}")
    print(f"    {DIM}if you definitely said it, try --threshold 0.35; "
          f"these models vary a lot by voice{RESET}")
    return 1


async def run_checks() -> int:
    results = {
        "devices": show_devices(),
        "mic": check_mic(),
        "stt": check_stt(),
        "tts": await check_tts(),
        "brain": await check_brain(),
    }
    bad = [k for k, v in results.items() if v != 0]
    print(f"\n  {GREEN if not bad else RED}"
          f"{len(results) - len(bad)} ok{RESET}  "
          f"{RED if bad else DIM}{len(bad)} failed{RESET}"
          f"{'  (' + ', '.join(bad) + ')' if bad else ''}\n")
    return 1 if bad else 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "devices":
        code = show_devices()
    elif cmd == "all":
        code = asyncio.run(run_checks())
    elif cmd == "wake":
        code = asyncio.run(run_wake_check())
    elif cmd == "calibrate":
        code = calibrate()
    elif cmd == "mic":
        code = check_mic()
    elif cmd == "stt":
        code = check_stt()
    elif cmd == "tts":
        code = asyncio.run(check_tts())
    else:
        print(f"unknown command: {cmd}\n"
              "usage: test_voice.py [all|devices|mic|stt|tts|wake|calibrate]")
        code = 2
    sys.exit(code)
