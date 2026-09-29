"""Text to speech: Edge TTS first, Kokoro as the offline fallback.

The ordering is a measured one, not a preference. On this hardware:

    edge-tts   1.5s for a short phrase, 24 kHz neural voices
    kokoro     6.8s for the same phrase on CPU (real-time factor 5-8)

Kokoro stays because it is Apache-2.0, runs fully offline, and is the only
option that still works with no network at all. Edge TTS is a reverse-
engineered Microsoft endpoint with no formal ToS, so it is treated as
replaceable rather than load-bearing: if it breaks or the user prefers
offline-only, `prefer_local` flips the order.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Settings, get_settings

log = logging.getLogger("jarvis.tts")

EDGE_SR = 24_000
KOKORO_SR = 24_000

DEFAULT_EDGE_VOICE = "en-GB-SoniaNeural"
DEFAULT_KOKORO_VOICE = "af_sarah"

MODEL_DIRNAME = "models"
KOKORO_MODEL = "kokoro-v1.0.int8.onnx"
KOKORO_VOICES = "voices-v1.0.bin"


@dataclass
class Speech:
    audio: np.ndarray
    sample_rate: int
    engine: str
    latency_ms: int
    text: str

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate if self.sample_rate else 0.0


def _split_sentences(text: str, *, max_chars: int = 160) -> list[str]:
    """Split for playback, not for grammar.

    Speaking the first clause while the rest is still being synthesised is what
    makes the assistant feel responsive, so the split happens at sentence and
    then clause boundaries and never mid-clause where it would sound clipped.
    """
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        while len(part) > max_chars:
            cut = part.rfind(", ", 0, max_chars)
            if cut < max_chars // 2:
                cut = part.rfind(" ", 0, max_chars)
            if cut < max_chars // 2:
                break
            out.append(part[:cut].strip())
            part = part[cut:].lstrip(", ").strip()
        if part:
            out.append(part)
    return out


class Speaker:
    """Synthesises and plays replies, preferring the fastest engine."""

    def __init__(self, settings: Settings | None = None, *, voice: str | None = None,
                 prefer_local: bool = False, playback_device=None):
        self.settings = settings or get_settings()
        self.voice = voice
        self.prefer_local = prefer_local
        self.playback_device = playback_device
        self._kokoro = None
        self._kokoro_failed = False
        self._player = None

    # --- model discovery --------------------------------------------------

    def model_paths(self) -> tuple[Path, Path]:
        d = self.settings.data_dir / MODEL_DIRNAME
        return d / KOKORO_MODEL, d / KOKORO_VOICES

    def kokoro_available(self) -> bool:
        model, voices = self.model_paths()
        return model.is_file() and voices.is_file()

    def kokoro_setup_hint(self) -> str | None:
        model, voices = self.model_paths()
        if model.is_file() and voices.is_file():
            return None
        base = ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
                "model-files-v1.1")
        return (
            "Kokoro (offline TTS) is not installed. To add it, download into "
            f"{model.parent}:\n"
            f"  {base}/kokoro-v1.0.int8.onnx\n"
            f"  {base}/voices-v1.0.bin"
        )

    # --- synthesis -------------------------------------------------------

    async def synthesize(self, text: str) -> Speech:
        """One audio buffer for the whole reply."""
        text = (text or "").strip()
        if not text:
            return Speech(np.zeros(0, np.float32), EDGE_SR, "none", 0, "")

        order = (("kokoro", "edge"), ("edge", "kokoro")) if self.prefer_local \
            else (("edge", "kokoro"), ("kokoro", "edge"))
        chunks: list[Speech] = []
        started = time.perf_counter()
        for engine in order[0]:
            for piece in _split_sentences(text):
                try:
                    if engine == "edge":
                        chunks.append(await self._edge(piece))
                    else:
                        chunks.append(await asyncio.to_thread(self._kokoro_sync, piece))
                except Exception as exc:
                    log.info("tts engine %s failed on a chunk: %s", engine, exc)
                    break
            if chunks:
                if len(chunks) < len(_split_sentences(text)):
                    # Partial synthesis is still far better than silence.
                    log.info("tts fell back mid-reply (%s)", engine)
                break

        if not chunks:
            return Speech(np.zeros(0, np.float32), EDGE_SR, "none", 0, text)

        sr = chunks[0].sample_rate
        audio = np.concatenate([c.audio for c in chunks])
        return Speech(
            audio=audio,
            sample_rate=sr,
            engine=chunks[0].engine,
            latency_ms=int((time.perf_counter() - started) * 1000),
            text=text,
        )

    async def _edge(self, text: str) -> Speech:
        import edge_tts

        voice = self.voice or DEFAULT_EDGE_VOICE
        started = time.perf_counter()
        buf = bytearray()
        async for chunk in edge_tts.Communicate(text, voice).stream():
            if chunk["type"] == "audio":
                buf.extend(chunk["data"])
        if not buf:
            raise RuntimeError("edge-tts returned no audio")
        audio = await asyncio.to_thread(_mp3_to_pcm, bytes(buf))
        return Speech(audio, EDGE_SR, "edge", int((time.perf_counter() - started) * 1000), text)

    def _kokoro_sync(self, text: str) -> Speech:
        if self._kokoro is None:
            if self._kokoro_failed:
                raise RuntimeError("kokoro unavailable")
            model, voices = self.model_paths()
            if not (model.is_file() and voices.is_file()):
                self._kokoro_failed = True
                raise RuntimeError(self.kokoro_setup_hint() or "kokoro model missing")
            from kokoro_onnx import Kokoro

            self._kokoro = Kokoro(str(model), str(voices))
        started = time.perf_counter()
        samples, sr = self._kokoro.create(
            text, voice=DEFAULT_KOKORO_VOICE, speed=1.0, lang="en-us"
        )
        audio = np.concatenate(samples) if isinstance(samples, list) else np.asarray(samples)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        return Speech(audio, sr, "kokoro", int((time.perf_counter() - started) * 1000), text)

    # --- playback --------------------------------------------------------

    def play(self, speech: Speech) -> None:
        from .audio import play as play_blocking

        play_blocking(speech.audio, speech.sample_rate, device=self.playback_device)

    def start(self, speech: Speech) -> None:
        """Begin playback without blocking, so the mic can stay live."""
        from .audio import Player

        if self._player is None:
            self._player = Player(self.playback_device)
        self._player.start(speech.audio, speech.sample_rate)

    def speaking(self) -> bool:
        return bool(self._player and not self._player.finished())

    def stop(self) -> None:
        if self._player:
            self._player.stop()

    async def speak(self, text: str) -> Speech:
        """Synthesise then block until finished. Returns what was said."""
        speech = await self.synthesize(text)
        if speech.audio.size:
            await asyncio.to_thread(self.play, speech)
        return speech


def _mp3_to_pcm(data: bytes) -> np.ndarray:
    """Edge returns mp3. Decode to mono float32 at 24 kHz for playback."""
    import io

    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != EDGE_SR and len(audio):
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * EDGE_SR / sr))
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio.astype(np.float32)
