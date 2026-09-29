"""Speech to text: local faster-whisper, Groq Whisper as fallback.

Local is the default because it costs nothing, has no quota, and keeps audio
on the machine. Groq is the fallback for when the local model is too slow or
missing: its `whisper-large-v3-turbo` is markedly more accurate than the small
local model, and the free tier allows about eight hours of audio a day.

The CUDA DLL path has to be fixed up before CTranslate2 is imported, so that
happens at module scope rather than inside the class.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Settings, get_settings
from .cuda import cuda_available, enable_cuda_dlls

log = logging.getLogger("jarvis.stt")


@dataclass
class Transcript:
    text: str
    engine: str
    model: str
    latency_ms: int
    audio_seconds: float

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())


class Transcriber:
    """Lazily-loaded local Whisper with a remote fallback."""

    def __init__(self, settings: Settings | None = None, *, model: str = "small",
                 prefer_gpu: bool = True, language: str | None = "en"):
        self.settings = settings or get_settings()
        self.model_name = model
        self.prefer_gpu = prefer_gpu
        self.language = language
        self._model = None
        self._device: str = "cpu"
        self._compute: str = "int8"

    # --- lifecycle --------------------------------------------------------

    def _load(self):
        if self._model is not None:
            return self._model
        enable_cuda_dlls()
        from faster_whisper import WhisperModel

        if self.prefer_gpu and cuda_available():
            self._device, self._compute = "cuda", "int8_float16"
        else:
            # CPU int8 is the safe default: 12 threads handle `small` at a
            # usable rate, and it avoids a 4 GB VRAM fight with Ollama.
            self._device, self._compute = "cpu", "int8"
        log.info("loading whisper %s on %s/%s", self.model_name, self._device, self._compute)
        t = time.perf_counter()
        self._model = WhisperModel(self.model_name, device=self._device,
                                   compute_type=self._compute)
        log.info("whisper ready in %.1fs", time.perf_counter() - t)
        return self._model

    def warmup(self) -> None:
        try:
            self._load()
        except Exception as exc:
            log.warning("whisper warmup failed: %s", exc)

    def unload(self) -> None:
        self._model = None

    @property
    def device(self) -> str:
        return self._device

    # --- transcription ---------------------------------------------------

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> Transcript:
        """Local first; fall back to Groq only if local fails outright."""
        started = time.perf_counter()
        try:
            return self._transcribe_local(audio, sample_rate, started)
        except Exception as exc:
            log.warning("local whisper failed (%s); trying groq", exc)
        try:
            return self._transcribe_groq(audio, sample_rate, started)
        except Exception as exc:
            log.warning("groq whisper failed: %s", exc)
            return Transcript("", "none", "none", 0, len(audio) / sample_rate)

    def _transcribe_local(self, audio: np.ndarray, sample_rate: int, started: float) -> Transcript:
        model = self._load()
        segments, info = model.transcribe(
            audio, language=self.language, beam_size=1, vad_filter=False,
            condition_on_previous_text=False,  # stops runaway repetition loops
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return Transcript(
            text=text,
            engine="local",
            model=self.model_name,
            latency_ms=int((time.perf_counter() - started) * 1000),
            audio_seconds=len(audio) / sample_rate,
        )

    def _transcribe_groq(self, audio: np.ndarray, sample_rate: int, started: float) -> Transcript:
        import io
        import wave

        import httpx

        cap = self.settings.by_name("groq_stt")
        key = self.settings.credential(cap) if cap else None
        if not key:
            raise RuntimeError("no groq key for the stt fallback")

        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767).astype("<i2")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())

        with httpx.Client(timeout=60.0) as client:
            r = client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                files={"file": ("utterance.wav", buf.getvalue(), "audio/wav")},
                data={"model": "whisper-large-v3-turbo", "response_format": "json"},
            )
            r.raise_for_status()
            text = (r.json().get("text") or "").strip()
        return Transcript(
            text=text,
            engine="groq",
            model="whisper-large-v3-turbo",
            latency_ms=int((time.perf_counter() - started) * 1000),
            audio_seconds=len(audio) / sample_rate,
        )


def transcribe_file(path: str | Path, settings: Settings | None = None) -> Transcript:
    """Transcribe a WAV on disk. Used by the tests and for debugging."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16_000:
        from .audio import TARGET_SR

        audio = _resample(audio, sr, TARGET_SR)
        sr = TARGET_SR
    return Transcriber(settings).transcribe(audio, sr)


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Linear resample. Good enough for speech at these rates, and avoids
    pulling scipy into the hot path."""
    if src == dst or len(audio) == 0:
        return audio
    n = int(round(len(audio) * dst / src))
    idx = np.linspace(0, len(audio) - 1, n, dtype=np.float64)
    return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
