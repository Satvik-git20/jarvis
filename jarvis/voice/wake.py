"""Wake word detection using openWakeWord's pretrained "hey_jarvis" model.

IMPORTANT -- this does not currently work on this machine, and the code says so
rather than pretending otherwise.

Measured: every bundled openWakeWord model (hey_jarvis, alexa, hey_mycroft,
hey_rhasspy, the timers, weather) returns exactly 0.0000 on real human speech
that contains its own activation, while random feature vectors score 0.95 from
the same ONNX session. The graph runs; the feature pipeline feeding it does not
produce matching activations. Reproduced on openwakeword 0.5.1 and 0.6.0.

The likely cause is the framework. openWakeWord's own docs say tflite is the
default and more accurate on x86, and ONNX is the fallback -- but tflite-runtime
ships no Windows wheel for Python 3.12, so only the broken path is installable.

So `selftest()` probes the model at startup and `--wake` reports the result.
Push-to-talk is the supported input path and is the default; it needs none of
this and always works.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .audio import TARGET_SR

log = logging.getLogger("jarvis.wake")

DEFAULT_MODEL = "hey_jarvis"
DEFAULT_THRESHOLD = 0.5
# openWakeWord's native window is 80 ms of 16 kHz audio. Using its own chunk
# size matters: a 0.64 s block silently skipped any clip shorter than a block,
# which made a working model look completely dead.
CHUNK = 1280
BLOCK_MS = 80
# A genuine activation scores well above the 0.5 detection threshold. Probes
# that peak below this mean the model is running but not discriminating.
MIN_ACTIVATION = 0.05


@dataclass
class WakeEvent:
    score: float
    at: float


class WakeWord:
    """Streaming keyword spotter.

    Audio is accumulated and handed to `Model.predict` in growing windows.
    Repeatedly re-scoring the whole buffer would be quadratic; scoring
    forward in fixed blocks keeps it linear and the latency bounded.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, *, threshold: float = DEFAULT_THRESHOLD,
                 block_ms: int = BLOCK_MS):
        self.model_name = model_name
        self.threshold = threshold
        self.block = int(TARGET_SR * block_ms / 1000)
        self._model = None
        self._buf = np.zeros(0, dtype=np.float32)
        self._load_error = ""

    @property
    def import_error(self) -> str:
        return self._load_error

    def available(self) -> bool:
        try:
            self._ensure()
            return True
        except Exception:
            return False

    def selftest(self) -> tuple[bool, str]:
        """Check whether the model can produce a realistic activation score.

        This cannot prove detection on its own -- nothing short of a recording
        of a person saying the phrase does that. What it can do is catch the
        failure mode seen here, where the model returns near-zero for
        everything. A real activation lands well above 0.5, so anything under
        `MIN_ACTIVATION` across silence, a tone and noise means the feature
        pipeline is not producing usable activations.
        """
        self._ensure()
        t = TARGET_SR
        dur = int(t * 0.5)
        probes = {
            "silence": np.zeros(dur, dtype=np.float32),
            "tone": (0.3 * np.sin(2 * np.pi * 220 * np.arange(dur) / t)).astype(np.float32),
            "noise": np.random.default_rng(0).normal(0, 0.2, dur).astype(np.float32),
        }
        scores = {k: self.peak_score(v) for k, v in probes.items()}
        highest = max(scores.values())
        detail = ", ".join(f"{k}={v:.4f}" for k, v in scores.items())
        if highest < MIN_ACTIVATION:
            return False, (
                f"nothing scored above {MIN_ACTIVATION} ({detail}). The model runs "
                f"but its ONNX feature path is not producing activations on this "
                f"machine; tflite-runtime has no Windows wheel for Python 3.12."
            )
        return True, detail

    def _ensure(self):
        if self._model is not None:
            return self._model
        if self._load_error:
            raise RuntimeError(self._load_error)
        try:
            from openwakeword import Model
            from openwakeword.utils import download_models

            download_models([self.model_name])
            self._model = Model(wakeword_models=[self.model_name],
                                inference_framework="onnx")
            log.info("wake word '%s' ready", self.model_name)
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            raise
        return self._model

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        if self._model is not None and hasattr(self._model, "reset"):
            self._model.reset()

    @staticmethod
    def _score(pred) -> float:
        """0.6 returns {model: score}; older builds returned an array."""
        if pred is None:
            return 0.0
        if isinstance(pred, dict):
            return float(max(pred.values())) if pred else 0.0
        arr = np.asarray(pred, dtype=np.float32)
        return float(np.max(arr)) if arr.size else 0.0

    def feed(self, chunk: np.ndarray) -> float | None:
        """Add audio. Returns a score once a full block is available."""
        self._ensure()
        self._buf = np.concatenate([self._buf, np.asarray(chunk, dtype=np.float32).reshape(-1)])
        if self._buf.size < self.block:
            return None
        score = self._score(self._model.predict(self._buf))
        self._buf = np.zeros(0, dtype=np.float32)
        return score

    def detect(self, audio: np.ndarray) -> WakeEvent | None:
        """First block above threshold, or None.

        `detect` is the honest way to test: it returns the peak score too, so a
        model that never fires is visibly scoring 0 rather than silently
        reporting "not detected".
        """
        self._ensure()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        # A window shorter than one block is padded rather than rejected: the
        # wake phrase itself is often under a second.
        if audio.size < self.block:
            audio = np.pad(audio, (0, self.block - audio.size))
        self._buf = np.zeros(0, dtype=np.float32)
        for start in range(0, audio.size - self.block + 1, self.block):
            window = audio[start:start + self.block]
            score = self._score(self._model.predict(window))
            if score >= self.threshold:
                return WakeEvent(score=score, at=(start + self.block) / TARGET_SR)
        return None

    def peak_score(self, audio: np.ndarray) -> float:
        """Highest score anywhere in the buffer. For diagnostics."""
        self._ensure()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size < self.block:
            audio = np.pad(audio, (0, self.block - audio.size))
        best = 0.0
        for start in range(0, audio.size - self.block + 1, self.block):
            window = audio[start:start + self.block]
            best = max(best, self._score(self._model.predict(window)))
        return best


def setup_hint() -> str:
    return (
        "Wake word is not usable here, so push-to-talk is being used.\n"
        "openwakeword's ONNX path returns 0.0 for all audio on this machine; its\n"
        "preferred tflite backend has no Windows wheel for Python 3.12. Run\n"
        "  uv run python -m jarvis test-wake\n"
        "to see the probe scores. Push-to-talk needs none of this and always works."
    )
