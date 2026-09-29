"""Wake word detection using openWakeWord's pretrained "hey_jarvis" model.

The model ships with the package, so there is no training step and no custom
phrase to record.

`Model.predict()` takes *raw 16 kHz audio* and does its own framing and feature
extraction. Feeding it precomputed features returns 0.0 for everything, which
looks like a working detector and is not one.

A caveat worth knowing: these models are trained on human speech, and
synthetic TTS voices do not reliably trigger them. A wake word verified against
text-to-speech proves nothing. `--test-wake` exists so the check can be done
with your own voice instead, and push-to-talk remains the default path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .audio import TARGET_SR

log = logging.getLogger("jarvis.wake")

DEFAULT_MODEL = "hey_jarvis"
DEFAULT_THRESHOLD = 0.5
CHUNK = 1280  # 80 ms at 16 kHz


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
                 block_ms: int = 640):
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
        if audio.size < self.block:
            return None
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
        best = 0.0
        for start in range(0, max(1, audio.size - self.block + 1), self.block):
            window = audio[start:start + self.block]
            if window.size < self.block:
                break
            best = max(best, self._score(self._model.predict(window)))
        return best


def setup_hint() -> str:
    return (
        "Wake word unavailable, so push-to-talk is being used.\n"
        "  uv sync --extra wake\n"
        "If a Windows Application Control policy is blocking onnxruntime, the error\n"
        "is shown above. Push-to-talk needs none of this and always works."
    )
