"""Microphone capture with voice-activity endpointing.

The hard part of a voice assistant is not transcription, it is knowing when the
user has *stopped talking*. Waiting for a fixed silence window is either
sluggish (long window) or cuts people off (short one). Silero VAD decides it
from the audio itself, and this module turns that into utterance boundaries.

Everything is normalised to 16 kHz mono float32 because that is what both
faster-whisper and Silero expect, and resampling once here keeps the rest of
the pipeline simple.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("jarvis.audio")

TARGET_SR = 16_000
FRAME_MS = 32  # Silero's native frame size at 16 kHz
FRAME_SAMPLES = TARGET_SR * FRAME_MS // 1000  # 512


@dataclass
class AudioDevice:
    index: int
    name: str
    channels: int
    samplerate: int
    is_default: bool = False


def _sd():
    """Import sounddevice lazily so `--help` and tests work without audio."""
    import sounddevice as sd

    return sd


def list_inputs() -> list[AudioDevice]:
    sd = _sd()
    default_idx = sd.query_devices(kind="input")["index"]
    out = []
    for d in sd.query_devices():
        if d["max_input_channels"] > 0:
            out.append(AudioDevice(
                index=d["index"],
                name=d["name"],
                channels=d["max_input_channels"],
                samplerate=int(d["default_samplerate"]),
                is_default=d["index"] == default_idx,
            ))
    return out


def list_outputs() -> list[AudioDevice]:
    sd = _sd()
    default_idx = sd.query_devices(kind="output")["index"]
    out = []
    for d in sd.query_devices():
        if d["max_output_channels"] > 0:
            out.append(AudioDevice(
                index=d["index"],
                name=d["name"],
                channels=d["max_output_channels"],
                samplerate=int(d["default_samplerate"]),
                is_default=d["index"] == default_idx,
            ))
    return out


@dataclass
class Utterance:
    audio: np.ndarray
    sample_rate: int = TARGET_SR
    started_at: float = field(default_factory=time.time)
    duration: float = 0.0
    reason: str = "silence"


class Recorder:
    """Streams microphone audio and yields complete utterances.

    Endpointing rules, tuned for conversation rather than dictation:
      - a short hangover before speech is accepted, so a plosive is not
        mistaken for silence;
      - trailing silence is trimmed, keeping a small pad so the final phoneme
        survives;
      - a hard cap stops a stuck-open microphone from buffering forever.
    """

    def __init__(
        self,
        device: int | str | None = None,
        *,
        threshold: float = 0.5,
        rms_floor: float = 0.004,
        noise_multiplier: float = 3.0,
        silence_ms: int = 700,
        min_speech_ms: int = 300,
        max_utterance_s: float = 30.0,
        pre_roll_ms: int = 300,
    ):
        self.device = device
        self.threshold = threshold
        # A fixed energy floor cannot be right everywhere. Measured in this
        # room: 0.0058 overall but 0.0000 across most 300 ms windows, so a
        # "calibrated" value of 0.015 (taken while someone was talking) would
        # reject ordinary quiet speech outright.
        #
        # So the floor is tracked instead of fixed. `rms_floor` is only a
        # backstop for a dead-silent input, and speech must clear
        # `noise_multiplier` times the running noise level on top of that.
        self.rms_floor = rms_floor
        self.noise_multiplier = noise_multiplier
        self._noise = rms_floor
        self.silence_frames = max(1, silence_ms // FRAME_MS)
        self.min_speech_frames = max(1, min_speech_ms // FRAME_MS)
        self.max_frames = int(max_utterance_s * 1000 / FRAME_MS)
        self.preroll_frames = max(1, pre_roll_ms // FRAME_MS)
        self._model = None
        self._interrupted = False

    @property
    def noise_floor(self) -> float:
        """Current estimate of the room's background level."""
        return self._noise

    # --- VAD -------------------------------------------------------------

    def _vad(self):
        if self._model is None:
            from silero_vad import load_silero_vad

            # onnx=True avoids running inference through torch even though the
            # wrapper still keeps its recurrent state as torch tensors.
            self._model = load_silero_vad(onnx=True)
            log.info("silero VAD loaded (onnx)")
        return self._model

    def _is_speech(self, frame: np.ndarray) -> bool:
        """Speech probability for one 32 ms frame.

        The wrapper calls `.dim()` and `.unsqueeze()` on its input and keeps its
        recurrent state as torch tensors, so it wants a Tensor rather than the
        ndarray that comes out of PortAudio. Feeding an ndarray raises
        "expected a value of type 'Tensor'".
        """
        import torch

        rms = float(np.sqrt(np.mean(np.square(frame))))
        gate = max(self.rms_floor, self._noise * self.noise_multiplier)
        if rms < gate:
            # Too quiet to be speech. Cheaper than running the model, and
            # immune to its false positives on steady background sound.
            self._update_noise(rms)
            return False

        chunk = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
        prob = float(self._vad()(chunk, TARGET_SR))
        if prob >= self.threshold:
            return True
        self._update_noise(rms)
        return False

    def _update_noise(self, rms: float) -> None:
        """Track the background level from frames the VAD called silence.

        An exponential average is enough: a slow rise absorbs a fan starting
        up, and a fast fall means the floor recovers quickly once the noise
        stops, so the next utterance is not gated by a stale estimate.
        """
        # Rise slowly, fall quickly.
        alpha = 0.02 if rms > self._noise else 0.15
        self._noise = (1 - alpha) * self._noise + alpha * rms
        # Never fall below the absolute backstop.
        self._noise = max(self._noise, self.rms_floor * 0.5)

    def reset_noise(self) -> None:
        self._noise = self.rms_floor

    # --- capture ---------------------------------------------------------

    def utterances(self, *, timeout: float | None = None) -> Iterator[Utterance]:
        """Capture from the microphone until one utterance completes.

        Returns after the first completed utterance, or after `timeout`
        seconds with nothing said. The stream is opened once per call rather
        than held open between utterances: keeping it open means `read()` can
        hand back buffered audio faster than it arrives, which made the
        duration cap fire on wall-clock guesses rather than on real speech.
        """
        sd = _sd()
        self._vad()  # load before opening the stream, so init is not captured

        def frames():
            with sd.InputStream(
                samplerate=TARGET_SR,
                blocksize=FRAME_SAMPLES,
                device=self.device,
                channels=1,
                dtype="float32",
            ) as stream:
                while True:
                    block, _ = stream.read(FRAME_SAMPLES)
                    yield np.asarray(block, dtype=np.float32).reshape(-1)

        yield from self.endpoint(frames(), timeout=timeout)

    def endpoint(
        self, frames: Iterator[np.ndarray], *, timeout: float | None = None
    ) -> Iterator[Utterance]:
        """Turn a stream of 32 ms frames into utterances.

        Split out from capture so the endpointing rules can be tested without
        a microphone. Two bugs lived here and were invisible to unit tests:
        the duration cap counted frames since the stream opened rather than
        since speech began, and the timeout path discarded buffered speech
        instead of yielding it.
        """
        started = time.time()
        ring: list[np.ndarray] = []
        collecting = False
        speech_frames = 0
        silent_frames = 0
        # Frames since speech started, not since the stream opened.
        collected_frames = 0

        for frame in frames:
            if timeout is not None and time.time() - started > timeout:
                if collecting and ring:
                    yield self._finalise(ring, reason="timeout")
                return
            if self._interrupted:
                self._interrupted = False
                return

            speech = self._is_speech(frame)

            if not collecting:
                ring.append(frame)
                if len(ring) > self.preroll_frames:
                    ring.pop(0)
                if speech:
                    collecting = True
                    speech_frames = 1
                    silent_frames = 0
                    collected_frames = 1
                continue

            ring.append(frame)
            collected_frames += 1
            if speech:
                speech_frames += 1
                silent_frames = 0
            else:
                silent_frames += 1

            if silent_frames >= self.silence_frames and speech_frames >= self.min_speech_frames:
                yield self._finalise(ring)
                return
            if collected_frames >= self.max_frames:
                # Hard cap: a stuck-open gate or a very long answer must not
                # buffer forever.
                yield self._finalise(ring, reason="max_duration")
                return

    def _finalise(self, frames: list[np.ndarray], *, reason: str = "silence") -> Utterance:
        audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
        # Drop the leading pre-roll silence but keep a little of it; starting
        # mid-phoneme is worse than starting a touch early.
        trim = min(self.preroll_frames, max(0, len(frames) - self.min_speech_frames))
        if trim:
            audio = audio[trim * FRAME_SAMPLES:]
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            # Always normalise to just under full scale, not only when the
            # signal is already hot. Whisper is markedly worse on quiet input,
            # and a user who speaks normally into a laptop mic produces peak
            # values well under 1.0.
            audio = audio * (0.95 / peak)
        return Utterance(
            audio=audio.astype(np.float32),
            duration=len(audio) / TARGET_SR,
            reason=reason,
        )

    def interrupt(self) -> None:
        """Ask an in-progress `utterances()` loop to stop (barge-in)."""
        self._interrupted = True

    def close(self) -> None:
        with contextlib.suppress(Exception):
            _sd().stop()


def play(audio: np.ndarray, sample_rate: int, *, device: int | str | None = None) -> None:
    """Block until playback finishes."""
    if audio is None or len(audio) == 0:
        return
    sd = _sd()
    with sd.OutputStream(samplerate=sample_rate, channels=1,
                         dtype="float32", device=device) as stream:
        stream.write(np.asarray(audio, dtype=np.float32).reshape(-1, 1))
        stream.stop()


class Player:
    """Non-blocking playback so speech can be interrupted mid-sentence."""

    def __init__(self, device: int | str | None = None):
        self.device = device
        self._stream = None
        self._buf = None

    def start(self, audio: np.ndarray, sample_rate: int) -> None:
        self.stop()
        sd = _sd()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1, 1)
        self._stream = sd.OutputStream(samplerate=sample_rate, channels=1,
                                      dtype="float32", device=self.device)
        self._stream.start()
        self._stream.write(audio)

    def finished(self) -> bool:
        if self._stream is None:
            return True
        return not self._stream.active

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def __enter__(self) -> Player:
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
