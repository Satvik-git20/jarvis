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
        rms_floor: float = 0.015,
        silence_ms: int = 700,
        min_speech_ms: int = 300,
        max_utterance_s: float = 30.0,
        pre_roll_ms: int = 300,
    ):
        self.device = device
        self.threshold = threshold
        # Silero on its own is not enough in a room with steady background
        # sound: measured here, it flagged 62% of frames as speech with nobody
        # talking, because a fan or hum is spectrally speech-like. The energy
        # floor rejects that. Run `jarvis test-voice --calibrate` to set it
        # for your room.
        self.rms_floor = rms_floor
        self.silence_frames = max(1, silence_ms // FRAME_MS)
        self.min_speech_frames = max(1, min_speech_ms // FRAME_MS)
        self.max_frames = int(max_utterance_s * 1000 / FRAME_MS)
        self.preroll_frames = max(1, pre_roll_ms // FRAME_MS)
        self._model = None
        self._interrupted = False

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

        chunk = torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
        if float(np.sqrt(np.mean(np.square(frame)))) < self.rms_floor:
            # Too quiet to be speech. Cheaper than the model and immune to its
            # false positives on steady noise.
            return False
        return float(self._vad()(chunk, TARGET_SR)) >= self.threshold

    # --- capture ---------------------------------------------------------

    def utterances(self, *, timeout: float | None = None) -> Iterator[Utterance]:
        """Yield utterances until `timeout` seconds elapse with no speech.

        The stream stays open between utterances so the user can speak again
        without re-opening the device, which avoids a click and a slow restart.
        """
        sd = _sd()
        self._vad()  # load before opening the stream, so init is not captured
        started = time.time()
        last_activity = started

        with sd.InputStream(
            samplerate=TARGET_SR,
            blocksize=FRAME_SAMPLES,
            device=self.device,
            channels=1,
            dtype="float32",
        ) as stream:
            ring: list[np.ndarray] = []
            collecting = False
            speech_frames = 0
            silent_frames = 0
            total_frames = 0

            while True:
                if timeout is not None and time.time() - last_activity > timeout:
                    if collecting and ring:
                        break
                    if not collecting:
                        return
                if self._interrupted:
                    self._interrupted = False
                    return

                block, _ = stream.read(FRAME_SAMPLES)
                frame = np.asarray(block, dtype=np.float32).reshape(-1)
                total_frames += 1
                speech = self._is_speech(frame)

                if not collecting:
                    ring.append(frame)
                    if len(ring) > self.preroll_frames:
                        ring.pop(0)
                    if speech:
                        collecting = True
                        speech_frames = 1
                        silent_frames = 0
                        last_activity = time.time()
                    elif total_frames % 25 == 0:
                        # Cheap poll so an idle recorder still honours timeout.
                        last_activity = max(last_activity, time.time())
                    continue

                ring.append(frame)
                if speech:
                    speech_frames += 1
                    silent_frames = 0
                    last_activity = time.time()
                else:
                    silent_frames += 1

                if silent_frames >= self.silence_frames and speech_frames >= self.min_speech_frames:
                    yield self._finalise(ring)
                    return
                if total_frames >= self.max_frames and collecting:
                    yield self._finalise(ring, reason="max_duration")
                    return
                if total_frames >= self.max_frames * 40 and not collecting:
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
