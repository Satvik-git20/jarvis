"""Tests for the voice pipeline.

These cover the parts that can be wrong without a microphone: sentence
splitting for playback, the CUDA DLL discovery, wake-word score handling, and
the endpointing state machine. Anything needing a real mic is exercised by
scripts/test_voice.py instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from jarvis.voice.audio import FRAME_SAMPLES, TARGET_SR, Recorder
from jarvis.voice.cuda import _nvidia_bin_dirs, cuda_available, enable_cuda_dlls
from jarvis.voice.tts import Speech, _mp3_to_pcm, _split_sentences
from jarvis.voice.wake import WakeWord

# --- TTS sentence splitting -------------------------------------------------


def test_split_sentences_returns_nothing_for_empty():
    assert _split_sentences("") == []
    assert _split_sentences("   ") == []


def test_split_sentences_splits_on_terminators():
    got = _split_sentences("Yes. No. Maybe!")
    assert got == ["Yes.", "No.", "Maybe!"]


def test_split_sentences_breaks_long_text_without_losing_words():
    text = "This is a long sentence with many words that keeps going and going " \
           "past the maximum chunk size limit for a single playback buffer."
    parts = _split_sentences(text, max_chars=40)
    assert all(len(p) <= 45 for p in parts), parts
    # The point of splitting is latency, not summarising: no words may vanish.
    rejoined = " ".join(parts).replace(" ,", ",")
    assert set(text.split()) <= set(rejoined.split())


def test_split_sentences_handles_a_single_very_long_word():
    got = _split_sentences("x" * 300, max_chars=50)
    assert got == ["x" * 300]  # unsplittable, but not dropped


def test_mp3_to_pcm_normalises_to_mono_float32():
    # A stereo int16 buffer, as edge-tts would hand back after decoding.
    import io

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.zeros((2400, 2), dtype=np.float32), 24000, format="WAV")
    out = _mp3_to_pcm(buf.getvalue())
    assert out.dtype == np.float32
    assert out.ndim == 1


def test_mp3_to_pcm_clamps_hot_signal():
    import io

    import soundfile as sf

    hot = np.full(2400, 4.0, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, hot, 24000, format="WAV")
    out = _mp3_to_pcm(buf.getvalue())
    assert float(np.max(np.abs(out))) <= 1.0


# --- CUDA discovery ---------------------------------------------------------


def test_enable_cuda_dlls_is_idempotent():
    enable_cuda_dlls()
    assert enable_cuda_dlls() == []  # second call is a no-op


def test_cuda_available_returns_bool_without_raising():
    assert isinstance(cuda_available(), bool)


def test_nvidia_bin_dirs_are_existing_directories():
    for d in _nvidia_bin_dirs():
        assert d.is_dir()


# --- wake word score handling -----------------------------------------------


@pytest.mark.parametrize("pred,expected", [
    ({"hey_jarvis": 0.91}, 0.91),
    ({"a": 0.2, "hey_jarvis": 0.8}, 0.8),   # take the max across models
    ({}, 0.0),
    (None, 0.0),
    (np.array([[0.1, 0.7]]), 0.7),           # older array-returning builds
    (np.array([]), 0.0),
])
def test_wake_score_normalises_every_shape(pred, expected):
    assert WakeWord._score(pred) == pytest.approx(expected)


def test_wake_records_and_surfaces_a_load_failure():
    """A bogus model name makes the real load fail, which must be remembered
    rather than retried on every loop iteration."""
    w = WakeWord(model_name="definitely_not_a_real_wake_word_xyz")
    assert w.available() is False
    assert w.import_error, "the failure reason must be recorded"
    # Second call must short-circuit on the cached error, not raise.
    assert w.available() is False


# --- endpointing ------------------------------------------------------------


def test_recorder_frame_math_is_consistent():
    r = Recorder(silence_ms=700, min_speech_ms=300, pre_roll_ms=300)
    assert r.silence_frames == 700 // 32
    assert r.min_speech_frames == 300 // 32
    assert FRAME_SAMPLES == 512
    assert TARGET_SR == 16_000


def test_recorder_finalise_trims_preroll_and_normalises():
    r = Recorder()
    quiet = np.full(FRAME_SAMPLES, 0.0001, dtype=np.float32)
    loud = np.full(FRAME_SAMPLES, 0.5, dtype=np.float32)
    frames = [quiet] * r.preroll_frames + [loud] * 40
    u = r._finalise(frames)
    # Leading pre-roll is dropped so the clip does not start in silence.
    assert u.duration < (len(frames) * FRAME_SAMPLES / TARGET_SR)
    # Normalised up to just under full scale, not merely left alone.
    assert float(np.max(np.abs(u.audio))) == pytest.approx(0.95, abs=1e-3)


def test_recorder_normalises_very_quiet_speech_upward():
    """Regression: normalisation used to divide only when the peak already
    exceeded 1.0, so ordinary quiet speech was never boosted."""
    r = Recorder()
    frames = [np.full(FRAME_SAMPLES, 0.02, dtype=np.float32)] * 40
    u = r._finalise(frames)
    assert float(np.max(np.abs(u.audio))) == pytest.approx(0.95, abs=1e-3)


def test_recorder_leaves_silence_alone():
    r = Recorder()
    u = r._finalise([np.zeros(FRAME_SAMPLES, dtype=np.float32)] * 40)
    assert u.audio.size > 0
    assert float(np.max(np.abs(u.audio))) == 0.0  # no division by zero


# --- the energy gate --------------------------------------------------------


def test_quiet_audio_is_rejected_without_calling_the_model():
    """Steady background noise is what fools a pure VAD, so the energy floor
    must short-circuit before the model runs."""
    r = Recorder(rms_floor=0.015)
    r._vad = lambda: (_ for _ in ()).throw(AssertionError("model should not run"))
    quiet = np.full(FRAME_SAMPLES, 0.005, dtype=np.float32)  # rms 0.005 < floor
    assert r._is_speech(quiet) is False


def test_loud_audio_reaches_the_model():
    r = Recorder(rms_floor=0.015, threshold=0.5)
    loud = np.full(FRAME_SAMPLES, 0.5, dtype=np.float32)  # rms 0.5 > floor
    seen = {}

    class FakeVad:
        def __call__(self, chunk, sr):
            seen["sr"] = sr
            seen["type"] = type(chunk).__name__
            return 0.9

    r._vad = FakeVad
    assert r._is_speech(loud) is True
    assert seen["sr"] == TARGET_SR
    # The wrapper calls .dim()/.unsqueeze(), so it must get a torch Tensor.
    assert seen["type"] == "Tensor"


def test_energy_gate_can_be_disabled():
    r = Recorder(rms_floor=0.0, threshold=0.5)

    class FakeVad:
        def __call__(self, chunk, sr):
            return 0.1  # below threshold

    r._vad = FakeVad
    quiet = np.full(FRAME_SAMPLES, 0.0001, dtype=np.float32)
    assert r._is_speech(quiet) is False  # decided by the vad, not the gate


def test_settings_carry_voice_defaults(tmp_path):
    from jarvis.config import Settings

    s = Settings(
        data_dir=tmp_path, ollama_host="", ollama_chat_model="",
        ollama_embed_model="", daemon_host="", daemon_port=0, daemon_token="",
        exclude_privacy_unsafe=False,
    )
    assert s.rms_floor > 0
    assert 0 < s.vad_threshold < 1
    assert s.whisper_model


# --- endpointing state machine (no microphone needed) -----------------------


def frames_of(n: int, level: float) -> list[np.ndarray]:
    """n frames of constant amplitude, as the capture loop would deliver."""
    return [np.full(FRAME_SAMPLES, level, dtype=np.float32) for _ in range(n)]


def recorder_with_stub(rec: Recorder, *, speech: bool) -> Recorder:
    """A Recorder whose gate always agrees with `speech`, bypassing the VAD."""
    rec._is_speech = lambda frame: speech
    return rec


def test_speech_then_silence_yields_one_utterance():
    r = recorder_with_stub(Recorder(), speech=False)
    stream = iter(frames_of(40, 0.3) + frames_of(200, 0.0))
    r._is_speech = lambda f: np.sqrt(np.mean(np.square(f))) > 0.1
    got = list(r.endpoint(stream, timeout=5))
    assert len(got) == 1
    assert got[0].reason == "silence"
    # 40 speech frames (1.28 s) plus the trailing silence.
    assert 1.0 < got[0].duration < 6.0


def test_long_idle_then_speech_does_not_hit_the_duration_cap():
    """Regression: the cap counted frames since the stream opened, so after a
    long idle it fired the instant the user began speaking.

    3000 idle frames is 96 seconds, far past the 62-frame cap. A short
    utterance after that must still end on silence.
    """
    r = Recorder(max_utterance_s=2.0)  # cap is only 62 frames
    stream = iter(frames_of(3000, 0.0) + frames_of(30, 0.3) + frames_of(200, 0.0))
    r._is_speech = lambda f: np.sqrt(np.mean(np.square(f))) > 0.1
    got = list(r.endpoint(stream, timeout=30))
    assert len(got) == 1
    assert got[0].reason == "silence", "cap must count from speech, not from open"
    assert got[0].duration < 2.0


def test_speech_longer_than_the_cap_is_truncated_not_dropped():
    r = Recorder(max_utterance_s=1.0)  # 31 frames
    stream = iter(frames_of(500, 0.3))
    r._is_speech = lambda f: np.sqrt(np.mean(np.square(f))) > 0.1
    got = list(r.endpoint(stream, timeout=30))
    assert len(got) == 1
    assert got[0].reason == "max_duration"
    assert got[0].duration <= 1.5


def test_timeout_while_speaking_still_yields_the_audio():
    """Regression: the timeout path broke out of the loop and discarded the
    buffered speech, so a slow speaker lost their question entirely."""
    r = Recorder()
    stream = iter(frames_of(200, 0.3))
    r._is_speech = lambda f: np.sqrt(np.mean(np.square(f))) > 0.1
    got = list(r.endpoint(stream, timeout=0.0))  # already expired
    assert len(got) == 1
    assert got[0].reason == "timeout"
    assert got[0].audio.size > 0


def test_timeout_with_no_speech_yields_nothing():
    r = Recorder()
    r._is_speech = lambda f: False
    got = list(r.endpoint(iter(frames_of(50, 0.0)), timeout=0.0))
    assert got == []


def test_speech_too_short_is_not_reported():
    """A click or a cough must not become a question."""
    r = Recorder()
    stream = iter(frames_of(3, 0.3) + frames_of(200, 0.0))
    r._is_speech = lambda f: np.sqrt(np.mean(np.square(f))) > 0.1
    got = list(r.endpoint(stream, timeout=5))
    assert got == []


# --- adaptive noise tracking ------------------------------------------------


def test_noise_floor_adapts_up_to_a_higher_room():
    r = Recorder(rms_floor=0.004, noise_multiplier=3.0)
    loud_quiet_frame = np.full(FRAME_SAMPLES, 0.05, dtype=np.float32)
    for _ in range(400):
        r._update_noise(0.05)  # a sustained hum, all below the multiplier
    assert r.noise_floor > 0.03
    # 0.05 background now, so 0.02 must read as silence.
    assert r._is_speech(loud_quiet_frame) is False


def test_noise_floor_recovers_after_noise_stops():
    r = Recorder(rms_floor=0.004)
    for _ in range(400):
        r._update_noise(0.05)
    for _ in range(400):
        r._update_noise(0.001)
    assert r.noise_floor < 0.01, "the gate must not stay shut after a fan stops"


def test_quiet_speech_is_not_rejected_in_a_quiet_room():
    """The measured room floor is ~0.002; ordinary speech at 0.03 must pass."""
    r = Recorder(rms_floor=0.004, noise_multiplier=3.0)
    r._update_noise(0.002)
    frame = np.full(FRAME_SAMPLES, 0.03, dtype=np.float32)
    r._vad = lambda: (lambda *a, **k: 0.95)
    assert r._is_speech(frame) is True


def test_dead_silent_input_is_never_speech():
    r = Recorder(rms_floor=0.004)
    r._vad = lambda: (lambda *a, **k: 0.99)  # even a confident VAD
    assert r._is_speech(np.zeros(FRAME_SAMPLES, dtype=np.float32)) is False


def test_recorder_finalise_handles_empty_input():
    u = Recorder()._finalise([])
    assert u.audio.size == 0
    assert u.duration == 0.0


def test_recorder_finalise_keeps_very_short_utterances():
    r = Recorder()
    loud = np.full(FRAME_SAMPLES, 0.4, dtype=np.float32)
    u = r._finalise([loud] * 4)  # shorter than pre_roll + min_speech
    assert u.audio.size > 0, "a short utterance must not be trimmed to nothing"


def test_speech_duration():
    s = Speech(np.zeros(24_000, dtype=np.float32), 24_000, "edge", 5, "hi")
    assert s.duration == pytest.approx(1.0)
