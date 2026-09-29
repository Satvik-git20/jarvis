"""Every command in `python -m jarvis ...` must construct its options cleanly.

Argument wiring is invisible in unit tests and only breaks when someone types
the command, which is exactly when it is most annoying. These parse real argv
so a field rename cannot silently break the CLI again.
"""

from __future__ import annotations

import pytest

from jarvis.main import main
from jarvis.ui.cli import Config


def parse(argv: list[str]):
    """Run main() far enough to build the options, without starting the loop.

    main() imports run_cli inside the function, so patching the cli module is
    what actually intercepts it.
    """
    captured: dict = {}

    async def fake_run_cli(**kwargs):
        captured.update(kwargs)
        return 0

    import jarvis.ui.cli as cli_mod

    original = cli_mod.run_cli
    cli_mod.run_cli = fake_run_cli
    try:
        rc = main(argv)
    finally:
        cli_mod.run_cli = original
    return rc, captured


def test_no_subcommand_defaults_to_the_voice_loop():
    rc, kwargs = parse([])
    assert rc == 0
    assert kwargs["local_brain"] is False
    assert kwargs["speak"] is True
    assert kwargs["wake"] is False


def test_run_subcommand_flags_map_onto_config():
    rc, kwargs = parse([
        "run", "--wake", "--no-speak", "--cpu", "--tts-local",
        "--whisper", "base", "--voice", "en-US-JennyNeural",
        "--mic", "3", "--speaker", "5", "--silence-ms", "900",
        "--rms-floor", "0.02", "--vad-threshold", "0.6",
    ])
    assert rc == 0
    assert kwargs["wake"] is True
    assert kwargs["speak"] is False
    assert kwargs["prefer_gpu"] is False
    assert kwargs["tts_local"] is True
    assert kwargs["whisper_model"] == "base"
    assert kwargs["tts_voice"] == "en-US-JennyNeural"
    assert kwargs["input_device"] == 3
    assert kwargs["output_device"] == 5
    assert kwargs["silence_ms"] == 900
    assert kwargs["rms_floor"] == 0.02
    assert kwargs["vad_threshold"] == 0.6


def test_local_brain_flag_reaches_run_cli():
    rc, kwargs = parse(["--local-brain"])
    assert kwargs["local_brain"] is True


@pytest.mark.parametrize("argv", [
    ["run"],
    ["run", "--wake", "--no-speak", "--cpu"],
    ["run", "--mic", "2", "--speaker", "3"],
    [],
])
def test_every_kwarg_builds_a_config(argv):
    """The regression: main() passed a duplicate and an unknown keyword."""
    _, kwargs = parse(argv)
    cfg = Config(**kwargs)  # local_brain is already in kwargs
    assert isinstance(cfg, Config)
    assert cfg.whisper_model
    assert cfg.silence_ms > 0


def test_bad_device_index_becomes_none_not_a_crash():
    _, kwargs = parse(["run", "--mic", "not-a-number"])
    assert kwargs["input_device"] is None
    Config(**kwargs)  # must not raise
