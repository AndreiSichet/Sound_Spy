"""``soundspy record-dataset`` with the fake microphone: files, sidecars, keys, numbering."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from soundspy.cli import record_dataset_cmd as cmd
from soundspy.cli.main import main
from soundspy.core.audio import AudioBuffer


@pytest.fixture
def custom_config(config, tmp_path: Path):
    return dataclasses.replace(config, paths={**config.paths, "custom": tmp_path / "custom"})


def keys(*presses: str):
    it: Iterator[str] = iter(presses)

    def fake_input(_prompt: str) -> str:
        return next(it)

    return fake_input


def take(seconds: float) -> AudioBuffer:
    n = round(seconds * 16_000)
    x = 0.1 * np.sin(np.arange(n) / 10).astype(np.float32)
    return AudioBuffer(np.stack([x, x], axis=1), 16_000, {"device": "fake"})


def sidecar(path: Path) -> dict:
    return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))


def test_session_keys_files_and_sidecars(custom_config) -> None:
    kept = cmd.record_session(
        custom_config, take, classes=["speech", "background"], clips=2, seconds=0.1,
        notes="kitchen", input_fn=keys("", "r", "x", "d", "", "", "s"),
        countdown_s=0, session_id="sess-1",
    )
    root = custom_config.paths["custom"]
    # speech: take, redo it, unknown key, discard it, two takes; background skipped
    assert [p.name for p in kept] == ["speech_0001.wav", "speech_0002.wav"]
    assert sorted(p.name for p in (root / "speech").glob("*.wav")) == [
        "speech_0001.wav", "speech_0002.wav"]
    assert not (root / "background").exists()
    meta = sidecar(kept[0])
    assert meta["class"] == "speech" and meta["session_id"] == "sess-1"
    assert meta["notes"] == "kitchen" and meta["channels"] == 2


def test_quit_keeps_what_was_recorded(custom_config) -> None:
    kept = cmd.record_session(custom_config, take, classes=["dishes"], clips=5, seconds=0.1,
                              input_fn=keys("", "q"), countdown_s=0)
    assert [p.name for p in kept] == ["dishes_0001.wav"]


def test_numbering_continues_across_runs(custom_config) -> None:
    for _ in range(2):
        cmd.record_session(custom_config, take, classes=["background"], clips=2, seconds=0.1,
                           input_fn=keys("", ""), countdown_s=0)
    names = sorted(p.name for p in (custom_config.paths["custom"] / "background").glob("*.wav"))
    assert names == [f"background_000{i}.wav" for i in range(1, 5)]
    sessions = {sidecar(custom_config.paths["custom"] / "background" / n)["session_id"]
                for n in names}
    assert len(sessions) == 2


def test_unknown_class_is_rejected(custom_config) -> None:
    with pytest.raises(ValueError, match="unknown class"):
        cmd.valid_classes(custom_config, ["speech", "laser"])
    assert cmd.valid_classes(custom_config, None)[-1] == "background"


def test_cli_with_fake_microphone(custom_config, monkeypatch, capsys) -> None:
    profile = custom_config.device_profile
    monkeypatch.setattr(cmd, "load_profile", lambda args: (custom_config, profile))
    monkeypatch.setattr("builtins.input", keys("", "", "", ""))
    argv = ["record-dataset", "--classes", "speech", "background", "--clips", "2",
            "--seconds", "0.2", "--countdown", "0"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "RMS dBFS" in out and "Saved speech_0002.wav" in out
    wav = custom_config.paths["custom"] / "background" / "background_0002.wav"
    meta = sidecar(wav)
    assert meta["class"] == "background"
    assert meta["sample_rate"] == 48_000 and meta["duration_s"] == pytest.approx(0.2)
