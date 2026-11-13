"""Smoke tests for the step-00 scaffold: imports, CLI wiring, configs."""

from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys

import pytest
import yaml

import soundfocus
from soundfocus.cli.main import main

SUBCOMMANDS = {
    "devices": "step-01",
    "check-mic": "step-01",
    "record": "step-01",
    "record-dataset": "step-02",
    "data": "step-03",
    "train": "step-05",
    "classify": "step-06",
    "enhance": "step-07",
}


def test_version() -> None:
    assert soundfocus.__version__ == "0.1.0"


def test_all_modules_import() -> None:
    names = [m.name for m in pkgutil.walk_packages(soundfocus.__path__, "soundfocus.")]
    assert "soundfocus.cli.main" in names
    for name in names:
        importlib.import_module(name)


def test_sounddevice_is_faked(fake_sounddevice) -> None:
    import sounddevice

    assert sounddevice is fake_sounddevice
    with pytest.raises(RuntimeError):
        sounddevice.rec(10)


def test_help_lists_all_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name in SUBCOMMANDS:
        assert name in out


@pytest.mark.parametrize(("name", "step"), SUBCOMMANDS.items())
def test_subcommand_placeholder(name: str, step: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([name]) == 0
    assert capsys.readouterr().out.strip() == f"not implemented yet ({step})"


def test_cli_runs_as_process() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "soundfocus.cli.main", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "record-dataset" in result.stdout


def test_configs_are_valid_yaml(project_root) -> None:
    files = sorted((project_root / "configs").rglob("*.yaml"))
    assert len(files) == 5
    for path in files:
        yaml.safe_load(path.read_text(encoding="utf-8"))
    default = yaml.safe_load((project_root / "configs" / "default.yaml").read_text("utf-8"))
    assert default["active_device_profile"] == "laptop"
    assert default["labels"] == [
        "speech", "music", "footsteps", "dishes", "chair_scrape", "alarm", "traffic"
    ]
