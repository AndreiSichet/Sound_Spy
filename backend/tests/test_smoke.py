"""Smoke tests: imports, CLI wiring, configs, library/API separation."""

from __future__ import annotations

import ast
import importlib
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import soundspy
from soundspy.cli.main import main

IMPLEMENTED_COMMANDS = [
    "serve", "devices", "check-mic", "record", "record-dataset", "data", "train",
    "evaluate", "detect",
]
PLACEHOLDER_COMMANDS = {
    "extract": "step-06",
}
ALL_COMMANDS = [*IMPLEMENTED_COMMANDS, *PLACEHOLDER_COMMANDS]
PACKAGE_DIR = Path(soundspy.__file__).parent
FORBIDDEN_IN_LIBRARY = ("fastapi", "starlette", "sqlalchemy", "soundspy.api")


def test_version() -> None:
    assert soundspy.__version__ == "0.1.0"


def test_all_modules_import() -> None:
    names = [m.name for m in pkgutil.walk_packages(soundspy.__path__, "soundspy.")]
    assert "soundspy.api.app" in names
    for name in names:
        importlib.import_module(name)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def test_library_does_not_import_api_frameworks() -> None:
    offenders = []
    for path in PACKAGE_DIR.rglob("*.py"):
        if "api" in path.relative_to(PACKAGE_DIR).parts[:1]:
            continue
        for module in _imported_modules(path):
            if any(module == f or module.startswith(f + ".") for f in FORBIDDEN_IN_LIBRARY):
                offenders.append(f"{path.relative_to(PACKAGE_DIR)}: {module}")
    assert not offenders, offenders


def test_library_import_does_not_load_api_frameworks() -> None:
    library = [
        m.name
        for m in pkgutil.walk_packages(soundspy.__path__, "soundspy.")
        if not m.name.startswith("soundspy.api")
    ]
    code = (
        "import importlib, sys\n"
        f"for name in {library!r}: importlib.import_module(name)\n"
        "print(','.join(m for m in ('fastapi', 'starlette', 'sqlalchemy') if m in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""


# Files allowed to index audio channels (rule 4): the reducer itself, level measurement (reads
# channels without changing the audio) and the sources applying a profile's raw_channels.
CHANNEL_INDEXING_ALLOWED = {
    "dsp/channels.py",
    "dsp/levels.py",
    "io/mic_source.py",
    "io/file_source.py",
}
CHANNEL_INDEXING = re.compile(r"\[\s*:\s*,|\.(mean|sum|average)\([^)]*axis\s*=\s*(1|-1)")


def test_channels_are_only_reduced_in_allowed_places() -> None:
    offenders = []
    for path in PACKAGE_DIR.rglob("*.py"):
        rel = path.relative_to(PACKAGE_DIR).as_posix()
        if rel in CHANNEL_INDEXING_ALLOWED:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if CHANNEL_INDEXING.search(line):
                offenders.append(f"{rel}:{number}: {line.strip()}")
    assert not offenders, offenders


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
    for name in ALL_COMMANDS:
        assert name in out


@pytest.mark.parametrize(("name", "step"), PLACEHOLDER_COMMANDS.items())
def test_subcommand_placeholder(name: str, step: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([name]) == 0
    assert capsys.readouterr().out.strip() == f"not implemented yet ({step})"


def test_cli_runs_as_process() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "soundspy.cli.main", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "serve" in result.stdout


def test_configs_are_valid_yaml(configs_dir: Path) -> None:
    files = sorted(configs_dir.rglob("*.yaml"))
    assert {p.relative_to(configs_dir).as_posix() for p in files} == {
        "default.yaml",
        "classes.yaml",
        "data/default.yaml",
        "data/esc50_to_audioset.yaml",
        "devices/laptop.yaml",
        "devices/respeaker_xvf3800.yaml",
        "detection/default.yaml",
        "extraction/mask_baseline.yaml",
        "extraction/neural_tse.yaml",
    }
    for path in files:
        yaml.safe_load(path.read_text(encoding="utf-8"))
    default = yaml.safe_load((configs_dir / "default.yaml").read_text("utf-8"))
    assert "labels" not in default, "classes.yaml is the single source of truth for classes"
