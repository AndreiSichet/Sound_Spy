"""Experiment folders: ``experiments/<name>/``.

=====================  =====================================================  ===========
file                   content                                                git
=====================  =====================================================  ===========
``config.json``        resolved config, git commit, data-index hash, options  tracked
``log.csv``            one row per epoch (losses, mAP, per-class AP, LR …)    ignored
``checkpoints/``       ``best.pt`` (best val mAP), ``last.pt`` (resume)       ignored
``thresholds.json``    per-class thresholds tuned on val (max F1)             tracked
``metrics.json``       val/test metrics, fine-tuned and zero-shot             tracked
``plots/*.png``        loss / mAP curves, precision-recall per class          tracked
``train.log``          console output of a detached run                       ignored
=====================  =====================================================  ===========
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def log(self) -> Path:
        return self.root / "log.csv"

    @property
    def best(self) -> Path:
        return self.root / "checkpoints" / "best.pt"

    @property
    def last(self) -> Path:
        return self.root / "checkpoints" / "last.pt"

    @property
    def thresholds(self) -> Path:
        return self.root / "thresholds.json"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.json"

    @property
    def plots(self) -> Path:
        return self.root / "plots"

    def ensure(self) -> ExperimentPaths:
        (self.root / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.plots.mkdir(parents=True, exist_ok=True)
        return self


def experiment_paths(experiments_dir: Path, name: str) -> ExperimentPaths:
    if not name or any(c in name for c in r'\/:*?"<>|') or name in (".", ".."):
        raise ValueError(f"invalid experiment name {name!r}")
    return ExperimentPaths(Path(experiments_dir) / name)


def git_commit(project_root: Path) -> dict[str, Any]:
    """Current commit and whether the working tree has changes (read-only git commands)."""
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=project_root, capture_output=True,
                                 text=True, timeout=10, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def file_hash(paths: list[Path]) -> str:
    """SHA-256 over the given files (in order); missing files hash as their name only."""
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode())
        if path.exists():
            with open(path, "rb") as fh:
                while block := fh.read(1 << 20):
                    h.update(block)
    return h.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=_json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def append_log(path: Path, row: dict[str, Any]) -> None:
    """Append one row to ``log.csv`` (header written on first use)."""
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def read_log(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def truncate_log(path: Path, epochs: int) -> None:
    """Keep only the first ``epochs`` rows (resuming after an interrupted epoch)."""
    rows = read_log(path)
    if len(rows) <= epochs:
        return
    path.unlink()
    for row in rows[:epochs]:
        append_log(path, row)
