"""Processing runs on disk: processed WAV + JSON sidecar.

A **run** is one offline processing of a recording (step 06: extraction of the selected classes).
It is two files with the same stem in ``outputs/processed/``:

- ``<run_id>.json``: the sidecar, written as soon as the run is created (status ``pending``) and
  rewritten when it starts, finishes or fails;
- ``<run_id>.wav``: the processed audio (32-bit float, one channel, time-aligned with the
  recording), present once the status is ``done``.

As for recordings, the **sidecar is the source of truth**; the app database only indexes it and
can be rebuilt from these files at any time (``soundspy db rebuild``).

Sidecar fields: ``run_id``, ``recording`` (path, relative to the project root when inside it),
``labels``, ``attenuation_db``, ``method``, ``parameters`` (the method's settings),
``activity_mode``, ``created``, ``status`` (``pending`` | ``running`` | ``done`` | ``failed``),
``error``, ``file``, ``duration_s``, ``sample_rate``, ``processing_seconds``, ``info`` (what the
method reports, e.g. where it found the target) and ``app_version``.

**Reading while writing.** A sidecar is rewritten (atomically: temp file + replace) while other
threads read it: the worker updates a run, a request re-indexes. On Windows a file cannot be
replaced while it is open, and cannot be opened in the instant it is replaced. Inside one process
a lock serialises both; across processes (``soundspy extract`` next to a running server) a locked
file is retried for a moment instead of failing.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import soundfile as sf

from soundspy import __version__
from soundspy.core.audio import AudioBuffer

RUN_STATUSES = ("pending", "running", "done", "failed")
FINISHED = ("done", "failed")
INTERRUPTED = "interrupted: the app stopped before this run finished"


class RunNotFoundError(FileNotFoundError):
    pass


def run_id_for(when: datetime) -> str:
    return f"run_{when.strftime('%Y%m%d-%H%M%S')}-{when.microsecond // 1000:03d}"


def sidecar_path(directory: str | Path, run_id: str) -> Path:
    return Path(directory) / f"{run_id}.json"


def audio_path(directory: str | Path, run_id: str) -> Path:
    return Path(directory) / f"{run_id}.wav"


_FILE_LOCK = threading.RLock()  # sidecar reads and rewrites within this process
_RETRIES, _RETRY_S = 20, 0.01  # a file locked by another process: up to ~0.2 s


def _retry(action: Callable[[], Any]) -> Any:
    for attempt in range(_RETRIES):
        try:
            return action()
        except PermissionError:
            if attempt == _RETRIES - 1:
                raise
            time.sleep(_RETRY_S)
    return None


def _write(path: Path, data: dict[str, Any]) -> None:
    """Replace the sidecar atomically (a reader never sees half a file)."""
    tmp = path.with_name(path.name + ".tmp")
    with _FILE_LOCK:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _retry(lambda: os.replace(tmp, path))


def create_run(directory: str | Path, *, recording: str, labels: list[str],
               attenuation_db: float, method: str, parameters: dict[str, Any],
               activity_mode: str) -> dict[str, Any]:
    """Write a new ``pending`` sidecar and return it. The run id is unique in ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    when = datetime.now().astimezone()
    data: dict[str, Any] = {
        "run_id": "",
        "recording": recording,
        "labels": sorted(labels),
        "attenuation_db": float(attenuation_db),
        "method": method,
        "parameters": parameters,
        "activity_mode": activity_mode,
        "created": when.isoformat(timespec="milliseconds"),
        "status": "pending",
        "error": None,
        "file": None,
        "duration_s": None,
        "sample_rate": None,
        "processing_seconds": None,
        "info": {},
        "app_version": __version__,
    }
    suffix = 0
    while True:
        run_id = run_id_for(when) + (f"-{suffix}" if suffix else "")
        data["run_id"] = run_id
        try:
            with sidecar_path(directory, run_id).open("x", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
            return data
        except FileExistsError:  # two runs in the same millisecond
            suffix += 1


def read_run(path: str | Path) -> dict[str, Any]:
    with _FILE_LOCK:
        text = _retry(lambda: Path(path).read_text(encoding="utf-8"))
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("run_id"), str):
        raise ValueError(f"{path}: not a run sidecar")
    return data


def load_run(directory: str | Path, run_id: str) -> dict[str, Any]:
    path = sidecar_path(directory, run_id)
    if not path.is_file():
        raise RunNotFoundError(f"no run {run_id!r} in {directory}")
    return read_run(path)


def update_run(directory: str | Path, run_id: str, **fields: Any) -> dict[str, Any]:
    if "status" in fields and fields["status"] not in RUN_STATUSES:
        raise ValueError(f"status must be one of {RUN_STATUSES}, got {fields['status']!r}")
    with _FILE_LOCK:  # read-modify-write as one step
        return _update(directory, run_id, fields)


def _update(directory: str | Path, run_id: str, fields: dict[str, Any]) -> dict[str, Any]:
    data = load_run(directory, run_id)
    data.update(fields)
    _write(sidecar_path(directory, run_id), data)
    return data


def finish_run(directory: str | Path, run_id: str, audio: AudioBuffer, *,
               processing_seconds: float, info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write the processed WAV, then mark the run ``done``."""
    wav = audio_path(directory, run_id)
    sf.write(str(wav), audio.samples, audio.sample_rate, format="WAV", subtype="FLOAT")
    return update_run(directory, run_id, status="done", error=None, file=wav.name,
                      duration_s=round(audio.num_frames / audio.sample_rate, 6),
                      sample_rate=audio.sample_rate,
                      processing_seconds=round(processing_seconds, 3), info=info or {})


def fail_run(directory: str | Path, run_id: str, error: str) -> dict[str, Any]:
    return update_run(directory, run_id, status="failed", error=error)


def iter_runs(directory: str | Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield ``(sidecar_path, sidecar)`` for every readable run sidecar in ``directory``."""
    directory = Path(directory)
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.json")):
        try:
            yield path, read_run(path)
        except (OSError, ValueError):
            continue


def interrupt_unfinished(directory: str | Path) -> int:
    """Mark runs left ``pending``/``running`` by a previous process as failed; return how many."""
    count = 0
    for _path, data in list(iter_runs(directory)):
        if data.get("status") not in FINISHED:
            fail_run(directory, data["run_id"], INTERRUPTED)
            count += 1
    return count
