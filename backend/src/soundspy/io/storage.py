"""Recording storage: WAV + JSON sidecar.

A recording is two files with the same stem in ``data/raw/`` (or any directory):

- ``<label or "rec">_<YYYYmmdd-HHMMSS-mmm>.wav``: 32-bit float, native sample rate, every kept
  channel, never resampled or downmixed;
- ``<same stem>.json``: the sidecar (device, host API, profile, channel indices, label, notes,
  overflows, app version, ...).

The sidecar is the **source of truth**; the app database is only an index built from it
(``soundspy.api.services.recordings``). Raw recordings are immutable: nothing here overwrites a
file.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import soundfile as sf

from soundspy import __version__
from soundspy.core.audio import AudioBuffer

SIDECAR_SUFFIX = ".json"


def _safe_label(label: str | None) -> str:
    if not label:
        return "rec"
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", label.strip()).strip("-")
    return cleaned or "rec"


def recording_stem(label: str | None, when: datetime) -> str:
    return f"{_safe_label(label)}_{when.strftime('%Y%m%d-%H%M%S')}-{when.microsecond // 1000:03d}"


def save_recording(
    buffer: AudioBuffer,
    directory: str | Path,
    *,
    label: str | None = None,
    notes: str = "",
    profile: str | None = None,
) -> Path:
    """Write ``buffer`` as WAV + JSON sidecar; return the WAV path. Never overwrites."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone()
    stem = recording_stem(label, now)
    wav_path = directory / f"{stem}.wav"
    sidecar_path = wav_path.with_suffix(SIDECAR_SUFFIX)
    for path in (wav_path, sidecar_path):
        if path.exists():
            raise FileExistsError(f"{path} already exists; raw recordings are never overwritten")

    meta = buffer.metadata
    sidecar: dict[str, Any] = {
        "file": wav_path.name,
        "sample_rate": buffer.sample_rate,
        "channels": buffer.num_channels,
        "duration_s": round(buffer.duration_s, 6),
        "created": now.isoformat(timespec="milliseconds"),
        "device": meta.get("device"),
        "hostapi": meta.get("hostapi"),
        "profile": profile if profile is not None else meta.get("profile"),
        "channel_indices": meta.get("channel_indices", list(range(buffer.num_channels))),
        "label": label,
        "notes": notes,
        "overflows": int(meta.get("overflows", 0)),
        "app_version": __version__,
    }

    # Exclusive creation ("x"): a concurrent writer can never clobber an existing recording.
    with wav_path.open("xb") as fh:
        sf.write(fh, buffer.samples, buffer.sample_rate, format="WAV", subtype="FLOAT")
    with sidecar_path.open("x", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return wav_path


def read_sidecar(wav_path: str | Path) -> dict[str, Any]:
    sidecar_path = Path(wav_path).with_suffix(SIDECAR_SUFFIX)
    with sidecar_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{sidecar_path}: sidecar must be a JSON object")
    return data


def load_recording(path: str | Path) -> tuple[AudioBuffer, dict[str, Any]]:
    """Load a recording and its sidecar. The buffer keeps every channel at the native rate."""
    path = Path(path)
    metadata = read_sidecar(path)
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    buffer = AudioBuffer(data, int(sample_rate), {**metadata, "source": "file", "path": str(path)})
    return buffer, metadata


def iter_recordings(directory: str | Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield ``(wav_path, sidecar)`` for every WAV in ``directory`` that has a readable sidecar."""
    directory = Path(directory)
    if not directory.is_dir():
        return
    for wav_path in sorted(directory.glob("*.wav")):
        if not wav_path.with_suffix(SIDECAR_SUFFIX).is_file():
            continue
        try:
            metadata = read_sidecar(wav_path)
        except (OSError, ValueError):
            continue
        yield wav_path, metadata
