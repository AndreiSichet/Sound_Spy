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


class RecordingWriter:
    """Writes one recording incrementally: WAV now, sidecar on ``close()``.

    Used by ``save_recording`` (one buffer) and by live sessions (chunk after chunk). The WAV is
    created exclusively when the writer is constructed, so an existing recording can never be
    overwritten. ``device`` / ``hostapi`` / ``channel_indices`` / ``profile`` are taken from
    ``metadata`` (normally the first chunk's metadata); ``overflows`` can be updated until close.
    ``stem`` fixes the file name (default: label + timestamp); ``extra`` adds sidecar fields
    (``record-dataset`` stores ``class`` and ``session_id`` this way).
    """

    def __init__(
        self,
        directory: str | Path,
        sample_rate: int,
        channels: int,
        *,
        label: str | None = None,
        notes: str = "",
        profile: str | None = None,
        metadata: dict[str, Any] | None = None,
        stem: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.created = datetime.now().astimezone()
        self.extra = dict(extra or {})
        name = stem if stem is not None else recording_stem(label, self.created)
        self.wav_path = directory / f"{name}.wav"
        self.sidecar_path = self.wav_path.with_suffix(SIDECAR_SUFFIX)
        if self.sidecar_path.exists():
            raise FileExistsError(
                f"{self.sidecar_path} already exists; raw recordings are never overwritten"
            )
        meta = metadata or {}
        self.sample_rate, self.channels = sample_rate, channels
        self.label, self.notes = label, notes
        self.profile = profile if profile is not None else meta.get("profile")
        self.device, self.hostapi = meta.get("device"), meta.get("hostapi")
        self.channel_indices = meta.get("channel_indices", list(range(channels)))
        self.overflows = int(meta.get("overflows", 0))
        self.frames = 0
        try:
            # Exclusive creation ("x"): fails instead of clobbering an existing file.
            self._fh = self.wav_path.open("xb")
        except FileExistsError as exc:
            raise FileExistsError(
                f"{self.wav_path} already exists; raw recordings are never overwritten"
            ) from exc
        self._file = sf.SoundFile(
            self._fh, mode="w", samplerate=sample_rate, channels=channels,
            format="WAV", subtype="FLOAT",
        )
        self._closed = False

    @property
    def duration_s(self) -> float:
        return self.frames / self.sample_rate

    def write(self, chunk: AudioBuffer) -> None:
        if self._closed:
            raise RuntimeError("recording already closed")
        if (chunk.sample_rate, chunk.num_channels) != (self.sample_rate, self.channels):
            raise ValueError("chunk format does not match the recording")
        n = int(chunk.metadata.get("valid_frames", chunk.num_frames))
        n = max(0, min(n, chunk.num_frames))
        if n:
            self._file.write(chunk.samples[:n])
            self.frames += n

    def close(self) -> Path:
        """Finish the WAV, write the sidecar, return the WAV path (idempotent)."""
        if self._closed:
            return self.wav_path
        self._closed = True
        self._file.close()
        self._fh.close()
        sidecar: dict[str, Any] = {
            "file": self.wav_path.name,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_s": round(self.duration_s, 6),
            "created": self.created.isoformat(timespec="milliseconds"),
            "device": self.device,
            "hostapi": self.hostapi,
            "profile": self.profile,
            "channel_indices": self.channel_indices,
            "label": self.label,
            "notes": self.notes,
            "overflows": self.overflows,
            "app_version": __version__,
        }
        sidecar.update({k: v for k, v in self.extra.items() if k not in sidecar})
        with self.sidecar_path.open("x", encoding="utf-8") as fh:
            json.dump(sidecar, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        return self.wav_path


def save_recording(
    buffer: AudioBuffer,
    directory: str | Path,
    *,
    label: str | None = None,
    notes: str = "",
    profile: str | None = None,
) -> Path:
    """Write ``buffer`` as WAV + JSON sidecar; return the WAV path. Never overwrites."""
    writer = RecordingWriter(
        directory,
        buffer.sample_rate,
        buffer.num_channels,
        label=label,
        notes=notes,
        profile=profile,
        metadata=buffer.metadata,
    )
    writer.write(buffer)
    return writer.close()


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


def iter_sidecars(directory: str | Path) -> Iterator[tuple[Path, dict[str, Any], bool]]:
    """Yield ``(wav_path, sidecar, wav_exists)`` for every readable recording sidecar.

    Unlike :func:`iter_recordings` this also reports sidecars whose WAV is gone, so an index can
    show such a recording as *missing* instead of forgetting it.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return
    for sidecar_path in sorted(directory.glob(f"*{SIDECAR_SUFFIX}")):
        wav_path = sidecar_path.with_suffix(".wav")
        try:
            metadata = read_sidecar(wav_path)
        except (OSError, ValueError):
            continue
        if "sample_rate" not in metadata or "channels" not in metadata:
            continue  # some other JSON file
        yield wav_path, metadata, wav_path.is_file()


def stored_path(path: str | Path, project_root: Path) -> str:
    """How a file is referred to in sidecars and the index: relative to the project root when it
    lives inside it (portable across machines), absolute otherwise; always forward slashes."""
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def absolute_path(stored: str, project_root: Path) -> Path:
    path = Path(stored)
    return path if path.is_absolute() else Path(project_root) / path
