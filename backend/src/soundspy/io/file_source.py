"""``FileSource``: replays a WAV file through the ``AudioSource`` interface.

Downstream code cannot tell a file from a microphone: audio arrives as ``AudioBuffer`` objects
(always 2-D float32) at the file's own sample rate. If a device profile is given, its
``raw_channels`` are applied exactly as ``MicSource`` would, so a raw 6-channel recording replays
as the same 4 channels a live array would give.

- ``chunks()`` yields exactly ``chunk_frames`` frames; the last chunk is zero-padded and carries
  ``metadata["valid_frames"]``.
- ``realtime=True`` paces the chunks at playback speed (simulates a live microphone).
- ``loop=True`` wraps around to the start instead of ending.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from soundspy.config import DeviceProfile
from soundspy.core.audio import AudioBuffer
from soundspy.io.base import AudioSource, frames_for


class FileSource(AudioSource):
    def __init__(
        self,
        path: str | Path,
        *,
        profile: DeviceProfile | None = None,
        realtime: bool = False,
        loop: bool = False,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.profile = profile
        self.realtime = realtime
        self.loop = loop
        info = sf.info(str(self.path))
        self.sample_rate = int(info.samplerate)
        self.file_channels = int(info.channels)
        self.total_frames = int(info.frames)
        if self.total_frames == 0:
            raise ValueError(f"{self.path} contains no audio")
        raw = profile.raw_channels if profile is not None else None
        if raw is not None and max(raw) >= self.file_channels:
            raise ValueError(
                f"profile {profile.name!r} keeps channels {list(raw)} but {self.path.name} "
                f"has only {self.file_channels} channel(s)"
            )
        self.channel_indices: tuple[int, ...] = (
            tuple(raw) if raw is not None else tuple(range(self.file_channels))
        )
        self.num_channels = len(self.channel_indices)
        self._keep = np.asarray(self.channel_indices, dtype=np.intp)
        self._file: sf.SoundFile | None = None
        self._closed = True

    def open(self) -> None:
        if self._file is None:
            self._file = sf.SoundFile(str(self.path))
            self._closed = False

    def close(self) -> None:
        self._closed = True
        if self._file is not None:
            self._file.close()
            self._file = None

    def _metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "source": "file",
            "file": str(self.path),
            "channel_indices": list(self.channel_indices),
        }
        if self.profile is not None:
            meta["profile"] = self.profile.name
        return meta

    def _take(self, n: int) -> tuple[np.ndarray, int]:
        """Next ``n`` frames (selected channels); zero-padded at the end unless looping."""
        if self._file is None:
            self.open()
        assert self._file is not None
        out = np.zeros((n, self.num_channels), dtype=np.float32)
        filled = 0
        while filled < n:
            block = self._file.read(n - filled, dtype="float32", always_2d=True)
            got = block.shape[0]
            if got:
                out[filled : filled + got] = block[:, self._keep]
                filled += got
            if filled < n:
                if not self.loop:
                    break
                self._file.seek(0)
        return out, filled

    def read(self, seconds: float) -> AudioBuffer:
        n = frames_for(seconds, self.sample_rate)
        data, valid = self._take(n)
        self.stats.frames_delivered += valid
        meta = self._metadata()
        if valid < n:
            meta["valid_frames"] = valid
        return AudioBuffer(data, self.sample_rate, meta)

    def chunks(self, chunk_frames: int) -> Iterator[AudioBuffer]:
        if chunk_frames < 1:
            raise ValueError("chunk_frames must be >= 1")
        self.open()
        meta = self._metadata()
        chunk_s = chunk_frames / self.sample_rate
        start = time.monotonic()
        index = 0
        try:
            while not self._closed:
                data, valid = self._take(chunk_frames)
                if valid == 0:
                    return
                if self.realtime:
                    delay = start + index * chunk_s - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                self.stats.frames_delivered += valid
                chunk_meta = {**meta, "chunk_index": index}
                if valid < chunk_frames:
                    chunk_meta["valid_frames"] = valid
                yield AudioBuffer(data, self.sample_rate, chunk_meta)
                index += 1
                if valid < chunk_frames:
                    return
        finally:
            self.close()
