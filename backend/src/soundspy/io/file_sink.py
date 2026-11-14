"""``FileSink``: writes processed audio to a WAV file, chunk by chunk.

Writes incrementally with ``soundfile.SoundFile`` (32-bit float WAV), so a long session never has
to sit in memory. Only the ``valid_frames`` of a zero-padded last chunk are written. An existing
file is never overwritten unless ``overwrite=True``.
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from soundspy.core.audio import AudioBuffer
from soundspy.io.base import AudioSink, valid_frames


class FileSink(AudioSink):
    def __init__(self, path: str | Path, *, overwrite: bool = False) -> None:
        self.path = Path(path)
        self.overwrite = overwrite
        self.sample_rate: int | None = None
        self.num_channels: int | None = None
        self.frames_written = 0
        self._file: sf.SoundFile | None = None

    def open(self, sample_rate: int, num_channels: int) -> None:
        if self._file is not None:
            raise RuntimeError(f"{self.path} is already open")
        if self.path.exists() and not self.overwrite:
            raise FileExistsError(f"{self.path} already exists (refusing to overwrite)")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = sf.SoundFile(
            str(self.path),
            mode="w",
            samplerate=sample_rate,
            channels=num_channels,
            format="WAV",
            subtype="FLOAT",
        )
        self.sample_rate, self.num_channels = sample_rate, num_channels

    def write(self, chunk: AudioBuffer) -> None:
        if self._file is None:
            raise RuntimeError("FileSink.open() must be called before write()")
        if (chunk.sample_rate, chunk.num_channels) != (self.sample_rate, self.num_channels):
            raise ValueError(
                f"chunk format (sr={chunk.sample_rate}, ch={chunk.num_channels}) does not match "
                f"the open file (sr={self.sample_rate}, ch={self.num_channels})"
            )
        n = valid_frames(chunk)
        if n:
            self._file.write(chunk.samples[:n])
            self.frames_written += n

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
