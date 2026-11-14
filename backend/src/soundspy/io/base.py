"""``AudioSource`` and ``AudioSink`` interfaces.

Every producer of audio (microphone, WAV file, later a simulated array) is an ``AudioSource``;
every consumer that takes audio out of the app (WAV file, later the speakers) is an
``AudioSink``. Code above ``io/`` only sees these interfaces and ``AudioBuffer`` objects.

Sources deliver audio two ways:

- ``read(seconds)``: block until exactly that much audio is available (record-then-process);
- ``chunks(chunk_frames)``: yield fixed-size chunks until closed or exhausted (streaming).

Both are usable as context managers: ``with MicSource(...) as mic: ...`` opens and always closes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

from soundspy.core.audio import AudioBuffer


@dataclass
class SourceStats:
    """Counters a source keeps while it runs."""

    overflows: int = 0  # the driver reported lost input (the app read too slowly)
    dropped_chunks: int = 0  # chunks discarded because the consumer fell behind
    frames_delivered: int = 0  # frames handed to the consumer
    warmup_frames_discarded: int = 0  # leading digital silence dropped after opening a mic


class AudioSource(ABC):
    sample_rate: int
    num_channels: int  # channels AFTER raw_channels selection

    def __init__(self) -> None:
        self.stats = SourceStats()

    @property
    def input_latency_s(self) -> float:
        """Delay between sound happening and the source handing it over (0 for files)."""
        return 0.0

    def describe(self) -> dict[str, Any]:
        """Short description for status displays (``source``, ``device``, ...)."""
        return {"source": type(self).__name__}

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def read(self, seconds: float) -> AudioBuffer:
        """Return exactly ``round(seconds * sample_rate)`` frames."""

    @abstractmethod
    def chunks(self, chunk_frames: int) -> Iterator[AudioBuffer]:
        """Yield chunks of exactly ``chunk_frames`` frames until closed or exhausted."""

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class AudioSink(ABC):
    @abstractmethod
    def open(self, sample_rate: int, num_channels: int) -> None: ...

    @abstractmethod
    def write(self, chunk: AudioBuffer) -> None:
        """Consume one chunk; honours ``chunk.metadata["valid_frames"]`` if present."""

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def frames_for(seconds: float, sample_rate: int) -> int:
    """Number of frames in ``seconds`` of audio (rounded), at least 1."""
    if seconds <= 0:
        raise ValueError(f"seconds must be > 0, got {seconds}")
    return max(1, round(seconds * sample_rate))


def valid_frames(chunk: AudioBuffer) -> int:
    """Frames of real audio in ``chunk`` (a zero-padded last chunk carries ``valid_frames``)."""
    n = chunk.metadata.get("valid_frames", chunk.num_frames)
    return max(0, min(int(n), chunk.num_frames))
