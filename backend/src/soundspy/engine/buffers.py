"""``Chunker``: turn blocks of any length into fixed-size chunks.

Sources and callers deliver audio in whatever block sizes suit them; the processing chain wants
exact ``chunk_frames`` chunks. ``push(block)`` returns every complete chunk now available and keeps
the remainder; ``flush()`` returns the remainder zero-padded to a full chunk, with
``metadata["valid_frames"]`` saying how much of it is real (or ``None`` if nothing is left).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from soundspy.core.audio import AudioBuffer


class Chunker:
    def __init__(self, chunk_frames: int) -> None:
        if chunk_frames < 1:
            raise ValueError("chunk_frames must be >= 1")
        self.chunk_frames = chunk_frames
        self._pending: np.ndarray | None = None
        self._sample_rate: int | None = None
        self._metadata: dict[str, Any] = {}

    @property
    def pending_frames(self) -> int:
        return 0 if self._pending is None else int(self._pending.shape[0])

    def push(self, block: AudioBuffer) -> list[AudioBuffer]:
        if self._pending is None:
            self._pending = np.empty((0, block.num_channels), dtype=np.float32)
            self._sample_rate = block.sample_rate
        elif (block.sample_rate, block.num_channels) != (self._sample_rate, self._pending.shape[1]):
            raise ValueError("all blocks pushed into a Chunker must share sample rate and channels")
        self._metadata = dict(block.metadata)
        self._metadata.pop("valid_frames", None)
        data = np.concatenate((self._pending, block.samples))
        n_full = data.shape[0] // self.chunk_frames
        out = [
            AudioBuffer(
                np.ascontiguousarray(data[i * self.chunk_frames : (i + 1) * self.chunk_frames]),
                block.sample_rate,
                dict(self._metadata),
            )
            for i in range(n_full)
        ]
        self._pending = data[n_full * self.chunk_frames :]
        return out

    def flush(self) -> AudioBuffer | None:
        if self._pending is None or self._pending.shape[0] == 0:
            return None
        assert self._sample_rate is not None
        valid = self._pending.shape[0]
        padded = np.zeros((self.chunk_frames, self._pending.shape[1]), dtype=np.float32)
        padded[:valid] = self._pending
        self._pending = self._pending[:0]
        return AudioBuffer(padded, self._sample_rate, {**self._metadata, "valid_frames": valid})
