"""``AudioBuffer``: the one audio format used everywhere in Sound Spy.

``samples`` is a float32 NumPy array of shape ``(frames, channels)`` — always 2-D, also for
mono — plus the ``sample_rate`` in Hz and a free-form ``metadata`` dict. No code may assume a
specific channel count, and channels are reduced only in ``dsp/channels.py`` or by a
``SpatialFilter``; this class deliberately has no downmix/select methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AudioBuffer:
    samples: np.ndarray
    sample_rate: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.samples, np.ndarray):
            raise TypeError(f"samples must be a numpy array, got {type(self.samples).__name__}")
        if self.samples.dtype != np.float32:
            raise TypeError(f"samples must be float32, got {self.samples.dtype}")
        if self.samples.ndim != 2:
            raise ValueError(
                f"samples must be 2-D (frames, channels), got shape {self.samples.shape}"
            )
        if self.samples.shape[1] < 1:
            raise ValueError("samples must have at least one channel")
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int | np.integer):
            raise TypeError(f"sample_rate must be an int, got {type(self.sample_rate).__name__}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        object.__setattr__(self, "sample_rate", int(self.sample_rate))

    @property
    def num_frames(self) -> int:
        return int(self.samples.shape[0])

    @property
    def num_channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration_s(self) -> float:
        return self.num_frames / self.sample_rate
