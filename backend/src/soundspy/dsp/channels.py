"""``ChannelReducer``: the only place where channels are reduced.

Some later stages genuinely need one channel (e.g. a single-channel model). They get it from a
``ChannelReducer`` in the processing chain, never by indexing the audio themselves. Together with
the spatial ``SpatialFilter`` (beamforming, module 3) this is the only place channels may be
dropped or mixed.

Modes:

- ``mean``: average all channels;
- ``select``: keep one channel (``channel=``).

The output is always 2-D ``(frames, 1)``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from soundspy.core.audio import AudioBuffer
from soundspy.engine.processor import Processor

ReduceMode = Literal["mean", "select"]


class ChannelReducer(Processor):
    def __init__(self, mode: ReduceMode = "mean", channel: int | None = None) -> None:
        if mode not in ("mean", "select"):
            raise ValueError(f"mode must be 'mean' or 'select', got {mode!r}")
        if mode == "select" and (channel is None or channel < 0):
            raise ValueError("mode 'select' needs a channel index >= 0")
        if mode == "mean" and channel is not None:
            raise ValueError("mode 'mean' takes no channel")
        self.mode = mode
        self.channel = channel

    def configure(self, sample_rate: int, num_channels: int) -> int:
        super().configure(sample_rate, num_channels)
        if self.mode == "select" and self.channel is not None and self.channel >= num_channels:
            raise ValueError(f"cannot select channel {self.channel} from {num_channels} channel(s)")
        return 1

    def process(self, chunk: AudioBuffer) -> AudioBuffer:
        if self.mode == "mean":
            samples = chunk.samples.mean(axis=1, keepdims=True, dtype=np.float32)
            reduction = "mean"
        else:
            assert self.channel is not None
            if self.channel >= chunk.num_channels:
                raise ValueError(
                    f"cannot select channel {self.channel} from {chunk.num_channels} channel(s)"
                )
            samples = chunk.samples[:, [self.channel]]
            reduction = f"select:{self.channel}"
        return AudioBuffer(
            np.ascontiguousarray(samples, dtype=np.float32),
            chunk.sample_rate,
            {**chunk.metadata, "channel_reduction": reduction},
        )
