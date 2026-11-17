"""Processing stage contracts.

- ``Processor``: a stateful stage in the audio path; consumes one chunk, returns one chunk.
  Spatial filters and extractors are processors.
- ``SideProcessor``: observes chunks without changing them and emits events. Detectors,
  analyzers, localizers and trackers are side processors.

Both are chunk-based and stateful, so offline processing (a file fed chunk by chunk) and live
processing use the same code. No stage may require the whole file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from soundspy.core.audio import AudioBuffer
from soundspy.core.events import Event
from soundspy.core.target import TargetSpec


class Processor(ABC):
    """Stateful stage in the audio path. Consumes and returns one chunk."""

    sample_rate: int | None = None
    num_channels: int | None = None
    target: TargetSpec | None = None

    @property
    def latency_samples(self) -> int:
        """Algorithmic delay of this stage in samples at the configured rate (0 = none).

        A stage that buffers (e.g. an STFT) reports it here, so the session's latency estimate
        and the delay-matched bypass signal include it.
        """
        return 0

    def configure(self, sample_rate: int, num_channels: int) -> int:
        """Prepare for chunks with this format; return the output channel count.

        The default keeps the channel count. A stage that reduces channels (e.g. a beamformer)
        returns its output count here.
        """
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        return num_channels

    def set_target(self, target: TargetSpec) -> None:
        self.target = target

    @abstractmethod
    def process(self, chunk: AudioBuffer) -> AudioBuffer: ...

    def start(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Called once **on the audio thread**, after ``configure`` and before the first chunk.

        For work that must happen on the thread that will call ``process``: e.g. warming a GPU
        model up (CUDA initialises per thread). The source is not delivering audio yet.
        """

    def reset(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Clear internal state (e.g. filter memories, model state buffers)."""

    def close(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Release resources (threads, devices) when the session ends."""


class SideProcessor(ABC):
    """Observes chunks without changing them; emits events."""

    sample_rate: int | None = None
    num_channels: int | None = None
    target: TargetSpec | None = None

    def configure(self, sample_rate: int, num_channels: int) -> None:
        self.sample_rate = sample_rate
        self.num_channels = num_channels

    def set_target(self, target: TargetSpec) -> None:
        self.target = target

    @abstractmethod
    def observe(self, chunk: AudioBuffer) -> list[Event]: ...

    def start(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Called once on the audio thread before the first chunk (see ``Processor.start``)."""

    def reset(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Clear internal state."""

    def close(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Release resources (e.g. the detector's inference thread) when the session ends."""


class Passthrough(Processor):
    """Returns the chunk unchanged (stand-in for stages not implemented yet)."""

    def process(self, chunk: AudioBuffer) -> AudioBuffer:
        return chunk
