"""``ProcessorChain``: main processors in order plus side processors observing the input.

``process(chunk)`` runs every side processor on the **input** chunk (what the microphone
heard), then the main processors in order, and returns the output chunk together with all
emitted events. Target changes are applied by the session at chunk boundaries via
``set_target``; the chain itself is not thread-safe and is driven by one worker thread.
"""

from __future__ import annotations

from collections.abc import Sequence

from soundspy.core.audio import AudioBuffer
from soundspy.core.events import Event
from soundspy.core.target import TargetSpec
from soundspy.engine.processor import Processor, SideProcessor


class ProcessorChain:
    def __init__(
        self,
        main: Sequence[Processor] = (),
        side: Sequence[SideProcessor] = (),
    ) -> None:
        self.main: list[Processor] = list(main)
        self.side: list[SideProcessor] = list(side)
        self.target: TargetSpec | None = None
        self._input_format: tuple[int, int] | None = None
        self._output_channels: int | None = None

    def configure(self, sample_rate: int, num_channels: int) -> int:
        """Configure every stage; channel counts propagate through the main chain.

        Returns the channel count of the chain's output.
        """
        if sample_rate <= 0 or num_channels < 1:
            raise ValueError("sample_rate must be > 0 and num_channels >= 1")
        for observer in self.side:
            observer.configure(sample_rate, num_channels)
        channels = num_channels
        for stage in self.main:
            channels = stage.configure(sample_rate, channels)
            if channels < 1:
                raise ValueError(f"{type(stage).__name__} reported {channels} output channels")
        self._input_format = (sample_rate, num_channels)
        self._output_channels = channels
        return channels

    @property
    def output_channels(self) -> int | None:
        return self._output_channels

    def set_target(self, target: TargetSpec) -> None:
        self.target = target
        for stage in self.main:
            stage.set_target(target)
        for observer in self.side:
            observer.set_target(target)

    def process(self, chunk: AudioBuffer) -> tuple[AudioBuffer, list[Event]]:
        if self._input_format is None:
            raise RuntimeError("ProcessorChain.configure() must be called before process()")
        expected = self._input_format
        if (chunk.sample_rate, chunk.num_channels) != expected:
            raise ValueError(
                f"chunk format (sr={chunk.sample_rate}, ch={chunk.num_channels}) does not match "
                f"configured (sr={expected[0]}, ch={expected[1]})"
            )
        events: list[Event] = []
        for observer in self.side:
            events.extend(observer.observe(chunk))
        out = chunk
        for stage in self.main:
            out = stage.process(out)
        return out, events

    def reset(self) -> None:
        for stage in self.main:
            stage.reset()
        for observer in self.side:
            observer.reset()
