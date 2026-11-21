"""``ProcessorChain``: main processors in order, with observers on the input and on the output.

``process(chunk)`` runs

1. every ``side`` processor on the **input** chunk (what the microphone heard),
2. the ``main`` processors in order,
3. every ``post_side`` processor on the **output** chunk (what the user hears),

and returns the output chunk together with all emitted events.

**Latency.** ``latency_samples`` is the sum of the main stages' algorithmic delays (e.g. the
extractor's STFT), so the session can report the true listening delay.

**Dry signal (for bypass / A-B listening).** With ``dry_after="<stage name>"`` the chain also
keeps the signal as it leaves that stage (in ``SEMANTIC``: after the channel reducer, before the
extractor), delayed by the latency of the stages after it. ``chain.dry`` is therefore
sample-aligned with the output: a listener can switch between the two without a jump in time.
A stage whose delay changes while it runs (the hybrid extractor since step 08: each route has its
own delay) provides that signal itself as ``aligned_dry`` after each ``process``; when it is the
only stage after the tap, the chain uses it instead of its fixed delay line.

Target changes are applied by the session at chunk boundaries via ``set_target``; the chain
itself is not thread-safe and is driven by one worker thread.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from soundspy.core.audio import AudioBuffer
from soundspy.core.events import Event
from soundspy.core.target import TargetSpec
from soundspy.engine.processor import Processor, SideProcessor


class ProcessorChain:
    def __init__(
        self,
        main: Sequence[Processor] = (),
        side: Sequence[SideProcessor] = (),
        *,
        post_side: Sequence[SideProcessor] = (),
        names: Mapping[str, Processor | SideProcessor] | None = None,
        dry_after: str | None = None,
    ) -> None:
        self.main: list[Processor] = list(main)
        self.side: list[SideProcessor] = list(side)
        self.post_side: list[SideProcessor] = list(post_side)
        self.target: TargetSpec | None = None
        self.dry: AudioBuffer | None = None
        # the DetectionState shared by the detector and the extractor (set by the factory)
        self.detection_state = None
        self._input_format: tuple[int, int] | None = None
        self._output_channels: int | None = None
        self._names: dict[str, Processor | SideProcessor] = {}
        # (code, message) pairs a session emits as warnings when it starts (e.g. no detector)
        self.warnings: list[tuple[str, str]] = []
        for name, stage in (names or {}).items():
            if not any(stage is s for s in self._stages()):
                raise ValueError(f"named stage {name!r} is not part of the chain")
            self._names[name] = stage
        if dry_after is not None and not isinstance(self._names.get(dry_after), Processor):
            raise ValueError(f"dry_after={dry_after!r} must name a main (Processor) stage")
        self._dry_after = dry_after
        self._dry_delay = np.zeros((0, 1), dtype=np.float32)

    def _stages(self) -> list[Processor | SideProcessor]:
        return [*self.main, *self.side, *self.post_side]

    def has(self, name: str) -> bool:
        return name in self._names

    def start(self) -> None:
        """Run every stage's ``start`` hook; call it on the thread that will call ``process``."""
        for stage in self._stages():
            stage.start()

    def close(self) -> None:
        """Release every stage's resources (called once when a session ends)."""
        for stage in self._stages():
            stage.close()

    def stage(self, name: str) -> Processor | SideProcessor:
        """A stage by name (e.g. ``"extractor"``)."""
        try:
            return self._names[name]
        except KeyError:
            names = sorted(self._names)
            raise KeyError(f"no stage named {name!r}; named stages: {names}") from None

    def replace(self, name: str, stage: Processor | SideProcessor) -> None:
        """Swap a named stage for another of the same kind (call ``configure`` again after)."""
        old = self.stage(name)
        if isinstance(old, Processor) != isinstance(stage, Processor):
            raise TypeError("a Processor can only be replaced by a Processor (same for side)")
        group: list = next(g for g in (self.main, self.side, self.post_side)
                           if any(s is old for s in g))
        group[next(i for i, s in enumerate(group) if s is old)] = stage
        self._names[name] = stage
        self._input_format = None  # must be configured again

    def configure(self, sample_rate: int, num_channels: int) -> int:
        """Configure every stage; channel counts propagate through the main chain.

        Returns the channel count of the chain's output.
        """
        if sample_rate <= 0 or num_channels < 1:
            raise ValueError("sample_rate must be > 0 and num_channels >= 1")
        for observer in self.side:
            observer.configure(sample_rate, num_channels)
        channels = num_channels
        dry_stage = self._names.get(self._dry_after) if self._dry_after else None
        dry_channels, delay = None, 0
        for stage in self.main:
            channels = stage.configure(sample_rate, channels)
            if channels < 1:
                raise ValueError(f"{type(stage).__name__} reported {channels} output channels")
            if dry_channels is not None:
                delay += stage.latency_samples
            if stage is dry_stage:
                dry_channels = channels
        if dry_channels is not None and dry_channels != channels:
            raise ValueError(
                f"the dry signal after {self._dry_after!r} has {dry_channels} channel(s) but the "
                f"chain output has {channels}: they must match for bypass")
        for observer in self.post_side:
            observer.configure(sample_rate, channels)
        self._dry_delay = np.zeros((delay, dry_channels or 1), dtype=np.float32)
        self.dry = None
        self._input_format = (sample_rate, num_channels)
        self._output_channels = channels
        return channels

    @property
    def output_channels(self) -> int | None:
        return self._output_channels

    @property
    def latency_samples(self) -> int:
        """Algorithmic delay input -> output in samples (sum over the main stages)."""
        return sum(stage.latency_samples for stage in self.main)

    def set_target(self, target: TargetSpec) -> None:
        self.target = target
        for stage in self._stages():
            stage.set_target(target)

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
        dry_stage = self._names.get(self._dry_after) if self._dry_after else None
        out = chunk
        for stage in self.main:
            out = stage.process(out)
            if stage is dry_stage:
                self.dry = self._delayed(out)
        if dry_stage is not None and self.main[-1] is not dry_stage:
            after = self.main[self.main.index(dry_stage) + 1:]
            aligned = getattr(after[0], "aligned_dry", None) if len(after) == 1 else None
            if aligned is not None:  # the stage's own delay-matched input (variable delay)
                self.dry = aligned
        for observer in self.post_side:
            events.extend(observer.observe(out))
        return out, events

    def _delayed(self, dry: AudioBuffer) -> AudioBuffer:
        """``dry`` delayed by the latency of the stages after the tap (aligned with the output)."""
        if not self._dry_delay.shape[0]:
            return dry
        n = dry.num_frames
        joined = np.concatenate((self._dry_delay, dry.samples))
        self._dry_delay = joined[n:]
        return AudioBuffer(np.ascontiguousarray(joined[:n]), dry.sample_rate, dry.metadata)

    def reset(self) -> None:
        for stage in self._stages():
            stage.reset()
        self._dry_delay = np.zeros_like(self._dry_delay)
        self.dry = None
