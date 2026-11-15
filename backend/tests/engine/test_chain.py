"""Tests for ProcessorChain, Passthrough and modes."""

from __future__ import annotations

import numpy as np
import pytest

from soundspy.core import AudioBuffer, Detection, Event, TargetSpec
from soundspy.engine import Mode, Passthrough, Processor, ProcessorChain, SideProcessor, StageKind
from soundspy.engine.modes import main_stages_for, requires_array, side_stages_for, stages_for


class CountingSide(SideProcessor):
    def __init__(self) -> None:
        self.seen: list[AudioBuffer] = []
        self.resets = 0

    def observe(self, chunk: AudioBuffer) -> list[Event]:
        self.seen.append(chunk)
        return [Detection(t=0.0, label="speech", probability=0.5, t_start=0.0, t_end=0.1)]

    def reset(self) -> None:
        self.resets += 1


class HalveChannels(Processor):
    """Fake channel-reducing stage (stands in for a beamformer)."""

    def configure(self, sample_rate: int, num_channels: int) -> int:
        super().configure(sample_rate, num_channels)
        return max(1, num_channels // 2)

    def process(self, chunk: AudioBuffer) -> AudioBuffer:
        n = max(1, chunk.num_channels // 2)
        return AudioBuffer(np.ascontiguousarray(chunk.samples[:, :n]), chunk.sample_rate)


class Gain(Processor):
    def process(self, chunk: AudioBuffer) -> AudioBuffer:
        return AudioBuffer(chunk.samples * np.float32(2.0), chunk.sample_rate)


def _chunk(channels: int = 2, frames: int = 256) -> AudioBuffer:
    rng = np.random.default_rng(0)
    return AudioBuffer(rng.standard_normal((frames, channels)).astype(np.float32), 16000)


def test_passthrough_chain_output_equals_input() -> None:
    side = CountingSide()
    chain = ProcessorChain([Passthrough(), Passthrough()], [side, CountingSide()])
    assert chain.configure(16000, 2) == 2
    chunk = _chunk()
    out, events = chain.process(chunk)
    np.testing.assert_array_equal(out.samples, chunk.samples)
    assert len(events) == 2 and all(isinstance(e, Detection) for e in events)
    assert side.seen == [chunk]


def test_side_processors_observe_input_not_output() -> None:
    side = CountingSide()
    chain = ProcessorChain([Gain()], [side])
    chain.configure(16000, 1)
    chunk = _chunk(1)
    out, _ = chain.process(chunk)
    np.testing.assert_allclose(out.samples, chunk.samples * 2)
    assert side.seen[0] is chunk


def test_configure_propagates_channel_counts() -> None:
    first, second, third = HalveChannels(), Passthrough(), HalveChannels()
    side = CountingSide()
    chain = ProcessorChain([first, second, third], [side])
    assert chain.configure(16000, 8) == 2
    assert (first.num_channels, second.num_channels, third.num_channels) == (8, 4, 4)
    assert side.num_channels == 8 and side.sample_rate == 16000
    out, _ = chain.process(_chunk(8))
    assert out.num_channels == 2


def test_set_target_reaches_every_stage() -> None:
    stages = [Passthrough(), Gain()]
    sides = [CountingSide(), CountingSide()]
    chain = ProcessorChain(stages, sides)
    target = TargetSpec(labels={"speech"}, attenuation_db=20)
    chain.set_target(target)
    assert chain.target is target
    assert all(s.target is target for s in [*stages, *sides])


def test_process_requires_configure_and_matching_format() -> None:
    chain = ProcessorChain([Passthrough()])
    with pytest.raises(RuntimeError):
        chain.process(_chunk())
    chain.configure(16000, 2)
    with pytest.raises(ValueError):
        chain.process(_chunk(channels=3))


def test_reset_reaches_side_processors() -> None:
    side = CountingSide()
    chain = ProcessorChain([Passthrough()], [side])
    chain.reset()
    assert side.resets == 1


def test_processor_is_abstract() -> None:
    with pytest.raises(TypeError):
        Processor()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        SideProcessor()  # type: ignore[abstract]


# --- modes ------------------------------------------------------------------------------------


def test_mode_stage_mapping() -> None:
    semantic = {StageKind.EXTRACTOR, StageKind.DETECTOR, StageKind.ANALYZER}
    assert stages_for(Mode.SEMANTIC) == semantic
    assert stages_for(Mode.SPATIAL) == stages_for(Mode.SEMANTIC) | {
        StageKind.SPATIAL_FILTER,
        StageKind.LOCALIZER,
    }
    assert stages_for(Mode.SOURCES) == stages_for(Mode.SPATIAL) | {StageKind.TRACKER}


def test_main_stage_order() -> None:
    assert main_stages_for(Mode.SEMANTIC) == [StageKind.EXTRACTOR]
    assert main_stages_for(Mode.SPATIAL) == [StageKind.SPATIAL_FILTER, StageKind.EXTRACTOR]
    assert side_stages_for(Mode.SOURCES)[-1] == StageKind.TRACKER


def test_requires_array() -> None:
    assert not requires_array(Mode.SEMANTIC)
    assert requires_array(Mode.SPATIAL) and requires_array(Mode.SOURCES)
    assert Mode("semantic") is Mode.SEMANTIC


class Delay(Processor):
    """Delays by ``n`` samples and doubles the signal (a stand-in for an STFT stage)."""

    def __init__(self, n: int) -> None:
        self.n = n

    @property
    def latency_samples(self) -> int:
        return self.n

    def configure(self, sample_rate: int, num_channels: int) -> int:
        super().configure(sample_rate, num_channels)
        self._buffer = np.zeros((self.n, num_channels), dtype=np.float32)
        return num_channels

    def process(self, chunk: AudioBuffer) -> AudioBuffer:
        joined = np.concatenate((self._buffer, chunk.samples * np.float32(2.0)))
        self._buffer = joined[chunk.num_frames:]
        return AudioBuffer(np.ascontiguousarray(joined[: chunk.num_frames]), chunk.sample_rate)


def test_post_side_processors_observe_the_output() -> None:
    before, after = CountingSide(), CountingSide()
    chain = ProcessorChain([HalveChannels(), Gain()], [before], post_side=[after],
                           names={"out": after})
    assert chain.configure(16000, 2) == 1
    assert (after.num_channels, before.num_channels) == (1, 2)  # configured with the output
    chunk = _chunk()
    out, events = chain.process(chunk)
    assert before.seen[0] is chunk and after.seen[0] is out
    assert len(events) == 2
    target = TargetSpec(labels={"speech"})
    chain.set_target(target)
    chain.reset()
    assert after.target is target and after.resets == 1
    assert chain.stage("out") is after
    replacement = CountingSide()
    chain.replace("out", replacement)
    assert chain.post_side == [replacement]


def test_latency_is_the_sum_of_the_main_stages() -> None:
    chain = ProcessorChain([Passthrough(), Delay(100), Delay(28)])
    chain.configure(16000, 1)
    assert chain.latency_samples == 128
    assert ProcessorChain([Passthrough()]).latency_samples == 0


@pytest.mark.parametrize("frames", [64, 100, 1000])
def test_dry_signal_is_aligned_with_the_output(frames: int) -> None:
    reducer, delay = HalveChannels(), Delay(100)
    chain = ProcessorChain([reducer, delay], names={"reducer": reducer}, dry_after="reducer")
    chain.configure(16000, 2)
    assert chain.dry is None
    rng = np.random.default_rng(1)
    outs, dries = [], []
    for _ in range(8):
        chunk = AudioBuffer(rng.standard_normal((frames, 2)).astype(np.float32), 16000)
        out, _ = chain.process(chunk)
        outs.append(out.samples)
        dries.append(chain.dry.samples)
    out, dry = np.concatenate(outs), np.concatenate(dries)
    assert dry.shape == out.shape
    np.testing.assert_allclose(out, 2.0 * dry)  # same delay: only the processing differs
    assert not dry[:100].any() and dry[100:].any()
    chain.reset()
    assert chain.dry is None


def test_dry_tap_must_be_a_main_stage_with_the_output_channel_count() -> None:
    side = CountingSide()
    with pytest.raises(ValueError, match="dry_after"):
        ProcessorChain([Passthrough()], [side], names={"side": side}, dry_after="side")
    with pytest.raises(ValueError, match="dry_after"):
        ProcessorChain([Passthrough()], dry_after="missing")
    first = Passthrough()
    chain = ProcessorChain([first, HalveChannels()], names={"first": first}, dry_after="first")
    with pytest.raises(ValueError, match="must match for bypass"):
        chain.configure(16000, 2)


def test_close_reaches_every_stage() -> None:
    closed = []

    class Closing(SideProcessor):
        def observe(self, chunk: AudioBuffer) -> list[Event]:
            return []

        def close(self) -> None:
            closed.append(self)

    a, b = Closing(), Closing()
    ProcessorChain([Passthrough()], [a], post_side=[b]).close()
    assert closed == [a, b]
