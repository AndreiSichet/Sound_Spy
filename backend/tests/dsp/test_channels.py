"""ChannelReducer: the only place where channels are reduced."""

from __future__ import annotations

import numpy as np
import pytest

from soundspy.core.audio import AudioBuffer
from soundspy.dsp.channels import ChannelReducer
from soundspy.engine import Passthrough, ProcessorChain


def _chunk() -> AudioBuffer:
    samples = np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], dtype=np.float32)
    return AudioBuffer(samples, 16000, {"device": "x"})


def test_mean_mode() -> None:
    reducer = ChannelReducer("mean")
    assert reducer.configure(16000, 4) == 1
    out = reducer.process(_chunk())
    assert out.samples.shape == (2, 1) and out.samples.dtype == np.float32
    np.testing.assert_allclose(out.samples[:, 0], [1.5, 5.5])
    assert out.metadata == {"device": "x", "channel_reduction": "mean"}


def test_select_mode() -> None:
    reducer = ChannelReducer("select", channel=2)
    assert reducer.configure(16000, 4) == 1
    out = reducer.process(_chunk())
    assert out.samples.shape == (2, 1)
    np.testing.assert_array_equal(out.samples[:, 0], [2.0, 6.0])
    assert out.metadata["channel_reduction"] == "select:2"


def test_mono_input_stays_2d() -> None:
    mono = AudioBuffer(np.ones((5, 1), dtype=np.float32), 16000)
    assert ChannelReducer("mean").process(mono).samples.shape == (5, 1)


@pytest.mark.parametrize(
    "kwargs",
    [{"mode": "sum"}, {"mode": "select"}, {"mode": "select", "channel": -1},
     {"mode": "mean", "channel": 0}],
)
def test_invalid_arguments(kwargs) -> None:
    with pytest.raises(ValueError):
        ChannelReducer(**kwargs)


def test_select_out_of_range() -> None:
    with pytest.raises(ValueError, match="cannot select channel 4"):
        ChannelReducer("select", channel=4).configure(16000, 4)


def test_in_a_chain_reduces_to_one_channel() -> None:
    chain = ProcessorChain([ChannelReducer("mean"), Passthrough()])
    assert chain.configure(16000, 4) == 1
    out, _ = chain.process(_chunk())
    assert out.num_channels == 1
