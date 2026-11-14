"""Chunker and build_chain."""

from __future__ import annotations

import numpy as np
import pytest

from soundspy.analysis.analyzer import Analyzer
from soundspy.core.audio import AudioBuffer
from soundspy.dsp.channels import ChannelReducer
from soundspy.engine.buffers import Chunker
from soundspy.engine.factory import ModeUnavailableError, build_chain
from soundspy.engine.processor import Passthrough


def _ramp(start: int, n: int, channels: int = 2) -> AudioBuffer:
    col = np.arange(start, start + n, dtype=np.float32)[:, None]
    return AudioBuffer(np.repeat(col, channels, axis=1), 16000, {"device": "x"})


@pytest.mark.parametrize("block_sizes", [[7] * 50, [1000, 1, 999, 3], [256] * 4, [1], [0, 500]])
def test_chunker_yields_exact_contiguous_chunks(block_sizes) -> None:
    chunker = Chunker(256)
    chunks, start = [], 0
    for n in block_sizes:
        chunks += chunker.push(_ramp(start, n))
        start += n
    assert all(c.samples.shape == (256, 2) for c in chunks)
    joined = np.concatenate([c.samples[:, 0] for c in chunks]) if chunks else np.empty(0)
    np.testing.assert_array_equal(joined, np.arange(len(chunks) * 256))
    assert chunker.pending_frames == start - len(chunks) * 256
    assert all(c.metadata == {"device": "x"} for c in chunks)


def test_chunker_flush_pads_and_marks_valid_frames() -> None:
    chunker = Chunker(100)
    chunker.push(_ramp(0, 130))
    last = chunker.flush()
    assert last is not None and last.samples.shape == (100, 2)
    assert last.metadata["valid_frames"] == 30
    np.testing.assert_array_equal(last.samples[:30, 0], np.arange(100, 130))
    assert not last.samples[30:].any()
    assert chunker.flush() is None


def test_chunker_rejects_format_changes() -> None:
    chunker = Chunker(10)
    chunker.push(_ramp(0, 5))
    with pytest.raises(ValueError):
        chunker.push(_ramp(0, 5, channels=1))
    with pytest.raises(ValueError):
        Chunker(0)


def test_build_semantic_chain(config) -> None:
    chain = build_chain("SEMANTIC", config, config.labels)
    assert isinstance(chain.stage("reducer"), ChannelReducer)
    assert isinstance(chain.stage("extractor"), Passthrough)
    assert isinstance(chain.stage("analyzer"), Analyzer)
    assert chain.configure(48000, 2) == 1  # reduced to mono for the output
    out, events = chain.process(AudioBuffer(np.zeros((960, 2), dtype=np.float32), 48000))
    assert out.num_channels == 1


@pytest.mark.parametrize(("mode", "fragment"), [("spatial", "module 3"), ("SOURCES", "module 5"),
                                                ("karaoke", "unknown mode")])
def test_unavailable_modes(config, mode, fragment) -> None:
    with pytest.raises(ModeUnavailableError, match=fragment):
        build_chain(mode, config, config.labels)


def test_named_stage_can_be_replaced(config) -> None:
    chain = build_chain("semantic", config, config.labels)
    new = Passthrough()
    chain.replace("extractor", new)
    assert chain.stage("extractor") is new and chain.main[1] is new
    with pytest.raises(RuntimeError):  # must be configured again after a replacement
        chain.process(AudioBuffer(np.zeros((10, 2), dtype=np.float32), 48000))
    with pytest.raises(TypeError):
        chain.replace("extractor", Analyzer(config.analysis))
    with pytest.raises(KeyError):
        chain.stage("detector")
