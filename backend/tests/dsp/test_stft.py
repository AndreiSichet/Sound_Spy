"""Streaming STFT: perfect reconstruction, reported delay, offline helpers, framing."""

from __future__ import annotations

import numpy as np
import pytest

from soundspy.dsp.stft import StftSpec, StreamingISTFT, StreamingSTFT, istft, stft


def run_identity(x: np.ndarray, spec: StftSpec, chunk: int) -> np.ndarray:
    forward, inverse = StreamingSTFT(spec, x.shape[1]), StreamingISTFT(spec, x.shape[1])
    out = []
    for i in range(0, x.shape[0], chunk):
        block = x[i:i + chunk]
        inverse.push(forward.push(block))
        out.append(inverse.read(block.shape[0]))
    return np.concatenate(out)


@pytest.mark.parametrize("chunk", [1, 100, 960, 4096])
@pytest.mark.parametrize(("n_fft", "hop"), [(1024, 256), (512, 256), (960, 240)])
def test_identity_reconstructs_after_the_reported_delay(chunk: int, n_fft: int, hop: int) -> None:
    spec = StftSpec(n_fft, hop)
    n = 6000 if chunk == 1 else 20_000
    x = np.random.default_rng(chunk + n_fft).standard_normal((n, 1)).astype(np.float32)
    y = run_identity(x, spec, chunk)
    delay = spec.latency
    assert y.shape == x.shape and delay == n_fft
    np.testing.assert_allclose(y[delay:], x[:-delay], atol=1e-5)
    np.testing.assert_allclose(y[:delay], 0.0, atol=1e-6)  # nothing before the delay


def test_any_channel_count_and_mixed_chunk_sizes() -> None:
    spec = StftSpec(1024, 256)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((30_000, 3)).astype(np.float32)
    forward, inverse = StreamingSTFT(spec, 3), StreamingISTFT(spec, 3)
    out, i = [], 0
    while i < x.shape[0]:
        n = int(rng.integers(1, 3000))
        block = x[i:i + n]
        spectra = forward.push(block)
        assert spectra.shape[1:] == (3, spec.bins)
        inverse.push(spectra)
        out.append(inverse.read(block.shape[0]))
        i += n
    y = np.concatenate(out)
    np.testing.assert_allclose(y[spec.latency:], x[:-spec.latency], atol=1e-5)
    assert forward.frames_out == x.shape[0] // spec.hop  # one frame per hop from the start


def test_offline_helpers_are_aligned_and_match_the_streaming_frames() -> None:
    spec = StftSpec(1024, 256)
    x = np.random.default_rng(5).standard_normal(10_000).astype(np.float32)
    spectra = stft(x, spec)
    assert spectra.shape == (spec.frames_for(x.size), spec.bins)
    np.testing.assert_allclose(istft(spectra, spec, x.size), x, atol=1e-5)  # zero delay
    streamed = StreamingSTFT(spec).push(x)
    k = streamed.shape[0]
    np.testing.assert_allclose(streamed.reshape(k, -1), spectra[:k], atol=1e-4)
    # 2-D in -> 3-D out, same numbers
    np.testing.assert_allclose(stft(x.reshape(-1, 1), spec).reshape(-1, spec.bins), spectra)
    assert stft(np.zeros(0, np.float32), spec).shape == (0, spec.bins)


def test_a_gain_applied_to_frames_scales_the_output() -> None:
    spec = StftSpec(512, 128)
    x = np.random.default_rng(9).standard_normal((8000, 1)).astype(np.float32)
    forward, inverse = StreamingSTFT(spec), StreamingISTFT(spec)
    inverse.push(forward.push(x) * 0.25)
    y = inverse.read(x.shape[0])
    np.testing.assert_allclose(y[spec.latency:], 0.25 * x[:-spec.latency], atol=1e-5)


def test_invalid_specs_and_reads() -> None:
    for n_fft, hop in [(1024, 300), (1024, 1024), (1023, 341), (2, 1)]:
        with pytest.raises(ValueError):
            StftSpec(n_fft, hop)
    inverse = StreamingISTFT(StftSpec(1024, 256))
    with pytest.raises(RuntimeError):
        inverse.read(257)  # only the priming hop is available before any frame
    with pytest.raises(ValueError):
        StreamingSTFT(StftSpec(1024, 256), 2).push(np.zeros((10, 1), np.float32))
