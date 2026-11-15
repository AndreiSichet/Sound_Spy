"""Streaming-friendly STFT / inverse STFT (any channel count).

One framing for live and offline use:

* **Window:** square-root Hann (periodic) for both analysis and synthesis. Their product is a
  Hann window, whose shifted copies add up to a constant when ``n_fft`` is a multiple of ``hop``
  and ``n_fft / hop >= 2``, so a frame that is not modified is reconstructed exactly
  (overlap-add). The synthesis window carries the ``1 / constant`` factor.
* **Framing:** frame ``k`` covers the input samples ``[k*hop - (n_fft - hop), k*hop + hop)``:
  the signal is treated as preceded by ``n_fft - hop`` zeros, so a new frame is complete every
  ``hop`` samples from the very first sample on.

Arrays follow the ``AudioBuffer`` layout: time first. Samples are ``(frames, channels)``,
spectra are ``(stft_frames, channels, bins)`` complex64 with ``bins = n_fft // 2 + 1``.

**Streaming** (``StreamingSTFT`` + ``StreamingISTFT``): chunks of any length go in, complete
frames come out; the inverse overlap-adds them and ``read(n)`` returns exactly ``n`` samples.
Used as ``frames = stft.push(chunk)``, ``istft.push(modified)``, ``out = istft.read(len(chunk))``,
the output is the (processed) input delayed by exactly ``StftSpec.latency`` = ``n_fft`` samples,
whatever the chunk sizes: ``n_fft - hop`` because a sample is final only once the last frame
overlapping it has arrived, plus ``hop`` so that a read never has to wait for a frame boundary.

**Offline** (``stft`` / ``istft``): the same frames for a whole signal, with zero delay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StftSpec:
    n_fft: int = 1024
    hop: int = 256

    def __post_init__(self) -> None:
        n_fft, hop = self.n_fft, self.hop
        if n_fft < 4 or n_fft % 2:
            raise ValueError(f"n_fft must be an even number >= 4, got {n_fft}")
        if hop < 1 or n_fft % hop or n_fft // hop < 2:
            raise ValueError(
                f"hop must divide n_fft with n_fft / hop >= 2 (perfect reconstruction with a "
                f"sqrt-Hann window), got n_fft={n_fft}, hop={hop}")

    @property
    def bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def overlap(self) -> int:
        """Samples shared by consecutive frames (also the zeros assumed before the signal)."""
        return self.n_fft - self.hop

    @property
    def latency(self) -> int:
        """Delay in samples of ``StreamingSTFT`` -> ``StreamingISTFT.read``."""
        return self.n_fft

    def frames_for(self, num_samples: int) -> int:
        """Frames the offline ``stft`` returns for a signal of this length."""
        return 0 if num_samples <= 0 else -(-(num_samples + self.overlap) // self.hop)

    def frame_centre(self, k: int) -> float:
        """Centre of frame ``k`` in input samples (may be negative for the first frames)."""
        return k * self.hop + self.hop - self.n_fft / 2


def analysis_window(spec: StftSpec) -> np.ndarray:
    """Periodic square-root Hann window (float32)."""
    n = np.arange(spec.n_fft, dtype=np.float64)
    return np.sqrt(0.5 - 0.5 * np.cos(2.0 * np.pi * n / spec.n_fft)).astype(np.float32)


def synthesis_window(spec: StftSpec) -> np.ndarray:
    """The analysis window divided by the overlap-add constant (``n_fft / (2 * hop)``)."""
    return (analysis_window(spec).astype(np.float64) / (spec.n_fft / (2.0 * spec.hop))).astype(
        np.float32)


def _frames_to_spectra(segments: np.ndarray, window: np.ndarray) -> np.ndarray:
    """``(k, n_fft, channels)`` time segments -> ``(k, channels, bins)`` spectra."""
    windowed = segments.transpose(0, 2, 1) * window  # (k, channels, n_fft)
    return np.fft.rfft(windowed, axis=-1).astype(np.complex64, copy=False)


def _segments(samples: np.ndarray, spec: StftSpec) -> np.ndarray:
    """Every ``hop``-spaced window of ``n_fft`` samples: ``(k, n_fft, channels)`` (a view)."""
    if samples.shape[0] < spec.n_fft:
        return np.zeros((0, spec.n_fft, samples.shape[1]), dtype=samples.dtype)
    view = np.lib.stride_tricks.sliding_window_view(samples, spec.n_fft, axis=0)
    return view[:: spec.hop].transpose(0, 2, 1)  # (k, channels, n_fft) -> (k, n_fft, channels)


def _overlap_add(spectra: np.ndarray, spec: StftSpec, window: np.ndarray) -> np.ndarray:
    """``(k, channels, bins)`` -> overlap-added samples ``(k*hop + n_fft - hop, channels)``."""
    k, channels = spectra.shape[0], spectra.shape[1]
    hop, parts = spec.hop, spec.n_fft // spec.hop
    out = np.zeros((k * hop + spec.overlap, channels), dtype=np.float32)
    if k == 0:
        return out
    frames = np.fft.irfft(spectra, n=spec.n_fft, axis=-1).astype(np.float32, copy=False) * window
    # (k, channels, n_fft) -> (parts, k, hop, channels): the r-th hop-long piece of every frame
    pieces = frames.reshape(k, channels, parts, hop).transpose(2, 0, 3, 1)
    for r in range(parts):  # ... lands r hops later
        out[r * hop:r * hop + k * hop] += pieces[r].reshape(k * hop, channels)
    return out


def _as_2d(samples: np.ndarray) -> tuple[np.ndarray, bool]:
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 1:
        return x.reshape(-1, 1), True
    if x.ndim != 2:
        raise ValueError(f"expected (frames,) or (frames, channels) samples, got shape {x.shape}")
    return x, False


class StreamingSTFT:
    """Chunks of any length in, complete frames out."""

    def __init__(self, spec: StftSpec, channels: int = 1) -> None:
        if channels < 1:
            raise ValueError("channels must be >= 1")
        self.spec, self.channels = spec, channels
        self._window = analysis_window(spec)
        self.reset()

    def reset(self) -> None:
        self._buffer = np.zeros((self.spec.overlap, self.channels), dtype=np.float32)
        self.frames_out = 0

    def push(self, samples: np.ndarray) -> np.ndarray:
        """``(n, channels)`` samples -> ``(k, channels, bins)`` spectra (``k`` may be 0)."""
        x, _ = _as_2d(samples)
        if x.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channel(s), got {x.shape[1]}")
        buffer = np.concatenate((self._buffer, x)) if x.shape[0] else self._buffer
        segments = _segments(buffer, self.spec)
        k = segments.shape[0]
        spectra = _frames_to_spectra(segments, self._window)
        self._buffer = buffer[k * self.spec.hop:].copy() if k else buffer
        self.frames_out += k
        return spectra


class StreamingISTFT:
    """Frames in (``push``), exactly as many samples out as asked for (``read``)."""

    def __init__(self, spec: StftSpec, channels: int = 1) -> None:
        if channels < 1:
            raise ValueError("channels must be >= 1")
        self.spec, self.channels = spec, channels
        self._window = synthesis_window(spec)
        self.reset()

    def reset(self) -> None:
        self._tail = np.zeros((self.spec.overlap, self.channels), dtype=np.float32)
        # ``hop`` zeros up front: a read of n samples is then always possible after n samples
        # have gone into the matching StreamingSTFT (total delay: n_fft samples).
        self._ready = np.zeros((self.spec.hop, self.channels), dtype=np.float32)

    @property
    def available(self) -> int:
        return int(self._ready.shape[0])

    def push(self, spectra: np.ndarray) -> None:
        spectra = np.asarray(spectra)
        if spectra.ndim != 3 or spectra.shape[1:] != (self.channels, self.spec.bins):
            raise ValueError(f"expected (k, {self.channels}, {self.spec.bins}) spectra, "
                             f"got shape {spectra.shape}")
        k = spectra.shape[0]
        if k == 0:
            return
        y = _overlap_add(spectra, self.spec, self._window)
        y[: self.spec.overlap] += self._tail
        done = k * self.spec.hop
        self._ready = np.concatenate((self._ready, y[:done]))
        self._tail = y[done:].copy()

    def read(self, n: int) -> np.ndarray:
        """The next ``n`` output samples, ``(n, channels)`` float32."""
        if n > self._ready.shape[0]:
            raise RuntimeError(f"asked for {n} samples but only {self._ready.shape[0]} are "
                               "ready (push the frames of the matching input first)")
        out, self._ready = self._ready[:n], self._ready[n:]
        return np.ascontiguousarray(out)


def stft(samples: np.ndarray, spec: StftSpec) -> np.ndarray:
    """Offline STFT with the streaming framing, padded so every sample is fully covered.

    ``(n, channels)`` -> ``(k, channels, bins)``; a 1-D input gives ``(k, bins)``.
    """
    x, squeeze = _as_2d(samples)
    k = spec.frames_for(x.shape[0])
    total = (k - 1) * spec.hop + spec.n_fft if k else 0
    padded = np.zeros((total, x.shape[1]), dtype=np.float32)
    padded[spec.overlap:spec.overlap + x.shape[0]] = x
    spectra = _frames_to_spectra(_segments(padded, spec), analysis_window(spec))
    return spectra.reshape(k, spec.bins) if squeeze else spectra


def istft(spectra: np.ndarray, spec: StftSpec, length: int) -> np.ndarray:
    """Inverse of :func:`stft`: ``length`` samples, time-aligned with the original signal."""
    spectra = np.asarray(spectra)
    squeeze = spectra.ndim == 2
    if squeeze:
        spectra = spectra.reshape(spectra.shape[0], 1, spectra.shape[1])
    y = _overlap_add(spectra, spec, synthesis_window(spec))
    out = np.zeros((length, spectra.shape[1]), dtype=np.float32)
    part = y[spec.overlap:spec.overlap + length]
    out[: part.shape[0]] = part
    return out.ravel() if squeeze else out
