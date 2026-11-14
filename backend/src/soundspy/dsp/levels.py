"""Level measurements and channel comparison.

All functions take an ``AudioBuffer`` and return one value per channel (a NumPy array), except
``channel_similarity`` which compares two 1-D signals. Reading individual channels here is
*measurement*, not reduction: the audio itself is never changed.

dBFS = decibels relative to full scale: a sample of 1.0 is 0 dBFS. A full-scale sine has a peak
of 0 dBFS and an RMS of -3.01 dBFS. Silence is reported as ``DB_FLOOR`` rather than -infinity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from soundspy.core.audio import AudioBuffer

DB_FLOOR = -120.0
_EPS = 10 ** (DB_FLOOR / 20)


def amplitude_to_dbfs(values: np.ndarray) -> np.ndarray:
    """20·log10(|x|), clamped at ``DB_FLOOR``."""
    return np.maximum(20.0 * np.log10(np.maximum(np.abs(values), _EPS)), DB_FLOOR)


def rms_dbfs(buf: AudioBuffer) -> np.ndarray:
    samples = buf.samples.astype(np.float64)
    return amplitude_to_dbfs(np.sqrt(np.mean(samples**2, axis=0)))


def peak_dbfs(buf: AudioBuffer) -> np.ndarray:
    return amplitude_to_dbfs(np.max(np.abs(buf.samples), axis=0))


def clipping_ratio(buf: AudioBuffer, threshold: float = 0.999) -> np.ndarray:
    """Fraction of samples per channel at or above ``threshold`` in magnitude."""
    return np.mean(np.abs(buf.samples) >= threshold, axis=0)


def digital_silence_ratio(buf: AudioBuffer, min_run_ms: float = 5.0) -> np.ndarray:
    """Fraction of samples per channel inside runs of exact 0.0 lasting at least ``min_run_ms``.

    Isolated zeros are normal: a 16-bit converter outputs its "0" code whenever the signal
    passes near zero (measured on the dev laptop: ~1.5 % of samples, in runs of at most 0.1 ms,
    exactly as often as the neighbouring codes ±1). Only *stretches* of exact zeros mean that
    something upstream (a noise gate, noise suppression, a muted input) replaced the sound.
    """
    min_run = max(1, round(buf.sample_rate * min_run_ms / 1000))
    zero = buf.samples == 0.0
    padded = np.zeros((zero.shape[0] + 2, zero.shape[1]), dtype=np.int8)
    padded[1:-1] = zero
    ratios = np.empty(buf.num_channels)
    for ch in range(buf.num_channels):
        edges = np.flatnonzero(np.diff(padded[:, ch]))
        runs = edges[1::2] - edges[0::2]
        ratios[ch] = runs[runs >= min_run].sum() / buf.num_frames
    return ratios


def noise_floor_dbfs(
    buf: AudioBuffer, window_ms: float = 50.0, percentile: float = 10
) -> np.ndarray:
    """Estimate the background level: the ``percentile`` of short-window RMS values.

    The quietest windows of a recording are (mostly) the noise floor: room tone, fan, the
    microphone's own hiss.
    """
    window = max(1, round(buf.sample_rate * window_ms / 1000))
    usable = (buf.num_frames // window) * window
    if usable == 0:
        return rms_dbfs(buf)
    samples = buf.samples[:usable].astype(np.float64)
    windows = samples.reshape(-1, window, buf.num_channels)
    window_rms = np.sqrt(np.mean(windows**2, axis=1))  # (n_windows, channels)
    return amplitude_to_dbfs(np.percentile(window_rms, percentile, axis=0))


@dataclass(frozen=True)
class ChannelSimilarity:
    identical: bool  # equal sample by sample (within 1e-6)
    corr_zero_lag: float  # normalized correlation with no time shift
    peak_corr: float  # normalized correlation at the best lag (signed)
    best_lag_samples: int  # positive = b receives the sound later than a
    best_lag_s: float


def channel_similarity(
    a: np.ndarray, b: np.ndarray, sample_rate: int, max_lag_s: float = 0.001
) -> ChannelSimilarity:
    """Compare two 1-D signals with an FFT-based, normalized cross-correlation.

    Lags are searched within ±``max_lag_s``. A positive ``best_lag_samples`` means ``b`` is a
    delayed copy of ``a``: the sound reached ``b``'s microphone later.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape or a.size == 0:
        raise ValueError("channel_similarity needs two non-empty signals of equal length")
    identical = bool(np.allclose(a, b, rtol=0.0, atol=1e-6))
    a0, b0 = a - a.mean(), b - b.mean()
    norm = float(np.sqrt(np.sum(a0**2) * np.sum(b0**2)))
    if norm == 0.0:
        return ChannelSimilarity(identical, 0.0, 0.0, 0, 0.0)

    corr = signal.correlate(b0, a0, mode="full", method="fft") / norm
    lags = signal.correlation_lags(b0.size, a0.size, mode="full")
    max_lag = max(1, int(np.ceil(max_lag_s * sample_rate)))
    window = np.abs(lags) <= max_lag
    corr, lags = corr[window], lags[window]
    best = int(np.argmax(np.abs(corr)))
    zero = float(corr[lags == 0][0])
    lag = int(lags[best])
    return ChannelSimilarity(
        identical=identical,
        corr_zero_lag=zero,
        peak_corr=float(corr[best]),
        best_lag_samples=lag,
        best_lag_s=lag / sample_rate,
    )


def compare_to_reference(
    buf: AudioBuffer, reference: int = 0, max_lag_s: float = 0.001
) -> dict[int, ChannelSimilarity]:
    """``channel_similarity`` of every other channel against ``reference``."""
    ref = buf.samples[:, reference]
    return {
        ch: channel_similarity(ref, buf.samples[:, ch], buf.sample_rate, max_lag_s)
        for ch in range(buf.num_channels)
        if ch != reference
    }
