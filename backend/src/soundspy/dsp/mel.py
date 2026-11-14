"""Mel scale and mel filterbank in plain NumPy (no torch).

The mel scale spaces frequencies the way hearing does: finely at low frequencies, coarsely at
high ones. HTK formula: ``mel = 2595 * log10(1 + f / 700)``.

``mel_filterbank`` returns triangular filters with a peak of 1, one row per band, to be applied to
an FFT **power** spectrum. Bands are equally spaced on the mel scale between ``fmin`` and ``fmax``.
"""

from __future__ import annotations

import numpy as np


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_band_edges_hz(n_bands: int, fmin: float, fmax: float) -> np.ndarray:
    """``n_bands + 2`` frequencies: left edge, the ``n_bands`` centres, right edge."""
    return mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_bands + 2))


def mel_filterbank(
    sample_rate: int, n_fft: int, n_bands: int, fmin: float, fmax: float | None = None
) -> np.ndarray:
    """Triangular mel filters, shape ``(n_bands, n_fft // 2 + 1)``.

    Every band gets at least one FFT bin (narrow low bands fall back to their nearest bin), so no
    band is ever empty even when the FFT resolution is coarser than the band spacing.
    """
    fmax = sample_rate / 2 if fmax is None else fmax
    if not 0 <= fmin < fmax <= sample_rate / 2:
        raise ValueError(f"need 0 <= fmin < fmax <= Nyquist, got fmin={fmin}, fmax={fmax}")
    freqs = np.linspace(0.0, sample_rate / 2, n_fft // 2 + 1)
    edges = mel_band_edges_hz(n_bands, fmin, fmax)
    lower, centre, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rising = (freqs[None, :] - lower) / (centre - lower)
    falling = (upper - freqs[None, :]) / (upper - centre)
    bank = np.clip(np.minimum(rising, falling), 0.0, None)
    empty = bank.max(axis=1) == 0  # a band whose triangle covers no FFT bin
    if np.any(empty):
        nearest = np.abs(freqs[None, :] - centre[empty]).argmin(axis=1)
        bank[np.flatnonzero(empty), nearest] = 1.0
    return bank
