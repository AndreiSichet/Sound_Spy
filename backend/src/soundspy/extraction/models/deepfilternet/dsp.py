# Copyright (c) 2021 Hendrik Schröter
#
# Licensed under either of the Apache License, Version 2.0 (LICENSE-APACHE in this folder) or
# the MIT license (LICENSE-MIT in this folder), at your option.
#
# A NumPy re-implementation of the signal processing of libDF (DeepFilterNet's Rust library,
# https://github.com/Rikorose/DeepFilterNet, version 0.5.6): the analysis / synthesis STFT, the
# ERB band layout and the two feature normalisations. Same algorithms and constants; checked
# against libDF's own output (tests/fixtures/deepfilternet_dsp_reference.npz).
"""DeepFilterNet's signal processing, streaming, in NumPy.

* **STFT** (``Analysis`` / ``Synthesis``): a 960-sample Vorbis window, hop 480 (10 ms at
  48 kHz). Analysis takes exactly one hop of new samples per frame (the window covers the
  previous hop and the new one), scales the spectrum by ``wnorm = 2 * hop / fft_size**2``;
  synthesis inverse-transforms without scaling, windows again and overlap-adds. The Vorbis window
  is power-complementary at 50 % overlap, so analysis + synthesis returns the input delayed by
  ``fft_size - hop`` samples.
* **ERB features**: the power of each of 32 ERB bands in dB (``10 log10(mean |X|^2 + 1e-10)``),
  minus its running mean (``erb_norm``), divided by 40.
* **Complex features** of the lowest ``nb_df`` bins (96 = up to 4.8 kHz): each bin divided by
  the square root of its running mean magnitude (``unit_norm``).

Both running means are exponential with ``alpha = exp(-hop / (sr * norm_tau))`` rounded to the
first precision below 1 (0.99 for 10 ms and 1 s), and start from libDF's fixed values, so a
stream and a whole file give the same features frame by frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MEAN_NORM_INIT = (-60.0, -90.0)  # libDF: initial running mean of the ERB features (dB)
UNIT_NORM_INIT = (0.001, 0.0001)  # libDF: initial running mean magnitude of the complex bins


@dataclass(frozen=True)
class DfParams:
    """The ``[df]`` and ``[deepfilternet]`` settings of a DeepFilterNet model (``config.ini``)."""

    sr: int = 48000
    fft_size: int = 960
    hop_size: int = 480
    nb_erb: int = 32
    nb_df: int = 96
    norm_tau: float = 1.0
    lsnr_max: float = 35.0
    lsnr_min: float = -15.0
    min_nb_erb_freqs: int = 2
    df_order: int = 5
    df_lookahead: int = 2
    conv_lookahead: int = 2
    conv_ch: int = 64
    conv_kernel: tuple[int, int] = (1, 3)
    convt_kernel: tuple[int, int] = (1, 3)
    conv_kernel_inp: tuple[int, int] = (3, 3)
    emb_hidden_dim: int = 256
    emb_num_layers: int = 3
    enc_linear_groups: int = 32
    linear_groups: int = 16
    df_hidden_dim: int = 256
    df_num_layers: int = 2
    df_pathway_kernel_size_t: int = 5
    df_gru_skip: str = "groupedlinear"
    emb_gru_skip: str = "none"
    emb_gru_skip_enc: str = "none"
    enc_concat: bool = False
    pf_beta: float = 0.02

    @property
    def freq_bins(self) -> int:
        return self.fft_size // 2 + 1

    @property
    def lookahead(self) -> int:
        """Frames of future input an output frame needs."""
        return max(self.conv_lookahead, self.df_lookahead)


def norm_alpha(p: DfParams) -> float:
    """``exp(-dt / tau)``, rounded like DeepFilterNet's ``get_norm_alpha``."""
    a_ = math.exp(-(p.hop_size / p.sr) / p.norm_tau)
    precision, a = 3, 1.0
    while a >= 1.0:
        a = round(a_, precision)
        precision += 1
    return a


def vorbis_window(fft_size: int) -> np.ndarray:
    half = fft_size // 2
    i = np.arange(fft_size, dtype=np.float64)
    s = np.sin(0.5 * np.pi * (i + 0.5) / half)
    return np.sin(0.5 * np.pi * s * s).astype(np.float32)


def _freq2erb(f: float) -> float:
    return 9.265 * math.log1p(f / (24.7 * 9.265))


def _erb2freq(n: float) -> float:
    return 24.7 * 9.265 * (math.exp(n / 9.265) - 1.0)


def erb_widths(p: DfParams) -> np.ndarray:
    """Number of frequency bins per ERB band (libDF ``erb_fb``)."""
    nyq = p.sr // 2
    freq_width = p.sr / p.fft_size
    low, high = _freq2erb(0.0), _freq2erb(float(nyq))
    step = (high - low) / p.nb_erb
    widths = np.zeros(p.nb_erb, dtype=np.int64)
    prev, over = 0, 0
    for i in range(1, p.nb_erb + 1):
        fb = int(round(_erb2freq(low + i * step) / freq_width))
        n = fb - prev - over
        if n < p.min_nb_erb_freqs:
            over = p.min_nb_erb_freqs - n
            n = p.min_nb_erb_freqs
        else:
            over = 0
        widths[i - 1] = n
        prev = fb
    widths[-1] += 1  # fft_size / 2 + 1 bins
    too_large = int(widths.sum()) - p.freq_bins
    if too_large > 0:
        widths[-1] -= too_large
    return widths


def erb_matrices(widths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(fb, inverse)``: bins -> band means ``[F, E]`` and bands -> bins ``[E, F]`` (the
    model's ``erb_fb`` / ``erb_inv_fb`` buffers)."""
    n_freqs = int(np.sum(widths))
    fb = np.zeros((n_freqs, len(widths)), dtype=np.float32)
    start = 0
    for band, width in enumerate(widths):
        fb[start:start + int(width), band] = 1.0
        start += int(width)
    return fb / fb.sum(axis=0, keepdims=True), fb.T.copy()


class Analysis:
    """Streaming analysis STFT: one hop of samples in, one spectrum out."""

    def __init__(self, p: DfParams) -> None:
        self.p = p
        self.window = vorbis_window(p.fft_size)
        self.wnorm = np.float32(2 * p.hop_size / p.fft_size ** 2)
        self.reset()

    def reset(self) -> None:
        self._mem = np.zeros(self.p.fft_size - self.p.hop_size, dtype=np.float32)

    def frames(self, hops: np.ndarray) -> np.ndarray:
        """``hops``: ``(k, hop_size)`` new samples -> ``(k, freq_bins)`` complex64 spectra."""
        hops = np.asarray(hops, dtype=np.float32).reshape(-1, self.p.hop_size)
        k = hops.shape[0]
        if not k:
            return np.zeros((0, self.p.freq_bins), dtype=np.complex64)
        keep = self.p.fft_size - self.p.hop_size
        stream = np.concatenate((self._mem, hops.reshape(-1)))
        idx = np.arange(k)[:, None] * self.p.hop_size + np.arange(self.p.fft_size)[None, :]
        buf = stream[idx] * self.window
        self._mem = stream[stream.size - keep:].copy()
        return (np.fft.rfft(buf, axis=1) * self.wnorm).astype(np.complex64)


class Synthesis:
    """Streaming synthesis: spectra in, one hop of samples per spectrum out (overlap-add)."""

    def __init__(self, p: DfParams) -> None:
        self.p = p
        self.window = vorbis_window(p.fft_size)
        self.reset()

    def reset(self) -> None:
        self._mem = np.zeros(self.p.fft_size - self.p.hop_size, dtype=np.float32)

    def frames(self, spectra: np.ndarray) -> np.ndarray:
        """``(k, freq_bins)`` complex -> ``k * hop_size`` samples."""
        spectra = np.asarray(spectra).reshape(-1, self.p.freq_bins)
        hop, n = self.p.hop_size, self.p.fft_size
        out = np.empty(spectra.shape[0] * hop, dtype=np.float32)
        frames = (np.fft.irfft(spectra, n=n, axis=1) * n).astype(np.float32) * self.window
        for i, x in enumerate(frames):  # n - hop == hop for DeepFilterNet's 50 % overlap
            out[i * hop:(i + 1) * hop] = x[:hop] + self._mem[:hop]
            self._mem = x[hop:].copy()
        return out


class Features:
    """The model's two inputs from spectra, with the running means as explicit state."""

    def __init__(self, p: DfParams) -> None:
        self.p = p
        self.alpha = np.float32(norm_alpha(p))
        widths = erb_widths(p)
        self.widths = widths
        fb, _ = erb_matrices(widths)
        self._fb = fb  # [F, E], columns = band means
        self.reset()

    def reset(self) -> None:
        p = self.p
        self.erb_state = np.linspace(*MEAN_NORM_INIT, p.nb_erb).astype(np.float32)
        self.unit_state = np.linspace(*UNIT_NORM_INIT, p.nb_df).astype(np.float32)

    def frames(self, spectra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(k, F)`` complex spectra -> ``erb (k, nb_erb)`` float32, ``spec (k, nb_df)``
        complex64."""
        spectra = np.asarray(spectra, dtype=np.complex64).reshape(-1, self.p.freq_bins)
        power = (spectra.real ** 2 + spectra.imag ** 2).astype(np.float32)
        erb_db = (10.0 * np.log10(power @ self._fb + np.float32(1e-10))).astype(np.float32)
        low = spectra[:, :self.p.nb_df]
        mag = np.abs(low).astype(np.float32)
        a, b = self.alpha, np.float32(1.0) - self.alpha
        erb_out = np.empty_like(erb_db)
        spec_out = np.empty_like(low)
        for i in range(spectra.shape[0]):  # running means: sequential by nature
            self.erb_state = erb_db[i] * b + self.erb_state * a
            erb_out[i] = (erb_db[i] - self.erb_state) / np.float32(40.0)
            self.unit_state = mag[i] * b + self.unit_state * a
            spec_out[i] = low[i] / np.sqrt(self.unit_state)
        return erb_out, spec_out
