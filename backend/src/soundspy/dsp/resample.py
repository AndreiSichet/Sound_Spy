"""Sample-rate conversion (all channels), polyphase: whole buffers and streams.

**Whole buffers** (``resample``, ``resample_array``): ``scipy.signal.resample_poly`` with the
reduced ratio ``up/down`` (44.1 kHz -> 16 kHz is 160/441). It filters with a Kaiser-windowed FIR,
so no aliasing, and the output length is ``ceil(frames * up / down)``: whole-second inputs keep
their duration exactly. Zero delay (it looks at the future), so it needs the whole signal.

**Streams** (``StreamingResampler``, step 07): the same filter, but causal and with memory, for
audio that arrives in chunks of any size. Conceptually the input is upsampled by ``up`` (zeros
in between), low-pass filtered, and every ``down``-th sample kept; a polyphase structure
computes only the samples that are kept: output sample ``m`` is

    y[m] = sum_j  h[(m * down) mod up + j * up] * x[(m * down) // up - j]

so it needs input samples only up to ``(m * down) // up``. After ``n`` input samples exactly
``ceil(n * up / down)`` output samples exist, whatever the chunk sizes, and they are the same
numbers as for one big chunk (the filter state is the last few input samples).

The filter is symmetric (linear phase): the output is the ideal resampled signal delayed by
``delay = (taps - 1) / (2 * down)`` output samples. With the default length that is 10 samples
when downsampling. ``taps`` can be chosen by the caller to make the delay of a *chain* of two
resamplers a whole number of samples (``extraction/neural_tse.py`` does that).
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import firwin, resample_poly

from soundspy.core.audio import AudioBuffer

ZERO_CROSSINGS = 10  # filter half-length in periods of the lower rate (as scipy's resample_poly)
KAISER_BETA = 5.0
BLOCK = 1 << 16  # input samples handled at once (bounds memory for long one-shot calls)


def ratio(source_rate: int, target_rate: int) -> tuple[int, int]:
    """``(up, down)`` in lowest terms."""
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    g = gcd(int(source_rate), int(target_rate))
    return int(target_rate) // g, int(source_rate) // g


def resample_array(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample a 2-D ``(frames, channels)`` float array along the frame axis."""
    if samples.ndim != 2:
        raise ValueError("expected a 2-D (frames, channels) array")
    if source_rate == target_rate:
        return samples.astype(np.float32, copy=False)
    up, down = ratio(source_rate, target_rate)
    if samples.shape[0] == 0:
        return np.zeros((0, samples.shape[1]), dtype=np.float32)
    out = resample_poly(samples.astype(np.float64), up, down, axis=0)
    return out.astype(np.float32)


def resample(buffer: AudioBuffer, target_rate: int) -> AudioBuffer:
    """``buffer`` at ``target_rate`` (same channels; returned as-is if the rate matches)."""
    if buffer.sample_rate == target_rate:
        return buffer
    return AudioBuffer(
        resample_array(buffer.samples, buffer.sample_rate, target_rate),
        int(target_rate),
        {**buffer.metadata, "resampled_from": buffer.sample_rate},
    )


def default_taps(up: int, down: int) -> int:
    """Filter length used when none is given: ``2 * 10 * max(up, down) + 1`` (odd)."""
    return 2 * ZERO_CROSSINGS * max(up, down) + 1


class StreamingResampler:
    """Causal polyphase resampler with state: any chunk sizes, exact continuity (see module)."""

    def __init__(self, source_rate: int, target_rate: int, channels: int = 1, *,
                 taps: int | None = None) -> None:
        if channels < 1:
            raise ValueError("channels must be >= 1")
        self.source_rate, self.target_rate = int(source_rate), int(target_rate)
        self.channels = channels
        self.up, self.down = ratio(source_rate, target_rate)
        if self.up == self.down:  # same rate: one tap, the samples pass unchanged
            self.taps = 1
            h = np.ones(1)
        else:
            self.taps = default_taps(self.up, self.down) if taps is None else int(taps)
            if self.taps < 3 or self.taps % 2 == 0:
                raise ValueError(f"taps must be an odd number >= 3, got {self.taps}")
            cutoff = 1.0 / max(self.up, self.down)
            h = firwin(self.taps, cutoff, window=("kaiser", KAISER_BETA)) * self.up
        self._per_phase = -(-self.taps // self.up)  # taps of one polyphase branch
        padded = np.zeros(self._per_phase * self.up, dtype=np.float64)
        padded[: self.taps] = h
        # phases[p, j] = h[p + j * up]
        self._phases = padded.reshape(self._per_phase, self.up).T.astype(np.float32).copy()
        self._steps = np.arange(self._per_phase)
        self.reset()

    @property
    def delay(self) -> float:
        """Delay of the output against the ideal resampled signal, in **output** samples."""
        return (self.taps - 1) / (2.0 * self.down)

    def reset(self) -> None:
        self._history = np.zeros((self._per_phase - 1, self.channels), dtype=np.float32)
        self._consumed = 0  # input samples seen
        self._produced = 0  # output samples returned

    def output_count(self, total_input: int) -> int:
        """Output samples that exist after ``total_input`` input samples."""
        return -(-total_input * self.up // self.down)

    def push(self, samples: np.ndarray) -> np.ndarray:
        """``(n, channels)`` in, every output sample that is now computable out."""
        x = np.asarray(samples, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.channels:
            raise ValueError(f"expected (frames, {self.channels}) samples, got shape {x.shape}")
        if x.shape[0] <= BLOCK:
            return self._push(x)
        parts = [self._push(x[i:i + BLOCK]) for i in range(0, x.shape[0], BLOCK)]
        return np.concatenate(parts)

    def _push(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        if n == 0:
            return np.zeros((0, self.channels), dtype=np.float32)
        end = self.output_count(self._consumed + n)
        m = np.arange(self._produced, end, dtype=np.int64)
        position = m * self.down
        base = position // self.up  # newest input sample each output needs
        phase = position % self.up
        # buffer[i] is input sample (consumed - (per_phase - 1) + i)
        buffer = np.concatenate((self._history, x))
        first = self._consumed - (self._per_phase - 1)
        index = (base - first).reshape(-1, 1) - self._steps.reshape(1, -1)
        out = np.einsum("mj,mjc->mc", self._phases[phase], buffer[index], optimize=True)
        keep = self._per_phase - 1
        self._history = buffer[buffer.shape[0] - keep:].copy() if keep else self._history
        self._consumed += n
        self._produced = end
        return out.astype(np.float32, copy=False)


def taps_for_whole_delay(first: StreamingResampler, source_rate: int, target_rate: int) -> int:
    """Filter length for a resampler ``target_rate -> source_rate`` that follows ``first``
    (``source_rate -> target_rate``), such that the two delays together are a **whole number of
    samples at** ``source_rate``.

    ``first`` delays by ``(N1 - 1) / (2 * up)`` source samples and the second by
    ``(N2 - 1) / (2 * up)`` (its ``down`` is the first one's ``up``), so the sum is whole when
    ``N1 + N2 - 2`` is a multiple of ``2 * up``. The smallest such ``N2`` not below the default
    length is returned.
    """
    up, down = ratio(source_rate, target_rate)
    if (first.up, first.down) != (up, down):
        raise ValueError("`first` must convert source_rate -> target_rate")
    modulus = 2 * up
    taps = default_taps(down, up)
    taps += (-(first.taps + taps - 2)) % modulus
    return taps
