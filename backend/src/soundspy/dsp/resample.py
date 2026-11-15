"""Sample-rate conversion of ``AudioBuffer`` objects (all channels), polyphase.

``scipy.signal.resample_poly`` with the reduced ratio ``up/down`` (44.1 kHz -> 16 kHz is
160/441). It filters with a Kaiser-windowed FIR, so no aliasing, and the output length is
``ceil(frames * up / down)``: whole-second inputs keep their duration exactly.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

from soundspy.core.audio import AudioBuffer


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
