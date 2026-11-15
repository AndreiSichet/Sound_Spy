"""Training windows: where to crop and how to label them (NumPy only).

A window of ``window_s`` is cut from a rendered mixture. A class is **positive** in the window if
it is active for at least ``min_active_s`` inside it, measured on the per-class activity returned
by ``render()`` (``activity_rate_hz`` frames per second). Classes not present, and windows with
only background, are negative.
"""

from __future__ import annotations

import numpy as np


def window_labels(
    class_activity: np.ndarray,
    rate_hz: float,
    start_s: float,
    window_s: float,
    min_active_s: float,
) -> np.ndarray:
    """Multi-hot ``(classes,)`` float32 for the window ``[start_s, start_s + window_s)``."""
    a = max(0, int(round(start_s * rate_hz)))
    b = min(class_activity.shape[0], int(round((start_s + window_s) * rate_hz)))
    if b <= a:
        return np.zeros(class_activity.shape[1], dtype=np.float32)
    active_s = class_activity[a:b].sum(axis=0) / rate_hz
    need = max(min_active_s, 0.5 / rate_hz)  # at least half a frame, so 0 never counts
    return (active_s >= need - 1e-9).astype(np.float32)


def window_starts(duration_s: float, window_s: float, n: int, rng: np.random.Generator
                  ) -> np.ndarray:
    """``n`` random window starts (seconds) inside a mixture of ``duration_s``."""
    span = max(0.0, duration_s - window_s)
    return rng.uniform(0.0, span, size=n) if span > 0 else np.zeros(n)
