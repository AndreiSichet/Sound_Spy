"""Where the extractor learns whether the target is sounding: ``ActivitySource``.

The baseline extractor asks one question per STFT frame: *is any of the selected classes active
now?* Three sources answer it:

* ``LiveDetectorActivity``: the shared ``DetectionState`` the live ``Detector`` writes. Causal and
  lagged: a class switches on ~0.3 s after it starts and off 0.6-1 s after it ends.
* ``TimelineActivity``: decisions computed beforehand for a whole file, as a step function of
  time. Two builders: :func:`centred_timeline` (offline mode: each decision comes from a window
  **centred** on its time, so there is no lag) and :func:`causal_timeline` (evaluation: reproduces
  the live detector's lag without threads).
* ``OracleActivity``: the true per-class activity of a rendered recipe (evaluation only).

Every question carries two times, in seconds on the processed stream's clock: ``centre_s``, the
centre of the frame being decided, and ``now_s``, the newest sample that has arrived. A source
that knows the whole file answers for ``centre_s``; a causal source can only know what was
decided up to ``now_s``.

``lag_s`` tells the extractor how late "active" may arrive after the target really started. It
only treats a frame as noise once the target has stayed inactive for that long *after* it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING

import numpy as np

from soundspy.core.audio import AudioBuffer
from soundspy.engine.detection_state import DetectionState

if TYPE_CHECKING:
    from soundspy.detection.inference import DetectionCore


class ActivitySource(ABC):
    lag_s: float = 0.0

    @abstractmethod
    def active(self, labels: Collection[str], centre_s: float, now_s: float) -> bool:
        """True if any of ``labels`` is active for the frame centred at ``centre_s``."""

    def reset(self) -> None:  # noqa: B027 - optional hook
        """Forget anything tied to a previous stream."""


class LiveDetectorActivity(ActivitySource):
    """The live detector's current decision (whatever the frame time: it is all we know)."""

    def __init__(self, state: DetectionState, lag_s: float = 0.5) -> None:
        self.state = state
        self.lag_s = float(lag_s)

    def active(self, labels: Collection[str], centre_s: float, now_s: float) -> bool:
        return not self.state.snapshot().active.isdisjoint(labels)


class TimelineActivity(ActivitySource):
    """A step function: decision ``i`` holds from ``times[i]`` until ``times[i + 1]``.

    ``clock`` says which time a question is answered for: ``"centre"`` (the frame's own time;
    offline, no lag) or ``"now"`` (the newest sample; causal simulation of the live detector).
    Before ``times[0]`` nothing is active.
    """

    def __init__(self, labels: Sequence[str], times: Sequence[float], active: np.ndarray, *,
                 clock: str = "centre", lag_s: float = 0.0) -> None:
        self.labels = list(labels)
        self.times = np.asarray(times, dtype=np.float64)
        self.matrix = np.asarray(active, dtype=bool).reshape(len(self.times), len(self.labels))
        if clock not in ("centre", "now"):
            raise ValueError(f"clock must be 'centre' or 'now', got {clock!r}")
        if np.any(np.diff(self.times) < 0):
            raise ValueError("times must be non-decreasing")
        self.clock, self.lag_s = clock, float(lag_s)
        self._index = {lab: j for j, lab in enumerate(self.labels)}

    def active(self, labels: Collection[str], centre_s: float, now_s: float) -> bool:
        t = centre_s if self.clock == "centre" else now_s
        i = int(np.searchsorted(self.times, t, side="right")) - 1
        if i < 0:
            return False
        row = self.matrix[i]
        return any(row[self._index[lab]] for lab in labels if lab in self._index)

    def segments(self, label: str) -> list[tuple[float, float]]:
        """``(start, end)`` of every active stretch of ``label`` (end = next decision time)."""
        column = self.matrix.T[self._index[label]]
        out, start = [], None
        for i, on in enumerate(column):
            if on and start is None:
                start = float(self.times[i])
            if start is not None and not on:
                out.append((start, float(self.times[i])))
                start = None
        if start is not None:
            step = float(self.times[-1] - self.times[-2]) if len(self.times) > 1 else 0.0
            out.append((start, float(self.times[-1]) + step))
        return out


class OracleActivity(ActivitySource):
    """Ground truth: ``class_activity`` as returned by ``render()`` (frames x classes)."""

    def __init__(self, labels: Sequence[str], class_activity: np.ndarray, rate_hz: float) -> None:
        self.labels = list(labels)
        self.matrix = np.asarray(class_activity) > 0.5
        if self.matrix.ndim != 2 or self.matrix.shape[1] != len(self.labels):
            raise ValueError("class_activity must be (frames, classes)")
        self.rate_hz = float(rate_hz)
        self._index = {lab: j for j, lab in enumerate(self.labels)}

    def active(self, labels: Collection[str], centre_s: float, now_s: float) -> bool:
        i = int(np.floor(centre_s * self.rate_hz + 1e-9))
        if i < 0 or i >= self.matrix.shape[0]:
            return False
        row = self.matrix[i]
        return any(row[self._index[lab]] for lab in labels if lab in self._index)


# --------------------------------------------------------------------------------------------
# Timelines from the detection model


def _hysteresis(core: DetectionCore, probs: np.ndarray, hysteresis: float) -> np.ndarray:
    on = np.array([core.thresholds[lab] for lab in core.labels], dtype=np.float64)
    off = np.maximum(on - hysteresis, 0.0)
    active = np.zeros(probs.shape, dtype=bool)
    state = np.zeros(len(core.labels), dtype=bool)
    for i, p in enumerate(probs):
        state = np.where(state, p >= off, p >= on)
        active[i] = state
    return active


def _ema(probs: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(probs, dtype=np.float64)
    p = None
    for i, row in enumerate(probs):
        p = row if p is None else alpha * row + (1 - alpha) * p
        out[i] = p
    return out


def _centres(audio: AudioBuffer, hop_s: float) -> np.ndarray:
    return np.arange(0.0, audio.num_frames / audio.sample_rate + hop_s / 2, hop_s)


def _ends(audio: AudioBuffer, hop_s: float) -> np.ndarray:
    return np.arange(hop_s, audio.num_frames / audio.sample_rate + 1e-9, hop_s)


def timeline_probabilities(core: DetectionCore, audio: AudioBuffer, hop_s: float, *,
                           batch: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Raw probabilities for both timelines of a file with one pass over the model.

    Returns ``(centred, causal)``, to pass as ``raw=`` to :func:`centred_timeline` and
    :func:`causal_timeline`. A window centred on ``t`` is the window ending at ``t + window/2``,
    so when half a window is a whole number of hops the two sets overlap almost completely and
    only their union is evaluated (the evaluation runs both for every recipe).
    """
    from soundspy.detection.inference import probabilities_at

    centred_ends = _centres(audio, hop_s) + core.window_s / 2
    causal_ends = _ends(audio, hop_s)
    keys = lambda ends: [int(round(e / hop_s * 1000)) for e in ends]  # noqa: E731
    union = sorted(set(keys(centred_ends)) | set(keys(causal_ends)))
    raw = probabilities_at(core, audio, [k * hop_s / 1000 for k in union], batch=batch)
    row = {k: i for i, k in enumerate(union)}
    pick = lambda ends: raw[[row[k] for k in keys(ends)]]  # noqa: E731
    return pick(centred_ends), pick(causal_ends)


def centred_timeline(core: DetectionCore, audio: AudioBuffer, *, hop_s: float, ema_alpha: float,
                     hysteresis: float, batch: int = 32, raw: np.ndarray | None = None
                     ) -> TimelineActivity:
    """Offline activity without lag: the decision for time ``t`` uses the window centred on ``t``.

    Windows are centred every ``hop_s`` from 0 to the end of the file (zero-padded outside it).
    The probabilities are smoothed with the live EMA run forwards **and** backwards (averaged, so
    the smoothing does not shift them in time), then the live thresholds and hysteresis apply.
    Decision ``i`` holds for ``[i*hop - hop/2, i*hop + hop/2)``.
    """
    from soundspy.detection.inference import probabilities_at

    centres = _centres(audio, hop_s)
    if raw is None:
        raw = probabilities_at(core, audio, centres + core.window_s / 2, batch=batch)
    smooth = 0.5 * (_ema(raw, ema_alpha) + _ema(raw[::-1], ema_alpha)[::-1])
    return TimelineActivity(core.labels, centres - hop_s / 2,
                            _hysteresis(core, smooth, hysteresis), clock="centre", lag_s=0.0)


def causal_timeline(core: DetectionCore, audio: AudioBuffer, *, hop_s: float, ema_alpha: float,
                    hysteresis: float, delay_s: float, lag_s: float, batch: int = 32,
                    raw: np.ndarray | None = None) -> TimelineActivity:
    """The live detector's decisions for a file, reproduced without threads.

    Same windows (the last ``window_s`` before every hop; silence before the file starts), same
    EMA and hysteresis as ``Detector``. The decision for the window ending at ``e`` is in use from
    ``e + delay_s`` (inference time plus the wait for the next chunk).
    """
    from soundspy.detection.inference import Smoother, probabilities_at

    ends = _ends(audio, hop_s)
    if raw is None:
        raw = probabilities_at(core, audio, ends, batch=batch)
    smoother = Smoother(core.labels, core.thresholds, ema_alpha, hysteresis)
    active = np.array([smoother.update(row)[1] for row in raw], dtype=bool).reshape(
        len(ends), len(core.labels))
    return TimelineActivity(core.labels, ends + delay_s, active, clock="now", lag_s=lag_s)
