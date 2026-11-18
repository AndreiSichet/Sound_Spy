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

**Confident absence** (step 07c, :meth:`ActivitySource.absent`): "not active" only says the
probability is below the threshold. The hybrid extractor's gate closes only when the detector is
*sure*: every selected class's smoothed probability below ``factor`` x its own threshold. A
source that knows probabilities answers that (live state and both timelines); the oracle knows
the truth, so for it "absent" is simply "not active".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping, Sequence
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

    def absent(self, labels: Collection[str], centre_s: float, now_s: float,
               factor: float) -> bool:
        """True if the source is *sure* that none of ``labels`` is there (see the module
        docstring). Without probabilities: not active."""
        return not self.active(labels, centre_s, now_s)

    def known(self, centre_s: float, now_s: float) -> bool:
        """False before the source has decided anything (the live detector's first ~0.3 s)."""
        return True

    def reset(self) -> None:  # noqa: B027 - optional hook
        """Forget anything tied to a previous stream."""


def _below(probabilities: Mapping[str, float], labels: Collection[str],
           thresholds: Mapping[str, float], factor: float) -> bool:
    """Every label known and below ``factor`` x its threshold."""
    return all(lab in probabilities and lab in thresholds
               and probabilities[lab] < factor * thresholds[lab] for lab in labels)


class LiveDetectorActivity(ActivitySource):
    """The live detector's current decision (whatever the frame time: it is all we know).
    ``thresholds``: the detector's per-class thresholds (for :meth:`absent`)."""

    def __init__(self, state: DetectionState, lag_s: float = 0.5,
                 thresholds: Mapping[str, float] | None = None) -> None:
        self.state = state
        self.lag_s = float(lag_s)
        self.thresholds = dict(thresholds) if thresholds is not None else None

    def active(self, labels: Collection[str], centre_s: float, now_s: float) -> bool:
        return not self.state.snapshot().active.isdisjoint(labels)

    def known(self, centre_s: float, now_s: float) -> bool:
        return self.state.snapshot().available

    def absent(self, labels: Collection[str], centre_s: float, now_s: float,
               factor: float) -> bool:
        snap = self.state.snapshot()
        if not snap.available or not snap.active.isdisjoint(labels):
            return False  # no detection yet, or a selected class is on
        if self.thresholds is None:
            return True
        return _below(snap.probabilities, labels, self.thresholds, factor)


class TimelineActivity(ActivitySource):
    """A step function: decision ``i`` holds from ``times[i]`` until ``times[i + 1]``.

    ``clock`` says which time a question is answered for: ``"centre"`` (the frame's own time;
    offline, no lag) or ``"now"`` (the newest sample; causal simulation of the live detector).
    Before ``times[0]`` nothing is active.
    """

    def __init__(self, labels: Sequence[str], times: Sequence[float], active: np.ndarray, *,
                 clock: str = "centre", lag_s: float = 0.0,
                 probabilities: np.ndarray | None = None,
                 thresholds: Mapping[str, float] | None = None) -> None:
        self.labels = list(labels)
        self.times = np.asarray(times, dtype=np.float64)
        shape = (len(self.times), len(self.labels))
        self.matrix = np.asarray(active, dtype=bool).reshape(shape)
        if clock not in ("centre", "now"):
            raise ValueError(f"clock must be 'centre' or 'now', got {clock!r}")
        if np.any(np.diff(self.times) < 0):
            raise ValueError("times must be non-decreasing")
        self.clock, self.lag_s = clock, float(lag_s)
        self.probabilities = (None if probabilities is None
                              else np.asarray(probabilities, dtype=np.float64).reshape(shape))
        self.thresholds = dict(thresholds) if thresholds is not None else None
        self._index = {lab: j for j, lab in enumerate(self.labels)}

    def _row(self, centre_s: float, now_s: float) -> int:
        t = centre_s if self.clock == "centre" else now_s
        return int(np.searchsorted(self.times, t, side="right")) - 1

    def active(self, labels: Collection[str], centre_s: float, now_s: float) -> bool:
        i = self._row(centre_s, now_s)
        if i < 0:
            return False
        row = self.matrix[i]
        return any(row[self._index[lab]] for lab in labels if lab in self._index)

    def known(self, centre_s: float, now_s: float) -> bool:
        return self._row(centre_s, now_s) >= 0

    def absent(self, labels: Collection[str], centre_s: float, now_s: float,
               factor: float) -> bool:
        i = self._row(centre_s, now_s)
        if i < 0:
            return False  # before the first decision nothing is known
        if self.active(labels, centre_s, now_s):
            return False
        if self.probabilities is None or self.thresholds is None:
            return True
        probs = {lab: float(self.probabilities[i, self._index[lab]]) for lab in labels
                 if lab in self._index}
        return _below(probs, labels, self.thresholds, factor)

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
                            _hysteresis(core, smooth, hysteresis), clock="centre", lag_s=0.0,
                            probabilities=smooth, thresholds=core.thresholds)


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
    rows = [smoother.update(row) for row in raw]
    shape = (len(ends), len(core.labels))
    active = np.array([r[1] for r in rows], dtype=bool).reshape(shape)
    smooth = np.array([r[0] for r in rows], dtype=np.float64).reshape(shape)
    return TimelineActivity(core.labels, ends + delay_s, active, clock="now", lag_s=lag_s,
                            probabilities=smooth, thresholds=core.thresholds)
