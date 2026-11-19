"""``SpeechRouteGate``: the speech route's gate, driven by DeepFilterNet3's own output (step 07d).

**Why.** The 07c :class:`~soundspy.extraction.hybrid_gate.HybridGate` closes only when the
detector is *sure* nothing selected is there, for 400 ms. In short pauses while someone talks
over music the live detector never becomes sure (it still hears the last word for 0.6-1 s), so
the gate stayed open and only DeepFilterNet3's own suppression worked. But on the speech route
the extractor itself is a reliable speech / no-speech indicator: the level of DeepFilterNet3's
output against its input,

    r = L(output) - L(input)    (dB; 1 ms blocks, envelopes ratio_attack_ms / ratio_release_ms)

is about -30 to -45 dB on music alone and about -5 dB on the owner's speech (measured on the
tuning session, 19 Nov). A 20+ dB gap: this gate follows ``r`` and asks the detector only how
long to wait before closing. (The Waveformer's confident mistakes leak at -12 to -20 dB, so its
route keeps the detector-driven gate.)

::

                 r < close_db without a break for the hold
       ┌────────┐ ──────────────────────────────────────► ┌────────┐
       │  OPEN  │              (close_ms ramp)             │ CLOSED │ ◄── start of a stream /
       │ gain 1 │ ◄────────────────────────────────────── │ gain g │     of the speech route
       └────────┘          r > open_db  (open_ms ramp)     └────────┘

       hold = close_hold_confident_ms   the detector is sure nothing selected is there
              detector_override_max_ms  the detector says a selected class is on
              close_hold_ms             otherwise (unsure, not decided yet, no detector)

* **Hysteresis.** Between ``close_db`` and ``open_db`` nothing changes: a signal hovering
  around one threshold cannot make the gate chatter.
* **The detector is secondary.** It can shorten the hold (sure: absent) or lengthen it (on),
  but it can never close the gate while ``r`` is above ``close_db``, never open it, and never
  keep it open for longer than ``detector_override_max_ms`` while ``r`` stays low.
* **Start.** A stream starts closed (``start_closed``), and so does the speech route when it
  starts in a running session (:meth:`restart`); it opens when ``r > open_db``, i.e. when
  speech appears. For ``warmup_ms`` after a start ``r`` is not used at all (the gate keeps its
  start position; the envelopes keep running).
* **Closed** = the session's attenuation on the whole output, ``g = 10^(-attenuation_db/20)``.
* **No added latency**, any chunking gives the same gains (blocks counted from the start).
* An input quieter than ``min_level_dbfs`` cannot open the gate (nothing to judge) and counts
  as "low" for closing.

Times (for the detector): block ``k`` holds output samples ``[origin + k b, origin + (k+1) b)``
of the stream, i.e. input samples ``latency`` earlier (as in ``HybridGate``).
"""

from __future__ import annotations

import math
from collections.abc import Collection

import numpy as np

from soundspy.config import SpeechGateConfig
from soundspy.extraction.activity import ActivitySource

_EPS = 1e-12
PRESENCE_EVERY = 10  # blocks a detector answer is reused for (10 ms; decisions change per hop)


class SpeechRouteGate:
    def __init__(self, config: SpeechGateConfig, sample_rate: int, latency: int,
                 activity: ActivitySource | None = None) -> None:
        self.config = config
        self.sample_rate = int(sample_rate)
        self.latency = int(latency)
        self.activity = activity
        self.block = max(1, int(round(config.block_ms * sample_rate / 1000.0)))
        block_ms = 1000.0 * self.block / sample_rate
        self._attack = 1.0 - math.exp(-block_ms / max(config.ratio_attack_ms, 1e-6))
        self._release = 1.0 - math.exp(-block_ms / max(config.ratio_release_ms, 1e-6))
        self._open_step = 1.0 if config.open_ms <= 0 else min(1.0, block_ms / config.open_ms)
        self._close_step = 1.0 if config.close_ms <= 0 else min(1.0, block_ms / config.close_ms)
        self._hold = config.close_hold_ms / block_ms  # in blocks
        self._hold_sure = config.close_hold_confident_ms / block_ms
        self._hold_on = max(self._hold, config.detector_override_max_ms / block_ms)
        self._warmup = int(round(config.warmup_ms / block_ms))
        self._close = 10.0 ** (config.close_db / 10.0)  # power ratios: no logarithm per block
        self._open = 10.0 ** (config.open_db / 10.0)
        self._min_level = 10.0 ** (config.min_level_dbfs / 10.0)
        self._origin = 0
        self.reset()

    # --- state -----------------------------------------------------------------------------------

    def reset(self) -> None:
        """A new stream: closed (``start_closed``), the block grid from sample 0."""
        self._start_at(0, warmup=False)

    def restart(self, at_sample: int = 0, warmup: bool = True) -> None:
        """The speech route (re)starts at stream sample ``at_sample``: closed again, ``r`` not
        used for ``warmup_ms`` (with ``warmup``)."""
        self._start_at(at_sample, warmup)

    def _start_at(self, origin: int, warmup: bool) -> None:
        self._origin = int(origin)
        self._k = 0
        self._fill = 0
        self._sum_est = self._sum_dry = 0.0
        self._env_est = self._env_dry = 0.0
        closed = self.config.enabled and self.config.start_closed
        self.is_open = not closed
        self.position = 0.0 if closed else 1.0
        self._start = self._end = self.position
        self._low = 0  # blocks in a row with r below close_db
        self._warm = self._warmup if warmup else 0
        self._presence: tuple[int, tuple[bool, bool]] | None = None
        self.blocks = self.closed_blocks = self.reopened = 0
        self.last_ratio_db: float | None = None

    @property
    def closed(self) -> bool:
        return self.position < 0.5

    @property
    def closed_share(self) -> float:
        return self.closed_blocks / self.blocks if self.blocks else 0.0

    # --- decisions -------------------------------------------------------------------------------

    def _detector(self, labels: Collection[str]) -> tuple[bool, bool]:
        """``(on, sure absent)`` for the current block, asked at most every PRESENCE_EVERY."""
        group = self._k // PRESENCE_EVERY
        if self._presence is not None and self._presence[0] == group:
            return self._presence[1]
        on = absent = False
        if self.activity is not None:
            b, sr = self.block, self.sample_rate
            centre = (self._origin + (self._k + 0.5) * b - self.latency) / sr
            now = (self._origin + (self._k + 1) * b) / sr
            if self.activity.known(centre, now):
                on = self.activity.active(labels, centre, now)
                absent = not on and self.activity.absent(labels, centre, now,
                                                         self.config.close_prob_factor)
        self._presence = (group, (on, absent))
        return on, absent

    def _hold_for(self, labels: Collection[str]) -> float:
        on, absent = self._detector(labels)
        if absent:
            return self._hold_sure
        return self._hold_on if on else self._hold

    def _decide(self, gated: bool, labels: Collection[str], sum_est: float,
                sum_dry: float) -> None:
        """End of block ``self._k``: envelopes, ``r``, the state and the position."""
        b = self.block
        ms = sum_est / b
        self._env_est += (self._attack if ms > self._env_est else self._release) * (
            ms - self._env_est)
        ms = sum_dry / b
        self._env_dry += (self._attack if ms > self._env_dry else self._release) * (
            ms - self._env_dry)
        ratio = (self._env_est + _EPS) / (self._env_dry + _EPS)
        if not gated:  # nothing selected / switched off: open, at once
            self.is_open, self.position = True, 1.0
        elif self._warm > 0:
            self._warm -= 1  # r is not used yet: the start position stays
        else:
            quiet = self._env_dry <= self._min_level
            if quiet or ratio < self._close:
                self._low += 1
            else:
                self._low = 0
            if self.is_open:
                # the detector is asked only once r has been low for the shortest hold
                if self._low >= self._hold_sure and self._low >= self._hold_for(labels):
                    self.is_open = False
            elif not quiet and ratio > self._open:
                self.is_open = True
                self.reopened += 1
        if self.is_open:
            self.position = min(1.0, self.position + self._open_step)
        else:
            self.position = max(0.0, self.position - self._close_step)
        self.blocks += 1
        self.closed_blocks += self.position < 0.5
        self._k += 1
        self._ratio = ratio

    # --- gains -----------------------------------------------------------------------------------

    def positions(self, est: np.ndarray, dry: np.ndarray, labels: Collection[str]
                  ) -> np.ndarray:
        """Gate positions (1 = open, 0 = closed) for these output samples; ``est`` and ``dry``
        are DeepFilterNet3's output (input gain undone) and its input, aligned. Any length; the
        block grid runs on across calls. Within a block the position moves linearly from its
        value at the block's start to the one decided at the end of the previous block."""
        est = np.asarray(est, dtype=np.float64).reshape(-1)
        dry = np.asarray(dry, dtype=np.float64).reshape(-1)
        n = est.size
        if not n:
            return np.zeros(0, dtype=np.float64)
        gated = bool(labels) and self.config.enabled
        if not gated:
            self.is_open, self.position = True, 1.0
            self._start = self._end = 1.0
        b = self.block
        ce = np.concatenate(([0.0], np.cumsum(est * est)))
        cd = np.concatenate(([0.0], np.cumsum(dry * dry)))
        s0 = -self._fill  # start of the current block, relative to this call
        xs, ys = [float(s0)], [self._start]
        end = s0 + b
        decided = False
        while end <= n:
            lo = max(0, end - b)
            sum_e = ce[end] - ce[lo] + (self._sum_est if lo == 0 else 0.0)
            sum_d = cd[end] - cd[lo] + (self._sum_dry if lo == 0 else 0.0)
            xs.append(float(end))
            ys.append(self._end)
            self._decide(gated, labels, sum_e, sum_d)
            decided = True
            self._start, self._end = self._end, self.position
            end += b
        lo = max(0, end - b)
        self._sum_est = ce[n] - ce[lo] + (self._sum_est if lo == 0 and end - b < 0 else 0.0)
        self._sum_dry = cd[n] - cd[lo] + (self._sum_dry if lo == 0 and end - b < 0 else 0.0)
        self._fill = n - (end - b)
        xs.append(float(end))
        ys.append(self._end)
        if decided:
            self.last_ratio_db = 10.0 * math.log10(self._ratio)
        return np.interp(np.arange(1, n + 1, dtype=np.float64), xs, ys)

    def gains(self, est: np.ndarray, dry: np.ndarray, labels: Collection[str],
              attenuation_db: float) -> np.ndarray:
        """Linear gains (1 open .. ``g`` closed, ramped in dB) for these output samples."""
        return position_gains(self.positions(est, dry, labels), attenuation_db)


def position_gains(positions: np.ndarray, attenuation_db: float) -> np.ndarray:
    """Gate positions -> linear gains: closed = the slider's attenuation, ramped in dB."""
    depth = max(0.0, float(attenuation_db))
    return (10.0 ** (-(1.0 - np.asarray(positions)) * depth / 20.0)).astype(np.float32)
