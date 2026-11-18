"""``HybridGate``: the detector closes it, the extractor reopens it (step 07c).

Two earlier gates each solved half of the owner's listening test (classical music, someone
talking with pauses, Speech selected at 40 dB):

* the **baseline's** gate follows the detector, which is reliably *sure* when nobody talks:
  pauses were silent (music-only parts -40 dB), but it opens ~0.3 s late (lost word starts);
* the **neural output gate** (step 07b) follows the extractor's own output, which reacts at
  once, but it cannot see a confident mistake and it ducked quiet syllables.

This gate takes the strength of each::

                 detector sure nothing selected is there          output rises against the
                 for close_hold_ms, and no output rise            input (> reopen_ratio_db),
       ┌────────┐   for close_hold_ms either   ┌────────┐          or the detector switches
       │  OPEN  │ ───────────────────────────► │ CLOSED │          a selected class on
       │ gain 1 │ ◄─────────────────────────── │ gain g │ ◄── (opens within open_ms)
       └────────┘         open_ms ramp         └────────┘
                      (closing: close_ms ramp)

* **Closing** needs the detector to be *confident*: every selected class's smoothed
  probability below ``close_prob_factor`` x its own threshold (half of it by default), for
  ``close_hold_ms`` in a row, and no output rise during that time either.
* **Reopening** needs no detector: the level of the extractor's estimate, measured against the
  input's level (envelopes with ``attack_ms`` / ``release_ms``), rising above
  ``reopen_ratio_db`` opens the gate at once (``open_ms``). That is what keeps word starts: the
  extractor hears the first syllable 0.3 s before the detector says "speech".
* **Closed** = the session's attenuation: ``g = 10^(-attenuation_db / 20)`` on the **whole**
  output (estimate and residual), so a closed gate at 40 dB gives the baseline's -40 dB.
* **No added latency.** The decision is made every ``block_ms`` (1 ms) from everything up to the
  end of the previous block and applies to the next block, ramped sample by sample in dB (no
  clicks). The blocks are counted from the start of the stream, so any chunking gives the same
  gains (streaming = whole file, tested).
* **Open unless sure.** The gate is closed only *while* the closing condition holds: a detector
  that becomes unsure opens it as well (the spec's gate would stay closed until (a) or (b)).
* **Start-up.** ``restart(warmup=True)`` (the hybrid calls it when the speech enhancer starts):
  for ``warmup_ms`` the output cannot reopen the gate, because DeepFilterNet3 leaks in its first
  second or two (measured: −9 / −18 dB, then below −75 dB). ``start_closed``: a stream that
  starts with a selection and a detector starts closed. Before the detector's first decision
  (``ActivitySource.known``) the gate keeps its position.
* Nothing selected, or no detector: the gate stays open.

Times: block ``k`` holds output samples ``[k b, (k+1) b)`` of the extractor, i.e. input samples
``latency`` earlier. The activity source is asked for ``centre_s`` = the input time of the
block's centre and ``now_s`` = the newest input sample (as the baseline asks it).
"""

from __future__ import annotations

import math
from collections.abc import Collection

import numpy as np

from soundspy.config import HybridGateConfig
from soundspy.extraction.activity import ActivitySource

_EPS = 1e-12
PRESENCE_EVERY = 10  # blocks between two questions to the activity source (10 ms)


class HybridGate:
    def __init__(self, config: HybridGateConfig, sample_rate: int, latency: int,
                 activity: ActivitySource | None = None) -> None:
        self.config = config
        self.sample_rate = int(sample_rate)
        self.latency = int(latency)
        self.activity = activity
        self.block = max(1, int(round(config.block_ms * sample_rate / 1000.0)))
        block_ms = 1000.0 * self.block / sample_rate
        self._attack = 1.0 - math.exp(-block_ms / max(config.attack_ms, 1e-6))
        self._release = 1.0 - math.exp(-block_ms / max(config.release_ms, 1e-6))
        self._open_step = 1.0 if config.open_ms <= 0 else min(1.0, block_ms / config.open_ms)
        self._close_step = 1.0 if config.close_ms <= 0 else min(1.0, block_ms / config.close_ms)
        self._hold = config.close_hold_ms / block_ms  # in blocks
        self._warmup = config.warmup_ms / block_ms
        self._ratio = config.reopen_ratio_db
        self._min_level = 10.0 ** (config.min_level_dbfs / 10.0)  # mean square
        self.reset()

    def reset(self) -> None:
        """A new stream. It starts **closed** when ``start_closed`` and a detector is there (a
        stream that starts with a selection: offline runs, a session started with chips
        selected): before anything is known, the extractor's first second is not trusted."""
        self._k = 0  # blocks completed since the start of the stream
        self._fill = 0  # samples of the current block seen so far
        self._sum_est = self._sum_dry = 0.0
        self._env_est = self._env_dry = 0.0
        closed = self.config.start_closed and self.config.enabled and self.activity is not None
        self._absent = self._hold if closed else 0.0  # blocks of confident absence in a row
        self._since_rise = math.inf  # blocks since the output last rose
        self._warm = 0  # blocks left in which an output rise does not count (restart())
        self.position = 0.0 if closed else 1.0  # 1 = open, 0 = closed
        self._start = self._end = self.position  # position at the start / end of the block
        self.blocks = self.closed_blocks = self.reopened = 0
        self.last_ratio_db: float | None = None
        self._presence_cache: tuple[bool, bool] | None = None

    def restart(self, warmup: bool = True) -> None:
        """A route has just started: with ``warmup`` (the speech enhancer's routes), for
        ``warmup_ms`` its output does not reopen the gate: DeepFilterNet3 leaks in its first
        second or two, until its running means and recurrent state have heard enough. (The
        Waveformer has no such start: measured on the same recording, its leaks are spread
        over the whole file.)"""
        self._warm = self._warmup if warmup else 0

    @property
    def closed(self) -> bool:
        return self.position < 0.5

    @property
    def closed_share(self) -> float:
        return self.closed_blocks / self.blocks if self.blocks else 0.0

    def _presence(self, labels: Collection[str]) -> tuple[bool, bool, bool]:
        """``(active, confidently absent, known)`` for block ``self._k``, asked every
        ``PRESENCE_EVERY`` blocks (decisions change every detector hop, 250 ms). ``known`` is
        False before the source's first decision (the live detector's first ~0.3 s)."""
        if self._k % PRESENCE_EVERY == 0 or self._presence_cache is None:
            active = absent = False
            known = True
            if labels and self.activity is not None:
                b, sr = self.block, self.sample_rate
                centre = ((self._k + 0.5) * b - self.latency) / sr
                now = (self._k + 1) * b / sr
                known = self.activity.known(centre, now)
                active = known and self.activity.active(labels, centre, now)
                absent = known and not active and self.activity.absent(
                    labels, centre, now, self.config.close_prob_factor)
            self._presence_cache = (active, absent, known)
        return self._presence_cache

    def _decide(self, labels: Collection[str], sum_est: float, sum_dry: float) -> None:
        """End of block ``self._k``: update the envelopes and the position for the next one."""
        b = self.block
        ms = sum_est / b
        self._env_est += (self._attack if ms > self._env_est else self._release) * (
            ms - self._env_est)
        ms = sum_dry / b
        self._env_dry += (self._attack if ms > self._env_dry else self._release) * (
            ms - self._env_dry)
        self.last_ratio_db = 10.0 * math.log10((self._env_est + _EPS) / (self._env_dry + _EPS))
        rise = (self._env_dry > self._min_level and self.last_ratio_db > self._ratio
                and self._warm <= 0)
        self._warm -= 1
        active, absent, known = self._presence(labels) if labels else (False, False, True)
        if known:  # nothing decided yet: the count of sure-absent blocks stays as it is
            self._absent = self._absent + 1 if absent else 0
        self._since_rise = 0 if rise else self._since_rise + 1
        close = (bool(labels) and self.config.enabled and self._absent >= self._hold
                 and self._since_rise >= self._hold)
        before = self.position
        if close:
            self.position = max(0.0, self.position - self._close_step)
        elif known or rise:  # nothing decided yet and no rise: stay where it is
            self.position = min(1.0, self.position + self._open_step)
        if before < 0.5 <= self.position:
            self.reopened += 1
        self.blocks += 1
        self.closed_blocks += self.position < 0.5
        self._k += 1

    def gains(self, est: np.ndarray, dry: np.ndarray, labels: Collection[str],
              attenuation_db: float) -> np.ndarray:
        """Gains (linear) for these output samples; ``est`` and ``dry`` are the estimate and
        the input aligned with it. Any length; the block grid runs on across calls.

        Within a block the position moves linearly from its value at the block's start to the
        one decided at the end of the previous block (in dB of gain: no steps, no clicks)."""
        est = np.asarray(est, dtype=np.float64).reshape(-1)
        dry = np.asarray(dry, dtype=np.float64).reshape(-1)
        n = est.size
        if not n:
            return np.zeros(0, dtype=np.float32)
        b = self.block
        ce = np.concatenate(([0.0], np.cumsum(est * est)))
        cd = np.concatenate(([0.0], np.cumsum(dry * dry)))
        s0 = -self._fill  # start of the current block, relative to this call
        xs, ys = [float(s0)], [self._start]
        end = s0 + b
        while end <= n:
            lo = max(0, end - b)
            sum_e = ce[end] - ce[lo] + (self._sum_est if lo == 0 else 0.0)
            sum_d = cd[end] - cd[lo] + (self._sum_dry if lo == 0 else 0.0)
            xs.append(float(end))
            ys.append(self._end)
            self._decide(labels, sum_e, sum_d)
            self._start, self._end = self._end, self.position
            end += b
        # the block still open at the end of this call
        lo = max(0, end - b)
        self._sum_est = ce[n] - ce[lo] + (self._sum_est if lo == 0 and end - b < 0 else 0.0)
        self._sum_dry = cd[n] - cd[lo] + (self._sum_dry if lo == 0 and end - b < 0 else 0.0)
        self._fill = n - (end - b)
        xs.append(float(end))
        ys.append(self._end)
        pos = np.interp(np.arange(1, n + 1, dtype=np.float64), xs, ys)
        depth = max(0.0, float(attenuation_db))
        return (10.0 ** (-(1.0 - pos) * depth / 20.0)).astype(np.float32)
