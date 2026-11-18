"""The output-driven soft gate of the neural extractor (step 07b).

Even a well-trained extractor leaves a faint residue of the other sounds when the selected class
is not sounding: music at -15 dB under "Speech" is still clearly audible at the 40 dB setting.
The gate turns the model's estimate further down while the estimate itself says "there is
nothing of the target here". Because it follows the **model's output**, not the detector, it
opens as soon as the model lets the target through (within one model step), with none of the
baseline gate's detector lag.

Per model step (416 samples = 9.4 ms at 44.1 kHz), on the estimate ``y`` and the model input
``x`` of the same samples:

1. smoothed short-term levels of both: the mean square of the step, followed by a one-pole
   smoother that rises with ``attack_ms`` and falls with ``release_ms``;
2. their ratio ``r = L_y - L_x`` in dB: how much of the input the model kept;
3. the wanted gain: 0 dB when ``r >= open_db``, ``-depth_db`` when ``r <= close_db``, a straight
   line in dB between;
4. the gain follows it quickly upwards (``attack_ms``) and slowly downwards (``release_ms``), and
   only after it has been open for ``hold_ms`` without a reason to close: short pauses between
   words do not make it pump;
5. inside the step the gain moves linearly from its previous value to the new one (no steps,
   no clicks).

The gate sees exactly the samples it changes, so it adds **no latency**; it works block by block
from the start of the signal, so streaming and whole-file processing give the same output.
"""

from __future__ import annotations

import math

import numpy as np

from soundspy.config import NeuralGateConfig

_POWER_FLOOR = 1e-12  # -120 dBFS: two silent signals give r = 0 dB (gate open)


class OutputGate:
    """One gate. ``process(estimate, reference)`` works on whole blocks of ``block`` samples
    (a shorter last block is allowed). State is kept between calls."""

    def __init__(self, cfg: NeuralGateConfig, sample_rate: int, block: int) -> None:
        if block < 1:
            raise ValueError("block must be >= 1")
        self.cfg = cfg
        self.block = int(block)
        step_s = block / float(sample_rate)
        self._attack = math.exp(-step_s / (cfg.attack_ms / 1000.0))
        self._release = math.exp(-step_s / (cfg.release_ms / 1000.0))
        self._hold_steps = math.ceil(cfg.hold_ms / 1000.0 / step_s - 1e-9)
        self.reset()

    def reset(self) -> None:
        """Start open: the first words after a selection must not be cut."""
        self._env_y = 0.0
        self._env_x = 0.0
        self.gain_db = 0.0
        self._hold = self._hold_steps
        self._closed_steps = 0
        self._steps = 0

    @property
    def closed_share(self) -> float:
        """Share of the steps so far with more than half of the depth applied (status)."""
        return self._closed_steps / self._steps if self._steps else 0.0

    def wanted_gain_db(self, ratio_db: float) -> float:
        cfg = self.cfg
        if ratio_db >= cfg.open_db:
            return 0.0
        if ratio_db <= cfg.close_db:
            return -cfg.depth_db
        return -cfg.depth_db * (cfg.open_db - ratio_db) / (cfg.open_db - cfg.close_db)

    def _smooth(self, env: float, power: float) -> float:
        a = self._attack if power > env else self._release
        return a * env + (1.0 - a) * power

    def block_gains_db(self, power_y: np.ndarray, power_x: np.ndarray) -> np.ndarray:
        """The gain (dB) after each block, from the blocks' mean squares; updates the state."""
        out = np.empty(len(power_y), dtype=np.float64)
        depth, attack, release = self.cfg.depth_db, self._attack, self._release
        env_y, env_x, gain_db, hold = self._env_y, self._env_x, self.gain_db, self._hold
        closed = 0
        for k, (py, px) in enumerate(zip(power_y.tolist(), power_x.tolist(), strict=True)):
            env_y = self._smooth(env_y, py)
            env_x = self._smooth(env_x, px)
            ratio = 10.0 * math.log10((env_y + _POWER_FLOOR) / (env_x + _POWER_FLOOR))
            wanted = self.wanted_gain_db(ratio)
            if wanted >= gain_db:  # opening (or staying): fast, and the hold starts again
                gain_db = attack * gain_db + (1.0 - attack) * wanted
                hold = self._hold_steps
            elif hold > 0:  # a reason to close, but not for long enough yet
                hold -= 1
            else:
                gain_db = release * gain_db + (1.0 - release) * wanted
            out[k] = gain_db
            closed += gain_db < -0.5 * depth
        self._env_y, self._env_x, self.gain_db, self._hold = env_y, env_x, gain_db, hold
        self._steps += len(out)
        self._closed_steps += closed
        return out

    def process(self, estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """The gated estimate (1-D float32, same length). ``reference`` = the model input of the
        same samples."""
        y = np.asarray(estimate, dtype=np.float32).reshape(-1)
        x = np.asarray(reference, dtype=np.float32).reshape(-1)
        if x.shape != y.shape:
            raise ValueError(f"estimate {y.shape} and reference {x.shape} differ in length")
        n = y.shape[0]
        if n == 0:
            return y.copy()
        starts = np.arange(0, n, self.block)
        sizes = np.minimum(self.block, n - starts)
        power_y = np.add.reduceat(np.square(y, dtype=np.float64), starts) / sizes
        power_x = np.add.reduceat(np.square(x, dtype=np.float64), starts) / sizes
        previous = 10.0 ** (self.gain_db / 20.0)
        gains = 10.0 ** (self.block_gains_db(power_y, power_x) / 20.0)
        before = np.concatenate(([previous], gains[:-1]))
        which = np.repeat(np.arange(starts.size), sizes)
        position = (np.arange(n) - starts[which] + 1) / sizes[which]  # 1/size .. 1 in a block
        ramp = before[which] + (gains - before)[which] * position
        return (y * ramp).astype(np.float32)
