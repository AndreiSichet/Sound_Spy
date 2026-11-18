"""Extraction metrics: SI-SDR, SNR, their improvements over the mixture, and suppression.

All functions take plain arrays (any shape, compared sample by sample) that are already
**time-aligned**; use :func:`align` first when the estimate comes out of a stage with a known
delay.

* **SI-SDR** (scale-invariant signal-to-distortion ratio, dB): how much of the estimate is the
  reference, however loud the estimate is. The estimate is split into the part that is a scaled
  copy of the reference and everything else; the ratio of their energies is the SI-SDR.
  Undefined for a silent reference (see ``suppression_db``).
* **SNR** (dB): reference energy over the energy of ``reference - estimate``. Not scale-invariant:
  an estimate that is right but too quiet scores badly, which SI-SDR would not notice.
* **Improvement** (SI-SDRi, SNRi): the metric of the estimate minus the metric of the unprocessed
  mixture. 0 dB = no better than doing nothing.
* **Suppression** (dB): output energy relative to the mixture's. For recipes whose target class is
  absent the correct output is silence, so the more negative the better.

Values are capped at +-``CAP_DB`` so perfect or silent signals give finite numbers.
"""

from __future__ import annotations

import math

import numpy as np

CAP_DB = 100.0
_TINY = 1e-20


def _flat(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


def _db(ratio: float) -> float:
    if ratio <= 0.0:
        return -CAP_DB
    return float(min(CAP_DB, max(-CAP_DB, 10.0 * math.log10(ratio))))


def _ratio_db(numerator: float, denominator: float) -> float:
    if denominator <= _TINY:
        return CAP_DB if numerator > _TINY else -CAP_DB
    return _db(numerator / denominator)


def align(estimate: np.ndarray, delay: int, length: int | None = None) -> np.ndarray:
    """Undo a known delay: drop the first ``delay`` samples, zero-pad the end to ``length``."""
    x = np.asarray(estimate)
    if delay < 0:
        raise ValueError("delay must be >= 0")
    length = x.shape[0] if length is None else length
    out = np.zeros((length, *x.shape[1:]), dtype=x.dtype)
    part = x[delay:delay + length]
    out[: part.shape[0]] = part
    return out


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    est, ref = _flat(estimate), _flat(reference)
    if est.shape != ref.shape:
        raise ValueError(f"shapes differ: {est.shape} vs {ref.shape}")
    est, ref = est - est.mean(), ref - ref.mean()
    ref_energy = float(ref @ ref)
    if ref_energy <= _TINY:
        raise ValueError("SI-SDR is undefined for a silent reference (use suppression_db)")
    scaled = (float(est @ ref) / ref_energy) * ref
    noise = est - scaled
    return _ratio_db(float(scaled @ scaled), float(noise @ noise))


def snr(estimate: np.ndarray, reference: np.ndarray) -> float:
    est, ref = _flat(estimate), _flat(reference)
    if est.shape != ref.shape:
        raise ValueError(f"shapes differ: {est.shape} vs {ref.shape}")
    ref_energy = float(ref @ ref)
    if ref_energy <= _TINY:
        raise ValueError("SNR is undefined for a silent reference (use suppression_db)")
    error = ref - est
    return _ratio_db(ref_energy, float(error @ error))


def si_sdr_improvement(estimate: np.ndarray, mixture: np.ndarray, reference: np.ndarray) -> float:
    return si_sdr(estimate, reference) - si_sdr(mixture, reference)


def snr_improvement(estimate: np.ndarray, mixture: np.ndarray, reference: np.ndarray) -> float:
    return snr(estimate, reference) - snr(mixture, reference)


def suppression_db(output: np.ndarray, mixture: np.ndarray) -> float:
    """Output energy relative to the mixture energy in dB (0 = untouched, negative = quieter)."""
    out, mix = _flat(output), _flat(mixture)
    mix_energy = float(mix @ mix)
    if mix_energy <= _TINY:
        return 0.0
    return _db(float(out @ out) / mix_energy)


def is_silent(reference: np.ndarray) -> bool:
    ref = _flat(reference)
    return float(ref @ ref) <= _TINY


def frame_power(x: np.ndarray, n: int, frames: int) -> np.ndarray:
    """Mean power of ``frames`` consecutive frames of ``n`` samples (a 1-D signal)."""
    y = np.asarray(x[:frames * n], dtype=np.float64).reshape(frames, n)
    return np.einsum("fn,fn->f", y, y) / n


def interruption_rate(output: np.ndarray, reference: np.ndarray, sample_rate: int, *,
                      frame_ms: float = 50.0, drop_db: float = 6.0, range_db: float = 20.0,
                      floor_dbfs: float = -70.0) -> float | None:
    """Share of the reference's active frames that the output turned down by more than
    ``drop_db`` (step 07c: "interruption rate": cut word starts, ducked syllables).

    Frames of ``frame_ms``; a frame is active when the reference's energy in it is within
    ``range_db`` of its loudest frame and above ``floor_dbfs``. ``None`` when no frame is
    active. For a speech-only input (reference = input) it is exactly "output more than 6 dB
    below the input".

    The dataset's speech clips carry their own background (a room, a street), which a speech
    enhancer removes too: on synthetic mixtures the rate therefore also counts removed noise in
    quiet frames of the reference. ``range_db`` (20 dB) keeps such frames out as far as their
    level allows; the real-room benchmark (quiet room) measures interruptions cleanly.
    """
    ref, out = _flat(reference), _flat(output)
    n = max(1, int(round(frame_ms * sample_rate / 1000.0)))
    frames = min(ref.size, out.size) // n
    if frames == 0:
        return None
    r, o = frame_power(ref, n, frames), frame_power(out, n, frames)
    level = 10.0 * np.log10(r + _TINY)
    active = (level > level.max() - range_db) & (level > floor_dbfs)
    if not active.any():
        return None
    drop = level - 10.0 * np.log10(o + _TINY)
    return float(np.mean(drop[active] > drop_db))
