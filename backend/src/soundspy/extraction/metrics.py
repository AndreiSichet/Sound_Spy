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
