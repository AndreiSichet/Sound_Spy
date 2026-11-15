"""Extraction metrics on cases with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from soundspy.extraction import metrics as M


@pytest.fixture
def signals() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(48_000)
    noise = rng.standard_normal(48_000)
    ref -= ref.mean()
    noise -= noise.mean()
    noise -= (noise @ ref) / (ref @ ref) * ref  # exactly orthogonal to the reference
    return ref, noise


def test_si_sdr_ignores_scale_and_is_capped(signals) -> None:
    ref, _ = signals
    assert M.si_sdr(ref, ref) == M.CAP_DB
    assert M.si_sdr(0.01 * ref, ref) == M.CAP_DB  # a scaled copy is "perfect"
    assert M.si_sdr(np.zeros_like(ref), ref) == -M.CAP_DB
    assert M.si_sdr(-ref, ref) == M.CAP_DB  # sign is a scale too


def test_si_sdr_of_orthogonal_noise_is_the_energy_ratio(signals) -> None:
    ref, noise = signals
    noise = noise * np.sqrt((ref @ ref) / (noise @ noise) / 10.0)  # 10 dB below the reference
    assert M.si_sdr(ref + noise, ref) == pytest.approx(10.0, abs=1e-6)
    assert M.si_sdr(3.0 * (ref + noise), ref) == pytest.approx(10.0, abs=1e-6)
    assert M.snr(ref + noise, ref) == pytest.approx(10.0, abs=1e-6)


def test_snr_notices_a_wrong_level(signals) -> None:
    ref, _ = signals
    assert M.snr(0.5 * ref, ref) == pytest.approx(6.0206, abs=1e-3)  # error = half the reference
    assert M.snr(ref, ref) == M.CAP_DB


def test_improvements_are_relative_to_the_mixture(signals) -> None:
    ref, noise = signals
    mixture = ref + noise  # 0 dB
    better = ref + 0.1 * noise  # 20 dB
    assert M.si_sdr_improvement(better, mixture, ref) == pytest.approx(20.0, abs=1e-6)
    assert M.snr_improvement(better, mixture, ref) == pytest.approx(20.0, abs=1e-6)
    assert M.si_sdr_improvement(mixture, mixture, ref) == 0.0
    # uniformly quieter: SI-SDR does not care, SNR does (silence scores 0 dB, the mixture 20 dB)
    assert M.si_sdr_improvement(0.03 * better, better, ref) == pytest.approx(0.0, abs=1e-6)
    assert M.snr_improvement(0.03 * better, better, ref) == pytest.approx(-19.7, abs=0.2)
    # ... so for a mixture below 0 dB, SNR rewards plain silence
    worse = ref + 10.0 * noise
    assert M.snr_improvement(np.zeros_like(ref), worse, ref) == pytest.approx(20.0, abs=0.1)


def test_delay_compensation(signals) -> None:
    ref, _ = signals
    delayed = np.concatenate((np.zeros(1024), ref))[: ref.size]
    assert M.si_sdr(delayed, ref) < 0.0  # misaligned: looks like noise
    aligned = M.align(delayed, 1024)
    assert aligned.shape == ref.shape
    assert M.si_sdr(aligned[:-1024], ref[:-1024]) == M.CAP_DB
    np.testing.assert_array_equal(aligned[-1024:], 0.0)
    two_d = M.align(delayed.reshape(-1, 1), 1024, length=100)
    assert two_d.shape == (100, 1)
    with pytest.raises(ValueError):
        M.align(ref, -1)


def test_suppression_for_empty_targets(signals) -> None:
    ref, _ = signals
    assert M.suppression_db(ref, ref) == 0.0
    assert M.suppression_db(10 ** (-30 / 20) * ref, ref) == pytest.approx(-30.0, abs=1e-6)
    assert M.suppression_db(np.zeros_like(ref), ref) == -M.CAP_DB
    assert M.suppression_db(ref, np.zeros_like(ref)) == 0.0  # nothing to suppress
    assert M.is_silent(np.zeros(10)) and not M.is_silent(ref)


def test_silent_reference_and_shape_errors(signals) -> None:
    ref, _ = signals
    with pytest.raises(ValueError, match="silent reference"):
        M.si_sdr(ref, np.zeros_like(ref))
    with pytest.raises(ValueError, match="silent reference"):
        M.snr(ref, np.zeros_like(ref))
    with pytest.raises(ValueError, match="shapes differ"):
        M.si_sdr(ref[:10], ref)
