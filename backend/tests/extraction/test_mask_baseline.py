"""The detector-gated Wiener baseline on synthetic signals (no model, no data)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from soundspy.core.audio import AudioBuffer
from soundspy.core.target import TargetSpec
from soundspy.engine.processor import Passthrough
from soundspy.extraction import metrics as M
from soundspy.extraction.activity import ActivitySource
from soundspy.extraction.mask_baseline import MaskBaselineExtractor
from soundspy.extraction.offline import build_extractor, run_offline

SR = 48_000
SPEECH = TargetSpec(labels=frozenset({"speech"}), attenuation_db=30.0)


class Schedule(ActivitySource):
    """Active inside the given ``(start, stop)`` spans of the frame's own time."""

    def __init__(self, *spans: tuple[float, float], lag_s: float = 0.0) -> None:
        self.spans, self.lag_s = spans, lag_s

    def active(self, labels, centre_s: float, now_s: float) -> bool:
        return any(a <= centre_s < b for a, b in self.spans)


def buf(x: np.ndarray) -> AudioBuffer:
    return AudioBuffer(np.asarray(x, dtype=np.float32).reshape(-1, 1), SR)


def db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.square(x)) + 1e-30))


@pytest.fixture
def cfg(config):
    return config.extraction.mask_baseline


@pytest.fixture
def noise() -> np.ndarray:
    return 0.05 * np.random.default_rng(0).standard_normal(SR * 4)


@pytest.mark.parametrize("attenuation_db", [10.0, 30.0])
def test_inactive_target_is_attenuated_by_the_floor(cfg, noise, attenuation_db: float) -> None:
    target = TargetSpec(labels=frozenset({"speech"}), attenuation_db=attenuation_db)
    out = run_offline(MaskBaselineExtractor(cfg, Schedule()), buf(noise), target).samples.ravel()
    settled = slice(SR, None)  # after the ramps
    assert db(out[settled]) - db(noise[settled]) == pytest.approx(-attenuation_db, abs=1.0)


def test_noise_is_reduced_and_the_tone_kept_while_the_target_is_active(cfg, noise) -> None:
    t = np.arange(noise.size) / SR
    on = (t >= 1.5) & (t < 3.5)
    tone = 0.1 * np.sin(2 * np.pi * 1000 * t) * on
    mixture = tone + noise
    extractor = MaskBaselineExtractor(cfg, Schedule((1.5, 3.5)))
    out = run_offline(extractor, buf(mixture), SPEECH).samples.ravel()
    assert M.si_sdr_improvement(out, mixture, tone) > 5.0
    assert M.snr_improvement(out, mixture, tone) > 5.0
    inside = slice(int(2.0 * SR), int(3.0 * SR))
    assert db(out[inside] - tone[inside]) < db(noise[inside]) - 6.0  # noise reduced
    assert db(out[inside]) == pytest.approx(db(tone[inside]), abs=1.0)  # tone preserved


def test_gate_ramps_cause_no_discontinuity(cfg) -> None:
    t = np.arange(SR * 3) / SR
    hum = 0.3 * np.sin(2 * np.pi * 200 * t)  # always there: "everything else"
    fade = np.clip(np.minimum(t - 0.5, 1.0 - t) / 0.05, 0, 1) + np.clip(
        np.minimum(t - 1.8, 2.2 - t) / 0.05, 0, 1)
    wanted = 0.3 * np.sin(2 * np.pi * 1000 * t) * fade  # only inside the active spans
    x = hum + wanted  # smooth input: any click would stand out
    extractor = MaskBaselineExtractor(cfg, Schedule((0.5, 1.0), (1.8, 2.2)))
    out = run_offline(extractor, buf(x), SPEECH).samples.ravel()
    largest_input_step = float(np.max(np.abs(np.diff(x))))
    assert float(np.max(np.abs(np.diff(out)))) <= 1.1 * largest_input_step
    # the gate really moved: the wanted tone comes through, the hum ends up at the floor
    inside = slice(int(0.7 * SR), int(0.9 * SR))
    assert db(out[inside]) == pytest.approx(db(wanted[inside]), abs=1.5)
    assert db(out[int(2.9 * SR):]) == pytest.approx(db(hum) - 30.0, abs=1.0)


def test_hold_keeps_the_gate_open_after_the_target_stops(cfg, noise) -> None:
    extractor = MaskBaselineExtractor(dataclasses.replace(cfg, hold_ms=300, close_ms=200),
                                      Schedule((0.5, 1.0)))
    extractor.configure(SR, 1)
    extractor.set_target(SPEECH)
    hop = cfg.hop
    gates = []
    for i in range(0, 2 * SR, hop):
        extractor.process(buf(noise[i:i + hop]))
        gates.append(extractor.gate)
    at = lambda s: gates[int(s * SR / hop)]  # noqa: E731
    assert at(0.4) == 0.0 and at(0.6) == 1.0
    assert at(1.25) == 1.0  # still inside the 300 ms hold
    assert 0.0 < at(1.4) < 1.0  # closing ramp
    assert at(1.6) == 0.0


def test_empty_labels_pass_the_sound_through_unchanged(cfg, noise) -> None:
    extractor = MaskBaselineExtractor(cfg, Schedule())
    out = run_offline(extractor, buf(noise), TargetSpec()).samples.ravel()
    np.testing.assert_allclose(out, noise, atol=1e-5)
    # streaming: the same samples, delayed by the reported latency (still through the STFT)
    extractor = MaskBaselineExtractor(cfg, Schedule())
    extractor.configure(SR, 1)
    extractor.set_target(TargetSpec())
    out = np.concatenate([extractor.process(buf(noise[i:i + 960])).samples.ravel()
                          for i in range(0, noise.size, 960)])
    delay = extractor.latency_samples
    assert delay == cfg.n_fft
    np.testing.assert_allclose(out[delay:], noise[:-delay], atol=1e-5)
    assert extractor.status_info() == {"method": "mask_baseline", "latency_ms": 21.33,
                                       "gating": True, "target_active": False}


def test_zero_attenuation_changes_nothing(cfg, noise) -> None:
    target = TargetSpec(labels=frozenset({"speech"}), attenuation_db=0.0)
    out = run_offline(MaskBaselineExtractor(cfg, Schedule()), buf(noise), target).samples.ravel()
    np.testing.assert_allclose(out, noise, atol=1e-5)


def test_selecting_and_deselecting_is_ramped(cfg) -> None:
    t = np.arange(SR * 2) / SR
    x = 0.5 * np.sin(2 * np.pi * 200 * t)
    extractor = MaskBaselineExtractor(cfg, Schedule())
    extractor.configure(SR, 1)
    extractor.set_target(TargetSpec())
    out = []
    for i in range(0, x.size, 960):
        if i == 960 * 25:
            extractor.set_target(SPEECH)  # never active: the sound drops to the floor
        if i == 960 * 75:
            extractor.set_target(TargetSpec())  # back to pass-through
        out.append(extractor.process(buf(x[i:i + 960])).samples.ravel())
    out = np.concatenate(out)
    assert float(np.max(np.abs(np.diff(out)))) <= 1.1 * float(np.max(np.abs(np.diff(x))))
    assert db(out[int(1.2 * SR):int(1.4 * SR)]) == pytest.approx(db(x) - 30.0, abs=1.0)
    assert db(out[int(1.8 * SR):]) == pytest.approx(db(x), abs=0.5)


def test_without_an_activity_source_there_is_no_gating(cfg, noise) -> None:
    """No detection model: plain noise reduction (minimum statistics), the gate stays open."""
    t = np.arange(noise.size) / SR
    bursts = 0.3 * np.sin(2 * np.pi * 800 * t) * ((t % 1.0) < 0.3)  # short tones, 1 per second
    mixture = bursts + noise
    extractor = MaskBaselineExtractor(cfg, None)
    out = run_offline(extractor, buf(mixture), SPEECH).samples.ravel()
    assert extractor.status_info()["gating"] is False and extractor.gate == 1.0
    on = (t % 1.0) < 0.3
    on[:SR] = False
    assert db(out[on]) == pytest.approx(db(mixture[on]), abs=2.0)  # tones are kept
    quiet = ~((t % 1.0) < 0.4)
    quiet[:2 * SR] = False
    assert db(out[quiet]) < db(noise[quiet]) - 3.0  # the steady noise is reduced


def test_a_late_detector_does_not_teach_the_filter_the_target(cfg, noise) -> None:
    """The lag guard: frames just before "active" arrives are not used as noise."""
    t = np.arange(noise.size) / SR
    tone = 0.1 * np.sin(2 * np.pi * 1000 * t) * ((t >= 1.5) & (t < 3.5))
    mixture = tone + noise
    late = (1.8, 3.5)  # the detector notices 0.3 s late
    results = {}
    for lag in (0.0, 0.5):
        extractor = MaskBaselineExtractor(cfg, Schedule(late, lag_s=lag))
        out = run_offline(extractor, buf(mixture), SPEECH).samples.ravel()
        inside = slice(int(2.0 * SR), int(3.0 * SR))
        results[lag] = db(out[inside]) - db(tone[inside])
    assert results[0.5] == pytest.approx(0.0, abs=1.0)  # tone intact with the guard
    assert results[0.0] < results[0.5] - 3.0  # without it, the tone was learnt as noise


def test_new_labels_forget_the_noise_estimate(cfg, noise) -> None:
    extractor = MaskBaselineExtractor(cfg, Schedule())
    extractor.configure(SR, 1)
    extractor.set_target(SPEECH)
    extractor.process(buf(noise[:SR]))
    assert extractor._noise is not None
    extractor.set_target(TargetSpec(labels=frozenset({"speech"}), attenuation_db=12.0))
    assert extractor._noise is not None  # only the attenuation changed
    extractor.set_target(TargetSpec(labels=frozenset({"music"})))
    assert extractor._noise is None


def test_only_one_channel_and_configure_first(cfg) -> None:
    extractor = MaskBaselineExtractor(cfg, None)
    with pytest.raises(RuntimeError, match="configure"):
        extractor.process(buf(np.zeros(10)))
    with pytest.raises(ValueError, match="one channel"):
        extractor.configure(SR, 2)


def test_build_extractor_follows_the_configured_method(config) -> None:
    assert isinstance(build_extractor(config, None), MaskBaselineExtractor)
    assert isinstance(build_extractor(config, None, method="passthrough"), Passthrough)
    with pytest.raises(ValueError, match="unknown extraction method"):
        build_extractor(config, None, method="magic")
    assert Passthrough().latency_samples == 0
