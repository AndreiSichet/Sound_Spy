"""``HybridGate`` (step 07c): closes only after confident absence + hold, reopens at once on an
output rise even while the detector still says absent, closed gain = the slider, no clicks,
any chunking gives the same gains, warm-up, closed start, the detector timeline offline."""

from __future__ import annotations

import dataclasses

import numpy as np

from soundspy.config import HybridGateConfig
from soundspy.extraction.activity import ActivitySource, TimelineActivity
from soundspy.extraction.hybrid_gate import HybridGate

SR = 48000
LABELS = frozenset({"speech"})


class Scripted(ActivitySource):
    """``active`` / ``absent`` as functions of the block's input time (``centre_s``)."""

    def __init__(self, active=lambda t: False, absent=lambda t: True) -> None:
        self._active, self._absent = active, absent

    def active(self, labels, centre_s, now_s):
        return self._active(centre_s)

    def absent(self, labels, centre_s, now_s, factor):
        return not self._active(centre_s) and self._absent(centre_s)


def cfg(**kw) -> HybridGateConfig:
    base = dict(close_hold_ms=400, close_ms=150, open_ms=5, reopen_ratio_db=-10, attack_ms=5,
                release_ms=50, min_level_dbfs=-60, block_ms=1, warmup_ms=0, start_closed=False)
    return HybridGateConfig(**{**base, **kw})


def noise(seconds: float, level: float = 0.1, seed: int = 0) -> np.ndarray:
    return (level * np.random.default_rng(seed).standard_normal(int(seconds * SR))
            ).astype(np.float32)


def run(gate: HybridGate, est, dry, att=40.0, chunk=960) -> np.ndarray:
    return np.concatenate([gate.gains(est[i:i + chunk], dry[i:i + chunk], LABELS, att)
                           for i in range(0, est.size, chunk)])


def test_closes_only_after_confident_absence_and_hold() -> None:
    dry = noise(2.0)
    est = 0.001 * dry  # the extractor finds nothing (-60 dB)
    g = run(HybridGate(cfg(), SR, 0, Scripted()), est, dry)
    ms = lambda t: int(t * SR / 1000)  # noqa: E731
    assert np.all(g[:ms(395)] == 1.0)  # held open for close_hold_ms
    assert g[ms(420)] < 1.0  # then closing over close_ms
    assert np.allclose(g[ms(600):], 0.01)  # closed = the slider's 40 dB, on everything


def test_not_sure_is_not_absent() -> None:
    dry = noise(1.5)
    gate = HybridGate(cfg(), SR, 0, Scripted(absent=lambda t: False))  # below threshold, unsure
    assert np.all(run(gate, 0.001 * dry, dry) == 1.0)


def test_output_rise_reopens_while_the_detector_still_says_absent() -> None:
    dry = noise(2.0)
    est = 0.001 * dry.copy()
    onset = int(1.0 * SR)
    est[onset:] = dry[onset:]  # a word starts: the estimate jumps to the input's level
    g = run(HybridGate(cfg(), SR, 0, Scripted()), est, dry)  # detector: absent throughout
    assert np.allclose(g[onset - 10], 0.01)
    reopened = onset + np.argmax(g[onset:] > 0.99)
    assert (reopened - onset) / SR * 1000 < 15  # envelope attack + open ramp + one block


def test_detector_on_reopens() -> None:
    dry = noise(2.0)
    est = 0.001 * dry
    gate = HybridGate(cfg(), SR, 0, Scripted(active=lambda t: t >= 1.0))
    g = run(gate, est, dry)
    assert np.allclose(g[int(0.9 * SR)], 0.01) and np.all(g[int(1.03 * SR):] > 0.99)
    assert gate.reopened == 1


def test_closed_gain_follows_the_slider_and_has_no_clicks() -> None:
    dry = noise(1.5)
    for att in (15.0, 30.0, 40.0):
        g = run(HybridGate(cfg(), SR, 0, Scripted()), 0.001 * dry, dry, att=att)
        assert np.isclose(g[-1], 10 ** (-att / 20), rtol=1e-4)
        assert np.abs(np.diff(g)).max() < 0.01  # ramped sample by sample (in dB)


def test_any_chunking_gives_the_same_gains() -> None:
    dry = noise(2.0, seed=1)
    est = 0.001 * dry.copy()
    est[SR:] = 0.5 * dry[SR:]
    source = Scripted(active=lambda t: 1.5 <= t < 1.7)
    whole = HybridGate(cfg(), SR, 0, source).gains(est, dry, LABELS, 30.0)
    rng = np.random.default_rng(2)
    gate, parts, i = HybridGate(cfg(), SR, 0, source), [], 0
    while i < est.size:
        n = int(rng.integers(1, 3000))
        parts.append(gate.gains(est[i:i + n], dry[i:i + n], LABELS, 30.0))
        i += n
    assert np.allclose(np.concatenate(parts), whole, rtol=0, atol=1e-7)


def test_warmup_ignores_the_output_until_it_is_over() -> None:
    dry = noise(3.0)
    est = dry.copy()  # a leaking start: the estimate as loud as the input
    gate = HybridGate(cfg(warmup_ms=2000, start_closed=True), SR, 0, Scripted())
    gate.restart(warmup=True)
    g = run(gate, est, dry)
    assert np.allclose(g[:int(1.9 * SR)], 0.01)  # closed from the start, the leak not trusted
    assert np.all(g[int(2.05 * SR):] > 0.99)  # after the warm-up the output counts again


def test_starts_open_without_a_detector_and_never_closes() -> None:
    dry = noise(1.0)
    gate = HybridGate(cfg(start_closed=True), SR, 0, None)
    assert np.all(run(gate, 0.001 * dry, dry) == 1.0)


def test_nothing_selected_or_disabled_keeps_it_open() -> None:
    dry = noise(1.0)
    gate = HybridGate(cfg(), SR, 0, Scripted())
    assert np.all(gate.gains(0.001 * dry, dry, frozenset(), 40.0) == 1.0)
    off = HybridGate(cfg(enabled=False), SR, 0, Scripted())
    assert np.all(run(off, 0.001 * dry, dry) == 1.0)


def test_quiet_input_cannot_reopen() -> None:
    dry = noise(1.5, level=1e-4)  # -80 dBFS: below min_level_dbfs
    gate = HybridGate(cfg(start_closed=True), SR, 0, Scripted())
    assert np.allclose(run(gate, dry.copy(), dry), 0.01)


def test_timeline_probabilities_decide_confident_absence() -> None:
    times = np.array([0.0, 0.5, 1.0])
    probs = np.array([[0.10], [0.30], [0.90]])  # threshold 0.5: sure / unsure / on
    timeline = TimelineActivity(["speech"], times, probs >= 0.5, probabilities=probs,
                                thresholds={"speech": 0.5})
    assert timeline.absent(LABELS, 0.2, 0.2, 0.5)  # 0.10 < 0.25
    assert not timeline.absent(LABELS, 0.7, 0.7, 0.5)  # 0.30: below threshold, not sure
    assert not timeline.absent(LABELS, 1.2, 1.2, 0.5) and timeline.active(LABELS, 1.2, 1.2)
    plain = TimelineActivity(["speech"], times, probs >= 0.5)  # no probabilities: not active
    assert plain.absent(LABELS, 0.7, 0.7, 0.5)


def test_latency_shifts_the_detector_clock() -> None:
    seen: list[float] = []

    class Clock(Scripted):
        def active(self, labels, centre_s, now_s):
            seen.append(centre_s)
            return False

    dry = noise(0.1)
    HybridGate(dataclasses.replace(cfg(), block_ms=1), SR, 4800, Clock()).gains(
        dry, dry, LABELS, 30.0)
    assert np.isclose(seen[0], (0.5 * 48 - 4800) / SR)  # block 0's input time: 100 ms earlier


def test_nothing_decided_yet_keeps_the_position() -> None:
    class Late(Scripted):  # the live detector's first decision arrives at 0.3 s
        def known(self, centre_s, now_s):
            return now_s >= 0.3

    dry = noise(1.0)
    est = 0.001 * dry
    on = Late(active=lambda t: t >= 0.0, absent=lambda t: False)  # speech from the start
    g = run(HybridGate(cfg(start_closed=True), SR, 0, on), est, dry)
    assert np.allclose(g[:int(0.29 * SR)], 0.01)  # closed until the detector has decided
    assert np.all(g[int(0.33 * SR):] > 0.99)  # then "speech on" opens it
    opened = HybridGate(cfg(start_closed=False), SR, 0, Late(absent=lambda t: False))
    assert np.all(run(opened, est, dry)[:int(0.29 * SR)] == 1.0)  # an open start stays open
