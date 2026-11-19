"""``SpeechRouteGate`` (step 07d): closes after ``close_hold_ms`` with DeepFilterNet3's output
low, never on the detector alone, reopens within ``open_ms``, hysteresis (no chattering), the
closed start and the warm-up, the detector's hold, any chunking = the whole signal."""

from __future__ import annotations

import numpy as np

from soundspy.config import SpeechGateConfig
from soundspy.extraction.activity import ActivitySource
from soundspy.extraction.speech_gate import SpeechRouteGate

SR = 48000
LABELS = frozenset({"speech"})
G40 = 10 ** (-40 / 20)


class Scripted(ActivitySource):
    def __init__(self, active=lambda t: False, absent=lambda t: False) -> None:
        self._active, self._absent = active, absent

    def active(self, labels, centre_s, now_s):
        return self._active(centre_s)

    def absent(self, labels, centre_s, now_s, factor):
        return not self._active(centre_s) and self._absent(centre_s)


def cfg(**kw) -> SpeechGateConfig:
    base = dict(close_db=-15, open_db=-12, close_hold_ms=150, close_hold_confident_ms=50,
                detector_override_max_ms=1000, close_ms=50, open_ms=5, ratio_attack_ms=5,
                ratio_release_ms=50, min_level_dbfs=-60, block_ms=1, warmup_ms=0,
                start_closed=True)
    return SpeechGateConfig(**{**base, **kw})


def noise(seconds: float, level: float = 0.1, seed: int = 0) -> np.ndarray:
    return (level * np.random.default_rng(seed).standard_normal(int(seconds * SR))
            ).astype(np.float32)


def db(value: float) -> float:
    return 10 ** (value / 20)


def ms(t: float) -> int:
    return int(t * SR / 1000)


def run(gate: SpeechRouteGate, est, dry, att=40.0, chunk=960) -> np.ndarray:
    return np.concatenate([gate.gains(est[i:i + chunk], dry[i:i + chunk], LABELS, att)
                           for i in range(0, est.size, chunk)])


def speech_then_music(seconds: float = 2.0, switch: float = 0.5) -> tuple[np.ndarray, ...]:
    """The output as loud as the input (speech) until ``switch``, then -40 dB (music only)."""
    dry = noise(seconds)
    est = dry.copy()
    est[int(switch * SR):] *= db(-40)
    return est, dry


def test_closes_after_the_hold_with_the_output_low() -> None:
    est, dry = speech_then_music()
    g = run(SpeechRouteGate(cfg(), SR, 0), est, dry)
    assert np.all(g[ms(20):ms(500)] > 0.999)  # speech: open
    # the output's envelope falls 4.3 dB per 50 ms (release): r < -15 dB after ~173 ms; then
    # the 150 ms hold, then the 50 ms ramp
    assert np.all(g[ms(500):ms(810)] > 0.999)
    assert np.allclose(g[ms(880):], G40)  # closed = the slider's 40 dB, on everything


def test_the_detector_alone_never_closes_it() -> None:
    dry = noise(2.0)
    est = db(-6) * dry  # r = -6 dB: above close_db the whole time
    sure = Scripted(absent=lambda t: True)  # the detector: sure nothing is there
    g = run(SpeechRouteGate(cfg(start_closed=False), SR, 0, sure), est, dry)
    assert np.all(g == 1.0)


def test_reopens_within_open_ms_when_the_output_rises() -> None:
    dry = noise(2.0)
    est = db(-40) * dry
    onset = SR
    est[onset:] = dry[onset:]  # a word starts
    gate = SpeechRouteGate(cfg(), SR, 0)
    g = run(gate, est, dry)
    assert np.allclose(g[:onset], G40)  # closed from the start, music only
    opened = np.argmax(g[onset:] > 0.99)
    assert opened / SR * 1000 < 12  # envelope attack + open ramp + one block
    assert gate.reopened == 1


def _alternating(levels_db: tuple[float, float], seconds: float, period_ms: float = 40.0
                 ) -> tuple[np.ndarray, np.ndarray]:
    dry = noise(seconds, seed=3)
    t = np.arange(dry.size) / SR * 1000
    level = np.where((t // (period_ms / 2)) % 2 == 0, levels_db[0], levels_db[1])
    return (dry * 10 ** (level / 20)).astype(np.float32), dry


def test_hysteresis_no_chattering_around_either_threshold() -> None:
    # around close_db (-14 / -16 dB, 20 ms each): never 150 ms below it in a row: stays open
    est, dry = _alternating((-14.0, -16.0), 2.0)
    gate = SpeechRouteGate(cfg(start_closed=False), SR, 0)
    g = run(gate, est, dry)
    assert np.all(g == 1.0) and gate.reopened == 0
    # around open_db (-11 / -13 dB) from a closed start: opens once, then stays open (-13 is
    # above close_db)
    est, dry = _alternating((-11.0, -13.0), 2.0)
    gate = SpeechRouteGate(cfg(), SR, 0)
    g = run(gate, est, dry)
    assert gate.reopened == 1 and np.all(g[ms(100):] == 1.0)
    # between the two thresholds the state never changes: closed stays closed
    est = db(-13.5) * dry
    assert np.allclose(run(SpeechRouteGate(cfg(), SR, 0), est, dry), G40)


def test_start_closed_until_speech_appears() -> None:
    dry = noise(2.0)
    est = db(-26) * dry  # a leaky start: DeepFilterNet3 keeps some music
    est[int(1.5 * SR):] = dry[int(1.5 * SR):]
    g = run(SpeechRouteGate(cfg(), SR, 0), est, dry)
    assert np.allclose(g[:int(1.5 * SR)], G40)
    assert np.all(g[int(1.52 * SR):] > 0.99)
    opened = run(SpeechRouteGate(cfg(start_closed=False), SR, 0), est, dry)
    assert opened[0] == 1.0  # an open start (start_closed false) begins open


def test_warmup_keeps_the_start_position() -> None:
    dry = noise(1.0)
    est = dry.copy()  # speech from the first sample
    gate = SpeechRouteGate(cfg(warmup_ms=300), SR, 0)
    gate.restart(0, warmup=True)
    g = run(gate, est, dry)
    assert np.allclose(g[:ms(299)], G40)
    assert np.all(g[ms(312):] > 0.99)
    plain = run(SpeechRouteGate(cfg(warmup_ms=300), SR, 0), est, dry)  # reset(): no warm-up
    assert np.all(plain[ms(12):] > 0.99)


def test_the_detector_sets_the_hold() -> None:
    est, dry = speech_then_music(3.0)
    unsure = run(SpeechRouteGate(cfg(), SR, 0), est, dry)
    sure = SpeechRouteGate(cfg(), SR, 0, Scripted(absent=lambda t: t >= 0.5))
    g = run(sure, est, dry)
    assert np.all(g[ms(500):ms(715)] > 0.999)  # r < -15 dB after ~173 ms, then 50 ms
    assert np.allclose(g[ms(785):], G40)  # 50 ms hold: closed ...
    assert unsure[ms(785)] > 0.999  # ... while the 150 ms hold still waits
    on = SpeechRouteGate(cfg(), SR, 0, Scripted(active=lambda t: True))  # the detector: speech
    g = run(on, est, dry)
    assert np.all(g[ms(500):ms(1660)] > 0.999)  # it keeps the gate open ...
    assert np.allclose(g[ms(1740):], G40)  # ... at most detector_override_max_ms


def test_any_chunking_gives_the_same_gains() -> None:
    est, dry = speech_then_music(3.0)
    est[int(2.0 * SR):int(2.3 * SR)] = dry[int(2.0 * SR):int(2.3 * SR)]
    source = Scripted(active=lambda t: 0.4 <= t < 1.1, absent=lambda t: t > 2.5)
    whole = SpeechRouteGate(cfg(), SR, 480, source).gains(est, dry, LABELS, 30.0)
    rng = np.random.default_rng(2)
    gate, parts, i = SpeechRouteGate(cfg(), SR, 480, source), [], 0
    while i < est.size:
        n = int(rng.integers(1, 3000))
        parts.append(gate.gains(est[i:i + n], dry[i:i + n], LABELS, 30.0))
        i += n
    assert np.allclose(np.concatenate(parts), whole, rtol=0, atol=1e-7)


def test_closed_gain_follows_the_slider_without_clicks() -> None:
    est, dry = speech_then_music()
    for att in (15.0, 30.0, 40.0):
        g = run(SpeechRouteGate(cfg(), SR, 0), est, dry, att=att)
        assert np.isclose(g[-1], 10 ** (-att / 20), rtol=1e-4)
        assert np.abs(np.diff(g)).max() < 0.25  # ramped sample by sample in dB (open: 5 ms)


def test_disabled_or_nothing_selected_stays_open() -> None:
    est, dry = speech_then_music()
    assert np.all(SpeechRouteGate(cfg(), SR, 0).gains(est, dry, frozenset(), 40.0) == 1.0)
    assert np.all(run(SpeechRouteGate(cfg(enabled=False), SR, 0), est, dry) == 1.0)


def test_quiet_input_cannot_open_it() -> None:
    dry = noise(1.0, level=1e-4)  # -80 dBFS: below min_level_dbfs
    assert np.allclose(run(SpeechRouteGate(cfg(), SR, 0), dry.copy(), dry), G40)


def test_restart_moves_the_detector_clock() -> None:
    seen: list[float] = []

    class Clock(Scripted):
        def active(self, labels, centre_s, now_s):
            seen.append(centre_s)
            return False

    est, dry = speech_then_music(0.5, switch=0.0)
    gate = SpeechRouteGate(cfg(start_closed=False), SR, 4800, Clock())
    gate.restart(at_sample=SR, warmup=False)  # the route started one second into the stream
    gate.gains(est, dry, LABELS, 30.0)
    assert seen and np.isclose(seen[0], (SR + 50.5 * 48 - 4800) / SR, atol=0.011)
