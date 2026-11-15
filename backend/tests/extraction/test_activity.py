"""Activity sources: shared live state, timelines (centred / causal), oracle."""

from __future__ import annotations

import time

import numpy as np
import pytest

from soundspy.core.audio import AudioBuffer
from soundspy.detection.inference import Detector, probabilities_at
from soundspy.engine.detection_state import DetectionState
from soundspy.extraction.activity import (
    LiveDetectorActivity,
    OracleActivity,
    TimelineActivity,
    causal_timeline,
    centred_timeline,
    timeline_probabilities,
)

SR = 16_000
LABELS = ["speech", "music"]


class LoudnessCore:
    """Fake model: "speech" probability follows the window's loudness, "music" is never there."""

    def __init__(self) -> None:
        self.labels = list(LABELS)
        self.thresholds = {"speech": 0.5, "music": 0.5}
        self.sample_rate, self.window_s, self.window_samples = SR, 1.0, SR
        self.device, self.name = "cpu", "fake"

    def predict(self, windows: np.ndarray) -> np.ndarray:
        loud = np.sqrt(np.mean(np.square(windows), axis=-1))
        speech = np.clip(loud / 0.2, 0.0, 1.0)
        return np.stack([speech, np.zeros_like(speech)], axis=-1).astype(np.float32)

    def warm_up(self) -> None:
        pass


@pytest.fixture
def audio() -> AudioBuffer:
    """6 s: silence, a tone from 2.0 to 4.0 s, silence."""
    t = np.arange(6 * SR) / SR
    x = 0.5 * np.sin(2 * np.pi * 440 * t) * ((t >= 2.0) & (t < 4.0))
    return AudioBuffer(x.astype(np.float32).reshape(-1, 1), SR)


def test_detection_state_and_live_activity() -> None:
    state = DetectionState()
    live = LiveDetectorActivity(state, lag_s=0.4)
    assert not state.snapshot().available and live.lag_s == 0.4
    assert live.active({"speech"}, 0.0, 0.0) is False  # nothing known yet
    state.update(1.25, {"speech": 0.9, "music": 0.1}, ["speech"])
    snapshot = state.snapshot()
    assert snapshot.available and snapshot.t == 1.25 and snapshot.active == {"speech"}
    assert live.active({"speech", "alarm"}, 5.0, 5.0) is True  # the time does not matter live
    assert live.active({"music"}, 5.0, 5.0) is False
    assert live.active(set(), 5.0, 5.0) is False
    state.update(1.5, {"speech": 0.2, "music": 0.1}, [])
    assert live.active({"speech"}, 5.0, 5.0) is False and state.updates == 2
    state.reset()
    assert not state.snapshot().available and state.updates == 0


def test_timeline_is_a_step_function_on_its_clock() -> None:
    active = np.array([[0, 0], [1, 0], [1, 1], [0, 0]], dtype=bool)
    centre = TimelineActivity(LABELS, [0.0, 1.0, 2.0, 3.0], active)
    assert centre.active({"speech"}, 0.5, 9.0) is False
    assert centre.active({"speech"}, 1.0, 0.0) is True  # uses centre_s, ignores now_s
    assert centre.active({"music"}, 1.9, 0.0) is False and centre.active({"music"}, 2.0, 0.0)
    assert centre.active({"speech"}, 3.5, 0.0) is False
    assert centre.active({"speech"}, -1.0, 0.0) is False  # before the first decision
    assert centre.active({"alarm"}, 2.0, 0.0) is False  # a class the model does not know
    assert centre.segments("speech") == [(1.0, 3.0)] and centre.segments("music") == [(2.0, 3.0)]
    now = TimelineActivity(LABELS, [0.0, 1.0, 2.0, 3.0], active, clock="now", lag_s=0.5)
    assert now.active({"speech"}, 1.5, 0.5) is False and now.active({"speech"}, 0.5, 1.5) is True
    assert now.lag_s == 0.5
    with pytest.raises(ValueError):
        TimelineActivity(LABELS, [0.0, 1.0], active[:2], clock="later")
    with pytest.raises(ValueError):
        TimelineActivity(LABELS, [1.0, 0.0], active[:2])
    open_end = TimelineActivity(LABELS, [0.0, 0.25], np.array([[0, 0], [1, 0]], dtype=bool))
    assert open_end.segments("speech") == [(0.25, 0.5)]


def test_oracle_reads_the_true_activity() -> None:
    truth = np.zeros((40, 2), dtype=np.float32)  # 10 frames per second
    truth[10:20, 0] = 1.0
    oracle = OracleActivity(LABELS, truth, rate_hz=10.0)
    assert oracle.lag_s == 0.0
    assert oracle.active({"speech"}, 0.95, 0.0) is False
    assert oracle.active({"speech"}, 1.0, 0.0) is True and oracle.active({"speech"}, 1.99, 0.0)
    assert oracle.active({"speech"}, 2.0, 0.0) is False
    assert oracle.active({"music"}, 1.5, 0.0) is False
    assert oracle.active({"speech"}, -0.1, 0.0) is False and not oracle.active({"speech"}, 9, 0)
    with pytest.raises(ValueError):
        OracleActivity(LABELS, np.zeros((4, 3)), 10.0)


def test_probabilities_at_pads_with_silence_outside_the_file(audio) -> None:
    core = LoudnessCore()
    probs = probabilities_at(core, audio, [0.25, 2.1, 3.0, 4.9, 6.5, 9.0])
    assert probs.shape == (6, 2)
    speech = probs.T[0]
    assert speech[0] == 0.0 and speech[5] == 0.0  # before the tone / entirely after the file
    assert speech[4] == 0.0  # 5.5-6.5 s: silence, half of it beyond the end of the file
    assert speech[2] == pytest.approx(1.0)  # the window 2.0-3.0 is all tone
    assert 0.0 < speech[1] < speech[2] and 0.0 < speech[3] < speech[2]  # 0.1 s of tone
    assert probabilities_at(core, audio, []).shape == (0, 2)
    stereo = AudioBuffer(np.repeat(audio.samples, 2, axis=1), SR)
    np.testing.assert_allclose(probabilities_at(core, stereo, [3.0]), probs[2:3], atol=1e-6)


def run_live_detector(core, audio: AudioBuffer, chunk: int = 320) -> list:
    """The real threaded ``Detector`` fed chunk by chunk (waiting for each result)."""
    state = DetectionState()
    detector = Detector(core, hop_s=0.25, ema_alpha=0.5, hysteresis=0.1, state=state)
    detector.configure(audio.sample_rate, 1)
    frames = []
    try:
        for i in range(0, audio.num_frames, chunk):
            frames += detector.observe(AudioBuffer(audio.samples[i:i + chunk], audio.sample_rate))
            deadline = time.monotonic() + 2.0
            while (detector._busy or detector._job is not None) and time.monotonic() < deadline:
                time.sleep(0.0005)
        frames += detector.observe(AudioBuffer(np.zeros((0, 1), np.float32), audio.sample_rate))
    finally:
        detector.close()
    assert state.snapshot().t == frames[-1].t  # the detector wrote the shared state
    assert state.snapshot().active == set(frames[-1].active)
    return frames


def test_causal_timeline_reproduces_the_live_detector_lag(audio, no_leaked_threads) -> None:
    core = LoudnessCore()
    live = run_live_detector(core, audio)
    simulated = causal_timeline(core, audio, hop_s=0.25, ema_alpha=0.5, hysteresis=0.1,
                                delay_s=0.0, lag_s=0.5)
    assert simulated.clock == "now" and simulated.lag_s == 0.5
    assert [f.t for f in live] == [round(1.0 + 0.25 * i, 3) for i in range(21)]
    for frame in live:  # same decision as the live detector for every window end
        assert simulated.active({"speech"}, 0.0, frame.t) == ("speech" in frame.active), frame.t
    # the lag itself: on a little after the tone starts (2.0 s), off well after it ends (4.0 s)
    (start, end), = simulated.segments("speech")
    assert 2.0 < start <= 2.75 and 4.5 <= end <= 5.25
    delayed = causal_timeline(core, audio, hop_s=0.25, ema_alpha=0.5, hysteresis=0.1,
                              delay_s=0.05, lag_s=0.5)
    assert delayed.segments("speech")[0][0] == pytest.approx(start + 0.05)
    assert simulated.segments("music") == []


def test_centred_timeline_has_no_lag(audio) -> None:
    core = LoudnessCore()
    centred = centred_timeline(core, audio, hop_s=0.25, ema_alpha=0.5, hysteresis=0.1)
    causal = causal_timeline(core, audio, hop_s=0.25, ema_alpha=0.5, hysteresis=0.1,
                             delay_s=0.0, lag_s=0.5)
    assert centred.clock == "centre" and centred.lag_s == 0.0
    (start, end), = centred.segments("speech")
    c_start, c_end = causal.segments("speech")[0]
    assert start <= 2.0 and end >= 4.0  # the whole tone is inside the active stretch
    assert start < c_start and end < c_end  # earlier than live on both sides
    # symmetric around the tone (2.0-4.0 s), within one hop
    assert abs((2.0 - start) - (end - 4.0)) <= 0.25 + 1e-9
    assert centred.active({"speech"}, 3.0, 0.0) and not centred.active({"speech"}, 0.5, 0.0)


def test_one_model_pass_serves_both_timelines(audio) -> None:
    class Counting(LoudnessCore):
        windows = 0

        def predict(self, windows: np.ndarray) -> np.ndarray:
            Counting.windows += windows.shape[0]
            return super().predict(windows)

    core = Counting()
    settings = dict(hop_s=0.25, ema_alpha=0.5, hysteresis=0.1)
    separate_c = centred_timeline(core, audio, **settings)
    separate_l = causal_timeline(core, audio, delay_s=0.05, lag_s=0.5, **settings)
    separately = Counting.windows
    Counting.windows = 0
    raw_c, raw_l = timeline_probabilities(core, audio, 0.25)
    assert Counting.windows == 26 < separately == 49  # the union instead of both sets
    shared_c = centred_timeline(core, audio, raw=raw_c, **settings)
    shared_l = causal_timeline(core, audio, delay_s=0.05, lag_s=0.5, raw=raw_l, **settings)
    assert Counting.windows == 26  # nothing more was evaluated
    np.testing.assert_array_equal(shared_c.matrix, separate_c.matrix)
    np.testing.assert_array_equal(shared_l.matrix, separate_l.matrix)
    np.testing.assert_allclose(shared_c.times, separate_c.times)
