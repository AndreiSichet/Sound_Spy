"""Detection: window labels, model head/zero-shot, metrics, checkpoints, live Detector."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import torch

from soundspy.core.audio import AudioBuffer
from soundspy.core.events import DetectionFrame
from soundspy.detection import metrics as M
from soundspy.detection.inference import (
    DetectionCore,
    Detector,
    DetectorUnavailable,
    Smoother,
    build_detector,
    detect_timeline,
    load_core,
    window_ends,
)
from soundspy.detection.model import (
    AudioSetRows,
    DetectorNet,
    ModelSpec,
    load_checkpoint,
    save_checkpoint,
)
from soundspy.detection.windows import window_labels

LABELS = ["speech", "music", "alarm"]


# ---- window labels ---------------------------------------------------------------------------

def test_window_labels_need_min_active_time() -> None:
    rate = 100.0
    act = np.zeros((600, 3), np.float32)
    act[100:110, 0] = 1  # speech active 0.10 s at 1.00-1.10 s
    act[100:109, 1] = 1  # music only 0.09 s
    assert window_labels(act, rate, 0.5, 1.0, 0.1).tolist() == [1.0, 0.0, 0.0]
    assert window_labels(act, rate, 1.05, 1.0, 0.1).tolist() == [0.0, 0.0, 0.0]  # 0.05 s left
    assert window_labels(act, rate, 0.5, 1.0, 0.0).tolist() == [1.0, 1.0, 0.0]


def test_background_only_window_is_all_zero() -> None:
    assert window_labels(np.zeros((600, 3), np.float32), 100, 2.0, 1.0, 0.1).sum() == 0


# ---- model -----------------------------------------------------------------------------------

def stub_spec(context_s: float = 1.0) -> ModelSpec:
    rows = AudioSetRows({"speech": [0], "music": [1, 2], "alarm": []},
                        {"speech": [0, 5], "music": [1], "alarm": []})
    return ModelSpec("stub", LABELS, 16_000, 1.0, rows, {}, context_s)


def test_head_initialised_from_mean_audioset_rows() -> None:
    model = DetectorNet(stub_spec())
    alarm_before = model.head.weight[2].clone()
    used = model.init_head_from_audioset()
    src = model.backbone.classifier[5]
    torch.testing.assert_close(model.head.weight[0], src.weight[0])
    torch.testing.assert_close(model.head.weight[1], src.weight[[1, 2]].mean(dim=0))
    torch.testing.assert_close(model.head.bias[1], src.bias[[1, 2]].mean())
    torch.testing.assert_close(model.head.weight[2], alarm_before)  # no rows -> stays random
    assert used == {"speech": 1, "music": 2, "alarm": 0}


def test_zero_shot_is_max_over_mapped_audioset_rows() -> None:
    model = DetectorNet(stub_spec()).eval()
    x = torch.randn(2, 16_000) * 0.1
    with torch.no_grad():
        probs = torch.sigmoid(model.audioset_logits(x))
        zs = model.zero_shot(x)
    torch.testing.assert_close(zs[:, 0], probs[:, [0, 5]].max(dim=1).values)
    torch.testing.assert_close(zs[:, 1], probs[:, 1])
    assert torch.all(zs[:, 2] == 0)


def test_short_windows_are_padded_to_the_context() -> None:
    model = DetectorNet(stub_spec(context_s=2.0))
    assert model.pad(torch.zeros(1, 16_000)).shape == (1, 32_000)
    assert model.pad(torch.zeros(1, 40_000)).shape == (1, 40_000)


def test_checkpoint_round_trip(tmp_path) -> None:
    model = DetectorNet(stub_spec()).eval()
    save_checkpoint(tmp_path / "best.pt", model, {"epoch": 3})
    loaded, payload = load_checkpoint(tmp_path / "best.pt")
    assert payload["epoch"] == 3 and loaded.spec.labels == LABELS
    x = torch.randn(1, 16_000)
    with torch.no_grad():
        torch.testing.assert_close(model(x), loaded(x))


# ---- metrics ---------------------------------------------------------------------------------

def test_metrics_match_hand_computed_values() -> None:
    y = np.array([[1, 0], [0, 0], [1, 1], [0, 1]], np.float32)
    s = np.array([[0.9, 0.2], [0.8, 0.1], [0.7, 0.9], [0.1, 0.3]], np.float32)
    # class 0: ranking 0.9(+) 0.8(-) 0.7(+) 0.1(-) -> AP = (1/1 + 2/3) / 2
    assert M.average_precision(y.T[0], s.T[0]) == pytest.approx((1 + 2 / 3) / 2)
    assert M.roc_auc(y.T[1], s.T[1]) == pytest.approx(1.0)
    p, r, f = M.prf_at(y.T[0], s.T[0], 0.75)
    assert (p, r, f) == pytest.approx((0.5, 0.5, 0.5))
    ev = M.evaluate(y, s, ["a", "b"], {"a": 0.75, "b": 0.25})
    assert ev["map"] == pytest.approx(((1 + 2 / 3) / 2 + 1.0) / 2)
    assert ev["per_class"]["b"]["f1"] == pytest.approx(1.0)


def test_threshold_tuning_maximises_f1() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.3).astype(np.float32)
    s = np.clip(y * 0.4 + rng.random(400) * 0.6, 0, 1)
    th, f1 = M.tune_threshold(y, s)
    grid = max(M.prf_at(y, s, t)[2] for t in np.linspace(0, 1, 201))
    assert f1 >= grid - 1e-9
    assert M.prf_at(y, s, th)[2] == pytest.approx(f1)


def test_false_positive_cooccurrence() -> None:
    y = np.array([[1, 0], [0, 0]], np.float32)
    s = np.array([[0.9, 0.9], [0.1, 0.9]], np.float32)
    table = M.false_positive_cooccurrence(y, s, {"a": 0.5, "b": 0.5}, ["a", "b"])
    assert table == {"a": {}, "b": {"a": 1, "background": 1}}


# ---- smoothing -------------------------------------------------------------------------------

def test_ema_and_hysteresis() -> None:
    sm = Smoother(["a"], {"a": 0.6}, ema_alpha=0.5, hysteresis=0.1)
    seq = [0.9, 0.9, 0.5, 0.5, 0.4, 0.0]
    out = [sm.update(np.array([p])) for p in seq]
    probs = [round(float(p[0]), 4) for p, _ in out]
    assert probs == [0.9, 0.9, 0.7, 0.6, 0.5, 0.25]
    active = [bool(a[0]) for _, a in out]
    # on at >= 0.6; stays on until below 0.5 (0.6 - 0.1)
    assert active == [True, True, True, True, True, False]


# ---- live Detector ---------------------------------------------------------------------------

class FakeCore:
    """Stands in for DetectionCore: records the windows it gets, optionally slowly."""

    def __init__(self, delay_s: float = 0.0, sample_rate: int = 16_000) -> None:
        self.labels = list(LABELS)
        self.thresholds = {lab: 0.5 for lab in LABELS}
        self.sample_rate, self.window_s = sample_rate, 1.0
        self.window_samples = sample_rate
        self.device, self.name = "cpu", "fake"
        self.delay_s = delay_s
        self.windows: list[np.ndarray] = []
        self.release = threading.Event()

    def predict(self, windows: np.ndarray) -> np.ndarray:
        self.windows.append(windows.copy())
        if self.delay_s:
            self.release.wait(self.delay_s)
        return np.tile(np.array([0.9, 0.1, 0.6], np.float32), (windows.shape[0], 1))

    def warm_up(self) -> None:
        pass


def feed(det: Detector, audio: np.ndarray, sr: int, chunk: int, settle: bool = True):
    events = []
    for i in range(0, audio.shape[0], chunk):
        events += det.observe(AudioBuffer(audio[i:i + chunk].reshape(-1, 1), sr))
        if settle:
            deadline = time.monotonic() + 2.0
            while det._busy or det._job is not None:  # let the worker finish (no skips)
                if time.monotonic() > deadline:
                    break
                time.sleep(0.001)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and (det._busy or det._job is not None):
        time.sleep(0.001)
    return events + det.observe(AudioBuffer(np.zeros((0, 1), np.float32), sr))


@pytest.mark.parametrize("chunk", [160, 960, 4000])
def test_hop_schedule_does_not_depend_on_chunk_size(chunk: int, no_leaked_threads) -> None:
    sr = 16_000
    det = Detector(FakeCore(), hop_s=0.25, ema_alpha=1.0, hysteresis=0.1)
    det.configure(sr, 1)
    try:
        audio = np.arange(3 * sr, dtype=np.float32) / sr  # 3 s ramp
        events = feed(det, audio, sr, chunk)
    finally:
        det.close()
    assert [e.t for e in events] == [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    assert all(isinstance(e, DetectionFrame) for e in events)
    assert events[0].active == ["speech", "alarm"]
    # each window is exactly the last second before its time stamp
    last = det.core.windows[-1].ravel()
    np.testing.assert_allclose(last[[0, -1]], [2.0, (3 * sr - 1) / sr], atol=1e-6)
    assert det.skipped == 0


def test_busy_worker_skips_windows_instead_of_blocking(no_leaked_threads) -> None:
    sr = 16_000
    core = FakeCore(delay_s=5.0)
    det = Detector(core, hop_s=0.25, ema_alpha=1.0, hysteresis=0.1)
    det.configure(sr, 1)
    try:
        started = time.perf_counter()
        for _ in range(3 * 25):  # 3 s of 40 ms chunks, all while the model "thinks"
            det.observe(AudioBuffer(np.zeros((640, 1), np.float32), sr))
        assert time.perf_counter() - started < 1.0  # the audio path never waited
        assert det.skipped >= 7  # 9 windows due, one accepted (maybe one queued)
        status = det.status_info()
        assert status["skipped"] == det.skipped and status["model"] == "fake"
    finally:
        core.release.set()
        det.close()


def test_warm_up_runs_on_the_inference_thread_before_audio(no_leaked_threads) -> None:
    # CUDA sets up per-thread state on a thread's first call (~3 s on the laptop); a warm-up on
    # the session thread left that to the first live window and 12 hops were skipped.
    class SlowWarmUp(FakeCore):
        def warm_up(self) -> None:
            time.sleep(0.2)
            self.warmed_on = threading.current_thread().name

    core = SlowWarmUp()
    det = Detector(core, hop_s=0.25, ema_alpha=1.0, hysteresis=0.1)
    det.configure(16_000, 1)
    try:
        assert getattr(core, "warmed_on", None) == "soundspy-detector"  # done before returning
    finally:
        det.close()


def test_stereo_input_is_mixed_down(no_leaked_threads) -> None:
    sr = 16_000
    det = Detector(FakeCore(), hop_s=0.5, ema_alpha=1.0, hysteresis=0.1)
    det.configure(sr, 2)
    try:
        x = np.stack([np.ones(sr, np.float32), -np.ones(sr, np.float32)], axis=1)
        det.observe(AudioBuffer(x, sr))
        deadline = time.monotonic() + 2
        while not det.core.windows and time.monotonic() < deadline:
            time.sleep(0.001)
    finally:
        det.close()
    np.testing.assert_allclose(det.core.windows[0], 0.0, atol=1e-6)


def test_window_ends_for_offline_detection() -> None:
    assert window_ends(32_000, 16_000, 1.0, 0.25) == [16_000, 20_000, 24_000, 28_000, 32_000]
    assert window_ends(8_000, 16_000, 1.0, 0.25) == [8_000]  # shorter than one window
    assert window_ends(0, 16_000, 1.0, 0.25) == []


def test_offline_timeline_with_stub_model(stub_experiment) -> None:
    config, _exp = stub_experiment
    core = load_core(config.detection.checkpoint, "cpu")
    assert isinstance(core, DetectionCore) and core.sample_rate == 16_000
    audio = AudioBuffer(np.random.default_rng(0).normal(0, 0.1, (48_000 * 3, 2)).astype(
        np.float32), 48_000)
    timeline = detect_timeline(core, audio, hop_s=0.5, ema_alpha=0.5, hysteresis=0.1)
    assert timeline.duration_s == 3.0 and timeline.times[0] == 1.0 and timeline.times[-1] == 3.0
    assert set(timeline.probabilities) == set(core.labels)
    assert all(len(v) == len(timeline.times) for v in timeline.probabilities.values())
    for seg in timeline.segments:
        assert 0.0 <= seg.t_start < seg.t_end <= 3.0


def test_segments_after_a_short_gap_are_merged() -> None:
    from soundspy.detection.inference import _segments

    # on 1.0-2.0, off at 2.25 only, on again 2.5-3.0: the second run would start at 1.5 (its
    # window start), inside the first one (seen on a real recording: 21.25-26.75 + 26.25-32.75)
    times = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5,
             4.75, 5.0, 5.25]
    on = [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    probs = [[0.8 if a else 0.1] for a in on]
    segs = _segments(["speech"], times, probs, [[bool(a)] for a in on], 1.0)
    assert [(s.t_start, s.t_end) for s in segs] == [(0.0, 3.0), (4.25, 5.25)]
    assert segs[0].probability == 0.8


def test_missing_checkpoint_is_a_clear_reason(config, tmp_path) -> None:
    import dataclasses

    cfg = dataclasses.replace(config, detection=dataclasses.replace(
        config.detection, checkpoint=tmp_path / "nope" / "checkpoints" / "best.pt"))
    with pytest.raises(DetectorUnavailable, match="soundspy train detection"):
        build_detector(cfg)
    cfg = dataclasses.replace(config, detection=dataclasses.replace(
        config.detection, checkpoint=None))
    with pytest.raises(DetectorUnavailable, match="no detection checkpoint"):
        build_detector(cfg)


def test_chain_without_model_starts_with_a_warning(config, tmp_path) -> None:
    import dataclasses

    from soundspy.engine.factory import build_chain

    cfg = dataclasses.replace(config, detection=dataclasses.replace(
        config.detection, checkpoint=tmp_path / "missing.pt"))
    chain = build_chain("semantic", cfg, cfg.labels)
    assert not chain.has("detector")
    assert chain.warnings and chain.warnings[0][0] == "detector_unavailable"


def test_chain_with_stub_model_has_a_detector(stub_experiment, no_leaked_threads) -> None:
    from soundspy.engine.factory import build_chain

    config, _exp = stub_experiment
    chain = build_chain("semantic", config, config.labels)
    assert chain.has("detector") and chain.warnings == []
    chain.configure(48_000, 2)
    chain.close()
