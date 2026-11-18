"""The real-room benchmark (step 07c): ``record-benchmark`` with the fake microphone, the case
builder (mixing ratios, room tone), the scoring (interruption rate) and the whole run with a
stand-in for the models. No models, no hardware."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from soundspy.cli import record_benchmark_cmd as cmd
from soundspy.core.audio import AudioBuffer
from soundspy.extraction import metrics as M
from soundspy.training import real_benchmark as rb

SR = 48000


@pytest.fixture
def bench_config(config, tmp_path: Path):
    return dataclasses.replace(config, paths={
        **config.paths, "raw": tmp_path / "data" / "raw", "outputs": tmp_path / "outputs",
        "experiments": tmp_path / "experiments"})


def keys(*presses: str):
    it: Iterator[str] = iter(presses)
    return lambda _prompt: next(it)


def speechlike(seconds: float, seed: int = 0) -> np.ndarray:
    """Bursts (words) with pauses."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    return (0.2 * rng.standard_normal(t.size) * (t % 0.6 < 0.35)).astype(np.float32)


def fake_mic():
    parts = iter([speechlike(1.0), 0.05 * np.random.default_rng(1).standard_normal(SR),
                  1e-4 * np.random.default_rng(2).standard_normal(SR)])

    def take(seconds: float) -> AudioBuffer:
        x = np.asarray(next(parts), dtype=np.float32)
        return AudioBuffer(np.stack([x, x], axis=1), SR, {"device": "fake"})

    return take


def test_record_benchmark_writes_one_session(bench_config) -> None:
    kept = cmd.record_benchmark(bench_config, fake_mic(),
                                seconds={"speech": 1.0, "music": 1.0, "room": 1.0},
                                input_fn=keys("", "", ""), countdown_s=0,
                                session_id="sess-7", notes="living room")
    folder = cmd.benchmark_root(bench_config) / "sess-7"
    assert [p.name for p in kept] == ["speech_01.wav", "music_01.wav", "room_01.wav"]
    meta = json.loads((folder / "music_01.json").read_text(encoding="utf-8"))
    assert meta["part"] == "music" and meta["benchmark_session"] == "sess-7"
    assert meta["notes"] == "living room"


def test_record_benchmark_redo_skip_quit(bench_config) -> None:
    kept = cmd.record_benchmark(bench_config, fake_mic(), seconds={"speech": 1.0, "music": 1.0},
                                input_fn=keys("", "r", "s", "q"), countdown_s=0,
                                session_id="s2")
    assert [p.name for p in kept] == ["speech_01.wav"]  # redone once, music skipped
    assert not (cmd.benchmark_root(bench_config) / "s2" / "music_01.wav").exists()


def _takes() -> dict[str, list[rb.Take]]:
    rng = np.random.default_rng(5)
    speech = speechlike(3.0)
    music = (0.05 * rng.standard_normal(4 * SR)).astype(np.float32)
    room = (1e-4 * rng.standard_normal(3 * SR)).astype(np.float32)
    return {"speech": [rb.Take("speech", Path("speech_01.wav"), speech, SR)],
            "music": [rb.Take("music", Path("music_01.wav"), music, SR)],
            "music_pop": [], "room": [rb.Take("room", Path("room_01.wav"), room, SR)]}


def test_cases_mix_at_the_requested_ratios() -> None:
    takes = _takes()
    cases = rb.build_cases(takes, ["speech"])
    mixes = [c for c in cases if c.kind == "mix"]
    assert [c.ratio_db for c in mixes] == list(rb.RATIOS_DB)
    speech = takes["speech"][0].samples
    room = takes["room"][0].samples
    for case in mixes:
        music = case.mixture - case.reference - room[:case.mixture.size]
        ratio = 10 * np.log10(rb.active_power(speech, SR) / np.mean(music.astype(np.float64) ** 2))
        assert abs(ratio - case.ratio_db) < 0.05
        assert np.array_equal(case.reference, speech[:case.mixture.size])
    kinds = [c.kind for c in cases]
    assert kinds.count("music_only") == 1 and kinds.count("speech_only") == 1
    assert all(c.labels == frozenset({"speech"}) for c in cases)


def test_cases_need_speech_and_music() -> None:
    takes = _takes()
    takes["music"] = []
    with pytest.raises(rb.RealBenchmarkError, match="speech and one music"):
        rb.build_cases(takes, ["speech"])


def test_interruption_rate_counts_turned_down_speech_frames() -> None:
    x = speechlike(3.0)
    assert M.interruption_rate(x, x, SR) == 0.0
    y = x.copy()
    y[:SR] *= 0.1  # the first second -20 dB: about a third of the speech frames
    rate = M.interruption_rate(y, x, SR)
    assert 0.25 < rate < 0.45
    assert M.interruption_rate(np.zeros(100), np.zeros(100), SR) is None


def test_scoring_and_the_whole_run(bench_config) -> None:
    folder = cmd.benchmark_root(bench_config) / "20261118-120000-abcd"
    cmd.record_benchmark(bench_config, fake_mic(),
                         seconds={"speech": 1.0, "music": 1.0, "room": 1.0},
                         input_fn=keys("", "", ""), countdown_s=0, session_id=folder.name)

    def bypass(mixture, sr, labels, att):  # stands in for the models: the input, and silence
        return {"mixture": mixture, "hybrid_offline": np.zeros_like(mixture)}

    result = rb.evaluate_real(bench_config, systems_fn=bypass, progress=lambda m: None)
    assert result["session"] == folder.name and Path(result["path"]).is_file()
    summary = result["summary"]["40"]
    assert summary["speech_only"]["mixture"]["interruption"] == 0.0
    assert summary["speech_only"]["hybrid_offline"]["interruption"] == 1.0
    assert summary["music_only"]["mixture"]["suppression_db"] == 0.0
    assert summary["mix +0 dB"]["mixture"]["si_sdr_i"] == 0.0
    assert "Real-room benchmark" in rb.format_real(result)
    assert (Path(result["examples_dir"])).is_dir()


def test_no_session_yet_is_a_user_message(bench_config) -> None:
    with pytest.raises(rb.RealBenchmarkError, match="soundspy record-benchmark"):
        rb.evaluate_real(bench_config, systems_fn=lambda *a: {}, progress=lambda m: None)
