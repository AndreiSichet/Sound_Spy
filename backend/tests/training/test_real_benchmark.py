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


def test_room_part_says_quiet_is_fine_and_warns_only_for_a_dead_input(bench_config,
                                                                      capsys) -> None:
    """Step 07d: room tone is meant to be quiet; only a dead input (< -90 dBFS) warns."""
    def mic(level):
        def take(seconds: float) -> AudioBuffer:
            x = (level * np.random.default_rng(4).standard_normal(SR)).astype(np.float32)
            return AudioBuffer(np.stack([x, x], axis=1), SR, {"device": "fake"})
        return take

    cmd.record_benchmark(bench_config, mic(1e-3), seconds={"room": 1.0}, input_fn=keys(""),
                         countdown_s=0, session_id="quiet")  # -60 dBFS: a quiet room
    out = capsys.readouterr().out
    assert "Room tone: -60.0 dBFS" in out and "very quiet" not in out and "WARNING" not in out
    cmd.record_benchmark(bench_config, mic(1e-6), seconds={"room": 1.0}, input_fn=keys(""),
                         countdown_s=0, session_id="dead")  # -120 dBFS: nothing is connected
    out = capsys.readouterr().out
    assert "WARNING: all channels below -90 dBFS" in out and "dead input" in out


def _with_pauses() -> tuple[np.ndarray, list[float]]:
    """Room tone (-64 dBFS) with words between pauses of 0.3, 1.0 and 2.5 s; silence before
    the first word (0.4 s) and after the last (0.2 s) is not a pause."""
    rng = np.random.default_rng(9)
    parts, pauses = [np.zeros(int(0.4 * SR))], [0.3, 1.0, 2.5]
    for pause in (*pauses, None):
        parts.append(0.1 * rng.standard_normal(int(0.5 * SR)))
        parts.append(np.zeros(int((pause or 0.2) * SR)))
    x = np.concatenate(parts)
    room = 10 ** (-64 / 20) * rng.standard_normal(x.size)
    return (x + room).astype(np.float32), pauses


def test_pauses_are_found_grouped_and_pooled() -> None:
    speech, lengths = _with_pauses()
    found = rb.find_pauses(speech, SR, floor_db=-64.0)
    seconds = [(b - a) * rb.FRAME_MS / 1000 for a, b in found]
    assert np.allclose(seconds, lengths, atol=0.051)
    assert [rb.pause_group(s) for s in seconds] == ["brief", "short", "long"]
    music = (0.05 * np.random.default_rng(1).standard_normal(speech.size)).astype(np.float32)
    mixture = speech + music
    n = int(rb.FRAME_MS * SR / 1000)
    y = mixture.copy()
    for a, b in found:  # the system removes 40 dB in the pauses
        y[a * n:b * n] *= 0.01
    energies = rb.pause_energies(y, mixture, found, SR)
    assert energies["pause_short_n"] == 1 and energies["pause_long_n"] == 1
    row = {"kind": "mix", "ratio_db": 0.0, "music": "music",
           "40": {"system": {"si_sdr_i": 0.0, **energies}}}
    summary = rb.summarize_real([row, row], (40.0,))["40"]["mixes"]["system"]
    assert abs(summary["pause_short_db"] + 40.0) < 0.01 and summary["pause_long_n"] == 2
    # tiny energies (a gate closed in a quiet pause) are pooled unrounded, not as "silent"
    deep = {**energies, "pause_long_out": 1e-7, "pause_long_in": 1.0}
    row = {"kind": "mix", "ratio_db": 0.0, "music": "music", "40": {"system": deep}}
    summary = rb.summarize_real([row], (40.0,))["40"]["mixes"]["system"]
    assert abs(summary["pause_long_db"] + 70.0) < 0.01


def test_leak_seconds_counts_the_time_the_output_is_not_30_db_down() -> None:
    x = (0.1 * np.random.default_rng(3).standard_normal(2 * SR)).astype(np.float32)
    y = 0.001 * x  # -60 dB ...
    y[:int(0.4 * SR)] = x[:int(0.4 * SR)]  # ... except a 0.4 s leak at the start
    assert abs(rb.leak_seconds(y, x, SR) - 0.4) < 0.021
    assert rb.leak_seconds(0.001 * x, x, SR) == 0.0


def test_stream_start_cases() -> None:
    rng = np.random.default_rng(5)
    speech, _ = _with_pauses()
    music = (0.05 * rng.standard_normal(12 * SR)).astype(np.float32)
    takes = {"speech": [rb.Take("speech", Path("speech_01.wav"), speech, SR)],
             "music": [rb.Take("music", Path("music_01.wav"), music, SR)], "music_pop": [],
             "room": [rb.Take("room", Path("room_01.wav"),
                              (10 ** (-64 / 20) * rng.standard_normal(12 * SR)
                               ).astype(np.float32), SR)]}
    cases = rb.build_cases(takes, ["speech"])
    starts = [c for c in cases if c.kind == "music_start"]
    assert [c.meta["offset_s"] for c in starts] == [0.0, 6.0]  # every 6 s, 5 s long
    assert all(c.mixture.size == int(rb.START_SECONDS * SR) for c in starts)
    words = [c for c in cases if c.kind == "speech_start"]
    assert len(words) == 2  # the words after the 1.0 s and the 2.5 s pause
    mixes = [c for c in cases if c.kind == "mix"]
    assert all(len(c.meta["pauses"]) == 3 for c in mixes)


def test_two_sessions_trimmed_with_the_room_tone_repeated(bench_config) -> None:
    """Step 07d: a speech take recorded in its own session joins another session's music and
    room (``A+B``); ``max_seconds`` leaves out the end of every take; a room take shorter than
    the mixture is repeated."""
    rng = np.random.default_rng(6)

    def mic(parts):
        it = iter(parts)

        def take(seconds: float) -> AudioBuffer:
            x = np.asarray(next(it), dtype=np.float32)
            return AudioBuffer(np.stack([x, x], axis=1), SR, {"device": "fake"})
        return take

    speech, _ = _with_pauses()  # 5.2 s
    cmd.record_benchmark(bench_config, mic([0.05 * rng.standard_normal(6 * SR),
                                            1e-4 * rng.standard_normal(2 * SR)]),
                         seconds={"music": 6.0, "room": 2.0}, input_fn=keys("", ""),
                         countdown_s=0, session_id="20261119-000000-aaaa")
    cmd.record_benchmark(bench_config, mic([speech]), seconds={"speech": 5.2},
                         input_fn=keys(""), countdown_s=0, session_id="20261119-000001-bbbb")
    root = cmd.benchmark_root(bench_config)
    folders = rb.find_sessions(root, "20261119-000000-aaaa+20261119-000001-bbbb")
    takes = rb.load_takes(folders, max_seconds=5.0)
    assert takes["speech"][0].label == "bbbb_speech_01"
    assert all(t.samples.size == 5 * SR for parts in takes.values() for t in parts
               if t.part != "room")
    cases = rb.build_cases(takes, ["speech"], starts=False)
    mix = next(c for c in cases if c.kind == "mix")
    assert mix.mixture.size == 5 * SR  # not cut to the 2 s room take
    assert mix.name.startswith("bbbb_speech_01+aaaa_music_01")

    def bypass(mixture, sr, labels, att):
        return {"mixture": mixture}

    result = rb.evaluate_real(bench_config, session="20261119-000000-aaaa+20261119-000001-bbbb",
                              systems_fn=bypass, max_seconds=5.0, role="held-out",
                              name="ext_x", progress=lambda m: None)
    assert result["session"] == "20261119-000000-aaaa+20261119-000001-bbbb"
    assert result["role"] == "held-out" and "held-out session" in rb.format_real(result)
    assert np.allclose(rb.room_for(np.arange(3.0), 7), [0, 1, 2, 0, 1, 2, 0])


def test_no_session_yet_is_a_user_message(bench_config) -> None:
    with pytest.raises(rb.RealBenchmarkError, match="soundspy record-benchmark"):
        rb.evaluate_real(bench_config, systems_fn=lambda *a: {}, progress=lambda m: None)
