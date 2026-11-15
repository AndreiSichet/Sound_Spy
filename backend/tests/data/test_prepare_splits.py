"""Clip index: collectors, official splits, leakage groups, activity maps, path aliases."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from soundspy.config import resolve_path
from soundspy.core.audio import AudioBuffer
from soundspy.data.activity import ActivityStore, active_mask, active_regions, rms_envelope_db
from soundspy.data.aliases import AliasError, resolve_alias, to_alias
from soundspy.data.prepare import COLUMNS, load_clip_index, prepare
from soundspy.data.splits import leaking_groups, majority_split, split_groups


def by_id(index):
    return {row.clip_id: row for row in index.itertuples(index=False)}


def test_index_columns_roles_and_splits(prepared_env: Any, kit) -> None:
    index = load_clip_index(prepared_env.config.paths["processed"])
    assert tuple(index.columns) == COLUMNS
    rows = by_id(index)
    assert len(index) == 25 + 50  # FSD50K + ESC-50
    # FSD50K official split
    assert rows["fsd50k:101"].split == "train"
    assert rows["fsd50k:201"].split == "val"
    assert rows["fsd50k:301"].split == "test"
    # proposed val, but its uploader (anna) is in train: moved to train (uploader-disjoint)
    assert rows["fsd50k:207"].split == "train"
    # roles
    assert rows["fsd50k:101"].role == "fg:speech"  # Speech + Human_voice (smeared)
    assert rows["fsd50k:103"].role == "fg:music"
    assert rows["fsd50k:106"].role == "bg"
    assert rows["fsd50k:108"].role == "excluded"  # speech + music
    assert rows["fsd50k:109"].role == "excluded"  # alarm + traffic overlap
    assert rows["esc50:1-1011-A-25"].role == "fg:footsteps"
    assert rows["esc50:1-1010-A-0"].role == "bg"
    assert rows["esc50:1-1014-A-34"].role == "excluded"  # mapped to null
    # ESC-50 folds 1-3 train, 4 val, 5 test
    assert rows["esc50:3-1030-B-0"].split == "train"
    assert rows["esc50:4-1040-A-0"].split == "val"
    assert rows["esc50:5-1050-A-0"].split == "test"
    # header facts and metadata
    assert rows["fsd50k:101"].sample_rate == kit.SR
    assert rows["fsd50k:101"].duration_s == pytest.approx(1.2)
    assert rows["fsd50k:101"].group == "uploader:anna"
    assert rows["fsd50k:101"].license == "CC-BY 3.0"
    assert list(rows["fsd50k:101"].raw_labels) == ["Speech", "Human_voice"]
    assert rows["esc50:2-1020-A-0"].group == "2-1020"
    assert rows["esc50:2-1020-A-0"].path == "esc50://audio/2-1020-A-0.wav"


def test_no_group_in_two_splits(prepared_env: Any) -> None:
    index = load_clip_index(prepared_env.config.paths["processed"])
    assert leaking_groups(index[["dataset", "group", "split"]].itertuples(index=False)) == {}


def test_leak_detection_finds_a_shared_group() -> None:
    rows = [("fsd50k", "uploader:a", "train"), ("fsd50k", "uploader:a", "test"),
            ("esc50", "1-1", "train")]
    assert leaking_groups(rows) == {("fsd50k", "uploader:a"): {"train", "test"}}


def test_custom_split_is_by_session() -> None:
    sessions = [f"s{i}" for i in range(20)]
    assignment = split_groups(sessions * 3, val_fraction=0.15, test_fraction=0.15, seed=0)
    assert set(assignment) == set(sessions)
    counts = {s: list(assignment.values()).count(s) for s in ("train", "val", "test")}
    assert counts == {"train": 14, "val": 3, "test": 3}
    assert assignment == split_groups(sessions, val_fraction=0.15, test_fraction=0.15, seed=0)
    # few sessions: training keeps at least one
    small = split_groups(["a", "b"], val_fraction=0.15, test_fraction=0.15, seed=0)
    assert "train" in small.values()


def test_second_prepare_reads_no_audio(prepared_env: Any, monkeypatch) -> None:
    import soundspy.data.activity as activity

    calls = []
    monkeypatch.setattr(activity, "analyze_file", lambda *a: calls.append(a))
    config = prepared_env.config
    summary = prepare(config, workers=1, progress=False)
    assert calls == []
    assert summary.clips == {"fsd50k": 25, "esc50": 50}
    assert summary.moved_split == 1
    store = ActivityStore.load(config.paths["processed"] / "activity" / "fsd50k.npz",
                               config.data.activity_hop_s)
    assert len(store) == 25
    # 1 FSD50K horn clip (smeared to Alarm and Vehicle) + 10 ESC-50 car_horn clips
    assert summary.overlaps[("alarm", "traffic")] == 11
    assert summary.mixed == 1  # speech + music
    assert summary.bg_with_ancestor_labels["Human_voice"] == 1


def test_parallel_workers_give_the_same_envelopes(data_env: Any) -> None:
    config = data_env.config
    prepare(config, workers=2, progress=False)
    a = ActivityStore.load(config.paths["processed"] / "activity" / "esc50.npz")
    other = dataclasses.replace(
        config, paths={**config.paths, "processed": data_env.root / "processed2"})
    prepare(other, workers=1, progress=False)
    b = ActivityStore.load(other.paths["processed"] / "activity" / "esc50.npz")
    for cid in a.clip_ids():
        np.testing.assert_array_equal(a.envelope(cid), b.envelope(cid))


def test_activity_detects_silent_head_and_tail(kit) -> None:
    x = kit.burst(3.0, kit.SR, start=1.0, stop=2.0)
    env = rms_envelope_db(AudioBuffer(x.reshape(-1, 1), kit.SR), 0.1)
    assert env.shape == (30,)
    mask = active_mask(env, 30.0)
    assert not mask[:10].any() and mask[10:20].all() and not mask[20:].any()
    assert active_regions(env, 0.1, 30.0) == [pytest.approx((1.0, 2.0))]


def test_activity_envelope_averages_channels(kit) -> None:
    x = kit.burst(1.0, kit.SR, start=0.0, stop=1.0)
    stereo = np.stack([x, x], axis=1)
    env_mono = rms_envelope_db(AudioBuffer(x.reshape(-1, 1), kit.SR), 0.1)
    env_stereo = rms_envelope_db(AudioBuffer(stereo, kit.SR), 0.1)
    np.testing.assert_allclose(env_mono, env_stereo, atol=1e-4)


def test_aliases_resolve_with_absolute_and_relative_paths(tmp_path: Path) -> None:
    alias = to_alias("fsd50k", "FSD50K.dev_audio/1.wav")
    assert alias == "fsd50k://FSD50K.dev_audio/1.wav"
    absolute = {"external": tmp_path / "D" / "external"}
    assert resolve_alias(alias, absolute) == tmp_path / "D/external/fsd50k/FSD50K.dev_audio/1.wav"
    relative = {"external": resolve_path("data/external", tmp_path / "project")}
    assert resolve_alias(alias, relative) == (
        tmp_path / "project/data/external/fsd50k/FSD50K.dev_audio/1.wav").resolve()
    assert resolve_alias("custom://speech/a.wav", {"custom": tmp_path / "c"}) == (
        tmp_path / "c/speech/a.wav")
    with pytest.raises(AliasError):
        to_alias("fsd50k", "../escape.wav")
    with pytest.raises(AliasError):
        resolve_alias("nope://x.wav", absolute)


def test_moving_the_data_only_needs_a_config_change(prepared_env: Any, kit) -> None:
    import shutil

    from soundspy.data.mixtures import ClipLibrary

    config = prepared_env.config
    moved = prepared_env.root / "elsewhere"
    shutil.move(str(config.paths["external"]), str(moved))
    paths = {**config.paths, "external": moved}
    index = load_clip_index(config.paths["processed"])
    library = ClipLibrary(index, {}, paths, threshold_db=30)
    seg = library.load_segment("fsd50k:101", 0.0, 1.0, kit.SR)
    assert seg.shape == (kit.SR,)


def test_custom_collector_reads_record_dataset_sidecars(data_env: Any, kit) -> None:
    from soundspy.io.storage import RecordingWriter

    config = data_env.config
    for i, (cls, session) in enumerate([("speech", "s1"), ("speech", "s2"),
                                        ("background", "s1"), ("speech", "s3")]):
        folder = config.paths["custom"] / cls
        x = kit.burst(1.0, kit.SR, start=0.1, stop=0.9, seed=i)
        writer = RecordingWriter(folder, kit.SR, 2, stem=f"{cls}_{i + 1:04d}",
                                 extra={"class": cls, "session_id": session})
        writer.write(AudioBuffer(np.stack([x, x], axis=1), kit.SR))
        writer.close()
    prepare(config, workers=1, progress=False)
    rows = by_id(load_clip_index(config.paths["processed"]))
    assert rows["custom:speech_0001"].role == "fg:speech"
    assert rows["custom:background_0003"].role == "bg"
    assert rows["custom:speech_0001"].group == "session:s1"
    assert rows["custom:background_0003"].split == rows["custom:speech_0001"].split
    assert rows["custom:speech_0001"].channels == 2
    assert rows["custom:speech_0001"].path == "custom://speech/speech_0001.wav"


def test_unreadable_files_are_left_out(data_env: Any) -> None:
    config = data_env.config
    bad = config.paths["external"] / "fsd50k" / "FSD50K.dev_audio" / "101.wav"
    bad.write_bytes(b"not a wav")
    summary = prepare(config, workers=1, progress=False)
    assert summary.unreadable == ["fsd50k:101"]
    assert "fsd50k:101" not in by_id(load_clip_index(config.paths["processed"]))


def test_missing_dataset_is_skipped_with_a_hint(data_env: Any) -> None:
    (data_env.external / "fsd50k" / ".done").unlink()
    summary = prepare(data_env.config, workers=1, progress=False)
    assert any("soundspy data download fsd50k" in s for s in summary.skipped)
    assert "fsd50k" not in summary.clips


def test_write_wav_helper_roundtrip(tmp_path: Path, kit) -> None:
    kit.write_wav(tmp_path / "x.wav", np.zeros(10, np.float32), kit.SR)
    assert (tmp_path / "x.wav").exists()


def test_majority_split_moves_whole_groups_to_their_bigger_side() -> None:
    pairs = [("a", "train"), ("a", "train"), ("a", "val"),
             ("b", "val"), ("b", "val"), ("b", "train"),
             ("c", "val"), ("c", "train"),  # tie -> train
             ("d", "val")]
    assert majority_split(pairs) == {"a": "train", "b": "val", "c": "train", "d": "val"}
