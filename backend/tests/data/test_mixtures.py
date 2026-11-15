"""Mixture recipes (deterministic, rule-abiding) and rendering (exact, repeatable)."""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any

import numpy as np
import pytest

from soundspy.data.mixtures import (
    ClipLibrary,
    Event,
    Recipe,
    generate_recipes,
    iter_recipes,
    recipes_path,
    render,
    write_previews,
    write_recipes,
)
from soundspy.data.prepare import load_activity, load_clip_index


@pytest.fixture(scope="module")
def penv(shared_prepared_env: Any) -> Any:
    return shared_prepared_env


@pytest.fixture(scope="module")
def library(penv: Any) -> ClipLibrary:
    return ClipLibrary.from_config(penv.config)


def gen(env: Any, library: ClipLibrary, split: str = "train", count: int = 200, seed: int = 1,
        **mix_changes) -> list[Recipe]:
    mix = dataclasses.replace(env.config.data.mixtures, **mix_changes)
    return generate_recipes(library, env.config.labels, mix, split, count, seed)


def test_recipes_are_deterministic_for_a_seed(penv: Any, library: ClipLibrary) -> None:
    a = gen(penv, library, seed=7)
    b = gen(penv, library, seed=7)
    c = gen(penv, library, seed=8)
    assert [r.to_json() for r in a] == [r.to_json() for r in b]
    assert [r.to_json() for r in a] != [r.to_json() for r in c]


def test_counts_and_ranges_are_respected(penv: Any, library: ClipLibrary) -> None:
    mix = penv.config.data.mixtures
    recipes = gen(penv, library, count=300)
    assert len(recipes) == 300
    assert recipes[0].id == "train-000001"
    for r in recipes:
        fg = [e for e in r.events if e.role == "fg"]
        bg = [e for e in r.events if e.role == "bg"]
        assert mix.n_foreground[0] <= len(fg) <= mix.n_foreground[1]
        assert mix.n_background[0] <= len(bg) <= mix.n_background[1]
        assert len({e.cls for e in fg}) == len(fg)  # distinct fg classes
        assert sorted(r.present) == sorted(e.cls for e in fg)
        assert mix.target_to_rest_db[0] <= r.target_to_rest_db <= mix.target_to_rest_db[1]
        for e in r.events:
            info = library.info(e.clip_id)
            assert 0 <= e.onset_s and e.onset_s + e.dur_s <= r.duration_s + 1e-9
            assert e.clip_offset_s + e.dur_s <= info.duration_s + 1e-3
            assert e.dur_s >= min(mix.event_min_s, info.duration_s) - 1e-3
            lo, hi = mix.fg_gain_db if e.role == "fg" else mix.bg_gain_db
            assert lo <= e.gain_db <= hi
        if not r.empty_target:
            assert set(r.target) <= set(r.present)
            assert mix.target_size[0] <= len(r.target) <= mix.target_size[1]


def test_only_clips_of_the_same_split_are_used(penv: Any, library: ClipLibrary) -> None:
    for split in ("train", "val", "test"):
        for r in gen(penv, library, split=split, count=100):
            assert {library.info(e.clip_id).split for e in r.events} == {split}
            for e in r.events:
                role = library.info(e.clip_id).role
                assert role == ("bg" if e.role == "bg" else f"fg:{e.cls}")


def test_foreground_classes_are_sampled_uniformly(penv: Any,
                                                  library: ClipLibrary) -> None:
    recipes = gen(penv, library, count=2000, n_foreground=(1, 1))
    counts: dict[str, int] = {}
    for r in recipes:
        counts[r.present[0]] = counts.get(r.present[0], 0) + 1
    # train has 5 fg classes with very different clip counts; each gets ~ 1/5
    assert set(counts) == {"speech", "music", "footsteps", "alarm", "traffic"}
    assert all(330 <= n <= 470 for n in counts.values()), counts


def test_empty_targets_at_roughly_the_configured_rate(penv: Any,
                                                      library: ClipLibrary) -> None:
    recipes = gen(penv, library, count=2000, empty_target_prob=0.2)
    empty = [r for r in recipes if r.empty_target]
    assert 0.16 <= len(empty) / len(recipes) <= 0.24
    for r in empty:
        assert len(r.target) == 1 and r.target[0] not in r.present
        assert r.target[0] in penv.config.labels


def test_events_start_in_active_regions(penv: Any, library: ClipLibrary) -> None:
    for r in gen(penv, library, count=100):
        for e in r.events:
            regions = library.regions(e.clip_id)
            info = library.info(e.clip_id)
            if info.duration_s <= penv.config.data.mixtures.event_min_s:
                continue
            latest = info.duration_s - penv.config.data.mixtures.event_min_s
            assert any(min(s, latest) - 1e-3 <= e.clip_offset_s <= min(e2, latest) + 1e-3
                       for s, e2 in regions)


def test_jsonl_roundtrip_and_cli_writer(penv: Any) -> None:
    config = penv.config
    index = load_clip_index(config.paths["processed"])
    path, n = write_recipes(index, load_activity(config.paths["processed"]), config, "val")
    assert n == 20 and path == recipes_path(config.paths["mixtures"], "val")
    recipes = list(iter_recipes(path))
    assert len(recipes) == 20
    first = json.loads(path.read_text().splitlines()[0])
    assert set(first) == {"id", "duration_s", "events", "present", "target",
                          "target_to_rest_db"}
    assert set(first["events"][0]) == {"clip_id", "role", "class", "clip_offset_s", "onset_s",
                                       "dur_s", "gain_db"}
    assert Recipe.from_json(first) == recipes[0]


# ---- rendering ---------------------------------------------------------------------------

def do_render(env: Any, library: ClipLibrary, recipe: Recipe, sr: int = 16_000, **kw):
    mix = env.config.data.mixtures
    args = dict(reference_rms_dbfs=mix.reference_rms_dbfs, peak_limit=mix.peak_limit)
    args.update(kw)
    return render(recipe, sr, library, env.config.labels, **args)


def rms_db(x: np.ndarray) -> float:
    return 20 * math.log10(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))))


def test_mixture_is_target_plus_rest_and_repeatable(penv: Any,
                                                    library: ClipLibrary) -> None:
    for recipe in gen(penv, library, count=15):
        out = do_render(penv, library, recipe)
        assert out.mixture.samples.shape == (64_000, 1)
        assert out.mixture.samples.dtype == np.float32
        np.testing.assert_array_equal(out.mixture.samples,
                                      out.target.samples + out.rest.samples)
        again = do_render(penv, library, recipe)
        np.testing.assert_array_equal(out.mixture.samples, again.mixture.samples)
        np.testing.assert_array_equal(out.target.samples, again.target.samples)
        assert out.present.sum() == len(recipe.present)
        assert out.target_hot.sum() == len(recipe.target)
        if not recipe.empty_target and np.any(out.rest.samples):
            got = rms_db(out.target.samples) - rms_db(out.rest.samples)
            assert got == pytest.approx(recipe.target_to_rest_db, abs=0.1)


def test_peak_limit_scales_mixture_and_target_by_the_same_factor(penv: Any,
                                                                 library: ClipLibrary) -> None:
    recipe = next(r for r in gen(penv, library, count=50) if not r.empty_target)
    free = do_render(penv, library, recipe, peak_limit=1e9)
    peak = float(np.max(np.abs(free.mixture.samples)))
    limited = do_render(penv, library, recipe, peak_limit=peak / 4)
    assert limited.scale == pytest.approx(0.25, rel=1e-6)
    assert np.max(np.abs(limited.mixture.samples)) <= peak / 4 * (1 + 1e-6)
    np.testing.assert_allclose(limited.target.samples, free.target.samples * 0.25, atol=1e-7)
    np.testing.assert_allclose(limited.mixture.samples, free.mixture.samples * 0.25, atol=1e-7)


def test_empty_target_renders_silence(penv: Any, library: ClipLibrary) -> None:
    recipe = next(r for r in gen(penv, library, count=400, empty_target_prob=0.5)
                  if r.empty_target)
    out = do_render(penv, library, recipe)
    assert not np.any(out.target.samples)
    assert np.any(out.mixture.samples)
    np.testing.assert_array_equal(out.mixture.samples, out.rest.samples)
    assert out.target_hot.sum() == 1  # the requested (absent) class


def test_class_activity_marks_where_events_sound(penv: Any,
                                                 library: ClipLibrary) -> None:
    labels = penv.config.labels
    clip = "fsd50k:102"  # 3.0 s, tone from 0.2 to 2.8 s
    recipe = Recipe("t", 4.0, (Event(clip, "fg", "speech", 0.0, 1.0, 3.0, 0.0),),
                    ("speech",), ("speech",), 0.0)
    out = do_render(penv, library, recipe, activity_rate_hz=1.0)
    col = labels.index("speech")
    assert out.class_activity.shape == (4, len(labels))
    assert out.class_activity[:, col].tolist() == [0.0, 1.0, 1.0, 1.0]  # sounds 1.2 .. 3.8 s
    assert out.class_activity.sum() == 3.0
    fine = do_render(penv, library, recipe, activity_rate_hz=10.0)
    on = np.flatnonzero(fine.class_activity[:, col])
    assert on[0] == 12 and on[-1] == 37


@pytest.mark.parametrize("sr", [16_000, 32_000, 44_100])
def test_rendering_at_other_rates_keeps_durations_exact(penv: Any, library: ClipLibrary,
                                                        sr: int) -> None:
    recipe = gen(penv, library, count=1)[0]
    out = do_render(penv, library, recipe, sr=sr)
    assert out.mixture.num_frames == round(recipe.duration_s * sr)
    assert out.mixture.sample_rate == sr


def test_preview_writes_wavs_and_json(penv: Any) -> None:
    import soundfile as sf

    config = penv.config
    index = load_clip_index(config.paths["processed"])
    write_recipes(index, load_activity(config.paths["processed"]), config, "val")
    out_dir, ids = write_previews(config, "val", 3)
    assert ids == ["val-000001", "val-000002", "val-000003"]
    info = sf.info(str(out_dir / "val-000001_mix.wav"))
    assert info.samplerate == 44_100 and info.frames == 4 * 44_100
    assert (out_dir / "val-000003_target.wav").exists()
    meta = json.loads((out_dir / "val-000002.json").read_text())
    assert meta["id"] == "val-000002" and meta["render"]["sample_rate"] == 44_100
