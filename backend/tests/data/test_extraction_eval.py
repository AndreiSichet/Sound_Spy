"""Extraction on rendered recipes: oracle activity, IRM upper bound, the whole evaluation."""

from __future__ import annotations

import csv
import dataclasses
import json
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from soundspy.cli.main import main
from soundspy.core.target import TargetSpec
from soundspy.data.mixtures import (
    ClipLibrary,
    iter_recipes,
    recipes_path,
    render_with_config,
    write_recipes,
)
from soundspy.data.prepare import load_activity, load_clip_index
from soundspy.extraction import metrics as M
from soundspy.extraction.activity import OracleActivity
from soundspy.extraction.mask_baseline import MaskBaselineExtractor
from soundspy.extraction.offline import irm_oracle, run_offline
from soundspy.training.evaluate_extraction import (
    SYSTEMS,
    ExtractionEvalError,
    duration_bin,
    evaluate_extraction,
    select_recipes,
    summarize,
    t2r_bin,
    target_active_seconds,
)

SR = 16_000


@pytest.fixture(scope="module")
def recipe_env(shared_prepared_env: Any):
    config = shared_prepared_env.config
    index = load_clip_index(config.paths["processed"])
    activity = load_activity(config.paths["processed"])
    write_recipes(index, activity, config, "test", count=12)
    recipes = list(iter_recipes(recipes_path(config.paths["mixtures"], "test")))
    return config, ClipLibrary.from_config(config), recipes


def rendered_with_target(recipe_env, *, overlap: bool = False):
    config, library, recipes = recipe_env
    for recipe in recipes:
        if recipe.empty_target or (overlap and len(recipe.present) < 2):
            continue
        return render_with_config(recipe, SR, library, config, activity_rate_hz=100.0)
    raise AssertionError("no suitable recipe")


def test_oracle_activity_follows_the_rendered_recipe(recipe_env) -> None:
    config, _, _ = recipe_env
    rendered = rendered_with_target(recipe_env)
    oracle = OracleActivity(list(config.labels), rendered.class_activity,
                            rendered.activity_rate_hz)
    labels = set(rendered.recipe.target)
    truth = rendered.class_activity.T[np.flatnonzero(rendered.target_hot > 0.5)].max(axis=0)
    times = (np.arange(truth.size) + 0.5) / rendered.activity_rate_hz
    got = np.array([oracle.active(labels, float(t), 0.0) for t in times])
    np.testing.assert_array_equal(got, truth > 0.5)
    assert got.any() and not got.all()  # the target comes and goes within the mixture
    assert target_active_seconds(rendered) == pytest.approx(truth.sum() / 100.0)
    # where the oracle says "inactive", the target really is (almost) silent
    target = rendered.target.samples.ravel()
    silent = np.repeat(~got, SR // 100)[: target.size]
    assert np.mean(target[silent] ** 2) < 1e-3 * np.mean(target[~silent] ** 2)


def test_irm_oracle_gives_a_large_improvement(recipe_env) -> None:
    config, _, _ = recipe_env
    rendered = rendered_with_target(recipe_env, overlap=True)
    mixture = rendered.mixture.samples.ravel()
    target, rest = rendered.target.samples.ravel(), rendered.rest.samples.ravel()
    np.testing.assert_allclose(mixture, target + rest, atol=1e-6)
    out = irm_oracle(mixture, target, rest, config.extraction.mask_baseline)
    assert out.shape == mixture.shape
    assert M.si_sdr_improvement(out, mixture, target) > 10.0
    assert M.snr_improvement(out, mixture, target) > 10.0
    # with nothing but the target, the mask is 1 and the mixture comes back unchanged
    alone = irm_oracle(target, target, np.zeros_like(target), config.extraction.mask_baseline)
    np.testing.assert_allclose(alone, target, atol=1e-4)
    silent = irm_oracle(rest, np.zeros_like(rest), rest, config.extraction.mask_baseline)
    assert M.suppression_db(silent, rest) < -60.0


def test_baseline_with_oracle_activity_improves_a_recipe(recipe_env) -> None:
    config, _, _ = recipe_env
    rendered = rendered_with_target(recipe_env, overlap=True)
    mixture, target = rendered.mixture.samples.ravel(), rendered.target.samples.ravel()
    oracle = OracleActivity(list(config.labels), rendered.class_activity,
                            rendered.activity_rate_hz)
    spec = TargetSpec(labels=frozenset(rendered.recipe.target), attenuation_db=30.0)
    extractor = MaskBaselineExtractor(config.extraction.mask_baseline, oracle)
    out = run_offline(extractor, rendered.mixture, spec).samples.ravel()
    assert out.shape == mixture.shape  # delay-compensated: comparable sample by sample
    assert M.si_sdr(out, target) > M.si_sdr(mixture, target) - 1.0  # never much worse
    # the target is kept at its level where it sounds
    truth = rendered.class_activity.T[np.flatnonzero(rendered.target_hot > 0.5)].max(axis=0)
    on = np.repeat(truth > 0.5, SR // 100)[: target.size]
    kept = 10 * np.log10(np.mean(out[on] ** 2) / np.mean(mixture[on] ** 2))
    assert kept > -6.0


@pytest.fixture
def eval_config(recipe_env, tmp_path):
    """The fake dataset + a stub detection model that knows the fake label set."""
    from soundspy.detection.model import AudioSetRows, DetectorNet, ModelSpec, save_checkpoint

    config, _, _ = recipe_env
    labels = list(config.labels)
    rows = AudioSetRows({lab: [i] for i, lab in enumerate(labels)},
                        {lab: [i, i + 10] for i, lab in enumerate(labels)})
    exp = tmp_path / "experiments" / "stub_v1"
    save_checkpoint(exp / "checkpoints" / "best.pt",
                    DetectorNet(ModelSpec("stub", labels, SR, 1.0, rows, {}, 1.0)).eval(),
                    {"epoch": 1})
    (exp / "thresholds.json").write_text(
        json.dumps({"thresholds": {lab: 0.5 for lab in labels}}), encoding="utf-8")
    detection = dataclasses.replace(config.detection, checkpoint=exp / "checkpoints" / "best.pt",
                                    live=dataclasses.replace(config.detection.live, device="cpu"))
    paths = {**config.paths, "experiments": tmp_path / "experiments",
             "outputs": tmp_path / "outputs"}
    return dataclasses.replace(config, detection=detection, paths=paths)


def test_evaluate_extraction_end_to_end(eval_config, monkeypatch, capsys) -> None:
    logs: list[str] = []
    result = evaluate_extraction(eval_config, "mask_baseline", "test", n=8, sample_rate=SR,
                                 device="cpu", examples=3, progress=logs.append)
    overall = result["overall"]
    assert result["recipes"] == 8 and overall["n"] + overall["n_empty"] == 8
    assert set(SYSTEMS) <= set(overall)
    assert overall["mixture"]["si_sdr_i"] == 0.0 and overall["mixture"]["missed"] == 0.0
    assert overall["irm"]["si_sdr_i"] > 10.0  # the upper bound is far above everything
    assert overall["irm"]["si_sdr_i"] > overall["oracle"]["si_sdr_i"]
    assert overall["oracle"]["missed"] == 0.0  # true activity never misses the target
    if overall["n_empty"]:
        assert overall["oracle"]["suppression_db"] == pytest.approx(-30.0, abs=0.5)
        assert overall["mixture"]["suppression_db"] == 0.0
    for key in ("by_target_class", "by_target_to_rest_db", "by_foreground_sources",
                "by_target_duration", "empty_by_target_class"):
        assert result[key]
    assert sum(g["n"] for g in result["by_target_to_rest_db"].values()) == overall["n"]
    assert sum(g["n"] for g in result["by_target_duration"].values()) == overall["n"]
    assert any("8 of 12 test recipes" in m for m in logs)

    exp = eval_config.paths["experiments"] / "ext_mask_baseline"
    metrics = json.loads((exp / "metrics.json").read_text("utf-8"))
    assert metrics["test"]["overall"] == overall
    table = (exp / "metrics.md").read_text("utf-8")
    assert "| IRM oracle |" in table and "mask_baseline + live" in table
    assert "by target class" in table and "Target missed" in table
    with (exp / "per_recipe.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 8 and "live_si_sdr_i" in rows[0] and "target_active_s" in rows[0]
    cfg = json.loads((exp / "config.json").read_text("utf-8"))
    assert cfg["extraction"]["n_fft"] == 1024 and "git" in cfg

    examples = eval_config.paths["outputs"] / "eval" / "ext_mask_baseline"
    index = json.loads((examples / "index.json").read_text("utf-8"))
    assert len(index["examples"]) == 3 and (examples / "index.md").is_file()
    first = index["examples"][0]
    assert set(first["files"]) == {"target", *SYSTEMS}
    lengths = {sf.info(str(examples / name)).frames for name in first["files"].values()}
    assert len(lengths) == 1  # every file of an example is aligned and equally long

    # a later evaluation with another subset replaces the examples: no stale files are left
    (examples / "old-000001_live.wav").write_bytes(b"stale")
    (examples / "my_notes.txt").write_text("keep me", encoding="utf-8")
    evaluate_extraction(eval_config, "mask_baseline", "test", n=5, sample_rate=SR, device="cpu",
                        examples=2, progress=lambda m: None)
    index = json.loads((examples / "index.json").read_text("utf-8"))
    listed = {name for ex in index["examples"] for name in ex["files"].values()}
    assert {p.name for p in examples.glob("*.wav")} == listed and len(index["examples"]) == 2
    assert (examples / "my_notes.txt").is_file()  # only its own files are removed

    # the same seed gives the same subset and the same numbers
    again = evaluate_extraction(eval_config, "mask_baseline", "test", n=8, sample_rate=SR,
                                device="cpu", examples=0, progress=lambda m: None)
    assert again["overall"] == overall and again["examples_dir"] is None

    import soundspy.config

    monkeypatch.setattr(soundspy.config, "load_config", lambda *a, **k: eval_config)
    assert main(["evaluate", "extraction", "--method", "mask_baseline", "--n", "4",
                 "--sample-rate", str(SR), "--device", "cpu"]) == 0
    printed = capsys.readouterr().out
    assert "## Overall" in printed and "listening examples ->" in printed


def test_evaluation_needs_a_model_and_recipes(eval_config, tmp_path) -> None:
    no_model = dataclasses.replace(eval_config, detection=dataclasses.replace(
        eval_config.detection, checkpoint=None))
    with pytest.raises(ExtractionEvalError, match="no detection model"):
        evaluate_extraction(no_model, progress=lambda m: None)
    no_recipes = dataclasses.replace(eval_config, paths={**eval_config.paths,
                                                         "mixtures": tmp_path / "none"})
    with pytest.raises(ExtractionEvalError, match="soundspy data mix"):
        evaluate_extraction(no_recipes, device="cpu", progress=lambda m: None)
    with pytest.raises(ExtractionEvalError, match="unknown method"):
        evaluate_extraction(eval_config, "passthrough", progress=lambda m: None)


def test_subset_bins_and_summary_helpers(recipe_env) -> None:
    _, _, recipes = recipe_env
    subset = select_recipes(recipes, 5, seed=7)
    assert subset == select_recipes(recipes, 5, seed=7) and len(subset) == 5
    assert [r.id for r in subset] == sorted(r.id for r in subset)  # file order kept
    assert select_recipes(recipes, 99, seed=7) == recipes
    assert subset != select_recipes(recipes, 5, seed=8)
    assert [t2r_bin(x) for x in (-9, -5, -4.9, 4.9, 5, 12)] == [
        "<= -5 dB", "<= -5 dB", "-5 .. 5 dB", "-5 .. 5 dB", ">= 5 dB", ">= 5 dB"]
    assert [duration_bin(x) for x in (0.3, 1.0, 2.9, 3.0)] == [
        "< 1 s", "1 .. 3 s", "1 .. 3 s", ">= 3 s"]
    score = {"si_sdr": 1.0, "si_sdr_i": 2.0, "snr_i": 3.0, "missed": False,
             "suppression_db": None}
    absent = {"si_sdr": None, "si_sdr_i": None, "snr_i": None, "missed": None,
              "suppression_db": -20.0}
    records = [{"empty": False, **{s: score for s in SYSTEMS}},
               {"empty": False, **{s: {**score, "si_sdr_i": -10.0, "missed": True}
                                   for s in SYSTEMS}},
               {"empty": True, **{s: absent for s in SYSTEMS}}]
    summary = summarize(records)
    assert (summary["n"], summary["n_empty"]) == (2, 1)
    assert summary["live"] == {"si_sdr": 1.0, "si_sdr_i": -4.0, "si_sdr_i_median": -4.0,
                               "snr_i": 3.0, "missed": 0.5, "suppression_db": -20.0}
    assert summarize([])["live"]["si_sdr_i"] is None
