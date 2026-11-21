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
                               "snr_i": 3.0, "missed": 0.5, "suppression_db": -20.0,
                               "worse": 1,  # the -10 dB recipe
                               "interruption": None, "n": 2, "n_empty": 1}  # (step 07c)
    assert summarize([])["live"]["si_sdr_i"] is None


# --- step 07: the neural extractor in the same evaluation ---------------------------------------


def test_evaluate_neural_extraction_end_to_end(eval_config, neural_kit, tmp_path) -> None:
    """The five step-06 systems + the neural extractor offline / streaming / at the live
    attenuation, on the same recipes (tiny random model: the numbers only have to be
    consistent, not good)."""
    from soundspy.training.evaluate_extraction import NEURAL_SYSTEMS, systems_for

    config = neural_kit.make(eval_config, tmp_path)
    assert systems_for("neural_tse") == (*SYSTEMS, *NEURAL_SYSTEMS)
    assert systems_for("mask_baseline") == SYSTEMS
    logs: list[str] = []
    result = evaluate_extraction(config, "neural_tse", "test", n=6, sample_rate=SR,
                                 device="cpu", examples=2, reverberant=False,
                                 progress=logs.append)
    overall = result["overall"]
    assert result["systems"] == [*SYSTEMS, *NEURAL_SYSTEMS]
    assert result["extractor"]["model"] == "tse_tiny" and result["extractor"]["device"] == "cpu"
    assert result["attenuation_db"] == 30.0 and result["live_attenuation_db"] == 15.0
    assert "reverberant" not in result
    # streaming (chunk by chunk, with state) gives the same result as the whole file at once
    assert overall["neural_stream"]["si_sdr_i"] == pytest.approx(
        overall["neural_offline"]["si_sdr_i"], abs=0.05)
    # 15 dB of attenuation changes the mixture less than 30 dB does (random model: for the
    # worse or the better, but by less)
    assert abs(overall["neural_live_db"]["si_sdr_i"]) <= abs(overall["neural_stream"]
                                                             ["si_sdr_i"]) + 0.05
    assert overall["irm"]["si_sdr_i"] > 10.0 and overall["mixture"]["si_sdr_i"] == 0.0
    assert result["by_target_count"] and sum(
        g["n"] for g in result["by_target_count"].values()) == overall["n"]
    assert any("extractor tse_tiny on cpu" in m for m in logs)

    exp = config.paths["experiments"] / "ext_neural_tse"
    table = (exp / "metrics.md").read_text("utf-8")
    assert "neural_tse, streaming" in table or "streaming" in table
    with (exp / "per_recipe.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6 and "neural_stream_si_sdr_i" in rows[0]
    saved = json.loads((exp / "config.json").read_text("utf-8"))
    assert saved["neural_tse"]["model"] == "tse_tiny" and saved["method"] == "neural_tse"
    examples = config.paths["outputs"] / "eval" / "ext_neural_tse"
    index = json.loads((examples / "index.json").read_text("utf-8"))
    first = index["examples"][0]
    assert set(first["files"]) == {"target", *SYSTEMS, *NEURAL_SYSTEMS}
    lengths = {sf.info(str(examples / name)).frames for name in first["files"].values()}
    assert len(lengths) == 1
    # the baseline's own experiment folder is not touched by the neural evaluation
    assert not (config.paths["experiments"] / "ext_mask_baseline" / "metrics.json").exists()

    no_model = dataclasses.replace(config, extraction=dataclasses.replace(
        config.extraction, neural_tse=dataclasses.replace(config.extraction.neural_tse,
                                                          checkpoint=tmp_path / "none.pt")))
    with pytest.raises(ExtractionEvalError, match="soundspy train extraction"):
        evaluate_extraction(no_model, "neural_tse", progress=lambda m: None)


@pytest.mark.slow  # step 08: > 5 s
def test_reverberant_copy_of_the_evaluation(eval_config, neural_kit, tmp_path) -> None:
    pytest.importorskip("pyroomacoustics")
    config = neural_kit.make(eval_config, tmp_path)
    neural = config.extraction.neural_tse
    quick = dataclasses.replace(neural.data.reverb, max_order=4)  # fast rooms for the test
    config = dataclasses.replace(config, extraction=dataclasses.replace(
        config.extraction, neural_tse=dataclasses.replace(neural, data=dataclasses.replace(
            neural.data, reverb=quick))))
    result = evaluate_extraction(config, "neural_tse", "test", n=4, sample_rate=SR,
                                 device="cpu", examples=0, progress=lambda m: None)
    reverb = result["reverberant"]
    assert reverb["seed"] == config.extraction.evaluation.reverb_seed
    assert reverb["overall"]["n"] + reverb["overall"]["n_empty"] == 4
    assert reverb["overall"]["mixture"]["si_sdr_i"] == 0.0
    assert reverb["overall"]["irm"]["si_sdr_i"] > 8.0  # mixture is still target + rest
    assert reverb["by_target_class"]
    exp = config.paths["experiments"] / "ext_neural_tse"
    with (exp / "per_recipe_reverb.csv").open(encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 4
    assert "everberant" in (exp / "metrics.md").read_text("utf-8")
    # the same seed -> the same rooms -> the same numbers
    again = evaluate_extraction(config, "neural_tse", "test", n=4, sample_rate=SR,
                                device="cpu", examples=0, progress=lambda m: None)
    assert again["reverberant"]["overall"] == reverb["overall"]


@pytest.mark.slow  # step 08: > 5 s
def test_evaluation_with_the_gate_an_earlier_model_and_benchmarks(eval_config, neural_kit,
                                                                  tmp_path) -> None:
    """Step 07b rows (previous model, model without its gate, 40 dB) and the benchmark sets."""
    from soundspy.training.evaluate_extraction import NEURAL_SYSTEMS, bench_path

    config = neural_kit.make(eval_config, tmp_path)
    previous = neural_kit.make(eval_config, tmp_path / "old", seed=1)
    neural = config.extraction.neural_tse
    # thresholds the random tiny model's output really crosses
    gate = dataclasses.replace(neural.gate, enabled=True, open_db=10.0, close_db=-5.0)
    ev = dataclasses.replace(
        config.extraction.evaluation, bench_absent=5, bench_pair=("speech", "music"),
        bench_pair_n=2, bench_pair_min_clip_s=2.0, bench_librispeech=2, bench_examples=1,
        compare_checkpoint=previous.extraction.neural_tse.checkpoint)
    config = dataclasses.replace(config, extraction=dataclasses.replace(
        config.extraction, evaluation=ev,
        neural_tse=dataclasses.replace(neural, gate=gate)))
    logs: list[str] = []
    result = evaluate_extraction(config, "neural_tse", "test", n=4, sample_rate=SR,
                                 device="cpu", examples=1, reverberant=False,
                                 name="ext_v2_test", progress=logs.append)
    assert result["systems"] == [*SYSTEMS, "neural_v1", "neural_nogate", *NEURAL_SYSTEMS]
    overall = result["overall"]
    assert overall["neural_stream"]["si_sdr_i"] == pytest.approx(
        overall["neural_offline"]["si_sdr_i"], abs=0.05)  # streaming == whole file, gate on
    assert overall["neural_nogate"] != overall["neural_offline"]  # the gate changes the output
    assert all(isinstance(overall[s]["worse"], int) for s in result["systems"])
    assert result["high_attenuation_db"] == 40.0 and result["compared_with"]["model"] == "tse_tiny"
    assert any("output gate on" in m for m in logs)

    benches = result["benchmarks"]
    assert set(benches) == {"absent", "pair_absent", "pair_a", "pair_b"}  # no LibriSpeech here
    assert any("benchmark librispeech skipped" in m for m in logs)
    assert benches["absent"]["overall"]["n_empty"] == 5 and benches["absent"]["overall"]["n"] == 0
    assert benches["pair_absent"]["overall"]["n_empty"] == 2
    assert benches["pair_a"]["overall"]["n"] == 2 and benches["pair_b"]["overall"]["n"] == 2
    assert benches["absent"]["empty_by_target_class"]  # per selected class
    assert benches["absent"]["empty_by_dominant_other"]  # per loudest other sound
    path = bench_path(config, "absent")
    header = json.loads(path.read_text("utf-8").splitlines()[0])
    assert header["benchmark"] == "absent" and header["count"] == 5
    exp = config.paths["experiments"] / "ext_v2_test"
    for name in ("per_recipe_absent.csv", "per_recipe_pair_a.csv", "metrics.md"):
        assert (exp / name).is_file(), name
    table = (exp / "metrics.md").read_text("utf-8")
    assert "Benchmark `absent`" in table and "Worse than input" in table
    examples = config.paths["outputs"] / "eval" / "ext_v2_test"
    index = json.loads((examples / "index.json").read_text("utf-8"))
    assert {e.get("benchmark") for e in index["examples"]} >= {"absent", "pair_a"}
    first = next(e for e in index["examples"] if e.get("benchmark") is None)
    assert {"neural_v1", "neural_nogate", "neural_high_db"} <= set(first["files"])
    # the benchmark files are frozen: a second evaluation reads the same recipes
    before = path.read_text("utf-8")
    again = evaluate_extraction(config, "neural_tse", "test", n=2, sample_rate=SR, device="cpu",
                                examples=0, reverberant=False, name="ext_v2_test",
                                progress=lambda m: None)
    assert path.read_text("utf-8") == before
    assert again["benchmarks"]["absent"]["overall"] == benches["absent"]["overall"]
