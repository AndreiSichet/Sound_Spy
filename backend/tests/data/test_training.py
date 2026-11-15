"""Training loop and evaluation end to end on the fake dataset with the stub backbone (CPU)."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

from soundspy.data.mixtures import write_recipes
from soundspy.data.prepare import load_activity, load_clip_index
from soundspy.training.evaluate import evaluate_detection
from soundspy.training.experiment import read_log
from soundspy.training.train import TrainingError, TrainOptions, train_detection


@pytest.fixture
def train_config(prepared_env: Any, tmp_path):
    config = prepared_env.config
    index = load_clip_index(config.paths["processed"])
    activity = load_activity(config.paths["processed"])
    for split in ("train", "val", "test"):
        write_recipes(index, activity, config, split, count=16)
    det = config.detection
    det = dataclasses.replace(
        det, backbone="stub", model_sample_rate=16_000, input_context_s=1.0,
        training=dataclasses.replace(det.training, batch_size=8, num_workers=0, epochs=2),
        augment=dataclasses.replace(det.augment, freqm=8, timem=4, mel_range=False),
        eval_windows_per_recipe=2)
    paths = {**config.paths, "experiments": tmp_path / "experiments"}
    return dataclasses.replace(config, detection=det, paths=paths)


def opts(**kw) -> TrainOptions:
    base = dict(name="stub_run", device="cpu", allow_cpu=True, workers=0,
                pos_weight_windows=16)
    base.update(kw)
    return TrainOptions(**base)


def test_train_resume_and_evaluate(train_config) -> None:
    logs: list[str] = []
    first = train_detection(train_config, opts(epochs=1), log=logs.append)
    exp = first.paths
    assert [r["epoch"] for r in read_log(exp.log)] == ["1"]
    for path in (exp.best, exp.last, exp.thresholds, exp.metrics, exp.config,
                 exp.plots / "training_curves.png"):
        assert path.exists(), path
    cfg = json.loads(exp.config.read_text())
    assert cfg["labels"] == list(train_config.labels) and "git" in cfg
    assert set(json.loads(exp.thresholds.read_text())["thresholds"]) == set(train_config.labels)

    with pytest.raises(TrainingError, match="--resume"):
        train_detection(train_config, opts(epochs=2), log=logs.append)

    resumed = train_detection(train_config, opts(epochs=2, resume=True), log=logs.append)
    assert [r["epoch"] for r in read_log(exp.log)] == ["1", "2"]
    assert resumed.epochs_run == 1
    assert any("resuming stub_run after epoch 1" in m for m in logs)
    row = read_log(exp.log)[-1]
    assert {"train_loss", "val_loss", "val_map", "lr_backbone", "epoch_s"} <= set(row)

    result = evaluate_detection(train_config, "stub_run", "test", device="cpu", workers=0,
                                progress=logs.append)
    assert set(result) >= {"fine_tuned", "zero_shot", "windows"}
    assert result["windows"] == 32  # 16 recipes x 2 fixed windows
    metrics = json.loads(exp.metrics.read_text())
    assert "test" in metrics and "fine_tuned" in metrics["val"]
    assert (exp.root / "zero_shot_thresholds.json").exists()
    assert (exp.plots / "precision_recall_test.png").exists()


def test_cuda_is_required_unless_cpu_is_allowed(train_config, monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(TrainingError, match="NVIDIA driver"):
        train_detection(train_config, opts(allow_cpu=False, device="auto"), log=lambda m: None)


def test_resume_without_a_run_is_an_error(train_config) -> None:
    with pytest.raises(TrainingError, match="nothing to resume"):
        train_detection(train_config, opts(resume=True), log=lambda m: None)
