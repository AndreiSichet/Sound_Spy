"""Evaluation of detection models (``soundspy evaluate detection``).

Both models are scored on the **same fixed windows** of a split (``evaluation.windows_per_recipe``
windows per recipe, seeded with ``evaluation.seed``), so numbers are comparable between runs:

* **fine-tuned**: ``checkpoints/best.pt`` with the thresholds tuned on val (``thresholds.json``);
* **zero-shot**: the pretrained AudioSet model without fine-tuning, class probability = max over
  the AudioSet outputs inside ``F(c)``; its thresholds are tuned on val the same way
  (``zero_shot_thresholds.json``), never on test.

Results go to ``metrics.json`` (one entry per split), with a false-positive co-occurrence table
and per-class precision/recall plots in ``plots/``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from soundspy.config import AppConfig
from soundspy.data.mixtures import ClipLibrary, Recipe, iter_recipes, recipes_path
from soundspy.detection import metrics as M
from soundspy.detection.model import (
    DetectorNet,
    ModelSpec,
    audioset_rows,
    build_model,
    load_audioset_labels,
    load_checkpoint,
)
from soundspy.training.datasets import (
    DetectionDataset,
    WindowSpec,
    collate_windows,
    guarded,
    limit_worker_threads,
    worker_init,
)
from soundspy.training.experiment import ExperimentPaths, experiment_paths, read_json, write_json

log = logging.getLogger(__name__)

ZERO_SHOT_THRESHOLDS = "zero_shot_thresholds.json"


class EvaluationError(RuntimeError):
    pass


def window_spec(config: AppConfig) -> WindowSpec:
    det, mix = config.detection, config.data.mixtures
    return WindowSpec(det.model_sample_rate, det.window_s, det.min_active_s,
                      det.activity_rate_hz, mix.reference_rms_dbfs, mix.peak_limit)


def load_recipes(config: AppConfig, split: str, limit: int | None = None) -> list[Recipe]:
    recipes = []
    for recipe in iter_recipes(recipes_path(config.paths["mixtures"], split)):
        if limit is not None and len(recipes) >= limit:
            break
        recipes.append(recipe)
    if not recipes:
        raise EvaluationError(f"no {split} recipes: run `soundspy data mix` first")
    return recipes


def eval_dataset(config: AppConfig, library: ClipLibrary, split: str,
                 limit: int | None = None) -> DetectionDataset:
    det = config.detection
    return DetectionDataset(load_recipes(config, split, limit), library, config.labels,
                            window_spec(config), seed=det.eval_seed,
                            windows=det.eval_windows_per_recipe)


def make_loader(dataset: DetectionDataset, windows_per_batch: int, workers: int,
                sampler=None, *, timeout_s: float = 120.0) -> DataLoader:
    """Evaluation loader (persistent workers with one BLAS thread each; see datasets.py)."""
    per_batch = max(1, windows_per_batch // dataset.windows)
    limit_worker_threads()
    return DataLoader(dataset, batch_size=per_batch, sampler=sampler, shuffle=False,
                      num_workers=workers, persistent_workers=workers > 0,
                      pin_memory=torch.cuda.is_available(), collate_fn=collate_windows,
                      worker_init_fn=worker_init if workers else None,
                      timeout=timeout_s if workers else 0)


@torch.inference_mode()
def predict(models: Sequence[Callable[[torch.Tensor], torch.Tensor]], loader: DataLoader,
            device: torch.device, *, amp: bool = True,
            criterion: Callable | None = None) -> tuple[np.ndarray, list[np.ndarray], float]:
    """Run each model over the loader. Returns ``(y_true, [probabilities per model], loss)``.

    ``models`` return **probabilities**; ``criterion`` (on the first model's logits, optional)
    needs ``models[0]`` to expose ``.logits``.
    """
    ys, scores, losses = [], [[] for _ in models], []
    use_amp = amp and device.type == "cuda"
    for waves, labels in guarded(loader, what="evaluation batch"):
        waves = waves.to(device, non_blocking=True)
        ys.append(labels.numpy())
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            for k, model in enumerate(models):
                if k == 0 and criterion is not None:
                    logits = model.logits(waves)
                    losses.append(float(criterion(logits.float(), labels.to(device))))
                    scores[k].append(torch.sigmoid(logits.float()).cpu().numpy())
                else:
                    scores[k].append(model(waves).float().cpu().numpy())
    y = np.concatenate(ys) if ys else np.zeros((0, 0), np.float32)
    loss = float(np.mean(losses)) if losses else float("nan")
    return y, [np.concatenate(s) for s in scores], loss


class FineTuned:
    """Probability view of a ``DetectorNet`` (``.logits`` for the loss)."""

    def __init__(self, model: DetectorNet) -> None:
        self.model = model

    def logits(self, waves: torch.Tensor) -> torch.Tensor:
        return self.model(waves)

    def __call__(self, waves: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(waves).float())


class ZeroShot:
    def __init__(self, model: DetectorNet) -> None:
        self.model = model

    def __call__(self, waves: torch.Tensor) -> torch.Tensor:
        return self.model.zero_shot(waves)


def model_spec(config: AppConfig, *, augment: dict[str, Any] | None = None) -> ModelSpec:
    """Spec for a new model: AudioSet rows from the ontology (needs the downloaded ontology)."""
    from soundspy.data.prepare import load_ontology

    det = config.detection
    rows = audioset_rows(config.labels, load_ontology(config),
                         load_audioset_labels(det.audioset_labels))
    return ModelSpec(det.backbone, list(config.labels), det.model_sample_rate, det.window_s,
                     rows, augment or {}, det.input_context_s)


def pretrained_model(config: AppConfig, spec: ModelSpec | None = None) -> DetectorNet:
    """The untouched pretrained model (zero-shot reference)."""
    from soundspy.detection.weights import ensure_pretrained

    det = config.detection
    weights = (ensure_pretrained(det, config.paths["model_cache"])
               if det.backbone != "stub" else None)
    model, _ = build_model(spec or model_spec(config), pretrained_weights=weights,
                           head_init=det.head_init)
    return model.eval()


def resolve_device(name: str = "auto") -> torch.device:
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise EvaluationError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_detection(config: AppConfig, name: str, split: str, *, device: str = "auto",
                       workers: int | None = None, limit: int | None = None,
                       progress: Callable[[str], None] = print) -> dict[str, Any]:
    """Score fine-tuned (if trained) and zero-shot models on ``split``; update metrics.json."""
    det = config.detection
    exp = experiment_paths(config.paths["experiments"], name).ensure()
    dev = resolve_device(device)
    workers = det.training.num_workers if workers is None else workers
    library = ClipLibrary.from_config(config)
    labels = list(config.labels)

    fine = None
    if exp.best.exists():
        fine, _payload = load_checkpoint(exp.best, map_location="cpu")
        if not exp.thresholds.exists():
            raise EvaluationError(f"{exp.thresholds} missing: training did not finish")
        thresholds = read_json(exp.thresholds)["thresholds"]
        spec = fine.spec
    else:
        progress(f"no checkpoint in {exp.best.parent}: evaluating the zero-shot baseline only")
        thresholds, spec = None, model_spec(config)
    zero = pretrained_model(config, spec)

    zs_thresholds = _zero_shot_thresholds(config, exp, zero, library, dev, workers, limit,
                                          progress)
    progress(f"scoring {split} …")
    data = eval_dataset(config, library, split, limit)
    models: list[Any] = [ZeroShot(zero.to(dev))]
    if fine is not None:
        models.insert(0, FineTuned(fine.to(dev)))
    loader = make_loader(data, det.training.batch_size, workers,
                         timeout_s=det.training.loader_timeout_s)
    y, scores, _ = predict(models, loader, dev)

    result: dict[str, Any] = {"windows": int(y.shape[0]), "recipes": len(data)}
    if fine is not None:
        result["fine_tuned"] = M.evaluate(y, scores[0], labels, thresholds)
        result["fine_tuned"]["false_positives"] = M.false_positive_cooccurrence(
            y, scores[0], thresholds, labels)
    zs_scores = scores[-1]
    result["zero_shot"] = M.evaluate(y, zs_scores, labels, zs_thresholds)
    result["zero_shot"]["false_positives"] = M.false_positive_cooccurrence(
        y, zs_scores, zs_thresholds, labels)

    metrics = read_json(exp.metrics) if exp.metrics.exists() else {}
    metrics[split] = result
    write_json(exp.metrics, metrics)
    plot_precision_recall(exp, split, y, {"fine-tuned": scores[0]} if fine is not None else {},
                          zs_scores, labels)
    return result


def _zero_shot_thresholds(config, exp: ExperimentPaths, zero: DetectorNet, library, dev,
                          workers, limit, progress) -> dict[str, float]:
    path = exp.root / ZERO_SHOT_THRESHOLDS
    if path.exists() and limit is None:
        return read_json(path)["thresholds"]
    progress("tuning zero-shot thresholds on val …")
    data = eval_dataset(config, library, "val", limit)
    tr = config.detection.training
    y, (scores,), _ = predict([ZeroShot(zero.to(dev))],
                              make_loader(data, tr.batch_size, workers,
                                          timeout_s=tr.loader_timeout_s), dev)
    thresholds = M.tune_thresholds(y, scores, list(config.labels))
    if limit is None:
        write_json(path, {"thresholds": thresholds, "tuned_on": "val",
                          "windows": int(y.shape[0])})
    return thresholds


def plot_precision_recall(exp: ExperimentPaths, split: str, y: np.ndarray,
                          scores: dict[str, np.ndarray], zero_shot: np.ndarray,
                          labels: Sequence[str]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    cols = 4
    rows = (len(labels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.4 * rows), squeeze=False)
    for j, label in enumerate(labels):
        ax = axes[j // cols][j % cols]
        y_col = y.T[j]
        if y_col.sum() > 0:
            for name, s in [*scores.items(), ("zero-shot", zero_shot)]:
                p, r, _ = precision_recall_curve(y_col, s.T[j])
                ax.plot(r, p, label=name)
        ax.set_title(f"{label} ({int(y_col.sum())} pos.)")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    for k in range(len(labels), rows * cols):
        axes[k // cols][k % cols].axis("off")
    axes[0][0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Precision / recall per class — {split}")
    fig.tight_layout()
    path = exp.plots / f"precision_recall_{split}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
