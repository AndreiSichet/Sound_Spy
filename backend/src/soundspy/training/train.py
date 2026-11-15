"""Detection training (``soundspy train detection``).

One epoch = one random window from every training recipe (new positions each epoch). Every
epoch ends with a validation pass on fixed windows; the checkpoint with the best val mAP is
``best.pt``, the latest one ``last.pt`` (with optimizer/scheduler/scaler state for ``--resume``).
After the last epoch the best model's thresholds are tuned on val (max F1 per class) and the
val metrics, curves and thresholds are written to the experiment folder.

Optimisation (``configs/detection/default.yaml``, section ``training``): AdamW with two learning
rates (pretrained backbone 1e-4, new head 1e-3), linear warm-up for ``warmup_epochs`` then cosine
decay to 0 (per step), mixed precision on the GPU (``torch.autocast`` + ``GradScaler``), BCE loss
with optional per-class ``pos_weight = negatives / positives`` measured on training windows.
If the GPU runs out of memory the batch size is halved and the epoch restarted (reported).

Augmentation, on the GPU per window: random gain (±``gain_db``), white noise at a random SNR
(``noise_snr_db``); in the front-end: SpecAugment frequency/time masks and mel-range jitter.
"""

from __future__ import annotations

import math
import random
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from soundspy.config import AppConfig
from soundspy.data.mixtures import ClipLibrary
from soundspy.detection import metrics as M
from soundspy.detection.model import DetectorNet, build_model, load_checkpoint, save_checkpoint
from soundspy.training.datasets import (
    DetectionDataset,
    EpochSampler,
    guarded,
    limit_worker_threads,
    worker_init,
)
from soundspy.training.evaluate import (
    FineTuned,
    eval_dataset,
    load_recipes,
    make_loader,
    model_spec,
    predict,
    resolve_device,
    window_spec,
)
from soundspy.training.experiment import (
    ExperimentPaths,
    append_log,
    experiment_paths,
    file_hash,
    git_commit,
    read_log,
    truncate_log,
    write_json,
)


class TrainingError(RuntimeError):
    pass


@dataclass
class TrainOptions:
    name: str | None = None
    epochs: int | None = None
    smoke: bool = False
    resume: bool = False
    device: str = "auto"
    allow_cpu: bool = False  # tests / stub backbone
    pos_weight_windows: int = 1000  # training windows used to measure class frequencies
    workers: int | None = None


@dataclass
class TrainResult:
    paths: ExperimentPaths
    epochs_run: int
    best_epoch: int
    best_map: float
    batch_size: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_lambda(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    def f(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return f


def augment(waves: torch.Tensor, gain_db: float, snr_db: tuple[float, float] | None,
            generator: torch.Generator) -> torch.Tensor:
    """Random gain and low-level white noise per window (on the training device)."""
    n = waves.shape[0]
    dev = waves.device
    if gain_db > 0:
        g = (torch.rand(n, 1, generator=generator) * 2 - 1).to(dev) * gain_db
        waves = waves * torch.pow(10.0, g / 20.0)
    if snr_db is not None:
        snr = (torch.rand(n, 1, generator=generator) * (snr_db[1] - snr_db[0]) + snr_db[0]).to(dev)
        rms = waves.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-5)
        noise = torch.randn(waves.shape, generator=generator).to(dev)
        waves = waves + noise * rms * torch.pow(10.0, -snr / 20.0)
    return waves


def measure_pos_weight(dataset: DetectionDataset, n: int, workers: int, batch: int,
                       timeout_s: float = 120.0, log: Callable[[str], None] | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """``(pos_weight, positive fraction)`` per class from ``n`` training windows."""
    subset = torch.utils.data.Subset(dataset, [(i, 0) for i in range(min(n, len(dataset)))])
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch, num_workers=workers, collate_fn=_collate,
        worker_init_fn=worker_init if workers else None, timeout=timeout_s if workers else 0)
    labels = np.concatenate([y.numpy() for _, y in guarded(loader, log=log)])
    del loader  # not persistent: its workers are gone before training starts
    pos = labels.mean(axis=0)
    weight = np.where(pos > 0, (1 - pos) / np.maximum(pos, 1e-6), 1.0)
    return weight.astype(np.float32), pos


def _collate(batch):
    from soundspy.training.datasets import collate_windows

    return collate_windows(batch)


def plot_curves(exp: ExperimentPaths, labels: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_log(exp.log)
    if not rows:
        return
    ep = [int(r["epoch"]) for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    a.plot(ep, [float(r["train_loss"]) for r in rows], label="train")
    a.plot(ep, [float(r["val_loss"]) for r in rows], label="val")
    a.set_title("BCE loss")
    a.set_xlabel("epoch")
    a.legend()
    a.grid(alpha=0.3)
    b.plot(ep, [float(r["val_map"]) for r in rows], label="val mAP", linewidth=2.5, color="k")
    for lab in labels:
        b.plot(ep, [float(r[f"ap_{lab}"]) for r in rows], label=lab, alpha=0.6)
    b.set_title("validation AP")
    b.set_xlabel("epoch")
    b.set_ylim(0, 1)
    b.legend(fontsize=7, ncol=2)
    b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(exp.plots / "training_curves.png", dpi=110)
    plt.close(fig)


def train_detection(config: AppConfig, opts: TrainOptions,
                    log: Callable[[str], None] = print) -> TrainResult:
    det = config.detection
    tr = det.training
    labels = list(config.labels)
    name = opts.name or tr.name
    exp = experiment_paths(config.paths["experiments"], name).ensure()
    device = resolve_device(opts.device)
    if device.type != "cuda" and not opts.allow_cpu:
        raise TrainingError(
            "CUDA is not available: training on the CPU would take days. Update the NVIDIA "
            "driver, restart, and check `torch.cuda.is_available()` (or use --device cpu).")
    seed_everything(tr.seed)
    limit_worker_threads()
    import warnings

    # AMP's GradScaler skips the optimizer on the first step(s) (inf/NaN check); the scheduler
    # still steps, which is harmless but makes PyTorch warn every run.
    warnings.filterwarnings("ignore", message=re.escape("Detected call of `lr_scheduler.step()`"))
    epochs = opts.epochs or (tr.smoke_epochs if opts.smoke else tr.epochs)
    workers = tr.num_workers if opts.workers is None else opts.workers
    train_limit = tr.smoke_train_recipes if opts.smoke else None
    val_limit = tr.smoke_val_recipes if opts.smoke else None

    library = ClipLibrary.from_config(config)
    train_set = DetectionDataset(load_recipes(config, "train", train_limit), library,
                                 config.labels, window_spec(config), seed=tr.seed)
    val_set = eval_dataset(config, library, "val", val_limit)
    aug = det.augment
    augment_cfg = {"freqm": aug.freqm, "timem": aug.timem, "mel_range": aug.mel_range}

    state: dict[str, Any] | None = None
    if opts.resume:
        if not exp.last.exists():
            raise TrainingError(f"nothing to resume: {exp.last} does not exist")
        model, state = load_checkpoint(exp.last)
        log(f"resuming {name} after epoch {state['epoch']}")
    else:
        if exp.last.exists():
            raise TrainingError(f"{exp.root} already has a run; use --resume or another --name")
        from soundspy.detection.weights import ensure_pretrained

        weights = (ensure_pretrained(det, config.paths["model_cache"])
                   if det.backbone != "stub" else None)
        model, used = build_model(model_spec(config, augment=augment_cfg),
                                  pretrained_weights=weights, head_init=det.head_init)
        log(f"head initialised from AudioSet rows: {used}")

    if state is not None:
        pos_weight = np.array(state["pos_weight"], dtype=np.float32)
        batch = int(state["batch_size"])
    else:
        batch = tr.batch_size
        if tr.pos_weight:
            log(f"measuring class frequencies on {opts.pos_weight_windows} training windows …")
            pos_weight, pos = measure_pos_weight(train_set, opts.pos_weight_windows, workers,
                                                 batch, tr.loader_timeout_s, log)
            log("positive fraction: " + ", ".join(f"{k} {p:.3f}" for k, p in zip(labels, pos,
                                                                                    strict=True)))
        else:
            pos_weight = np.ones(len(labels), np.float32)
        write_json(exp.config, {
            "name": name, "task": "detection", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git": git_commit(config.project_root),
            "data_index_hash": file_hash([
                config.paths["processed"] / "clip_index.parquet",
                *(config.paths["mixtures"] / f"{s}.jsonl" for s in ("train", "val", "test"))]),
            "options": asdict(opts), "epochs": epochs, "device": str(device),
            "torch": torch.__version__, "labels": labels, "model": model.spec.to_json(),
            "pos_weight": dict(zip(labels, pos_weight.tolist(), strict=True)),
            "detection": config.detection, "mixtures": config.data.mixtures,
            "train_recipes": len(train_set), "val_recipes": len(val_set),
        })

    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weight).to(device))
    use_amp = tr.amp and device.type == "cuda"
    gen = torch.Generator().manual_seed(tr.seed)
    start_epoch = int(state["epoch"]) if state else 0
    best_map = float(state["best_map"]) if state else -1.0
    best_epoch = int(state["best_epoch"]) if state else 0
    truncate_log(exp.log, start_epoch)

    while True:  # restarted with a smaller batch on out-of-memory
        body, head = model.parameter_groups()
        optimizer = torch.optim.AdamW(
            [{"params": body, "lr": tr.lr_backbone}, {"params": head, "lr": tr.lr_head}],
            weight_decay=tr.weight_decay)
        steps_per_epoch = math.ceil(len(train_set) / batch)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda(int(tr.warmup_epochs * steps_per_epoch),
                                 epochs * steps_per_epoch))
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        if state is not None and int(state["batch_size"]) == batch:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
        elif start_epoch > 0:  # new batch size after OOM: continue the LR schedule where it was
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(start_epoch * steps_per_epoch):
                    scheduler.step()
        sampler = EpochSampler(len(train_set), seed=tr.seed)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch, sampler=sampler, num_workers=workers,
            persistent_workers=workers > 0, pin_memory=device.type == "cuda",
            collate_fn=_collate, drop_last=len(train_set) > batch,
            worker_init_fn=worker_init if workers else None,
            timeout=tr.loader_timeout_s if workers else 0)
        # validation uses half the workers: every worker costs ~1 GB of committed memory
        val_loader = make_loader(val_set, batch, (workers + 1) // 2 if workers else 0,
                                 timeout_s=tr.loader_timeout_s)
        try:
            for epoch in range(start_epoch, epochs):
                sampler.set_epoch(epoch)
                t0 = time.monotonic()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                model.train()
                losses = []
                for waves, y in guarded(train_loader, what="training batch",
                                        log=log if epoch == start_epoch else None):
                    waves = augment(waves.to(device, non_blocking=True), det.augment.gain_db,
                                    det.augment.noise_snr_db, gen)
                    y = y.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16,
                                        enabled=use_amp):
                        logits = model(waves)
                    loss = criterion(logits.float(), y)
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    losses.append(float(loss.detach()))
                model.eval()
                yv, (pv,), val_loss = predict([FineTuned(model)], val_loader, device,
                                              amp=tr.amp, criterion=criterion)
                ev = M.evaluate(yv, pv, labels)
                peak = (torch.cuda.max_memory_allocated() / 2**20
                        if device.type == "cuda" else 0.0)
                row = {"epoch": epoch + 1, "train_loss": round(float(np.mean(losses)), 5),
                       "val_loss": round(val_loss, 5), "val_map": round(ev["map"], 5),
                       **{f"ap_{k}": round(v["ap"], 5) for k, v in ev["per_class"].items()},
                       "lr_backbone": optimizer.param_groups[0]["lr"],
                       "lr_head": optimizer.param_groups[1]["lr"],
                       "epoch_s": round(time.monotonic() - t0, 1),
                       "gpu_mem_peak_mib": round(peak, 1), "batch_size": batch}
                append_log(exp.log, row)
                improved = ev["map"] > best_map
                if improved:
                    best_map, best_epoch = ev["map"], epoch + 1
                    save_checkpoint(exp.best, model, {"epoch": epoch + 1, "val_map": best_map})
                save_checkpoint(exp.last, model, {
                    "epoch": epoch + 1, "best_map": best_map, "best_epoch": best_epoch,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "batch_size": batch,
                    "pos_weight": pos_weight.tolist()})
                log(f"epoch {epoch + 1}/{epochs}: train {row['train_loss']:.4f}  val "
                    f"{row['val_loss']:.4f}  val mAP {row['val_map']:.4f}"
                    f"{'  *best*' if improved else ''}  ({row['epoch_s']:.0f} s, "
                    f"{row['gpu_mem_peak_mib']:.0f} MiB)")
                plot_curves(exp, labels)
            break
        except torch.cuda.OutOfMemoryError:
            if batch <= 1:
                raise
            del train_loader, val_loader
            torch.cuda.empty_cache()
            batch //= 2
            log(f"GPU out of memory: batch size lowered to {batch}, restarting the epoch")
            if exp.last.exists():
                model, state = load_checkpoint(exp.last)
                model.to(device)
                start_epoch = int(state["epoch"])  # optimizer state rebuilt for the new batch
            truncate_log(exp.log, start_epoch)

    finalize(config, exp, device, workers, val_set, log)
    return TrainResult(exp, epochs - (int(state["epoch"]) if state else 0), best_epoch,
                       best_map, batch)


def finalize(config: AppConfig, exp: ExperimentPaths, device: torch.device, workers: int,
             val_set: DetectionDataset, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Tune thresholds on val with ``best.pt`` and write thresholds.json + val metrics."""
    from soundspy.training.experiment import read_json

    labels = list(config.labels)
    model, payload = load_checkpoint(exp.best)
    model.to(device)
    tr = config.detection.training
    loader = make_loader(val_set, tr.batch_size, (workers + 1) // 2 if workers else 0,
                         timeout_s=tr.loader_timeout_s)
    y, (p,), _ = predict([FineTuned(model)], loader, device, amp=config.detection.training.amp)
    thresholds = M.tune_thresholds(y, p, labels)
    write_json(exp.thresholds, {"thresholds": thresholds, "tuned_on": "val",
                                "windows": int(y.shape[0]), "epoch": payload.get("epoch")})
    result = M.evaluate(y, p, labels, thresholds)
    metrics = read_json(exp.metrics) if exp.metrics.exists() else {}
    metrics.setdefault("val", {})["fine_tuned"] = result
    metrics["val"]["windows"] = int(y.shape[0])
    write_json(exp.metrics, metrics)
    plot_curves(exp, labels)
    log(f"thresholds tuned on val; val mAP {result['map']:.4f}, F1 {result['f1']:.4f}")
    return result


def describe(model: DetectorNet) -> str:
    n = sum(p.numel() for p in model.parameters())
    return f"{model.spec.backbone}, {n / 1e6:.2f} M parameters"


__all__ = ["TrainOptions", "TrainResult", "TrainingError", "train_detection", "finalize",
           "describe"]
