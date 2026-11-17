"""Fine-tuning the neural extractor (``soundspy train extraction``).

The published Waveformer is adapted to Sound Spy's classes (``extraction/neural_model.py``) and
trained on the pre-rendered mixtures (``data/cache.py``): random ``train_crop_s`` crops, a random
gain on mixture and target together, faint white or pink "microphone" noise on the mixture only.

* **Batch size.** The largest that fits the graphics card is found before training: batch sizes
  from ``batch_size`` up to ``max_batch_size`` are tried with one real training step each, and
  the largest whose peak memory stays under ``VRAM_FRACTION`` of the card is used. (On Windows a
  step that needs more than the card has does not fail, it silently spills into slow shared
  memory, so the peak is measured rather than waiting for an error.) Gradients are accumulated
  over several batches to reach ``effective_batch_size``.
* **Optimisation.** AdamW with two learning rates (the new first label layer learns faster),
  linear warm-up then cosine decay per optimiser step, mixed precision, gradient-norm clipping.
* **Validation** after every epoch on a fixed set of whole 6 s examples: mean **SI-SDRi** (how
  much closer to the target the output is than the unprocessed mixture) over the examples with a
  target, and the mean leak in dB over those without. SI-SDRi and SNRi are measured on **what
  the system outputs**, ``estimate + g * (mixture - estimate)`` at the evaluation's attenuation
  (30 dB), exactly as ``soundspy evaluate extraction`` does: the checkpoint that is kept is the
  best one for the number that is reported. (For a confident estimate the 30 dB residue changes
  nothing; for a nearly silent estimate it makes the output "the mixture, quieter", which is
  what the user hears, instead of an arbitrary direction with a wildly negative score.) The
  leak is measured on the bare estimate. ``best.pt`` is the epoch with the best SI-SDRi;
  training stops early after ``patience`` epochs without a better one.
* The experiment folder follows detection: ``config.json``, ``log.csv``, ``checkpoints/best.pt``
  and ``last.pt`` (with optimiser state for ``--resume``), ``metrics.json``, ``plots/``.
"""

from __future__ import annotations

import math
import re
import time
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from soundspy.config import AppConfig, ExtractionTrainingConfig
from soundspy.data.cache import ExtractionCache, split_dir
from soundspy.extraction.models.waveformer import Net
from soundspy.extraction.neural_model import (
    build_finetune_model,
    count_parameters,
    ensure_pretrained,
    load_checkpoint,
    parameter_groups,
    save_checkpoint,
)
from soundspy.training.datasets import guarded, limit_worker_threads, worker_init
from soundspy.training.evaluate import resolve_device
from soundspy.training.experiment import (
    ExperimentPaths,
    append_log,
    experiment_paths,
    git_commit,
    read_log,
    truncate_log,
    write_json,
)
from soundspy.training.extraction_data import (
    ExtractionEvalSet,
    ExtractionTrainSet,
    collate,
    extraction_loss,
    is_silent,
    si_snr_db,
    snr_db,
)
from soundspy.training.train import TrainingError, lr_lambda, seed_everything

VRAM_FRACTION = 0.72  # of the card's memory a training step may use (the rest: desktop, spikes)


class RotatingSampler(torch.utils.data.Sampler):
    """``per_epoch`` of ``n`` items per epoch, as ``(index, epoch)`` pairs.

    The items are shuffled and dealt out over consecutive epochs: with ``n = 20,000`` and
    ``per_epoch = 10,000`` epochs 0 and 1 together cover every item once, then a new shuffle
    starts. An epoch is therefore shorter than a pass over the data, but nothing is left out.
    """

    def __init__(self, n: int, per_epoch: int, *, seed: int) -> None:
        self.n, self.per_epoch, self.seed, self.epoch = n, min(n, per_epoch), seed, 0
        self.rounds = max(1, n // self.per_epoch)  # epochs per pass

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.per_epoch

    def __iter__(self):
        order = np.random.default_rng([self.seed, self.epoch // self.rounds]).permutation(self.n)
        start = (self.epoch % self.rounds) * self.per_epoch
        for index in order[start:start + self.per_epoch]:
            yield int(index), self.epoch


@dataclass
class ExtractionTrainOptions:
    name: str | None = None
    epochs: int | None = None
    smoke: bool = False
    resume: bool = False
    device: str = "auto"
    allow_cpu: bool = False  # tests
    workers: int | None = None
    pretrained: bool = True  # False: random weights (tests)
    batch_size: int | None = None  # skip the search


@dataclass
class ExtractionTrainResult:
    paths: ExperimentPaths
    epochs_run: int
    best_epoch: int
    best_si_sdri: float
    batch_size: int
    accumulate: int
    stopped_early: bool


# --------------------------------------------------------------------------------------------
# Pieces


def pink_noise(shape: tuple[int, int], generator: torch.Generator) -> torch.Tensor:
    """Unit-RMS noise whose power falls with 1/frequency (more rumble than hiss)."""
    white = torch.randn(shape, generator=generator)
    spectrum = torch.fft.rfft(white, dim=-1)
    scale = torch.arange(1, spectrum.shape[-1] + 1, dtype=torch.float32).rsqrt()
    noise = torch.fft.irfft(spectrum * scale, n=shape[-1], dim=-1)
    return noise / noise.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)


def augment(mixture: torch.Tensor, target: torch.Tensor, tr: ExtractionTrainingConfig,
            generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Random gain on both signals; microphone noise on the mixture only (CPU tensors)."""
    n = mixture.shape[0]
    if tr.gain_db > 0:
        gain = torch.pow(10.0, (torch.rand(n, 1, generator=generator) * 2 - 1) * tr.gain_db / 20)
        mixture, target = mixture * gain, target * gain
    if tr.mic_noise_db is not None:
        low, high = tr.mic_noise_db
        level = torch.pow(10.0, (torch.rand(n, 1, generator=generator) * (high - low) + low) / 20)
        white = torch.randn(mixture.shape, generator=generator)
        pink = pink_noise(tuple(mixture.shape), generator)
        use_pink = torch.rand(n, 1, generator=generator) < 0.5
        mixture = mixture + level * torch.where(use_pink, pink, white)
    return mixture, target


def training_step(model: Net, mixture: torch.Tensor, target: torch.Tensor, label: torch.Tensor,
                  tr: ExtractionTrainingConfig, use_amp: bool
                  ) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.autocast(device_type=mixture.device.type, dtype=torch.float16, enabled=use_amp):
        estimate = model(mixture.unsqueeze(1), label).squeeze(1)
    return extraction_loss(estimate, target, mixture, tr.loss)


def find_batch_size(model: Net, device: torch.device, crop: int, tr: ExtractionTrainingConfig,
                    use_amp: bool, log: Callable[[str], None]) -> int:
    """The largest batch size whose training step stays within ``VRAM_FRACTION`` of the card."""
    if device.type != "cuda":
        return tr.batch_size
    total = torch.cuda.get_device_properties(device).total_memory
    budget = VRAM_FRACTION * total
    # from small to large, stopping at the first that does not fit (``batch_size`` is only the
    # size hoped for: at 3 s crops this network needs ~0.55 GB per example to train)
    candidates = sorted({b for b in (1, 2, 3, 4, 6, tr.batch_size, 12, 16, 24, 32, 48, 64)
                         if b <= tr.max_batch_size})
    best = None
    labels = model.label_len
    for batch in candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            mixture = torch.randn(batch, crop, device=device) * 0.1
            target = torch.randn(batch, crop, device=device) * 0.05
            label = torch.zeros(batch, labels, device=device)
            label[..., 0] = 1.0
            started = time.perf_counter()
            for _ in range(2):
                loss, _ = training_step(model, mixture, target, label, tr, use_amp)
                loss.backward()
                model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            seconds = (time.perf_counter() - started) / 2
        except torch.cuda.OutOfMemoryError:
            log(f"  batch {batch}: out of memory")
            break
        peak = torch.cuda.max_memory_allocated(device)
        fits = peak <= budget
        log(f"  batch {batch}: peak {peak / 2**20:.0f} MiB of {total / 2**20:.0f}, "
            f"{seconds * 1000:.0f} ms per step ({batch / seconds:.0f} crops/s)"
            f"{'' if fits else '  -> too much'}")
        if not fits:
            break
        best = batch
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    if best is None:
        raise TrainingError(
            f"even batch size 1 needs more than {VRAM_FRACTION:.0%} of the graphics card: "
            "lower training.train_crop_s")
    return best


@torch.inference_mode()
def validate(model: Net, loader, device: torch.device, use_amp: bool,
             attenuation_db: float = math.inf) -> dict[str, float]:
    """Mean SI-SDRi / SNRi of the system output (the estimate with the rest of the mixture
    ``attenuation_db`` quieter; ``inf`` = the bare estimate) over the examples with a target;
    mean leak of the bare estimate over those without."""
    model.eval()
    residue = 0.0 if math.isinf(attenuation_db) else 10.0 ** (-attenuation_db / 20.0)
    si_sdri, snri, si_sdr, leak = [], [], [], []
    for mixture, target, label in guarded(loader, what="validation batch"):
        mixture = mixture.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            estimate = model(mixture.unsqueeze(1), label.to(device)).squeeze(1)
        estimate = estimate.float()
        silent = is_silent(target)
        sound = ~silent
        if sound.any():
            est, tgt, mix = estimate[sound], target[sound], mixture[sound]
            est = est + residue * (mix - est)
            after = si_snr_db(est, tgt)
            si_sdr += after.tolist()
            si_sdri += (after - si_snr_db(mix, tgt)).tolist()
            snri += (snr_db(est, tgt) - snr_db(mix, tgt)).tolist()
        if silent.any():
            est, mix = estimate[silent], mixture[silent]
            ratio = 10 * torch.log10((est.pow(2).sum(-1) + 1e-8) / (mix.pow(2).sum(-1) + 1e-8))
            leak += ratio.tolist()
    mean = lambda v: float(np.mean(v)) if v else float("nan")  # noqa: E731
    return {"si_sdri": mean(si_sdri), "snri": mean(snri), "si_sdr": mean(si_sdr),
            "leak_db": mean(leak), "with_target": len(si_sdri), "without_target": len(leak)}


def plot_curves(exp: ExperimentPaths) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_log(exp.log)
    if not rows:
        return
    epochs = [int(r["epoch"]) for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    a.plot(epochs, [float(r["train_loss"]) for r in rows], label="training loss")
    a.set_title("training loss (lower is better)")
    a.set_xlabel("epoch")
    a.grid(alpha=0.3)
    b.plot(epochs, [float(r["val_si_sdri"]) for r in rows], label="SI-SDRi", linewidth=2.5,
           color="k")
    b.plot(epochs, [float(r["val_snri"]) for r in rows], label="SNRi", alpha=0.7)
    b.set_title("validation improvement over the mixture (dB, higher is better)")
    b.set_xlabel("epoch")
    b.legend()
    b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(exp.plots / "training_curves.png", dpi=110)
    plt.close(fig)


def _loader(dataset, batch: int, workers: int, timeout_s: float, device: torch.device,
            sampler=None, drop_last: bool = False):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch, sampler=sampler, shuffle=False, num_workers=workers,
        persistent_workers=workers > 0, pin_memory=device.type == "cuda", collate_fn=collate,
        drop_last=drop_last, worker_init_fn=worker_init if workers else None,
        timeout=timeout_s if workers else 0)


# --------------------------------------------------------------------------------------------
# The run


def train_extraction(config: AppConfig, opts: ExtractionTrainOptions,
                     log: Callable[[str], None] = print) -> ExtractionTrainResult:
    cfg = config.extraction.neural_tse
    tr = cfg.training
    eval_db = config.extraction.evaluation.attenuation_db  # validation = the evaluated output
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
    warnings.filterwarnings("ignore", message=re.escape("Detected call of `lr_scheduler.step()`"))
    epochs = opts.epochs or (tr.smoke_epochs if opts.smoke else tr.epochs)
    workers = tr.num_workers if opts.workers is None else opts.workers
    use_amp = tr.amp and device.type == "cuda"

    rate = cfg.model.sample_rate
    train_cache = ExtractionCache(split_dir(config, rate, "train"))
    val_cache = ExtractionCache(split_dir(config, rate, "val"))
    for cache in (train_cache, val_cache):
        if cache.labels != labels:
            raise TrainingError(f"the cache at {cache.directory} was made for other classes; "
                                "rebuild it with `soundspy data cache --task extraction --force`")
    n_train = min(len(train_cache), tr.smoke_train_recipes if opts.smoke else tr.train_recipes)
    n_val = min(len(val_cache), tr.smoke_val_examples if opts.smoke else tr.val_examples)
    per_epoch = min(n_train, tr.epoch_recipes)
    train_set = ExtractionTrainSet(train_cache, range(n_train), crop_s=tr.train_crop_s,
                                   crop_target_prob=tr.crop_target_prob, seed=tr.seed)
    val_set = ExtractionEvalSet(val_cache, range(n_val))

    state: dict[str, Any] | None = None
    if opts.resume:
        if not exp.last.exists():
            raise TrainingError(f"nothing to resume: {exp.last} does not exist")
        model, _, saved_labels, state = load_checkpoint(exp.last)
        if saved_labels != labels:
            raise TrainingError(f"{exp.last} was trained for other classes: {saved_labels}")
        log(f"resuming {name} after epoch {state['epoch']}")
    else:
        if exp.last.exists():
            raise TrainingError(f"{exp.root} already has a run; use --resume or another --name")
        weights = (ensure_pretrained(cfg, config.paths["model_cache"]) if opts.pretrained
                   else None)
        model = build_finetune_model(cfg, labels, weights)
        log(f"Waveformer: {count_parameters(model):,} parameters, "
            f"{'pretrained, label layer adapted' if weights else 'random weights'}")
    model.to(device)

    if state is not None:
        batch = int(state["batch_size"])
    elif opts.batch_size:
        batch = opts.batch_size
    else:
        log("finding the batch size …")
        model.train()
        batch = find_batch_size(model, device, train_set.crop, tr, use_amp, log)
    accumulate = max(1, math.ceil(tr.effective_batch_size / batch))
    log(f"batch size {batch} x {accumulate} accumulated = {batch * accumulate} per update; "
        f"{per_epoch} crops of {train_set.crop / rate:.1f} s per epoch from {n_train} recipes, "
        f"{n_val} validation examples")

    if state is None:
        write_json(exp.config, {
            "name": name, "task": "extraction", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git": git_commit(config.project_root), "options": asdict(opts), "epochs": epochs,
            "device": str(device), "torch": torch.__version__, "labels": labels,
            "parameters": count_parameters(model), "batch_size": batch,
            "accumulate": accumulate, "neural_tse": cfg, "mixtures": config.data.mixtures,
            "train_recipes": n_train, "epoch_recipes": per_epoch, "val_examples": n_val,
            "train_cache": train_cache.index["signature"],
            "val_cache": val_cache.index["signature"],
        })

    optimizer = torch.optim.AdamW(parameter_groups(model, tr.lr_backbone, tr.lr_label),
                                  weight_decay=tr.weight_decay)
    batches = per_epoch // batch if per_epoch > batch else 1
    updates_per_epoch = math.ceil(batches / accumulate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(int(tr.warmup_epochs * updates_per_epoch),
                             epochs * updates_per_epoch))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, best, best_epoch, stale = 0, -math.inf, 0, 0
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best = int(state["epoch"]), float(state["best_si_sdri"])
        best_epoch, stale = int(state["best_epoch"]), int(state["stale_epochs"])
    truncate_log(exp.log, start_epoch)

    sampler = RotatingSampler(n_train, per_epoch, seed=tr.seed)
    train_loader = _loader(train_set, batch, workers, tr.loader_timeout_s, device,
                           sampler=sampler, drop_last=per_epoch > batch)
    val_loader = _loader(val_set, max(1, batch // 2), (workers + 1) // 2 if workers else 0,
                         tr.loader_timeout_s, device)
    generator = torch.Generator().manual_seed(tr.seed + start_epoch)
    stopped_early = False
    run_started = time.monotonic()

    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        t0 = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        losses, quality, leak = [], [], []
        optimizer.zero_grad(set_to_none=True)
        for step, (mixture, target, label) in enumerate(
                guarded(train_loader, what="training batch",
                        log=log if epoch == start_epoch else None), 1):
            mixture, target = augment(mixture, target, tr, generator)
            mixture = mixture.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            loss, parts = training_step(model, mixture, target, label.to(device), tr, use_amp)
            scaler.scale(loss / accumulate).backward()
            if step % accumulate == 0 or step == batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            losses.append(float(loss.detach()))
            if not math.isnan(parts["quality"]):
                quality.append(parts["quality"])
            if not math.isnan(parts["leak_db"]):
                leak.append(parts["leak_db"])
        train_s = time.monotonic() - t0
        val = validate(model, val_loader, device, use_amp, eval_db)
        peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
        row = {"epoch": epoch + 1, "train_loss": round(float(np.mean(losses)), 4),
               "train_quality": round(float(np.mean(quality)), 4) if quality else "",
               "train_leak_db": round(float(np.mean(leak)), 2) if leak else "",
               "val_si_sdri": round(val["si_sdri"], 4), "val_snri": round(val["snri"], 4),
               "val_si_sdr": round(val["si_sdr"], 4), "val_leak_db": round(val["leak_db"], 2),
               "lr_backbone": optimizer.param_groups[0]["lr"],
               "lr_label": optimizer.param_groups[1]["lr"],
               "epoch_s": round(time.monotonic() - t0, 1), "train_s": round(train_s, 1),
               "gpu_mem_peak_mib": round(peak, 1), "batch_size": batch}
        append_log(exp.log, row)
        improved = val["si_sdri"] > best
        if improved:
            best, best_epoch, stale = val["si_sdri"], epoch + 1, 0
            save_checkpoint(exp.best, model, cfg.model, labels,
                            {"epoch": epoch + 1, "val_si_sdri": best, "val": val})
        else:
            stale += 1
        save_checkpoint(exp.last, model, cfg.model, labels, {
            "epoch": epoch + 1, "best_si_sdri": best, "best_epoch": best_epoch,
            "stale_epochs": stale, "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "batch_size": batch})
        log(f"epoch {epoch + 1}/{epochs}: loss {row['train_loss']:.3f}  val SI-SDRi "
            f"{val['si_sdri']:+.2f} dB  SNRi {val['snri']:+.2f} dB  leak {val['leak_db']:.1f} dB"
            f"{'  *best*' if improved else ''}  ({row['epoch_s']:.0f} s, {peak:.0f} MiB)")
        plot_curves(exp)
        if epoch == start_epoch and epochs - epoch > 1:
            total_h = row["epoch_s"] * (epochs - start_epoch) / 3600
            note = "" if total_h <= tr.time_budget_h else \
                f"  (over the {tr.time_budget_h:g} h budget: fewer recipes, a shorter crop or " \
                "fewer epochs would be needed)"
            log(f"at this pace the {epochs - start_epoch} epochs of this run take about "
                f"{total_h:.1f} h in total{note}")
        if stale >= tr.patience:
            log(f"no better validation SI-SDRi for {tr.patience} epochs: stopping early")
            stopped_early = True
            break

    epochs_run = (epoch + 1 - start_epoch) if epochs > start_epoch else 0
    metrics: dict[str, Any] = {"best_epoch": best_epoch, "best_val_si_sdri": best,
                               "epochs_run_total": start_epoch + epochs_run,
                               "stopped_early": stopped_early, "batch_size": batch,
                               "accumulate": accumulate,
                               "hours": round((time.monotonic() - run_started) / 3600, 2)}
    if exp.best.exists():
        best_model, _, _, extra = load_checkpoint(exp.best, device)
        metrics["val"] = validate(best_model, val_loader, device, use_amp, eval_db)
        metrics["val_attenuation_db"] = eval_db
        metrics["val"]["epoch"] = extra.get("epoch")
    write_json(exp.metrics, metrics)
    log(f"best epoch {best_epoch}: val SI-SDRi {best:+.2f} dB -> {exp.root}")
    return ExtractionTrainResult(exp, epochs_run, best_epoch, best, batch, accumulate,
                                 stopped_early)
