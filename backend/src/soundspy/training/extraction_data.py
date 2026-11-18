"""PyTorch datasets over the pre-rendered extraction cache (``data/cache.py``), and the loss.

**Training items** are random crops: ``train_crop_s`` seconds out of the 6 s example (a full
example does not fit the 4 GB card at a useful batch size). With the cached training set of
step 07 (``ExtractionTrainSet``) a crop is addressed by ``(index, epoch)`` and its position comes
from a generator seeded with ``(seed, index, epoch)``; with dynamic mixing (step 07b,
``training/datasets.py: DynamicExtractionSet``) the whole mixture is new for every
``(index, epoch)``. With probability ``crop_target_prob`` the crop is placed so that it overlaps
the stretch where the target sounds; otherwise anywhere.

**Validation items** are whole examples, always the same ones.

**The loss** (:func:`extraction_loss`) has two parts, both in dB::

    quality (examples whose target sounds):
        -( snr_weight * SNR(est, target) + si_snr_weight * SI-SNR(est, target) )
    leakage (every frame of every example in which the target is silent):
        max( 10 log10( E_out / E_mix ), leak_floor_db ) - leak_floor_db

    loss = sum(quality) / batch  +  leak_weight * sum(leakage over silent frames) / (batch * frames)

The leakage of a frame is therefore 0 for a silent output (or one at least ``-leak_floor_db``
= 50 dB below the mixture) and 50 for an output that simply copies the mixture.

* *SNR* compares the estimate with the target sample by sample: both shape and **level** must be
  right. *SI-SNR* ignores the level and is what the evaluation reports. Sound Spy uses mostly
  SI-SNR (0.8) with a share of SNR (0.2): the SNR share is what fixes the level of the estimate
  (SI-SNR alone leaves it free, and the attenuation slider mixes the estimate with the input).
* **Leakage, frame by frame (step 07b).** SI-SNR over a 3 s crop is decided by the loud part: a
  model that keeps humming the background during the target's pauses loses almost nothing. So
  the crop is cut into ``leak_frame_ms`` frames; in every frame whose *true target* is below
  ``leak_silence_dbfs`` the correct output is silence, and the output's energy relative to the
  mixture's in that frame is penalised. Below ``leak_floor_db`` (−50 dB) there is no further
  reward, otherwise the easiest way to lower the loss would be to push already inaudible
  outputs towards minus infinity. An example whose target class is absent is simply all silent
  frames: it gets ``leak_weight`` times its mean leakage (step 07 used one whole-crop ratio for
  those examples only, and nothing for the silent stretches of the others).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from soundspy.config import ExtractionLossConfig
from soundspy.data.cache import ExtractionCache
from soundspy.data.dynamic import crop_start

SILENT_POWER = 1e-7  # mean square below this (-70 dBFS): the target does not sound in the crop
_EPS = 1e-8


class ExtractionTrainSet(Dataset):
    """Random crops ``(mixture, target, label)``; items are ``(index, epoch)`` pairs."""

    def __init__(self, cache: ExtractionCache, indices: Sequence[int], *, crop_s: float,
                 crop_target_prob: float, seed: int) -> None:
        self.cache = cache
        self.indices = list(indices)
        self.crop = min(cache.samples, round(crop_s * cache.sample_rate))
        self.crop_target_prob = crop_target_prob
        self.seed = seed

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, key: int | tuple[int, int]
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        position, epoch = key if isinstance(key, tuple) else (key, 0)
        i = self.indices[position]
        cache = self.cache
        rng = np.random.default_rng([self.seed, i, epoch])
        start = crop_start(rng, cache.samples, self.crop, int(cache.target_start[i]),
                           int(cache.target_end[i]), cache.sample_rate, self.crop_target_prob)
        pair = np.asarray(cache.pair(i)[..., start:start + self.crop], dtype=np.float32)
        return (torch.from_numpy(pair[0].copy()), torch.from_numpy(pair[1].copy()),
                torch.from_numpy(cache.target_hot[i].copy()))


class ExtractionEvalSet(Dataset):
    """Whole examples ``(mixture, target, label)`` in a fixed order."""

    def __init__(self, cache: ExtractionCache, indices: Sequence[int]) -> None:
        self.cache = cache
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = self.indices[position]
        pair = np.asarray(self.cache.pair(i), dtype=np.float32)
        return (torch.from_numpy(pair[0].copy()), torch.from_numpy(pair[1].copy()),
                torch.from_numpy(self.cache.target_hot[i].copy()))


def collate(batch):
    mixtures, targets, labels = zip(*batch, strict=True)
    return torch.stack(mixtures), torch.stack(targets), torch.stack(labels)


# --------------------------------------------------------------------------------------------
# Ratios in dB, per example: tensors are (batch, samples)


def _db(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10((numerator + _EPS) / (denominator + _EPS))


def energy(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).sum(dim=-1)


def snr_db(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return _db(energy(reference), energy(reference - estimate))


def si_snr_db(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Scale-invariant SNR (= SI-SDR): the estimate is first projected onto the reference."""
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    scale = (estimate * reference).sum(dim=-1, keepdim=True) / (energy(reference).unsqueeze(-1)
                                                                + _EPS)
    projection = scale * reference
    return _db(energy(projection), energy(estimate - projection))


def is_silent(reference: torch.Tensor) -> torch.Tensor:
    return reference.pow(2).mean(dim=-1) < SILENT_POWER


MIX_FLOOR_POWER = 1e-9  # a mixture frame is never taken to be quieter than -90 dBFS


def frame_power(x: torch.Tensor, frame: int) -> torch.Tensor:
    """Mean square per frame: ``(batch, samples)`` -> ``(batch, frames)`` (a rest is dropped)."""
    frames = x.shape[-1] // frame
    return x[..., :frames * frame].reshape(*x.shape[:-1], frames, frame).pow(2).mean(dim=-1)


def frame_leakage(estimate: torch.Tensor, target: torch.Tensor, mixture: torch.Tensor,
                  frame: int, silence_dbfs: float, floor_db: float
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(leak_db, silent)``, both ``(batch, frames)``: the output's level against the
    mixture's in every frame in dB (not below ``floor_db``), and where the true target is
    silent."""
    frame = max(1, min(frame, estimate.shape[-1]))
    silent = frame_power(target, frame) < 10.0 ** (silence_dbfs / 10.0)
    ratio = frame_power(estimate, frame) / frame_power(mixture, frame).clamp_min(MIX_FLOOR_POWER)
    leak = 10.0 * torch.log10(ratio + _EPS)
    return torch.clamp(leak, min=floor_db), silent


def extraction_loss(estimate: torch.Tensor, target: torch.Tensor, mixture: torch.Tensor,
                    cfg: ExtractionLossConfig, sample_rate: int
                    ) -> tuple[torch.Tensor, dict[str, float]]:
    """The batch loss (see the module docstring) and its parts for logging."""
    estimate, target, mixture = estimate.float(), target.float(), mixture.float()
    empty = is_silent(target)
    quality = -(cfg.snr_weight * snr_db(estimate, target)
                + cfg.si_snr_weight * si_snr_db(estimate, target))
    quality_sum = torch.where(empty, torch.zeros_like(quality), quality).sum()
    frame = round(cfg.leak_frame_ms * sample_rate / 1000.0)
    leak, silent = frame_leakage(estimate, target, mixture, frame, cfg.leak_silence_dbfs,
                                 cfg.leak_floor_db)
    penalty = leak - cfg.leak_floor_db  # 0 = silent (or below the floor)
    leak_sum = torch.where(silent, penalty, torch.zeros_like(penalty)).sum()
    batch = estimate.shape[0]
    loss = quality_sum / batch + cfg.leak_weight * leak_sum / (batch * leak.shape[-1])
    n_empty = int(empty.sum())
    n_silent = int(silent.sum())
    parts = {
        "quality": float(quality[~empty].mean().detach()) if n_empty < batch else float("nan"),
        "leak_db": float(leak[silent].mean().detach()) if n_silent else float("nan"),
        "silent": n_silent / max(1, silent.numel()),  # share of target-silent frames
        "empty": n_empty / max(1, batch),  # share of examples without any target
    }
    return loss, parts
