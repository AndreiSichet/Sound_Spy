"""PyTorch datasets over the pre-rendered extraction cache (``data/cache.py``), and the loss.

**Training items** are random crops: ``train_crop_s`` seconds out of the 6 s example (a full
example does not fit the 4 GB card at a useful batch size). A crop is addressed by
``(index, epoch)`` and its position comes from a generator seeded with ``(seed, index, epoch)``,
so every epoch sees new crops and a run can be repeated exactly. With probability
``crop_target_prob`` the crop is placed so that it overlaps the stretch where the target sounds;
otherwise anywhere. Without that, short targets would mostly fall outside the crop and most
training examples would ask for silence.

**Validation items** are whole examples, always the same ones.

**The loss** (:func:`extraction_loss`), per example, in dB::

    target sounds:     -( snr_weight * SNR(est, target) + si_snr_weight * SI-SNR(est, target) )
    target is silent:   max( 10 log10( |est|^2 / |mixture|^2 ), empty_floor_db )

and the batch loss is ``(sum of the first + empty_weight * sum of the second) / batch size``.

* *SNR* compares the estimate with the target sample by sample: both shape and **level** must be
  right. *SI-SNR* ignores the level and is what the evaluation reports. Sound Spy uses mostly
  SI-SNR (0.8) with a share of SNR (0.2): the SNR share is what fixes the level of the estimate
  (SI-SNR alone leaves it free, and the attenuation slider mixes the estimate with the input).
  The published Waveformer was trained with 0.9 SNR / 0.1 SI-SNR; that was tried first and
  left a tail of badly wrong outputs: extracting the *wrong* sound costs only ~3 dB of SNR
  (staying quiet is the safe choice) but 20-40 dB of SI-SNR, so the SI-SNR-heavy loss pushes
  much harder on exactly those cases.
* For a silent target neither ratio exists (division by zero energy). The correct output is
  silence, so the loss is the output's energy relative to the mixture's, in dB: lower is
  better. Below ``empty_floor_db`` (−40 dB) there is no further reward, otherwise the easiest
  way to lower the loss would be to push already inaudible outputs towards minus infinity.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from soundspy.config import ExtractionLossConfig
from soundspy.data.cache import ExtractionCache

SILENT_POWER = 1e-7  # mean square below this (-70 dBFS): the target does not sound in the crop
MIN_OVERLAP_S = 0.25  # a "target" crop holds at least this much of the target's stretch
_EPS = 1e-8


def crop_start(rng: np.random.Generator, samples: int, crop: int, target_start: int,
               target_end: int, sample_rate: int, target_prob: float) -> int:
    """Where a crop of ``crop`` samples starts inside an example of ``samples``."""
    last = samples - crop
    if last <= 0:
        return 0
    span = target_end - target_start
    if span > 0 and rng.random() < target_prob:
        overlap = min(span, crop, int(MIN_OVERLAP_S * sample_rate))
        low = max(0, target_start + overlap - crop)
        high = min(last, target_end - overlap)
        if high >= low:
            return int(rng.integers(low, high + 1))
    return int(rng.integers(0, last + 1))


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


def extraction_loss(estimate: torch.Tensor, target: torch.Tensor, mixture: torch.Tensor,
                    cfg: ExtractionLossConfig) -> tuple[torch.Tensor, dict[str, float]]:
    """The batch loss (see the module docstring) and its parts for logging."""
    estimate, target, mixture = estimate.float(), target.float(), mixture.float()
    silent = is_silent(target)
    quality = -(cfg.snr_weight * snr_db(estimate, target)
                + cfg.si_snr_weight * si_snr_db(estimate, target))
    leak = torch.clamp(_db(energy(estimate), energy(mixture)), min=cfg.empty_floor_db)
    per_example = torch.where(silent, cfg.empty_weight * leak, quality)
    n_silent = int(silent.sum())
    n_sound = silent.numel() - n_silent
    parts = {
        "quality": float(quality[~silent].mean().detach()) if n_sound else float("nan"),
        "leak_db": float(leak[silent].mean().detach()) if n_silent else float("nan"),
        "silent": n_silent / max(1, silent.numel()),
    }
    return per_example.mean(), parts
