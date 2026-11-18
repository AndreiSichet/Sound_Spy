"""Dynamic mixing (step 07b): training mixtures made on the fly, new ones every epoch.

Step 07 rendered 20,000 training mixtures once and trained on them over and over; after twelve
epochs the model knew them by heart (training loss falling, validation flat). Here nothing is
stored: training example ``(epoch, index)`` is **drawn** when it is needed,

1. a recipe from the same sampler that made the frozen files (``RecipeSampler``) with the
   ``mixtures.dynamic`` settings (more absent targets, hard negatives, partial targets, the
   speech source mix);
2. for half of them a room from the precomputed bank (``RirBank``): one position per event;
3. a random low-pass on some foreground events (``lowpass_prob``, cut-off log-uniform in
   ``lowpass_hz``);
4. rendered exactly like a file recipe (``mixtures.render``: ``mixture == target + rest``);
5. cropped to ``crop_s``: with probability ``crop_target_prob`` around a randomly chosen moment
   where the target really sounds (:func:`active_crop_start`; a crop that merely overlaps the
   target's first-to-last stretch can fall into a gap between two target events), or, for a
   partial target, around the moment it switches on or off; otherwise anywhere.

Everything random comes from ``(seed, epoch, index)`` only: the same triple always gives the
same example, on any DataLoader worker, and a new epoch gives new examples. The model therefore
never sees a mixture twice, while validation and test keep their fixed files.

This module has no PyTorch dependency (``training/datasets.py`` wraps it in a dataset).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from soundspy.config import AppConfig, MixtureConfig
from soundspy.core.labels import LabelSet
from soundspy.data.cache import TARGET_SPAN_FLOOR
from soundspy.data.mixtures import ClipLibrary, Recipe, RecipeSampler, render
from soundspy.data.reverb import RirBank

MIN_OVERLAP_S = 0.25  # a "target" crop holds at least this much of the target's stretch
PARTIAL_CROP_RANGE = (0.25, 0.75)  # where in the crop a partial target switches


ACTIVE_FRAME_S = 0.01  # resolution of "where the target sounds" for the crop


def active_crop_start(rng: np.random.Generator, target: np.ndarray, crop: int,
                      sample_rate: int, target_prob: float) -> int:
    """With probability ``target_prob``: a crop around a random moment at which the target
    sounds (that moment at least ``MIN_OVERLAP_S / 2`` inside the crop); otherwise anywhere."""
    samples = target.shape[0]
    last = samples - crop
    if last <= 0:
        return 0
    frame = max(1, round(ACTIVE_FRAME_S * sample_rate))
    frames = samples // frame
    level = np.abs(target[:frames * frame]).reshape(frames, frame).max(axis=1)
    peak = float(level.max(initial=0.0))
    active = np.flatnonzero(level > TARGET_SPAN_FLOOR * peak) if peak > 0 else level[:0]
    if active.size and rng.random() < target_prob:
        moment = (int(rng.choice(active)) + 0.5) * frame
        margin = min(crop // 4, round(MIN_OVERLAP_S * sample_rate / 2))
        position = rng.uniform(margin, crop - margin)
        return int(np.clip(round(moment - position), 0, last))
    return int(rng.integers(0, last + 1))


def crop_start(rng: np.random.Generator, samples: int, crop: int, target_start: int,
               target_end: int, sample_rate: int, target_prob: float) -> int:
    """Where a crop of ``crop`` samples starts inside an example of ``samples`` (the cached
    training set of step 07: only the target's first and last sample are stored)."""
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


@dataclass
class DynamicExample:
    mixture: np.ndarray  # (crop,) float32
    target: np.ndarray  # (crop,) float32
    rest: np.ndarray  # (crop,) float32: mixture == target + rest
    target_hot: np.ndarray  # multi-hot over the classes
    recipe: Recipe
    start: int  # first sample of the crop in the rendered mixture
    reverb: bool
    lowpass_hz: tuple[float | None, ...]  # per event


class DynamicMixer:
    """Makes training example ``(epoch, index)`` (see the module docstring)."""

    def __init__(self, library: ClipLibrary, labels: LabelSet, mix: MixtureConfig, split: str,
                 *, sample_rate: int, crop_s: float, crop_target_prob: float, seed: int,
                 rir_bank: RirBank | None = None, reverb_prob: float = 0.0) -> None:
        if rir_bank is not None and rir_bank.sample_rate != sample_rate:
            raise ValueError(f"the room bank is at {rir_bank.sample_rate} Hz, "
                             f"the mixtures at {sample_rate} Hz")
        self.library, self.labels, self.mix, self.split = library, labels, mix, split
        self.sampler = RecipeSampler(library, labels, mix, split)
        self.sample_rate = int(sample_rate)
        self.samples = round(mix.duration_s * sample_rate)
        self.crop = min(self.samples, round(crop_s * sample_rate))
        self.crop_target_prob = float(crop_target_prob)
        self.seed = int(seed)
        self.rir_bank = rir_bank
        self.reverb_prob = float(reverb_prob) if rir_bank is not None else 0.0

    @classmethod
    def from_config(cls, config: AppConfig, split: str = "train", *,
                    rir_bank: RirBank | None = None) -> DynamicMixer:
        cfg = config.extraction.neural_tse
        return cls(ClipLibrary.from_config(config), config.labels, config.data.dynamic, split,
                   sample_rate=cfg.model.sample_rate, crop_s=cfg.training.train_crop_s,
                   crop_target_prob=cfg.training.crop_target_prob, seed=cfg.training.seed,
                   rir_bank=rir_bank, reverb_prob=cfg.data.reverb.prob)

    def recipe(self, epoch: int, index: int) -> Recipe:
        rng = random.Random(f"{self.seed}/{epoch}/{index}")
        return self.sampler.sample(rng, f"dyn-{epoch}-{index}")

    def example(self, epoch: int, index: int) -> DynamicExample:
        recipe = self.recipe(epoch, index)
        rng = np.random.default_rng([self.seed, epoch, index])
        rirs = None
        if self.reverb_prob > 0 and recipe.events and rng.random() < self.reverb_prob:
            rirs = self.rir_bank.draw(rng, len(recipe.events))
        mix = self.mix
        cutoffs: list[float | None] = []
        for event in recipe.events:
            if event.role == "fg" and mix.lowpass_prob > 0 and rng.random() < mix.lowpass_prob:
                low, high = mix.lowpass_hz
                cutoffs.append(float(math.exp(rng.uniform(math.log(low), math.log(high)))))
            else:
                cutoffs.append(None)
        rendered = render(recipe, self.sample_rate, self.library, self.labels,
                          reference_rms_dbfs=mix.reference_rms_dbfs, peak_limit=mix.peak_limit,
                          rirs=rirs,
                          lowpass_hz=cutoffs if any(c is not None for c in cutoffs) else None)
        mixture = rendered.mixture.samples.ravel()
        target = rendered.target.samples.ravel()
        start = self._crop_start(rng, recipe, target)
        stop = start + self.crop
        return DynamicExample(
            mixture=np.ascontiguousarray(mixture[start:stop], dtype=np.float32),
            target=np.ascontiguousarray(target[start:stop], dtype=np.float32),
            rest=np.ascontiguousarray(rendered.rest.samples.ravel()[start:stop],
                                      dtype=np.float32),
            target_hot=rendered.target_hot.astype(np.float32), recipe=recipe, start=start,
            reverb=rirs is not None, lowpass_hz=tuple(cutoffs))

    def _crop_start(self, rng: np.random.Generator, recipe: Recipe, target: np.ndarray) -> int:
        last = self.samples - self.crop
        if last <= 0:
            return 0
        if recipe.switch_s is not None:  # the switch lands inside the crop, off its edges
            position = rng.uniform(*PARTIAL_CROP_RANGE) * self.crop
            return int(np.clip(round(recipe.switch_s * self.sample_rate - position), 0, last))
        return active_crop_start(rng, target, self.crop, self.sample_rate,
                                 self.crop_target_prob)

    def describe(self) -> dict[str, Any]:
        pool = self.sampler.pool
        return {"split": self.split, "sample_rate": self.sample_rate,
                "crop_s": self.crop / self.sample_rate, "reverb_prob": self.reverb_prob,
                "foreground_clips": {key: len(clips) for key, clips in pool.fg.items()},
                "background_clips": len(pool.bg),
                "rooms": self.rir_bank.rooms if self.rir_bank is not None else 0}
