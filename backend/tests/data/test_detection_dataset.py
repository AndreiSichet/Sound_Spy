"""DetectionDataset: determinism per (recipe, epoch), labels, and spawned DataLoader workers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from soundspy.data.mixtures import ClipLibrary, generate_recipes
from soundspy.training.datasets import (
    DetectionDataset,
    EpochSampler,
    WindowSpec,
    collate_windows,
    guarded,
    limit_worker_threads,
    worker_init,
)

SR = 16_000


@pytest.fixture(scope="module")
def dataset(shared_prepared_env: Any) -> DetectionDataset:
    config = shared_prepared_env.config
    library = ClipLibrary.from_config(config)
    recipes = generate_recipes(library, config.labels, config.data.mixtures, "train", 12, 1)
    spec = WindowSpec(SR, 1.0, 0.1, 100.0, -25.0, 0.99)
    return DetectionDataset(recipes, library, config.labels, spec, seed=7)


def test_same_recipe_and_epoch_give_the_same_window(dataset: DetectionDataset) -> None:
    a_wave, a_lab = dataset[(3, 0)]
    b_wave, b_lab = dataset[(3, 0)]
    torch.testing.assert_close(a_wave, b_wave)
    torch.testing.assert_close(a_lab, b_lab)
    assert a_wave.shape == (1, SR) and a_lab.shape == (1, len(dataset.labels))
    other_epoch, _ = dataset[(3, 1)]
    assert not torch.equal(a_wave, other_epoch)  # a new position every epoch


def test_eval_items_use_a_fixed_epoch(dataset: DetectionDataset) -> None:
    torch.testing.assert_close(dataset[2][0], dataset[2][0])


def test_epoch_sampler_shuffles_per_epoch_reproducibly() -> None:
    s = EpochSampler(10, seed=3)
    first = list(s)
    assert sorted(i for i, _ in first) == list(range(10)) and {e for _, e in first} == {0}
    assert list(s) == first
    s.set_epoch(1)
    second = list(s)
    assert {e for _, e in second} == {1} and [i for i, _ in second] != [i for i, _ in first]


def test_labels_are_multi_hot(dataset: DetectionDataset) -> None:
    labels = torch.cat([dataset[(i, 0)][1] for i in range(len(dataset))]).numpy()
    assert set(np.unique(labels)) <= {0.0, 1.0}
    assert labels.sum() > 0


def test_spawned_workers_deliver_the_same_batches(dataset: DetectionDataset) -> None:
    """num_workers=2 on Windows (spawn): the dataset pickles and workers produce the data."""
    limit_worker_threads()
    sampler = EpochSampler(len(dataset), seed=1, shuffle=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, sampler=sampler, num_workers=2, persistent_workers=True,
        collate_fn=collate_windows, worker_init_fn=worker_init, timeout=120)
    batches = list(guarded(loader))
    direct = collate_windows([dataset[(i, 0)] for i in range(len(dataset))])
    torch.testing.assert_close(torch.cat([b[0] for b in batches]), direct[0])
    torch.testing.assert_close(torch.cat([b[1] for b in batches]), direct[1])
