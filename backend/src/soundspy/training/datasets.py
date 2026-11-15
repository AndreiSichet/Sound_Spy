"""PyTorch datasets for training (kept out of ``soundspy.data`` so that stays torch-free).

``DetectionDataset`` renders a mixture recipe at the model sample rate (``data.mixtures.render``),
cuts ``windows`` windows of ``window_s`` from it and labels each from the per-class activity
(``detection.windows``). Everything random is seeded from ``(seed, recipe index, epoch)``:

* **training**: items are ``(index, epoch)`` pairs from ``EpochSampler``, so the window position
  changes every epoch but is reproducible; the sampler lives in the main process, which is what
  makes this work with persistent DataLoader workers;
* **evaluation**: ``epoch`` is fixed, so val/test always use the same windows.

The dataset is picklable (Windows DataLoader workers are *spawned*: each gets a pickled copy of
the clip library, ~27 MB, once per worker thanks to ``persistent_workers``).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from soundspy.core.labels import LabelSet
from soundspy.data.mixtures import ClipLibrary, Recipe, render
from soundspy.detection.windows import window_labels, window_starts

EVAL_EPOCH = -1


@dataclass(frozen=True)
class WindowSpec:
    sample_rate: int
    window_s: float
    min_active_s: float
    activity_rate_hz: float
    reference_rms_dbfs: float
    peak_limit: float


class DetectionDataset(Dataset):
    def __init__(self, recipes: Sequence[Recipe], library: ClipLibrary, labels: LabelSet,
                 spec: WindowSpec, *, seed: int, windows: int = 1) -> None:
        self.recipes = list(recipes)
        self.library = library
        self.labels = labels
        self.spec = spec
        self.seed = int(seed)
        self.windows = int(windows)

    def __len__(self) -> int:
        return len(self.recipes)

    def rng(self, index: int, epoch: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, index, epoch + 1])

    def item(self, index: int, epoch: int) -> tuple[np.ndarray, np.ndarray]:
        """``(waves (windows, samples), labels (windows, classes))`` for one recipe."""
        s = self.spec
        recipe = self.recipes[index]
        out = render(recipe, s.sample_rate, self.library, self.labels,
                     reference_rms_dbfs=s.reference_rms_dbfs, peak_limit=s.peak_limit,
                     activity_rate_hz=s.activity_rate_hz)
        mix = out.mixture.samples.ravel()
        n = round(s.window_s * s.sample_rate)
        starts = window_starts(recipe.duration_s, s.window_s, self.windows,
                               self.rng(index, epoch))
        waves = np.zeros((self.windows, n), dtype=np.float32)
        labels = np.zeros((self.windows, len(self.labels)), dtype=np.float32)
        for k, start in enumerate(starts):
            a = int(round(start * s.sample_rate))
            seg = mix[a:a + n]
            waves[k, :seg.shape[0]] = seg
            labels[k] = window_labels(out.class_activity, out.activity_rate_hz, float(start),
                                      s.window_s, s.min_active_s)
        return waves, labels

    def __getitem__(self, key: int | tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        index, epoch = (key if isinstance(key, tuple) else (key, EVAL_EPOCH))
        waves, labels = self.item(int(index), int(epoch))
        return torch.from_numpy(waves), torch.from_numpy(labels)


def collate_windows(batch: list[tuple[torch.Tensor, torch.Tensor]]
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack ``(windows, …)`` items into ``(batch * windows, …)``."""
    waves = torch.cat([b[0] for b in batch])
    labels = torch.cat([b[1] for b in batch])
    return waves, labels


class EpochSampler(Sampler):
    """Yields ``(index, epoch)``; shuffled with ``seed + epoch``. Call ``set_epoch`` each epoch."""

    def __init__(self, n: int, *, seed: int, shuffle: bool = True) -> None:
        self.n, self.seed, self.shuffle, self.epoch = n, seed, shuffle, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[tuple[int, int]]:
        order = (np.random.default_rng([self.seed, self.epoch]).permutation(self.n)
                 if self.shuffle else np.arange(self.n))
        for i in order:
            yield int(i), self.epoch


# --------------------------------------------------------------------------------------------
# DataLoader workers on Windows
#
# Workers are *spawned*: each re-imports numpy, scipy and torch, and every one of those starts a
# BLAS/OpenMP thread pool sized to all CPU cores, with per-thread buffers. Measured in step 05:
# 1.67 GB committed memory and 29 threads per worker by default, 0.95 GB and 7 threads with the
# pools limited to one thread. Eight default workers exhausted the commit limit, OpenBLAS failed
# to allocate inside a worker ("Memory allocation still failed after 10 retries") and training
# hung waiting for a batch. Rendering is single-threaded anyway: parallelism comes from the
# worker processes, so each worker gets one BLAS/OpenMP thread.

THREAD_ENV_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def limit_worker_threads() -> None:
    """One BLAS/OpenMP thread in processes spawned from now on (unless the user set them).

    Spawned workers inherit the environment, and these libraries read it when they load. The
    current process is not affected (its libraries are already initialised).
    """
    import os

    for var in THREAD_ENV_VARS:
        os.environ.setdefault(var, "1")


def worker_init(_worker_id: int) -> None:
    torch.set_num_threads(1)


class LoaderStalled(RuntimeError):
    """A DataLoader produced no batch within the timeout, or a worker died."""


def loader_error(exc: BaseException, timeout_s: float, workers: int) -> LoaderStalled:
    """A user-facing message for DataLoader timeouts and dead workers."""
    return LoaderStalled(
        f"the data loader delivered no batch within {timeout_s:.0f} s or a worker process died "
        f"({type(exc).__name__}: {exc}). With {workers} worker(s) on Windows this usually means "
        "the workers ran out of memory (check free RAM / close other programs) - retry with "
        "fewer workers, e.g. --workers 2, or --workers 0 to load in the main process.")


def guarded(loader, *, what: str = "batch", log=None):
    """Iterate a DataLoader, turning a timeout / dead worker into ``LoaderStalled``.

    With ``log``, reports how long the first batch took (worker start-up included), so a slow
    start is visible instead of silent.
    """
    import time

    workers = int(getattr(loader, "num_workers", 0))
    timeout = float(getattr(loader, "timeout", 0) or 0)
    started = time.monotonic()
    if log is not None and workers:
        log(f"waiting for the first {what} from {workers} worker(s) "
            f"(timeout {timeout:.0f} s) ...")
    it = iter(loader)
    first = True
    while True:
        try:
            item = next(it)
        except StopIteration:
            return
        except RuntimeError as exc:
            text = str(exc)
            if "timed out" in text or "exited unexpectedly" in text:
                raise loader_error(exc, timeout, workers) from exc
            raise
        if first and log is not None and workers:
            log(f"first {what} after {time.monotonic() - started:.1f} s")
        first = False
        yield item
