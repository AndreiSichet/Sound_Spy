"""Pre-rendered training examples for the neural extractor (step 07).

Rendering a mixture from its recipe (read the clips, resample, level, add up; reverberation for
half of them) takes ~50-200 ms, far longer than the GPU needs to learn from it. Detection
training in step 05 was limited by exactly that. So the mixtures are rendered **once** and
stored on D: as plain arrays the training reads at memory speed::

    <processed>/cache/extraction_<rate>/<split>/
        index.json            what is in here and how it was made (signature)
        shard_000.npy         (n, 2, samples) float16: [i, 0] = mixture, [i, 1] = target
        shard_000.meta.npz    per example: target / present multi-hot, flags, target span
        shard_001.npy ...

* **float16** halves the size (a 6 s mixture and its target at 44.1 kHz: 1.06 MB instead of
  2.1 MB) and is exact enough for training (about 3 decimal digits at every level).
* **Shards** of ``shard_size`` recipes (~270 MB) are written whole: a shard is rendered into a
  ``.part`` file and renamed when complete, and ``index.json`` lists the finished ones. An
  interrupted run therefore continues with the first missing shard (**resumable**), and running
  the command again when everything is there does nothing (**idempotent**).
* **Parallel**: shards are rendered by several processes at once.
* **Reverberation** (``data/reverb.py``) is decided per recipe from ``(seed, recipe id)``, so a
  rebuilt cache is identical.
* The **signature** in ``index.json`` (recipe file hash, rate, reverberation settings, levels,
  class list) is checked on every run: a cache made with other settings is refused, not mixed.

``ExtractionCache`` opens a split for reading; shards are memory-mapped, so nothing is loaded
until it is used. This module has no PyTorch dependency (``training/`` wraps it in a dataset).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from soundspy.config import AppConfig
from soundspy.data.mixtures import (
    ClipLibrary,
    Recipe,
    iter_recipes,
    recipes_path,
    render_with_config,
)
from soundspy.data.reverb import recipe_rirs

INDEX_FILE = "index.json"
CACHE_VERSION = 1
SIGNALS = 2  # mixture, target
TARGET_SPAN_FLOOR = 1e-3  # relative to the target's peak: where the target "sounds"
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


class CacheError(RuntimeError):
    pass


def cache_root(config: AppConfig, sample_rate: int) -> Path:
    return Path(config.paths["processed"]) / "cache" / f"extraction_{sample_rate}"


def split_dir(config: AppConfig, sample_rate: int, split: str) -> Path:
    return cache_root(config, sample_rate) / split


def shard_name(number: int) -> str:
    return f"shard_{number:03d}.npy"


def meta_name(number: int) -> str:
    return f"shard_{number:03d}.meta.npz"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def signature(config: AppConfig, split: str, sample_rate: int) -> dict[str, Any]:
    """Everything the cached samples depend on."""
    data, mix = config.extraction.neural_tse.data, config.data.mixtures
    return {
        "version": CACHE_VERSION,
        "split": split,
        "sample_rate": sample_rate,
        "recipes_sha256": file_sha256(recipes_path(config.paths["mixtures"], split)),
        "labels": list(config.labels),
        "duration_s": mix.duration_s,
        "reference_rms_dbfs": mix.reference_rms_dbfs,
        "peak_limit": mix.peak_limit,
        "shard_size": data.shard_size,
        "seed": data.seed,
        "reverb": dataclasses.asdict(data.reverb),
    }


def bytes_needed(n_recipes: int, samples: int) -> int:
    return n_recipes * SIGNALS * samples * 2  # float16


def free_bytes(path: Path) -> int:
    probe = Path(path)
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def check_free_space(directory: Path, needed: int, min_free_gb: float) -> None:
    free = free_bytes(directory)
    reserve = int(min_free_gb * 1e9)
    if free - needed < reserve:
        raise CacheError(
            f"not enough space for the cache at {directory}: {needed / 1e9:.1f} GB needed, "
            f"{free / 1e9:.1f} GB free, and {min_free_gb:g} GB must stay free "
            "(extraction.neural_tse.data.min_free_gb)")


def target_span(target: np.ndarray) -> tuple[int, int]:
    """First and one-past-last sample where the target sounds (``(0, 0)`` if it never does)."""
    magnitude = np.abs(target)
    peak = float(magnitude.max(initial=0.0))
    if peak <= 0.0:
        return 0, 0
    active = np.flatnonzero(magnitude > TARGET_SPAN_FLOOR * peak)
    return int(active[0]), int(active[-1]) + 1


# --------------------------------------------------------------------------------------------
# Rendering (worker processes)

_WORKER: dict[str, Any] = {}


def _init_worker(config: AppConfig) -> None:
    _WORKER["config"] = config
    _WORKER["library"] = ClipLibrary.from_config(config)


def render_example(config: AppConfig, library: ClipLibrary, recipe: Recipe, sample_rate: int
                   ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """``(mixture, target, meta)`` of one recipe as the cache stores it (float32 here)."""
    data = config.extraction.neural_tse.data
    rirs = recipe_rirs(recipe.id, len(recipe.events), sample_rate, data.reverb, data.seed)
    rendered = render_with_config(recipe, sample_rate, library, config, rirs=rirs)
    mixture = rendered.mixture.samples.ravel()
    target = rendered.target.samples.ravel()
    start, end = target_span(target)
    meta = {
        "target_hot": rendered.target_hot.astype(np.float32),
        "present_hot": rendered.present.astype(np.float32),
        "empty": end == start,
        "reverb": rirs is not None,
        "target_start": start,
        "target_end": end,
    }
    return mixture, target, meta


def _render_shard(directory: str, number: int, recipes: Sequence[Recipe], sample_rate: int,
                  samples: int) -> tuple[int, float]:
    """Render one shard and write it (``.part`` first); returns ``(number, seconds)``."""
    started = time.perf_counter()
    config, library = _WORKER["config"], _WORKER["library"]
    folder = Path(directory)
    final = folder / shard_name(number)
    part = final.with_name(final.name + ".part")
    array = np.lib.format.open_memmap(part, mode="w+", dtype=np.float16,
                                      shape=(len(recipes), SIGNALS, samples))
    metas = []
    for i, recipe in enumerate(recipes):
        mixture, target, meta = render_example(config, library, recipe, sample_rate)
        n = min(samples, mixture.shape[0])
        array[i, 0, :n] = mixture[:n]
        array[i, 1, :n] = target[:n]
        metas.append(meta)
    array.flush()
    del array
    np.savez(folder / meta_name(number),
             target_hot=np.stack([m["target_hot"] for m in metas]),
             present_hot=np.stack([m["present_hot"] for m in metas]),
             empty=np.array([m["empty"] for m in metas], dtype=bool),
             reverb=np.array([m["reverb"] for m in metas], dtype=bool),
             target_start=np.array([m["target_start"] for m in metas], dtype=np.int64),
             target_end=np.array([m["target_end"] for m in metas], dtype=np.int64))
    os.replace(part, final)  # the shard exists only when it is complete
    return number, time.perf_counter() - started


# --------------------------------------------------------------------------------------------
# Building


@dataclass
class CacheSummary:
    split: str
    directory: Path
    recipes: int
    shards: int
    built: int  # shards rendered in this run
    seconds: float
    bytes: int


def _write_index(directory: Path, index: dict[str, Any]) -> None:
    tmp = directory / (INDEX_FILE + ".tmp")
    tmp.write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, directory / INDEX_FILE)


def read_index(directory: Path) -> dict[str, Any] | None:
    path = Path(directory) / INDEX_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_extraction_cache(config: AppConfig, split: str, *, sample_rate: int | None = None,
                           workers: int | None = None, limit: int | None = None,
                           force: bool = False,
                           progress: Callable[[str], None] = print) -> CacheSummary:
    """Render every recipe of ``split`` into the cache (only what is still missing)."""
    sample_rate = sample_rate or config.extraction.neural_tse.model.sample_rate
    data = config.extraction.neural_tse.data
    path = recipes_path(config.paths["mixtures"], split)
    if not path.is_file():
        raise CacheError(f"no {split} recipes at {path}: run `soundspy data mix` first")
    recipes = list(iter_recipes(path))
    if limit is not None:
        recipes = recipes[:limit]
    if not recipes:
        raise CacheError(f"{path} holds no recipes")
    samples = round(config.data.mixtures.duration_s * sample_rate)
    directory = split_dir(config, sample_rate, split)
    # through JSON once, so it compares equal to the copy read back from index.json
    sig = json.loads(json.dumps({**signature(config, split, sample_rate), "limit": limit}))

    index = read_index(directory)
    if index is not None and index.get("signature") != sig:
        if not force:
            raise CacheError(
                f"the cache at {directory} was made with other settings (recipes, rate, "
                "reverberation or levels changed). Run again with --force to rebuild it.")
        shutil.rmtree(directory)
        index = None
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.part"):  # left by an interrupted run
        stale.unlink()

    size = data.shard_size
    n_shards = -(-len(recipes) // size)
    done = set(index["shards_done"]) if index else set()
    done = {n for n in done if (directory / shard_name(n)).is_file()
            and (directory / meta_name(n)).is_file()}
    todo = [n for n in range(n_shards) if n not in done]
    index = {"signature": sig, "recipes": len(recipes), "samples": samples,
             "sample_rate": sample_rate, "shard_size": size, "shards": n_shards,
             "recipe_ids": [r.id for r in recipes], "shards_done": sorted(done)}
    _write_index(directory, index)

    started = time.perf_counter()
    if todo:
        missing = sum(min(size, len(recipes) - n * size) for n in todo)
        check_free_space(directory, bytes_needed(missing, samples), data.min_free_gb)
        workers = max(1, min(workers or max(1, (os.cpu_count() or 2) - 2), len(todo)))
        progress(f"{split}: {len(todo)} of {n_shards} shards to render "
                 f"({missing} recipes, {bytes_needed(missing, samples) / 1e9:.1f} GB, "
                 f"{workers} processes)")
        tasks = [(str(directory), n, recipes[n * size:(n + 1) * size], sample_rate, samples)
                 for n in todo]
        if workers == 1:
            _init_worker(config)
            results = (_render_shard(*task) for task in tasks)
            _collect(results, directory, index, done, len(todo), started, progress)
        else:
            saved = {key: os.environ.get(key) for key in THREAD_ENV}
            os.environ.update({key: "1" for key in THREAD_ENV})  # inherited by the workers
            try:
                with ProcessPoolExecutor(workers, initializer=_init_worker,
                                         initargs=(config,)) as pool:
                    futures = [pool.submit(_render_shard, *task) for task in tasks]
                    results = (f.result() for f in as_completed(futures))
                    _collect(results, directory, index, done, len(todo), started, progress)
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
    total = sum((directory / shard_name(n)).stat().st_size for n in range(n_shards))
    return CacheSummary(split, directory, len(recipes), n_shards, len(todo),
                        time.perf_counter() - started, total)


def _collect(results, directory: Path, index: dict[str, Any], done: set[int], total: int,
             started: float, progress: Callable[[str], None]) -> None:
    for count, (number, _seconds) in enumerate(results, 1):
        done.add(number)
        index["shards_done"] = sorted(done)
        _write_index(directory, index)  # after every shard: a later run resumes from here
        if count % 5 == 0 or count == total:
            elapsed = time.perf_counter() - started
            progress(f"  {count}/{total} shards ({elapsed:.0f} s, "
                     f"about {elapsed / count * (total - count):.0f} s left)")


# --------------------------------------------------------------------------------------------
# Reading


class ExtractionCache:
    """A finished split of the cache. ``pair(i)`` returns views into memory-mapped files."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        index = read_index(self.directory)
        if index is None:
            raise CacheError(f"no extraction cache at {self.directory}: run "
                             "`soundspy data cache --task extraction` first")
        missing = sorted(
            n for n in range(index["shards"])
            if n not in index["shards_done"] or not (self.directory / shard_name(n)).is_file()
            or not (self.directory / meta_name(n)).is_file())
        if missing:
            raise CacheError(f"the cache at {self.directory} is incomplete ({len(missing)} of "
                             f"{index['shards']} shards missing): run `soundspy data cache "
                             "--task extraction` again, it continues where it stopped")
        self.index = index
        self.sample_rate: int = index["sample_rate"]
        self.samples: int = index["samples"]
        self.shard_size: int = index["shard_size"]
        self.recipe_ids: list[str] = index["recipe_ids"]
        self.labels: list[str] = index["signature"]["labels"]
        metas = [np.load(self.directory / meta_name(n)) for n in range(index["shards"])]
        self.target_hot = np.concatenate([m["target_hot"] for m in metas])
        self.present_hot = np.concatenate([m["present_hot"] for m in metas])
        self.empty = np.concatenate([m["empty"] for m in metas])
        self.reverb = np.concatenate([m["reverb"] for m in metas])
        self.target_start = np.concatenate([m["target_start"] for m in metas])
        self.target_end = np.concatenate([m["target_end"] for m in metas])
        self._shards: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.recipe_ids)

    def _shard(self, number: int) -> np.ndarray:
        shard = self._shards.get(number)
        if shard is None:
            shard = np.load(self.directory / shard_name(number), mmap_mode="r")
            self._shards[number] = shard
        return shard

    def pair(self, i: int) -> np.ndarray:
        """``(2, samples)`` float16: row 0 the mixture, row 1 the target (a read-only view)."""
        if not 0 <= i < len(self):
            raise IndexError(i)
        return self._shard(i // self.shard_size)[i % self.shard_size]

    def close(self) -> None:
        self._shards.clear()

    def __getstate__(self) -> dict[str, Any]:  # memory maps are reopened in worker processes
        state = dict(self.__dict__)
        state["_shards"] = {}
        return state
