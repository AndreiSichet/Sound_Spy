"""Clips converted once to the model's sample rate (step 07b): ``soundspy data cache --task clips``.

Dynamic mixing (``data/dynamic.py``) reads a few clip segments for every training example.
FSD50K and ESC-50 are 44.1 kHz WAV already: a segment is one seek and one read. LibriSpeech is
16 kHz FLAC: every segment would have to be decoded and resampled (x 441/160) again and again.
So its training utterances are resampled **once** and stored as 16-bit WAV::

    <processed>/cache/clips_<rate>/<dataset>/<the clip's path inside its dataset>.wav
    <processed>/cache/clips_<rate>/<dataset>/index.json

``ClipLibrary.load_segment`` uses such a copy whenever it exists and the wanted rate matches;
otherwise it reads the original file and resamples as before, so the cache changes speed, never
results (up to the 16-bit rounding, 96 dB below full scale).

A clip is written to ``<name>.part`` and renamed when complete: an interrupted run continues
with the missing files (**resumable**), a second run does nothing (**idempotent**).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from soundspy.config import AppConfig
from soundspy.core.audio import AudioBuffer
from soundspy.data.aliases import resolve_alias, split_alias
from soundspy.dsp.channels import ChannelReducer
from soundspy.dsp.resample import resample_array

INDEX_FILE = "index.json"
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
BYTES_PER_SAMPLE = 2  # PCM_16, mono


class ClipCacheError(RuntimeError):
    pass


def clip_cache_dir(config: AppConfig, sample_rate: int, dataset: str) -> Path:
    return Path(config.paths["processed"]) / "cache" / f"clips_{sample_rate}" / dataset


def cached_path(root: Path, alias: str) -> Path:
    """Where the converted copy of the clip with this aliased path lives under ``root``."""
    _dataset, relative = split_alias(alias)
    return root.joinpath(*relative.with_suffix(".wav").parts)


def convert_clip(source: Path, dest: Path, sample_rate: int) -> int:
    """Read ``source``, reduce to mono, resample, write 16-bit WAV. Returns the frames written."""
    data, native = sf.read(str(source), dtype="float32", always_2d=True)
    buf = AudioBuffer(data, int(native))
    if buf.num_channels > 1:
        buf = ChannelReducer("mean").process(buf)
    out = np.clip(resample_array(buf.samples, int(native), sample_rate), -1.0, 1.0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    sf.write(str(part), out, sample_rate, subtype="PCM_16", format="WAV")
    os.replace(part, dest)
    return int(out.shape[0])


def _convert_batch(jobs: Sequence[tuple[str, str]], sample_rate: int) -> tuple[int, list[str]]:
    frames, failed = 0, []
    for source, dest in jobs:
        try:
            frames += convert_clip(Path(source), Path(dest), sample_rate)
        except (OSError, RuntimeError, sf.LibsndfileError) as exc:
            failed.append(f"{source}: {exc}")
    return frames, failed


@dataclass
class ClipCacheSummary:
    dataset: str
    directory: Path
    clips: int
    built: int
    seconds: float
    bytes: int


def build_clip_cache(config: AppConfig, dataset: str, *, sample_rate: int | None = None,
                     splits: Sequence[str] | None = None, workers: int | None = None,
                     limit: int | None = None,
                     progress: Callable[[str], None] = print) -> ClipCacheSummary:
    """Convert every clip of ``dataset`` in ``splits`` that is not converted yet."""
    from soundspy.data.cache import check_free_space
    from soundspy.data.prepare import load_clip_index

    data = config.extraction.neural_tse.data
    sample_rate = sample_rate or config.extraction.neural_tse.model.sample_rate
    splits = tuple(splits or data.clip_cache_splits)
    index = load_clip_index(config.paths["processed"])
    rows = index[(index["dataset"] == dataset) & index["split"].isin(splits)
                 & (index["sample_rate"] != sample_rate)]
    if limit is not None:
        rows = rows.head(limit)
    if rows.empty:
        raise ClipCacheError(
            f"no {dataset} clips in {list(splits)} need converting to {sample_rate} Hz "
            "(is the dataset downloaded and `soundspy data prepare` run?)")
    root = clip_cache_dir(config, sample_rate, dataset)
    root.mkdir(parents=True, exist_ok=True)
    for stale in root.rglob("*.part"):
        stale.unlink()
    todo, missing_s = [], 0.0
    for rec in rows[["path", "duration_s"]].itertuples(index=False):
        dest = cached_path(root, rec.path)
        if not dest.is_file():
            todo.append((str(resolve_alias(rec.path, config.paths)), str(dest)))
            missing_s += float(rec.duration_s)
    started = time.perf_counter()
    if todo:
        needed = int(missing_s * sample_rate * BYTES_PER_SAMPLE)
        check_free_space(root, needed, data.min_free_gb)
        workers = max(1, min(workers or max(1, (os.cpu_count() or 2) - 2), len(todo)))
        progress(f"{dataset}: {len(todo)} of {len(rows)} clips to convert to {sample_rate} Hz "
                 f"({missing_s / 3600:.1f} h of audio, {needed / 1e9:.1f} GB, "
                 f"{workers} processes)")
        size = max(1, min(64, len(todo) // (workers * 4) or 1))
        batches = [todo[i:i + size] for i in range(0, len(todo), size)]
        failed: list[str] = []
        done = 0
        if workers == 1:
            results = (_convert_batch(batch, sample_rate) for batch in batches)
            done, failed = _gather(results, batches, len(todo), started, progress)
        else:
            saved = {key: os.environ.get(key) for key in THREAD_ENV}
            os.environ.update({key: "1" for key in THREAD_ENV})
            try:
                with ProcessPoolExecutor(workers) as pool:
                    results = pool.map(_convert_batch, batches, [sample_rate] * len(batches))
                    done, failed = _gather(results, batches, len(todo), started, progress)
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        if failed:
            raise ClipCacheError(f"{len(failed)} clip(s) could not be converted, e.g. "
                                 f"{failed[0]}; run the command again")
    total = sum(p.stat().st_size for p in root.rglob("*.wav"))
    payload = {"dataset": dataset, "sample_rate": sample_rate, "splits": list(splits),
               "clips": int(len(rows)), "bytes": total, "subtype": "PCM_16"}
    tmp = root / (INDEX_FILE + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, root / INDEX_FILE)
    return ClipCacheSummary(dataset, root, int(len(rows)), len(todo),
                            time.perf_counter() - started, total)


def _gather(results, batches, total: int, started: float,
            progress: Callable[[str], None]) -> tuple[int, list[str]]:
    done, failed, reported = 0, [], 0
    for batch, (_frames, errors) in zip(batches, results, strict=True):
        done += len(batch)
        failed += errors
        if done - reported >= max(500, total // 20) or done == total:
            reported = done
            elapsed = time.perf_counter() - started
            progress(f"  {done}/{total} clips ({elapsed:.0f} s, about "
                     f"{elapsed / done * (total - done):.0f} s left)")
    return done, failed
