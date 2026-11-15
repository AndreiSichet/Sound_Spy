"""Activity maps: where in each clip something actually sounds.

Many dataset clips have silent heads or tails (a dog barks once in 5 s). Mixtures should place
events where the sound is, so every clip gets a coarse **RMS envelope** in dB (one value per
``activity.hop_s``, 0.1 s by default). A frame is *active* when it is within
``threshold_db_below_peak`` (30 dB) of the clip's loudest frame and above an absolute floor.

The envelope is stored (not the active mask), so the threshold can change without recomputing.
One file per dataset: ``<processed>/activity/<dataset>.npz`` with the arrays

* ``clip_ids``  (n,) clip ids;
* ``offsets``   (n + 1,) start of each clip's envelope inside ``env_db``;
* ``env_db``    concatenated envelopes (float32, dBFS);
* ``hop_s``     the hop they were computed with (a different hop recomputes everything);
* ``sample_rate`` / ``frames`` / ``channels`` (n,) header facts read in the same pass.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from soundspy.core.audio import AudioBuffer
from soundspy.dsp.channels import ChannelReducer

SILENCE_FLOOR_DB = -90.0
"""Frames quieter than this are never active (digital silence, dither)."""
_EPS = 1e-12


def rms_envelope_db(buffer: AudioBuffer, hop_s: float) -> np.ndarray:
    """RMS per ``hop_s`` window in dBFS (last window may be shorter). Channels are averaged."""
    if buffer.num_channels > 1:
        buffer = ChannelReducer("mean").process(buffer)
    x = buffer.samples.ravel().astype(np.float64)
    hop = max(1, round(hop_s * buffer.sample_rate))
    n = max(1, math.ceil(len(x) / hop))
    padded = np.zeros(n * hop)
    padded[: len(x)] = x * x
    counts = np.full(n, hop, dtype=np.float64)
    if len(x) % hop:
        counts[-1] = len(x) % hop
    power = np.add.reduceat(padded, np.arange(0, n * hop, hop)) / counts
    return (10.0 * np.log10(power + _EPS)).astype(np.float32)


def active_mask(env_db: np.ndarray, threshold_db: float) -> np.ndarray:
    if env_db.size == 0:
        return np.zeros(0, dtype=bool)
    peak = float(env_db.max())
    return (env_db >= peak - threshold_db) & (env_db > SILENCE_FLOOR_DB)


def active_regions(
    env_db: np.ndarray, hop_s: float, threshold_db: float
) -> list[tuple[float, float]]:
    """Contiguous active stretches as ``(start_s, end_s)``."""
    mask = active_mask(env_db, threshold_db)
    regions = []
    start = None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            regions.append((start * hop_s, i * hop_s))
            start = None
    if start is not None:
        regions.append((start * hop_s, len(mask) * hop_s))
    return regions


@dataclass(frozen=True)
class FileFacts:
    sample_rate: int
    frames: int
    channels: int
    env_db: np.ndarray


def analyze_file(path: str, hop_s: float) -> FileFacts | None:
    """Header facts + envelope of one file (runs in worker processes). None if unreadable."""
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError, sf.LibsndfileError):
        return None
    buf = AudioBuffer(data, int(sr))
    return FileFacts(int(sr), int(data.shape[0]), int(data.shape[1]), rms_envelope_db(buf, hop_s))


def _analyze_job(job: tuple[str, float]) -> FileFacts | None:
    return analyze_file(*job)


class ActivityStore:
    """Envelopes + header facts for one dataset (see module docstring for the file layout)."""

    def __init__(self, hop_s: float) -> None:
        self.hop_s = float(hop_s)
        self._facts: dict[str, FileFacts] = {}

    def __contains__(self, clip_id: object) -> bool:
        return clip_id in self._facts

    def __len__(self) -> int:
        return len(self._facts)

    def clip_ids(self) -> list[str]:
        return list(self._facts)

    def facts(self, clip_id: str) -> FileFacts:
        return self._facts[clip_id]

    def envelope(self, clip_id: str) -> np.ndarray:
        return self._facts[clip_id].env_db

    def add(self, clip_id: str, facts: FileFacts) -> None:
        self._facts[clip_id] = facts

    def keep_only(self, clip_ids: Iterable[str]) -> int:
        """Drop clips that are no longer indexed; returns how many were dropped."""
        keep = set(clip_ids)
        gone = [c for c in self._facts if c not in keep]
        for c in gone:
            del self._facts[c]
        return len(gone)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ids = list(self._facts)
        envs = [self._facts[c].env_db for c in ids]
        offsets = np.zeros(len(ids) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(e) for e in envs])
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(
            tmp,
            clip_ids=np.array(ids, dtype=str),
            offsets=offsets,
            env_db=np.concatenate(envs).astype(np.float32) if envs else np.zeros(0, np.float32),
            hop_s=np.float64(self.hop_s),
            sample_rate=np.array([self._facts[c].sample_rate for c in ids], dtype=np.int64),
            frames=np.array([self._facts[c].frames for c in ids], dtype=np.int64),
            channels=np.array([self._facts[c].channels for c in ids], dtype=np.int64),
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path, hop_s: float | None = None) -> ActivityStore:
        """Load ``path``; returns an empty store if missing or computed with another hop."""
        if not path.exists():
            return cls(hop_s if hop_s is not None else 0.1)
        with np.load(path, allow_pickle=False) as npz:
            stored_hop = float(npz["hop_s"])
            if hop_s is not None and not math.isclose(stored_hop, hop_s):
                return cls(hop_s)
            store = cls(stored_hop)
            ids, offsets, env = npz["clip_ids"], npz["offsets"], npz["env_db"]
            srs, frames, chans = npz["sample_rate"], npz["frames"], npz["channels"]
        for i, cid in enumerate(ids):
            store._facts[str(cid)] = FileFacts(
                int(srs[i]), int(frames[i]), int(chans[i]), env[offsets[i]:offsets[i + 1]]
            )
        return store


def compute_missing(
    store: ActivityStore,
    files: Mapping[str, Path],
    *,
    workers: int | None = None,
    progress: bool = True,
    desc: str = "activity",
) -> list[str]:
    """Analyze every clip of ``files`` (``clip_id -> path``) not in ``store`` yet.

    Runs in ``workers`` processes (default: all CPU cores; 1 = in this process).
    Returns the clip ids whose file could not be read.
    """
    todo: Sequence[tuple[str, Path]] = [(c, p) for c, p in files.items() if c not in store]
    if not todo:
        return []
    from tqdm import tqdm

    jobs = [(str(p), store.hop_s) for _c, p in todo]
    failed = []
    bar = tqdm(total=len(jobs), desc=desc, unit="clip", disable=not progress, mininterval=1.0)
    try:
        if workers == 1 or len(jobs) < 64:
            results: Iterable[FileFacts | None] = (_analyze_job(j) for j in jobs)
            for (cid, _p), facts in zip(todo, results, strict=True):
                _collect(store, cid, facts, failed)
                bar.update()
        else:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = pool.map(_analyze_job, jobs, chunksize=64)
                for (cid, _p), facts in zip(todo, results, strict=True):
                    _collect(store, cid, facts, failed)
                    bar.update()
    finally:
        bar.close()
    return failed


def _collect(store: ActivityStore, cid: str, facts: FileFacts | None, failed: list[str]) -> None:
    if facts is None:
        failed.append(cid)
    else:
        store.add(cid, facts)
