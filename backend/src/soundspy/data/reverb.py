"""Simulated rooms for the training mixtures (step 07).

Dataset clips are recorded close and dry; a real room adds reflections. To narrow that gap, a
share of the mixtures is rendered **in a simulated room**: a random box-shaped room, one
microphone and one random position per event. Each event is convolved with the impulse response
(RIR) from its position to the microphone *before* the events are added up, so the mixture is
still exactly ``target + rest``, and the target is the **reverberant** target (the extractor is
not asked to remove the room, only the other sounds).

* :func:`sample_room`: room size, reverberation time (RT60) and positions from a seeded
  generator (``data.reverb`` in ``configs/extraction/neural_tse.yaml``).
* :func:`simulate_rirs`: the impulse responses, with ``pyroomacoustics`` (image-source method;
  the ``spatial`` extra). Each one is scaled to **unit energy**, so reverberation changes how an
  event sounds but not how loud it is (the recipe's levels keep their meaning).
* :func:`recipe_rirs`: deterministic per recipe: whether it is reverberant at all (``prob``) and
  with which room, from ``(seed, recipe id)`` only, so caches can be rebuilt identically.

**The bank** (step 07b). Simulating a room takes ~0.15 s, too slow to do for every training
example made on the fly. :func:`build_rir_bank` therefore simulates ``rooms`` rooms once, each
with ``positions`` source positions, and stores all impulse responses::

    <processed>/cache/rir_bank_<rate>.npz    the index: where each response starts, the rooms
    <processed>/cache/rir_bank_<rate>.npy    all responses one after the other, float16

Each response is cut where less than ``trim_db`` (−60 dB) of its energy is left, then scaled to
unit energy again. :class:`RirBank` opens the ``.npy`` memory-mapped (every DataLoader worker
shares the same pages) and :meth:`RirBank.draw` hands out one random room with a different
position for every event.

``TODO(module-3)``: the multi-channel version (one RIR per microphone of the array, from the
device profile's geometry) replaces the single microphone here.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
import zlib
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from soundspy.config import ReverbConfig

MIN_SOURCE_DISTANCE_M = 0.3  # a source is never placed closer than this to the microphone


class ReverbUnavailable(RuntimeError):
    """``pyroomacoustics`` is not installed."""


@dataclass(frozen=True)
class Room:
    dims: tuple[float, float, float]  # length, width, height in metres
    rt60_s: float
    mic: tuple[float, float, float]
    sources: tuple[tuple[float, float, float], ...]


def recipe_rng(seed: int, recipe_id: str) -> np.random.Generator:
    """A generator that depends only on ``seed`` and the recipe id (stable across runs)."""
    return np.random.default_rng([seed, zlib.crc32(recipe_id.encode("utf-8"))])


def _position(rng: np.random.Generator, dims: tuple[float, float, float], margin: float
              ) -> tuple[float, float, float]:
    return tuple(float(rng.uniform(margin, d - margin)) for d in dims)


def sample_room(rng: np.random.Generator, cfg: ReverbConfig, n_sources: int) -> Room:
    dims = tuple(float(rng.uniform(low, high)) for low, high in cfg.room_m)
    rt60 = float(rng.uniform(*cfg.rt60_s))
    mic = _position(rng, dims, cfg.wall_margin_m)
    sources = []
    for _ in range(n_sources):
        for _attempt in range(20):
            pos = _position(rng, dims, cfg.wall_margin_m)
            if float(np.linalg.norm(np.subtract(pos, mic))) >= MIN_SOURCE_DISTANCE_M:
                break
        sources.append(pos)
    return Room(dims, rt60, mic, tuple(sources))


def unit_energy(rir: np.ndarray) -> np.ndarray:
    energy = float(np.sum(np.square(rir, dtype=np.float64)))
    return (rir / np.sqrt(energy)).astype(np.float32) if energy > 0 else rir.astype(np.float32)


def simulate_rirs(room: Room, sample_rate: int, max_order: int) -> list[np.ndarray]:
    """One unit-energy impulse response per source, starting at the direct sound.

    The common delay of the direct path is removed (the earliest arrival over all sources is at
    sample 0 of its response and the others keep their relative delay to it), so reverberation
    does not shift the events in time by more than the difference in distance.
    """
    try:
        import pyroomacoustics as pra
    except ImportError as exc:
        raise ReverbUnavailable(
            'reverberation needs pyroomacoustics (pip install -e ".[spatial]")') from exc

    volume = float(np.prod(room.dims))
    surface = 2.0 * (room.dims[0] * room.dims[1] + room.dims[0] * room.dims[2]
                     + room.dims[1] * room.dims[2])
    # Sabine: RT60 = 0.161 V / (S a). The image-source order is capped (cost grows with its
    # cube), which shortens the latest part of long tails a little.
    absorption = min(0.99, max(0.01, 0.161 * volume / (surface * room.rt60_s)))
    shoebox = pra.ShoeBox(list(room.dims), fs=sample_rate, materials=pra.Material(absorption),
                          max_order=max_order)
    for source in room.sources:
        shoebox.add_source(list(source))
    shoebox.add_microphone(list(room.mic))
    shoebox.compute_rir()
    rirs = [np.asarray(shoebox.rir[0][i], dtype=np.float64) for i in range(len(room.sources))]
    # pyroomacoustics delays everything by its fractional-delay filter length; drop the samples
    # before the earliest direct sound
    distances = [float(np.linalg.norm(np.subtract(s, room.mic))) for s in room.sources]
    first = int(round(min(distances) / pra.constants.get("c") * sample_rate))
    lead = pra.constants.get("frac_delay_length") // 2 + first
    return [unit_energy(rir[max(0, lead - 2):]) for rir in rirs]


def recipe_rirs(recipe_id: str, n_events: int, sample_rate: int, cfg: ReverbConfig, seed: int
                ) -> list[np.ndarray] | None:
    """The impulse responses of a recipe's events, or ``None`` if this recipe stays dry."""
    rng = recipe_rng(seed, recipe_id)
    if n_events == 0 or rng.random() >= cfg.prob:
        return None
    return simulate_rirs(sample_room(rng, cfg, n_events), sample_rate, cfg.max_order)


# --------------------------------------------------------------------------------------------
# The bank of precomputed rooms (step 07b)

BANK_VERSION = 1
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


class RirBankError(RuntimeError):
    pass


def rir_bank_path(processed: Path, sample_rate: int) -> Path:
    return Path(processed) / "cache" / f"rir_bank_{sample_rate}.npz"


def _data_path(index_path: Path) -> Path:
    return Path(index_path).with_suffix(".npy")


def trim_tail(rir: np.ndarray, trim_db: float) -> np.ndarray:
    """``rir`` without the end that holds less than ``trim_db`` of its energy (unit energy)."""
    energy = np.cumsum(np.square(rir, dtype=np.float64)[::-1])[::-1]  # energy from i to the end
    total = float(energy[0]) if energy.size else 0.0
    if total <= 0.0:
        return rir.astype(np.float32)
    keep = int(np.searchsorted(-energy, -total * 10.0 ** (trim_db / 10.0)))
    return unit_energy(rir[:max(1, keep)])


def bank_signature(cfg: ReverbConfig, sample_rate: int, rooms: int, positions: int, seed: int,
                   trim_db: float) -> dict:
    reverb = dataclasses.asdict(cfg)
    reverb.pop("prob", None)  # how often a room is used does not change the rooms
    return json.loads(json.dumps({
        "version": BANK_VERSION, "sample_rate": sample_rate, "rooms": rooms,
        "positions": positions, "seed": seed, "trim_db": trim_db, "reverb": reverb}))


def _simulate_rooms(numbers: list[int], cfg: ReverbConfig, sample_rate: int, positions: int,
                    seed: int, trim_db: float):
    out = []
    for number in numbers:
        rng = np.random.default_rng([seed, number])
        room = sample_room(rng, cfg, positions)
        rirs = [trim_tail(r, trim_db).astype(np.float16)
                for r in simulate_rirs(room, sample_rate, cfg.max_order)]
        out.append((number, rirs, (*room.dims, room.rt60_s)))
    return out


def build_rir_bank(cfg: ReverbConfig, sample_rate: int, path: Path, *, rooms: int,
                   positions: int, seed: int, trim_db: float = -60.0,
                   workers: int | None = None, force: bool = False,
                   progress: Callable[[str], None] = print) -> dict:
    """Simulate the bank (nothing is done when a bank with the same settings exists)."""
    path = Path(path)
    signature = bank_signature(cfg, sample_rate, rooms, positions, seed, trim_db)
    if path.is_file() and _data_path(path).is_file():
        with np.load(path, allow_pickle=False) as npz:
            same = json.loads(str(npz["signature"])) == signature
        if same:
            return {"path": path, "built": False, "rooms": rooms, "positions": positions,
                    "bytes": _data_path(path).stat().st_size, "seconds": 0.0}
        if not force:
            raise RirBankError(f"the room bank at {path} was made with other settings; "
                               "run again with --force to rebuild it")
    started = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(workers or max(1, (os.cpu_count() or 2) - 2), rooms))
    size = max(1, min(25, rooms // (workers * 2) or 1))
    batches = [list(range(i, min(rooms, i + size))) for i in range(0, rooms, size)]
    progress(f"simulating {rooms} rooms x {positions} positions at {sample_rate} Hz "
             f"({workers} processes)")
    results: dict[int, tuple[list[np.ndarray], tuple[float, ...]]] = {}

    def gather(batch_results) -> None:
        for batch in batch_results:
            for number, rirs, facts in batch:
                results[number] = (rirs, facts)
            if len(results) % max(100, rooms // 10) < size or len(results) == rooms:
                progress(f"  {len(results)}/{rooms} rooms "
                         f"({time.perf_counter() - started:.0f} s)")

    args = (cfg, sample_rate, positions, seed, trim_db)
    if workers == 1:
        gather(_simulate_rooms(batch, *args) for batch in batches)
    else:
        saved = {key: os.environ.get(key) for key in THREAD_ENV}
        os.environ.update({key: "1" for key in THREAD_ENV})
        try:
            with ProcessPoolExecutor(workers) as pool:
                futures = [pool.submit(_simulate_rooms, batch, *args) for batch in batches]
                gather(f.result() for f in futures)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    mean_length = write_rir_bank(path, [results[n][0] for n in range(rooms)],
                                 [results[n][1] for n in range(rooms)], signature)
    return {"path": path, "built": True, "rooms": rooms, "positions": positions,
            "bytes": _data_path(path).stat().st_size,
            "seconds": time.perf_counter() - started,
            "mean_length_s": mean_length / sample_rate}


def write_rir_bank(path: Path, rirs: list[list[np.ndarray]], facts: list[tuple[float, ...]],
                   signature: dict) -> float:
    """Write a bank from ``rirs[room][position]``; returns the mean response length (samples).
    ``signature`` must hold ``sample_rate``."""
    path = Path(path)
    positions = len(rirs[0])
    if any(len(room) != positions for room in rirs):
        raise RirBankError("every room needs the same number of positions")
    lengths = np.array([len(r) for room in rirs for r in room], dtype=np.int64)
    offsets = np.zeros(lengths.size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = _data_path(path).with_name(_data_path(path).name + ".part")
    data = np.lib.format.open_memmap(part, mode="w+", dtype=np.float16, shape=(int(offsets[-1]),))
    k = 0
    for room in rirs:
        for rir in room:
            data[offsets[k]:offsets[k + 1]] = rir
            k += 1
    data.flush()
    del data
    os.replace(part, _data_path(path))
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, offsets=offsets, signature=np.array(json.dumps(signature)),
             rooms=np.array(facts, dtype=np.float32), positions=np.int64(positions))
    os.replace(tmp, path)
    return float(lengths.mean())


class RirBank:
    """A finished bank: ``rir(room, position)`` returns a float32 copy from the memory map."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_file() or not _data_path(self.path).is_file():
            raise RirBankError(f"no room bank at {self.path}: run "
                               "`soundspy data cache --task rirs` first")
        with np.load(self.path, allow_pickle=False) as npz:
            self.offsets = npz["offsets"]
            self.room_facts = npz["rooms"]  # (rooms, 4): length, width, height, RT60
            self.positions = int(npz["positions"])
            self.signature = json.loads(str(npz["signature"]))
        self.rooms = int(self.room_facts.shape[0])
        self.sample_rate = int(self.signature["sample_rate"])
        self._data: np.ndarray | None = None

    def rir(self, room: int, position: int) -> np.ndarray:
        if not (0 <= room < self.rooms and 0 <= position < self.positions):
            raise IndexError((room, position))
        if self._data is None:
            self._data = np.load(_data_path(self.path), mmap_mode="r")
        k = room * self.positions + position
        return np.asarray(self._data[self.offsets[k]:self.offsets[k + 1]], dtype=np.float32)

    def draw(self, rng: np.random.Generator, n_events: int) -> list[np.ndarray]:
        """One random room; every event gets its own position in it (reused only when there
        are more events than positions)."""
        room = int(rng.integers(self.rooms))
        chosen = rng.choice(self.positions, size=n_events, replace=n_events > self.positions)
        return [self.rir(room, int(p)) for p in chosen]

    def close(self) -> None:
        self._data = None

    def __getstate__(self) -> dict:  # the memory map is reopened in worker processes
        state = dict(self.__dict__)
        state["_data"] = None
        return state
