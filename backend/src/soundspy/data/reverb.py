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

``TODO(module-3)``: the multi-channel version (one RIR per microphone of the array, from the
device profile's geometry) replaces the single microphone here.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

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
