"""Synthetic mixtures (SemanticHearing style): recipes now, audio on the fly.

A **recipe** is a few lines of JSON that say exactly how to build one training example: which
clips, which part of each clip, where in the mixture, how loud, and which classes form the target.
Recipes are generated once, deterministically from the config seed, and stored as JSON lines in
``<mixtures>/<split>.jsonl``. Audio is rendered from a recipe when it is needed (training,
preview), at any sample rate: nothing is resampled or rendered ahead of time.

Generation rules (``mixtures`` in ``configs/data/default.yaml``):

* ``n_foreground`` **distinct** foreground classes, chosen uniformly among the classes that have
  clips in the split (not in proportion to clip counts, so rare classes are seen often);
  ``n_background`` background clips. Only clips of the recipe's own split are used.
* Each event starts inside an **active** region of its clip (activity map) and lasts at least
  ``event_min_s``; clips shorter than that are placed whole. Onsets are uniform in the mixture.
* ``target`` is a random non-empty subset (size ``target_size``) of the present classes; with
  probability ``empty_target_prob`` it is a class that is **not** present (correct output:
  silence). ``target_to_rest_db`` is sampled per recipe.

Rendering (:func:`render`): every event is normalised to the reference loudness (RMS over its
active frames), gets its ``gain_db``, and is added either to the *target* sum or to the *rest*.
The target is then scaled so that target vs. rest is ``target_to_rest_db``; ``mixture = target +
rest`` exactly. If the mixture peak exceeds ``peak_limit``, target and rest (and so the mixture)
are scaled by the same factor.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from soundspy.config import AppConfig, MixtureConfig
from soundspy.core.audio import AudioBuffer
from soundspy.core.labels import LabelSet
from soundspy.data.activity import ActivityStore, active_regions
from soundspy.data.aliases import resolve_alias
from soundspy.data.roles import BG, role_class
from soundspy.dsp.channels import ChannelReducer
from soundspy.dsp.resample import resample_array

_EPS = 1e-12
TIME_DECIMALS = 3  # recipes store times in ms precision


class RecipeError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------
# Recipe format


@dataclass(frozen=True)
class Event:
    clip_id: str
    role: str  # "fg" | "bg"
    cls: str | None  # class key for fg events, None for bg
    clip_offset_s: float
    onset_s: float
    dur_s: float
    gain_db: float

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["class"] = d.pop("cls")
        return {k: d[k] for k in ("clip_id", "role", "class", "clip_offset_s", "onset_s",
                                  "dur_s", "gain_db")}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Event:
        return cls(str(d["clip_id"]), str(d["role"]), d.get("class"),
                   float(d["clip_offset_s"]), float(d["onset_s"]), float(d["dur_s"]),
                   float(d["gain_db"]))


@dataclass(frozen=True)
class Recipe:
    id: str
    duration_s: float
    events: tuple[Event, ...]
    present: tuple[str, ...]
    target: tuple[str, ...]
    target_to_rest_db: float

    @property
    def empty_target(self) -> bool:
        return not set(self.target) & set(self.present)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "duration_s": self.duration_s,
            "events": [e.to_json() for e in self.events],
            "present": list(self.present),
            "target": list(self.target),
            "target_to_rest_db": self.target_to_rest_db,
        }

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Recipe:
        return cls(str(d["id"]), float(d["duration_s"]),
                   tuple(Event.from_json(e) for e in d["events"]),
                   tuple(d["present"]), tuple(d["target"]), float(d["target_to_rest_db"]))


def recipes_path(mixtures_dir: Path, split: str) -> Path:
    return Path(mixtures_dir) / f"{split}.jsonl"


def save_recipes(recipes: Iterable[Recipe], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        for recipe in recipes:
            fh.write(json.dumps(recipe.to_json(), separators=(",", ":")) + "\n")
            n += 1
    tmp.replace(path)
    return n


def iter_recipes(path: Path) -> Iterator[Recipe]:
    if not path.exists():
        raise RecipeError(f"{path} not found: run `soundspy data mix` first")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield Recipe.from_json(json.loads(line))


# --------------------------------------------------------------------------------------------
# Clip library: index rows + activity maps + file access


@dataclass(frozen=True)
class ClipInfo:
    clip_id: str
    path: str  # alias
    duration_s: float
    sample_rate: int
    role: str
    split: str


class ClipLibrary:
    """Everything needed to pick and load clips: the index, activity maps and ``paths``."""

    def __init__(
        self,
        index,
        activity: Mapping[str, ActivityStore],
        paths: Mapping[str, Path],
        *,
        threshold_db: float,
    ) -> None:
        self.paths = paths
        self.activity = activity
        self.threshold_db = threshold_db
        self.clips: dict[str, ClipInfo] = {}
        for rec in index[["clip_id", "path", "duration_s", "sample_rate", "role",
                          "split"]].itertuples(index=False):
            self.clips[rec.clip_id] = ClipInfo(rec.clip_id, rec.path, float(rec.duration_s),
                                               int(rec.sample_rate), rec.role, rec.split)
        self._regions: dict[str, list[tuple[float, float]]] = {}

    @classmethod
    def from_config(cls, config: AppConfig) -> ClipLibrary:
        from soundspy.data.prepare import load_activity, load_clip_index

        processed = config.paths["processed"]
        return cls(load_clip_index(processed), load_activity(processed), config.paths,
                   threshold_db=config.data.activity_threshold_db)

    def info(self, clip_id: str) -> ClipInfo:
        try:
            return self.clips[clip_id]
        except KeyError:
            raise RecipeError(f"clip {clip_id} is not in the clip index") from None

    def _store(self, clip_id: str) -> ActivityStore | None:
        store = self.activity.get(clip_id.split(":", 1)[0])
        return store if store is not None and clip_id in store else None

    def regions(self, clip_id: str) -> list[tuple[float, float]]:
        """Active regions in seconds (the whole clip if no activity map is available)."""
        if clip_id not in self._regions:
            store = self._store(clip_id)
            info = self.info(clip_id)
            regions = []
            if store is not None:
                regions = active_regions(store.envelope(clip_id), store.hop_s, self.threshold_db)
                regions = [(s, min(e, info.duration_s)) for s, e in regions
                           if s < info.duration_s]
            self._regions[clip_id] = regions or [(0.0, info.duration_s)]
        return self._regions[clip_id]

    def load_segment(
        self, clip_id: str, offset_s: float, dur_s: float, sample_rate: int
    ) -> np.ndarray:
        """Mono float32 samples of ``[offset, offset + dur)`` at ``sample_rate``.

        Always exactly ``round(dur_s * sample_rate)`` samples (zero-padded past the clip end).
        """
        info = self.info(clip_id)
        path = resolve_alias(info.path, self.paths)
        with sf.SoundFile(str(path)) as fh:
            native = fh.samplerate
            start = min(round(offset_s * native), fh.frames)
            fh.seek(start)
            data = fh.read(round(dur_s * native), dtype="float32", always_2d=True)
        buf = AudioBuffer(data, native) if data.shape[0] else AudioBuffer(
            np.zeros((0, data.shape[1]), np.float32), native)
        if buf.num_channels > 1:
            buf = ChannelReducer("mean").process(buf)
        out = resample_array(buf.samples, native, sample_rate).ravel()
        n = round(dur_s * sample_rate)
        if out.shape[0] < n:
            out = np.concatenate([out, np.zeros(n - out.shape[0], np.float32)])
        return out[:n].astype(np.float32, copy=False)


@dataclass
class SplitPool:
    split: str
    fg: dict[str, list[str]]  # class -> clip ids
    bg: list[str]

    @classmethod
    def from_library(cls, library: ClipLibrary, split: str, labels: LabelSet) -> SplitPool:
        fg: dict[str, list[str]] = {}
        bg: list[str] = []
        for info in library.clips.values():
            if info.split != split:
                continue
            key = role_class(info.role)
            if key is not None and key in labels:
                fg.setdefault(key, []).append(info.clip_id)
            elif info.role == BG:
                bg.append(info.clip_id)
        order = list(labels)
        fg = {k: sorted(fg[k]) for k in sorted(fg, key=order.index)}
        return cls(split, fg, sorted(bg))


# --------------------------------------------------------------------------------------------
# Generation


def _r(x: float) -> float:
    return round(float(x), TIME_DECIMALS)


def _place(
    rng: random.Random,
    library: ClipLibrary,
    clip_id: str,
    *,
    duration_s: float,
    event_min_s: float,
    fill: bool,
) -> tuple[float, float, float]:
    """``(clip_offset_s, onset_s, dur_s)`` for one event.

    The offset lies in an active region of the clip; ``fill`` (background) makes the event
    as long as possible, otherwise the length is uniform in ``[event_min_s, available]``.
    """
    length = library.info(clip_id).duration_s
    max_dur = min(length, duration_s)
    if length <= event_min_s:
        offset, dur = 0.0, length  # short clip: placed whole
    else:
        regions = library.regions(clip_id)
        weights = [max(e - s, 1e-3) for s, e in regions]
        start, end = rng.choices(regions, weights=weights)[0]
        latest = length - event_min_s
        lo = min(start, latest)
        hi = min(end, latest)
        offset = _r(rng.uniform(lo, hi) if hi > lo else lo)
        available = min(length - offset, duration_s)
        dur = available if fill else rng.uniform(min(event_min_s, available), available)
    dur = _r(min(dur, max_dur))
    onset = _r(rng.uniform(0.0, max(0.0, duration_s - dur)))
    return _r(offset), onset, dur


def generate_recipes(
    library: ClipLibrary,
    labels: LabelSet,
    mix: MixtureConfig,
    split: str,
    count: int,
    seed: int,
) -> list[Recipe]:
    """``count`` recipes for ``split``, deterministic for ``seed``."""
    pool = SplitPool.from_library(library, split, labels)
    classes = list(pool.fg)
    if not classes:
        raise RecipeError(f"split {split!r} has no foreground clips: run `soundspy data prepare`")
    if mix.n_background[1] > 0 and not pool.bg and mix.n_background[0] > 0:
        raise RecipeError(f"split {split!r} has no background clips")
    order = list(labels)
    rng = random.Random(seed)
    recipes = []
    width = max(6, len(str(count)))
    for i in range(count):
        hi = min(mix.n_foreground[1], len(classes))
        n_fg = rng.randint(min(mix.n_foreground[0], hi), hi)
        chosen = rng.sample(classes, n_fg)
        events = []
        for key in chosen:
            clip = rng.choice(pool.fg[key])
            offset, onset, dur = _place(rng, library, clip, duration_s=mix.duration_s,
                                        event_min_s=mix.event_min_s, fill=False)
            events.append(Event(clip, "fg", key, offset, onset, dur,
                                _r(rng.uniform(*mix.fg_gain_db))))
        n_bg = rng.randint(*mix.n_background) if pool.bg else 0
        for _ in range(n_bg):
            clip = rng.choice(pool.bg)
            offset, onset, dur = _place(rng, library, clip, duration_s=mix.duration_s,
                                        event_min_s=mix.event_min_s, fill=True)
            events.append(Event(clip, "bg", None, offset, onset, dur,
                                _r(rng.uniform(*mix.bg_gain_db))))
        present = sorted(chosen, key=order.index)
        absent = [k for k in order if k not in present]
        if absent and rng.random() < mix.empty_target_prob:
            target = [rng.choice(absent)]
        else:
            k = rng.randint(mix.target_size[0], min(mix.target_size[1], len(present)))
            target = sorted(rng.sample(present, max(1, min(k, len(present)))), key=order.index)
        recipes.append(Recipe(
            id=f"{split}-{i + 1:0{width}d}",
            duration_s=mix.duration_s,
            events=tuple(events),
            present=tuple(present),
            target=tuple(target),
            target_to_rest_db=_r(rng.uniform(*mix.target_to_rest_db)),
        ))
    return recipes


def write_recipes(
    index,
    activity: Mapping[str, ActivityStore],
    config: AppConfig,
    split: str,
    *,
    count: int | None = None,
) -> tuple[Path, int]:
    mix = config.data.mixtures
    library = ClipLibrary(index, activity, config.paths,
                          threshold_db=config.data.activity_threshold_db)
    n = mix.counts[split] if count is None else count
    if n < 0:
        raise RecipeError("--count must be >= 0")
    recipes = generate_recipes(library, config.labels, mix, split, n, mix.seed[split])
    path = recipes_path(config.paths["mixtures"], split)
    return path, save_recipes(recipes, path)


# --------------------------------------------------------------------------------------------
# Rendering


@dataclass
class RenderedMixture:
    recipe: Recipe
    mixture: AudioBuffer  # (frames, 1)
    target: AudioBuffer  # (frames, 1)
    rest: AudioBuffer  # (frames, 1); mixture == target + rest
    present: np.ndarray  # multi-hot over LabelSet
    target_hot: np.ndarray  # multi-hot over LabelSet
    class_activity: np.ndarray  # (n_frames, n_classes) float32, 1 = class sounds in frame
    activity_rate_hz: float
    scale: float = 1.0  # shared peak-limit factor applied to target and rest
    target_gain_db: float = 0.0  # gain applied to reach target_to_rest_db
    meta: dict[str, Any] = field(default_factory=dict)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def _active_rms(
    segment: np.ndarray, sample_rate: int, regions: Sequence[tuple[float, float]],
    offset_s: float,
) -> float:
    """RMS of ``segment`` over the parts that fall in the clip's active regions."""
    mask = np.zeros(segment.shape[0], dtype=bool)
    for start, end in regions:
        a = max(0, round((start - offset_s) * sample_rate))
        b = min(segment.shape[0], round((end - offset_s) * sample_rate))
        if b > a:
            mask[a:b] = True
    level = _rms(segment[mask]) if mask.any() else 0.0
    return level if level > 0 else _rms(segment)


def render(
    recipe: Recipe,
    sample_rate: int,
    library: ClipLibrary,
    labels: LabelSet,
    *,
    reference_rms_dbfs: float,
    peak_limit: float,
    activity_rate_hz: float = 10.0,
) -> RenderedMixture:
    """Render ``recipe`` at ``sample_rate`` (mono). Same recipe -> identical samples."""
    n = round(recipe.duration_s * sample_rate)
    target = np.zeros(n, dtype=np.float64)
    rest = np.zeros(n, dtype=np.float64)
    ref = 10.0 ** (reference_rms_dbfs / 20.0)
    n_frames = max(1, math.ceil(recipe.duration_s * activity_rate_hz - 1e-9))
    activity = np.zeros((n_frames, len(labels)), dtype=np.float32)
    target_set = set(recipe.target)

    for event in recipe.events:
        seg = library.load_segment(event.clip_id, event.clip_offset_s, event.dur_s,
                                   sample_rate).astype(np.float64)
        regions = library.regions(event.clip_id)
        level = _active_rms(seg, sample_rate, regions, event.clip_offset_s)
        if level > _EPS:
            seg *= ref / level
        seg *= 10.0 ** (event.gain_db / 20.0)
        # TODO(module-3): spatial rendering goes here: convolve `seg` with the room impulse
        # response from the event's source position to every microphone (pyroomacoustics)
        # instead of adding it to a single mono channel.
        start = round(event.onset_s * sample_rate)
        stop = min(n, start + seg.shape[0])
        if stop <= start:
            continue
        dest = target if (event.role == "fg" and event.cls in target_set) else rest
        dest[start:stop] += seg[: stop - start]
        if event.role == "fg" and event.cls in labels:
            col = labels.index(event.cls)
            for a, b in regions:
                a, b = max(a, event.clip_offset_s), min(b, event.clip_offset_s + event.dur_s)
                if b <= a:
                    continue
                t0 = event.onset_s + (a - event.clip_offset_s)
                t1 = min(recipe.duration_s, event.onset_s + (b - event.clip_offset_s))
                f0 = int(math.floor(t0 * activity_rate_hz))
                f1 = int(math.ceil(t1 * activity_rate_hz - 1e-9))
                activity[max(0, f0):min(n_frames, f1), col] = 1.0

    gain_db = 0.0
    p_target, p_rest = _rms(target), _rms(rest)
    if p_target > _EPS and p_rest > _EPS:
        gain_db = recipe.target_to_rest_db - 20.0 * math.log10(p_target / p_rest)
        target *= 10.0 ** (gain_db / 20.0)

    target32 = target.astype(np.float32)
    rest32 = rest.astype(np.float32)
    mixture32 = target32 + rest32
    peak = float(np.max(np.abs(mixture32))) if n else 0.0
    scale = 1.0
    if peak > peak_limit:
        scale = peak_limit / peak
        target32 = (target32 * np.float32(scale)).astype(np.float32)
        rest32 = (rest32 * np.float32(scale)).astype(np.float32)
        mixture32 = target32 + rest32

    def buf(x: np.ndarray) -> AudioBuffer:
        return AudioBuffer(x.reshape(-1, 1), sample_rate, {"recipe": recipe.id})

    return RenderedMixture(
        recipe=recipe,
        mixture=buf(mixture32),
        target=buf(target32),
        rest=buf(rest32),
        present=labels.to_multi_hot(recipe.present),
        target_hot=labels.to_multi_hot(recipe.target),
        class_activity=activity,
        activity_rate_hz=activity_rate_hz,
        scale=scale,
        target_gain_db=gain_db,
    )


def render_with_config(
    recipe: Recipe, sample_rate: int, library: ClipLibrary, config: AppConfig,
    *, activity_rate_hz: float = 10.0,
) -> RenderedMixture:
    mix = config.data.mixtures
    return render(recipe, sample_rate, library, config.labels,
                  reference_rms_dbfs=mix.reference_rms_dbfs, peak_limit=mix.peak_limit,
                  activity_rate_hz=activity_rate_hz)


# --------------------------------------------------------------------------------------------
# Preview

PREVIEW_RATE = 44_100


def write_previews(config: AppConfig, split: str, n: int) -> tuple[Path, list[str]]:
    """Render the first ``n`` recipes of ``split`` to ``outputs/previews`` (44.1 kHz WAV)."""
    library = ClipLibrary.from_config(config)
    out_dir = config.paths["outputs"] / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = []
    for recipe in iter_recipes(recipes_path(config.paths["mixtures"], split)):
        if len(ids) >= n:
            break
        rendered = render_with_config(recipe, PREVIEW_RATE, library, config)
        sf.write(out_dir / f"{recipe.id}_mix.wav", rendered.mixture.samples, PREVIEW_RATE,
                 subtype="FLOAT")
        sf.write(out_dir / f"{recipe.id}_target.wav", rendered.target.samples, PREVIEW_RATE,
                 subtype="FLOAT")
        info = {**recipe.to_json(),
                "render": {"sample_rate": PREVIEW_RATE, "peak_scale": round(rendered.scale, 4),
                           "target_gain_db": round(rendered.target_gain_db, 2)}}
        (out_dir / f"{recipe.id}.json").write_text(json.dumps(info, indent=2) + "\n",
                                                   encoding="utf-8")
        ids.append(recipe.id)
    return out_dir, ids
