"""Configuration loading.

Reads ``backend/configs/default.yaml``, the device profiles in ``backend/configs/devices/``,
``classes.yaml`` (-> ``LabelSet``) and the detection / extraction configs. Relative paths in the
configs are resolved against the **project root** (the folder that contains ``backend/`` and
``frontend/``), so ``data/``, ``models/``, ``outputs/`` and ``experiments/`` live at the root.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from soundspy.core.labels import LabelSet

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS_DIR = BACKEND_DIR / "configs"
DEFAULT_PROJECT_ROOT = BACKEND_DIR.parent


class ConfigError(ValueError):
    """A configuration file is missing a field or holds an invalid value."""


@dataclass(frozen=True)
class DeviceProfile:
    """Microphone description from ``configs/devices/<name>.yaml``.

    - ``device``: ``None`` = system default input, an ``int`` device index, or a
      case-insensitive substring of the device name.
    - ``hostapi``: ``None`` = host API preference order (WASAPI first on Windows), or a
      case-insensitive substring of a host API name.
    - ``sample_rate``: ``None`` = the device's native rate.
    - ``raw_channels``: ``None`` = every channel, or the 0-based device channels to keep.
    - ``geometry``: mic coordinates in metres, one ``(x, y, z)`` per kept channel.
    """

    name: str
    device: str | int | None = None
    hostapi: str | None = None
    sample_rate: int | None = None
    raw_channels: tuple[int, ...] | None = None
    geometry: tuple[tuple[float, ...], ...] | None = None

    @property
    def has_geometry(self) -> bool:
        return bool(self.geometry)


DEVICE_PROFILE_KEYS = ("device", "hostapi", "sample_rate", "raw_channels", "geometry")


def parse_device_profile(name: str, data: dict[str, Any], source: str = "") -> DeviceProfile:
    """Validate a device-profile mapping and build a ``DeviceProfile``."""
    where = f"device profile {name!r}" + (f" ({source})" if source else "")

    def fail(message: str) -> ConfigError:
        return ConfigError(f"{where}: {message}")

    unknown = sorted(set(data) - set(DEVICE_PROFILE_KEYS))
    if unknown:
        raise fail(f"unknown field(s) {unknown}; allowed: {list(DEVICE_PROFILE_KEYS)}")

    device = data.get("device")
    if isinstance(device, bool) or not (device is None or isinstance(device, int | str)):
        raise fail(f"'device' must be null, an index or a name, got {device!r}")
    if isinstance(device, int) and device < 0:
        raise fail(f"'device' index must be >= 0, got {device}")
    if isinstance(device, str) and not device.strip():
        raise fail("'device' name must not be empty")

    hostapi = data.get("hostapi")
    if hostapi is not None and (not isinstance(hostapi, str) or not hostapi.strip()):
        raise fail(f"'hostapi' must be null or a host API name, got {hostapi!r}")

    sample_rate = data.get("sample_rate")
    if sample_rate is not None and (
        isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0
    ):
        raise fail(f"'sample_rate' must be null or a positive integer, got {sample_rate!r}")

    raw = data.get("raw_channels")
    raw_channels: tuple[int, ...] | None = None
    if raw is not None:
        if (
            not isinstance(raw, list)
            or not raw
            or not all(isinstance(c, int) and not isinstance(c, bool) and c >= 0 for c in raw)
        ):
            raise fail(
                f"'raw_channels' must be null or a non-empty list of indices >= 0, got {raw!r}"
            )
        if len(set(raw)) != len(raw):
            raise fail(f"'raw_channels' contains duplicates: {raw}")
        raw_channels = tuple(raw)

    geo = data.get("geometry")
    geometry: tuple[tuple[float, ...], ...] | None = None
    if geo is not None:
        if not isinstance(geo, list) or not geo:
            raise fail("'geometry' must be null or a list of [x, y, z] positions in metres")
        for mic in geo:
            if (
                not isinstance(mic, list)
                or len(mic) != 3
                or not all(isinstance(v, int | float) and not isinstance(v, bool) for v in mic)
            ):
                raise fail(f"each 'geometry' entry must be [x, y, z] in metres, got {mic!r}")
        if raw_channels is not None and len(geo) != len(raw_channels):
            raise fail(
                f"'geometry' has {len(geo)} positions but 'raw_channels' keeps "
                f"{len(raw_channels)} channels"
            )
        geometry = tuple(tuple(float(v) for v in mic) for mic in geo)

    return DeviceProfile(
        name=name,
        device=device,
        hostapi=hostapi,
        sample_rate=sample_rate,
        raw_channels=raw_channels,
        geometry=geometry,
    )


@dataclass(frozen=True)
class ApiConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)
    ws_heartbeat_s: float = 2.0
    ws_client_queue: int = 64


# The sections below have no defaults in code: every value comes from default.yaml.


@dataclass(frozen=True)
class CaptureConfig:
    stall_timeout_s: float
    queue_chunks: int
    startup_silence_max_s: float


@dataclass(frozen=True)
class EngineConfig:
    chunk_ms: float
    rtf_warning: float
    rtf_window_s: float
    status_interval_s: float

    def chunk_frames(self, sample_rate: int) -> int:
        return max(1, round(self.chunk_ms * sample_rate / 1000))


@dataclass(frozen=True)
class OutputConfig:
    device: str | int | None
    prebuffer_chunks: int
    max_queue_chunks: int
    limiter_ceiling_dbfs: float
    limiter_release_ms: float
    default_gain_db: float


@dataclass(frozen=True)
class AnalysisConfig:
    levels_hz: float
    spectrogram_hz: float
    spectrogram_fft: int
    spectrogram_bands: int
    spectrogram_fmin_hz: float
    spectrogram_db_floor: float
    waveform_hz: float
    waveform_points: int


SPLITS: tuple[str, ...] = ("train", "val", "test")
DATASETS: tuple[str, ...] = ("esc50", "fsd50k", "custom")


@dataclass(frozen=True)
class MixtureConfig:
    duration_s: float
    counts: dict[str, int]
    seed: dict[str, int]
    n_foreground: tuple[int, int]
    n_background: tuple[int, int]
    target_size: tuple[int, int]
    empty_target_prob: float
    event_min_s: float
    fg_gain_db: tuple[float, float]
    bg_gain_db: tuple[float, float]
    target_to_rest_db: tuple[float, float]
    peak_limit: float
    reference_rms_dbfs: float


@dataclass(frozen=True)
class DataConfig:
    """``configs/data/default.yaml``: datasets, roles, activity, splits, mixtures."""

    enabled: dict[str, bool]
    min_fg_clips_warning: int
    activity_hop_s: float
    activity_threshold_db: float
    esc50_folds: dict[str, tuple[int, ...]]
    custom_val_fraction: float
    custom_test_fraction: float
    custom_seed: int
    fsd50k_min_val_clips_per_class: int
    mixtures: MixtureConfig


@dataclass(frozen=True)
class LiveDetectionConfig:
    hop_s: float
    ema_alpha: float
    hysteresis: float
    device: str  # auto | cuda | cpu
    cpu_threads: int  # PyTorch threads of the inference thread when it runs on the CPU


@dataclass(frozen=True)
class TrainingConfig:
    name: str
    epochs: int
    batch_size: int
    num_workers: int
    loader_timeout_s: float
    lr_backbone: float
    lr_head: float
    weight_decay: float
    warmup_epochs: float
    pos_weight: bool
    amp: bool
    seed: int
    smoke_epochs: int
    smoke_train_recipes: int
    smoke_val_recipes: int


@dataclass(frozen=True)
class AugmentConfig:
    gain_db: float
    noise_snr_db: tuple[float, float] | None
    freqm: int
    timem: int
    mel_range: bool


@dataclass(frozen=True)
class DetectionConfig:
    """``configs/detection/default.yaml`` (step 05)."""

    backbone: str
    model_sample_rate: int
    pretrained_file: str
    pretrained_url: str
    pretrained_sha256: str
    audioset_labels: Path  # absolute
    head_init: str  # audioset | random
    window_s: float
    input_context_s: float
    min_active_s: float
    activity_rate_hz: float
    checkpoint: Path | None  # absolute (resolved against the project root)
    live: LiveDetectionConfig
    offline_max_seconds: float
    training: TrainingConfig
    augment: AugmentConfig
    eval_windows_per_recipe: int
    eval_seed: int


BACKBONES = ("efficientat_mn10_as", "stub")


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    configs_dir: Path
    paths: dict[str, Path]
    device_profile: DeviceProfile
    labels: LabelSet
    capture: CaptureConfig
    engine: EngineConfig
    output: OutputConfig
    analysis: AnalysisConfig
    api: ApiConfig
    database_url: str
    data: DataConfig
    detection: DetectionConfig | None = None
    extraction: dict[str, dict[str, Any]] = field(default_factory=dict)


class _Section:
    """Reads one mapping from a config file with type and range checks (clear ConfigError).

    ``name`` may be dotted for nested sections (``"mixtures.counts"``); ``source`` names the file.
    """

    def __init__(
        self,
        data: dict[str, Any],
        name: str,
        keys: tuple[str, ...],
        source: str = "default.yaml",
    ) -> None:
        section = data.get(name.rsplit(".", 1)[-1])
        if not isinstance(section, dict):
            raise ConfigError(f"{source}: missing section '{name}'")
        missing = [k for k in keys if k not in section]
        unknown = sorted(set(section) - set(keys))
        if missing:
            raise ConfigError(f"{source}: '{name}' is missing {missing}")
        if unknown:
            raise ConfigError(f"{source}: '{name}' has unknown key(s) {unknown}")
        self.data, self.name, self.source = section, name, source

    def sub(self, key: str, keys: tuple[str, ...]) -> _Section:
        return _Section(self.data, f"{self.name}.{key}", keys, self.source)

    def _fail(self, key: str, rule: str) -> ConfigError:
        value = self.data[key]
        return ConfigError(f"{self.source}: '{self.name}.{key}' must be {rule}, got {value!r}")

    def boolean(self, key: str) -> bool:
        value = self.data[key]
        if not isinstance(value, bool):
            raise self._fail(key, "true or false")
        return value

    def pair(self, key: str, *, integer: bool = False, ge: float | None = None,
             le: float | None = None) -> tuple[Any, Any]:
        """A ``[low, high]`` range with low <= high."""
        value = self.data[key]
        kind = "integers" if integer else "numbers"
        ok = (
            isinstance(value, list)
            and len(value) == 2
            and all(
                not isinstance(v, bool) and isinstance(v, int if integer else int | float)
                for v in value
            )
        )
        if not ok or value[0] > value[1]:
            raise self._fail(key, f"[low, high] with two {kind}, low <= high")
        if (ge is not None and value[0] < ge) or (le is not None and value[1] > le):
            raise self._fail(key, f"within [{ge}, {le}]")
        cast = int if integer else float
        return cast(value[0]), cast(value[1])

    def int_list(self, key: str, *, ge: int, le: int) -> tuple[int, ...]:
        value = self.data[key]
        if (
            not isinstance(value, list)
            or not value
            or not all(isinstance(v, int) and not isinstance(v, bool) and ge <= v <= le
                       for v in value)
        ):
            raise self._fail(key, f"a non-empty list of integers in [{ge}, {le}]")
        return tuple(value)

    def number(self, key: str, *, gt: float | None = None, ge: float | None = None,
               le: float | None = None) -> float:
        value = self.data[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise self._fail(key, "a number")
        if gt is not None and not value > gt:
            raise self._fail(key, f"> {gt}")
        if ge is not None and not value >= ge:
            raise self._fail(key, f">= {ge}")
        if le is not None and not value <= le:
            raise self._fail(key, f"<= {le}")
        return float(value)

    def integer(self, key: str, *, ge: int) -> int:
        value = self.data[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < ge:
            raise self._fail(key, f"an integer >= {ge}")
        return int(value)


def _capture(data: dict[str, Any]) -> CaptureConfig:
    s = _Section(data, "capture", ("stall_timeout_s", "queue_chunks", "startup_silence_max_s"))
    return CaptureConfig(
        stall_timeout_s=s.number("stall_timeout_s", gt=0),
        queue_chunks=s.integer("queue_chunks", ge=1),
        startup_silence_max_s=s.number("startup_silence_max_s", ge=0),
    )


def _engine(data: dict[str, Any]) -> EngineConfig:
    s = _Section(data, "engine", ("chunk_ms", "rtf_warning", "rtf_window_s", "status_interval_s"))
    return EngineConfig(
        chunk_ms=s.number("chunk_ms", gt=0, le=1000),
        rtf_warning=s.number("rtf_warning", gt=0),
        rtf_window_s=s.number("rtf_window_s", gt=0),
        status_interval_s=s.number("status_interval_s", gt=0),
    )


def _output(data: dict[str, Any]) -> OutputConfig:
    keys = ("device", "prebuffer_chunks", "max_queue_chunks", "limiter_ceiling_dbfs",
            "limiter_release_ms", "default_gain_db")
    s = _Section(data, "output", keys)
    device = s.data["device"]
    if isinstance(device, bool) or not (device is None or isinstance(device, int | str)):
        raise s._fail("device", "null, a device index or a name substring")
    prebuffer = s.integer("prebuffer_chunks", ge=0)
    max_queue = s.integer("max_queue_chunks", ge=1)
    if max_queue < prebuffer:
        raise ConfigError("default.yaml: 'output.max_queue_chunks' must be >= prebuffer_chunks")
    return OutputConfig(
        device=device,
        prebuffer_chunks=prebuffer,
        max_queue_chunks=max_queue,
        limiter_ceiling_dbfs=s.number("limiter_ceiling_dbfs", le=0),
        limiter_release_ms=s.number("limiter_release_ms", gt=0),
        default_gain_db=s.number("default_gain_db", ge=-40, le=12),
    )


def _analysis(data: dict[str, Any]) -> AnalysisConfig:
    keys = ("levels_hz", "spectrogram_hz", "spectrogram_fft", "spectrogram_bands",
            "spectrogram_fmin_hz", "spectrogram_db_floor", "waveform_hz", "waveform_points")
    s = _Section(data, "analysis", keys)
    fft = s.integer("spectrogram_fft", ge=64)
    if fft & (fft - 1):
        raise s._fail("spectrogram_fft", "a power of two")
    return AnalysisConfig(
        levels_hz=s.number("levels_hz", gt=0, le=200),
        spectrogram_hz=s.number("spectrogram_hz", gt=0, le=200),
        spectrogram_fft=fft,
        spectrogram_bands=s.integer("spectrogram_bands", ge=8),
        spectrogram_fmin_hz=s.number("spectrogram_fmin_hz", ge=0),
        spectrogram_db_floor=s.number("spectrogram_db_floor", le=-10),
        waveform_hz=s.number("waveform_hz", gt=0, le=200),
        waveform_points=s.integer("waveform_points", ge=2),
    )


def load_data_config(path: Path) -> DataConfig:
    """Load and validate ``configs/data/default.yaml``."""
    source = f"data/{Path(path).name}"
    root = {"root": load_yaml(path)}
    top = _Section(root, "root", ("datasets", "roles", "activity", "splits", "mixtures"), source)
    top.name = ""  # error paths read "datasets.esc50.enabled", not "root.datasets..."

    def section(key: str, keys: tuple[str, ...]) -> _Section:
        return _Section(top.data, key, keys, source)

    datasets = section("datasets", DATASETS)
    enabled = {ds: datasets.sub(ds, ("enabled",)).boolean("enabled") for ds in DATASETS}
    roles = section("roles", ("min_fg_clips_warning",))
    activity = section("activity", ("hop_s", "threshold_db_below_peak"))
    splits = section("splits", ("esc50", "fsd50k", "custom"))
    esc = splits.sub("esc50", SPLITS)
    folds = {split: esc.int_list(split, ge=1, le=5) for split in SPLITS}
    all_folds = [f for fs in folds.values() for f in fs]
    if len(all_folds) != len(set(all_folds)):
        raise ConfigError(f"{source}: 'splits.esc50' uses a fold in more than one split")
    custom = splits.sub("custom", ("val_fraction", "test_fraction", "seed"))
    val_fraction = custom.number("val_fraction", ge=0, le=0.9)
    test_fraction = custom.number("test_fraction", ge=0, le=0.9)
    if val_fraction + test_fraction >= 1:
        raise ConfigError(f"{source}: 'splits.custom' val + test fractions must be < 1")

    mix_keys = ("duration_s", "counts", "seed", "n_foreground", "n_background", "target_size",
                "empty_target_prob", "event_min_s", "fg_gain_db", "bg_gain_db",
                "target_to_rest_db", "peak_limit", "reference_rms_dbfs")
    mix = section("mixtures", mix_keys)
    counts = mix.sub("counts", SPLITS)
    seeds = mix.sub("seed", SPLITS)
    duration = mix.number("duration_s", gt=0, le=120)
    event_min = mix.number("event_min_s", gt=0)
    if event_min > duration:
        raise ConfigError(f"{source}: 'mixtures.event_min_s' must be <= duration_s")
    mixtures = MixtureConfig(
        duration_s=duration,
        counts={s: counts.integer(s, ge=0) for s in SPLITS},
        seed={s: seeds.integer(s, ge=0) for s in SPLITS},
        n_foreground=mix.pair("n_foreground", integer=True, ge=1),
        n_background=mix.pair("n_background", integer=True, ge=0),
        target_size=mix.pair("target_size", integer=True, ge=1),
        empty_target_prob=mix.number("empty_target_prob", ge=0, le=1),
        event_min_s=event_min,
        fg_gain_db=mix.pair("fg_gain_db", ge=-60, le=20),
        bg_gain_db=mix.pair("bg_gain_db", ge=-60, le=20),
        target_to_rest_db=mix.pair("target_to_rest_db", ge=-60, le=60),
        peak_limit=mix.number("peak_limit", gt=0, le=1),
        reference_rms_dbfs=mix.number("reference_rms_dbfs", ge=-80, le=0),
    )
    return DataConfig(
        enabled=enabled,
        min_fg_clips_warning=roles.integer("min_fg_clips_warning", ge=0),
        activity_hop_s=activity.number("hop_s", gt=0, le=1),
        activity_threshold_db=activity.number("threshold_db_below_peak", gt=0, le=120),
        esc50_folds=folds,
        custom_val_fraction=val_fraction,
        custom_test_fraction=test_fraction,
        custom_seed=custom.integer("seed", ge=0),
        fsd50k_min_val_clips_per_class=splits.sub(
            "fsd50k", ("min_val_clips_per_class",)).integer("min_val_clips_per_class", ge=0),
        mixtures=mixtures,
    )


def load_detection_config(path: Path, project_root: Path) -> DetectionConfig:
    """Load and validate ``configs/detection/default.yaml``."""
    path = Path(path)
    source = f"detection/{path.name}"
    top = _Section({"root": load_yaml(path)}, "root", (
        "backbone", "model_sample_rate", "pretrained", "audioset_labels", "head_init",
        "window_s", "input_context_s", "min_active_s", "activity_rate_hz", "checkpoint", "live",
        "offline",
        "training", "augment", "evaluation"), source)
    top.name = ""
    d = top.data

    def sub(key: str, keys: tuple[str, ...]) -> _Section:
        return _Section(d, key, keys, source)

    def choice(sec: _Section, key: str, options: tuple[str, ...]) -> str:
        if sec.data[key] not in options:
            raise sec._fail(key, f"one of {list(options)}")
        return str(sec.data[key])

    def text(sec: _Section, key: str) -> str:
        if not isinstance(sec.data[key], str) or not sec.data[key]:
            raise sec._fail(key, "a non-empty string")
        return str(sec.data[key])

    pre = sub("pretrained", ("file", "url", "sha256"))
    sha = text(pre, "sha256").lower()
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise pre._fail("sha256", "a 64-character hex SHA-256")
    live = sub("live", ("hop_s", "ema_alpha", "hysteresis", "device", "cpu_threads"))
    off = sub("offline", ("max_seconds",))
    tr = sub("training", ("name", "epochs", "batch_size", "num_workers", "loader_timeout_s",
                          "lr_backbone",
                          "lr_head", "weight_decay", "warmup_epochs", "pos_weight", "amp",
                          "seed", "smoke_epochs", "smoke_train_recipes", "smoke_val_recipes"))
    aug = sub("augment", ("gain_db", "noise_snr_db", "freqm", "timem", "mel_range"))
    ev = sub("evaluation", ("windows_per_recipe", "seed"))
    window = top.number("window_s", gt=0, le=10)
    checkpoint = d["checkpoint"]
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint):
        raise top._fail("checkpoint", "a path or null")
    return DetectionConfig(
        backbone=choice(top, "backbone", BACKBONES),
        model_sample_rate=top.integer("model_sample_rate", ge=8000),
        pretrained_file=text(pre, "file"),
        pretrained_url=text(pre, "url"),
        pretrained_sha256=sha,
        audioset_labels=path.parent / text(top, "audioset_labels"),
        head_init=choice(top, "head_init", ("audioset", "random")),
        window_s=window,
        input_context_s=top.number("input_context_s", ge=window, le=60),
        min_active_s=top.number("min_active_s", ge=0, le=window),
        activity_rate_hz=top.number("activity_rate_hz", gt=0, le=1000),
        checkpoint=resolve_path(checkpoint, project_root) if checkpoint else None,
        live=LiveDetectionConfig(
            hop_s=live.number("hop_s", gt=0, le=window),
            ema_alpha=live.number("ema_alpha", gt=0, le=1),
            hysteresis=live.number("hysteresis", ge=0, le=1),
            device=choice(live, "device", ("auto", "cuda", "cpu")),
            cpu_threads=live.integer("cpu_threads", ge=1),
        ),
        offline_max_seconds=off.number("max_seconds", gt=0),
        training=TrainingConfig(
            name=text(tr, "name"),
            epochs=tr.integer("epochs", ge=1),
            batch_size=tr.integer("batch_size", ge=1),
            num_workers=tr.integer("num_workers", ge=0),
            loader_timeout_s=tr.number("loader_timeout_s", gt=0),
            lr_backbone=tr.number("lr_backbone", gt=0),
            lr_head=tr.number("lr_head", gt=0),
            weight_decay=tr.number("weight_decay", ge=0),
            warmup_epochs=tr.number("warmup_epochs", ge=0),
            pos_weight=tr.boolean("pos_weight"),
            amp=tr.boolean("amp"),
            seed=tr.integer("seed", ge=0),
            smoke_epochs=tr.integer("smoke_epochs", ge=1),
            smoke_train_recipes=tr.integer("smoke_train_recipes", ge=1),
            smoke_val_recipes=tr.integer("smoke_val_recipes", ge=1),
        ),
        augment=AugmentConfig(
            gain_db=aug.number("gain_db", ge=0, le=40),
            noise_snr_db=(None if aug.data["noise_snr_db"] is None
                          else aug.pair("noise_snr_db", ge=-20, le=120)),
            freqm=aug.integer("freqm", ge=0),
            timem=aug.integer("timem", ge=0),
            mel_range=aug.boolean("mel_range"),
        ),
        eval_windows_per_recipe=ev.integer("windows_per_recipe", ge=1),
        eval_seed=ev.integer("seed", ge=0),
    )


CACHE_ENV_VARS: dict[str, str] = {
    "TORCH_HOME": "torch",
    "HF_HOME": "huggingface",
    "XDG_CACHE_HOME": "cache",
}


def configure_cache_env(config: AppConfig) -> dict[str, str]:
    """Point library caches (PyTorch, Hugging Face, generic XDG) at ``paths.model_cache``.

    Only variables the user has not set already are changed; returns the ones that were set.
    Large pretrained weights then land on D: instead of in the user profile on C:.
    """
    cache = config.paths.get("model_cache")
    if cache is None:
        return {}
    applied = {}
    for var, sub in CACHE_ENV_VARS.items():
        if not os.environ.get(var):
            os.environ[var] = str(cache / sub)
            applied[var] = os.environ[var]
    return applied


def load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def resolve_path(path: str | Path, project_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (project_root / p).resolve()


def load_device_profile(name: str, configs_dir: Path = DEFAULT_CONFIGS_DIR) -> DeviceProfile:
    path = Path(configs_dir) / "devices" / f"{name}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in (Path(configs_dir) / "devices").glob("*.yaml"))
        raise ConfigError(f"device profile {name!r} not found at {path}; available: {available}")
    return parse_device_profile(name, load_yaml(path), source=str(path))


def load_label_set(path: Path) -> LabelSet:
    """Load ``classes.yaml``: ``{name: {display_name, description, audioset}}``."""
    try:
        return LabelSet.from_mapping(load_yaml(path))
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_config(
    configs_dir: Path = DEFAULT_CONFIGS_DIR,
    project_root: Path = DEFAULT_PROJECT_ROOT,
) -> AppConfig:
    configs_dir = Path(configs_dir)
    project_root = Path(project_root).resolve()
    default = load_yaml(configs_dir / "default.yaml")

    paths = {key: resolve_path(value, project_root) for key, value in default["paths"].items()}
    api = default.get("api", {})
    extraction_dir = configs_dir / "extraction"

    return AppConfig(
        project_root=project_root,
        configs_dir=configs_dir,
        paths=paths,
        device_profile=load_device_profile(default["active_device_profile"], configs_dir),
        labels=load_label_set(configs_dir / "classes.yaml"),
        capture=_capture(default),
        engine=_engine(default),
        output=_output(default),
        analysis=_analysis(default),
        api=ApiConfig(
            host=str(api.get("host", "127.0.0.1")),
            port=int(api.get("port", 8765)),
            cors_origins=tuple(api.get("cors_origins", ["http://localhost:5173"])),
            ws_heartbeat_s=float(api.get("ws_heartbeat_s", 2.0)),
            ws_client_queue=int(api.get("ws_client_queue", 64)),
        ),
        database_url=str(default["database"]["url"]),
        data=load_data_config(configs_dir / "data" / "default.yaml"),
        detection=load_detection_config(configs_dir / "detection" / "default.yaml",
                                        project_root),
        extraction={p.stem: load_yaml(p) for p in sorted(extraction_dir.glob("*.yaml"))},
    )
