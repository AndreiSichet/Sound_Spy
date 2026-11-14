"""Configuration loading.

Reads ``backend/configs/default.yaml``, the device profiles in ``backend/configs/devices/``,
``classes.yaml`` (-> ``LabelSet``) and the detection / extraction configs. Relative paths in the
configs are resolved against the **project root** (the folder that contains ``backend/`` and
``frontend/``), so ``data/``, ``models/``, ``outputs/`` and ``experiments/`` live at the root.
"""

from __future__ import annotations

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
    detection: dict[str, Any] = field(default_factory=dict)
    extraction: dict[str, dict[str, Any]] = field(default_factory=dict)


class _Section:
    """Reads one mapping from default.yaml with type and range checks (clear ConfigError)."""

    def __init__(self, data: dict[str, Any], name: str, keys: tuple[str, ...]) -> None:
        section = data.get(name)
        if not isinstance(section, dict):
            raise ConfigError(f"default.yaml: missing section '{name}'")
        missing = [k for k in keys if k not in section]
        unknown = sorted(set(section) - set(keys))
        if missing:
            raise ConfigError(f"default.yaml: '{name}' is missing {missing}")
        if unknown:
            raise ConfigError(f"default.yaml: '{name}' has unknown key(s) {unknown}")
        self.data, self.name = section, name

    def _fail(self, key: str, rule: str) -> ConfigError:
        value = self.data[key]
        return ConfigError(f"default.yaml: '{self.name}.{key}' must be {rule}, got {value!r}")

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
        detection=load_yaml(configs_dir / "detection" / "default.yaml"),
        extraction={p.stem: load_yaml(p) for p in sorted(extraction_dir.glob("*.yaml"))},
    )
