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


@dataclass(frozen=True)
class DeviceProfile:
    """Microphone description from ``configs/devices/<name>.yaml``."""

    name: str
    device: str | int | None = None
    raw_channels: tuple[int, ...] | None = None
    geometry: tuple[tuple[float, ...], ...] | None = None

    @property
    def has_geometry(self) -> bool:
        return bool(self.geometry)


@dataclass(frozen=True)
class ApiConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)
    ws_heartbeat_s: float = 2.0


@dataclass(frozen=True)
class EngineConfig:
    chunk_size: int = 1024


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    configs_dir: Path
    paths: dict[str, Path]
    device_profile: DeviceProfile
    labels: LabelSet
    engine: EngineConfig
    api: ApiConfig
    database_url: str
    detection: dict[str, Any] = field(default_factory=dict)
    extraction: dict[str, dict[str, Any]] = field(default_factory=dict)


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
        raise FileNotFoundError(f"device profile {name!r} not found at {path}")
    data = load_yaml(path)
    raw = data.get("raw_channels")
    geometry = data.get("geometry")
    return DeviceProfile(
        name=name,
        device=data.get("device"),
        raw_channels=tuple(int(c) for c in raw) if raw is not None else None,
        geometry=tuple(tuple(float(x) for x in mic) for mic in geometry) if geometry else None,
    )


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
    engine = default.get("engine", {})
    extraction_dir = configs_dir / "extraction"

    return AppConfig(
        project_root=project_root,
        configs_dir=configs_dir,
        paths=paths,
        device_profile=load_device_profile(default["active_device_profile"], configs_dir),
        labels=load_label_set(configs_dir / "classes.yaml"),
        engine=EngineConfig(chunk_size=int(engine.get("chunk_size", 1024))),
        api=ApiConfig(
            host=str(api.get("host", "127.0.0.1")),
            port=int(api.get("port", 8765)),
            cors_origins=tuple(api.get("cors_origins", ["http://localhost:5173"])),
            ws_heartbeat_s=float(api.get("ws_heartbeat_s", 2.0)),
        ),
        database_url=str(default["database"]["url"]),
        detection=load_yaml(configs_dir / "detection" / "default.yaml"),
        extraction={p.stem: load_yaml(p) for p in sorted(extraction_dir.glob("*.yaml"))},
    )
