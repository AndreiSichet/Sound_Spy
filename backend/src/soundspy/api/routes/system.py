"""System routes: health and app configuration."""

from __future__ import annotations

from fastapi import APIRouter

from soundspy import __version__
from soundspy.api.deps import ConfigDep
from soundspy.api.schemas import ClassInfo, ConfigResponse, HealthResponse, ModeInfo
from soundspy.engine.modes import Mode, requires_array

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(version=__version__)


@router.get("/config", response_model=ConfigResponse)
def app_config(config: ConfigDep) -> ConfigResponse:
    classes = [
        ClassInfo(name=c.name, display_name=c.display_name, description=c.description)
        for c in config.labels.classes
    ]
    profile = config.device_profile
    modes = []
    for mode in Mode:
        available = not requires_array(mode) or profile.has_geometry
        reason = None if available else "requires a microphone array (device profile geometry)"
        modes.append(ModeInfo(mode=mode, available=available, reason=reason))
    return ConfigResponse(classes=classes, modes=modes, device_profile=profile.name)
