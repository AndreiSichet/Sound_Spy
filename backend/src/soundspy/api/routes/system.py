"""System routes: health and app configuration."""

from __future__ import annotations

from fastapi import APIRouter

from soundspy import __version__
from soundspy.api.deps import ConfigDep
from soundspy.api.schemas import (
    ClassInfo,
    ConfigResponse,
    HealthResponse,
    MethodInfo,
    ModeInfo,
)
from soundspy.engine.modes import Mode, requires_array
from soundspy.extraction.offline import neural_available, resolve_method

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
    reason = neural_available(config)
    methods = [
        MethodInfo(name="neural_tse", display_name="Neural", available=reason is None,
                   reason=reason),
        MethodInfo(name="mask_baseline", display_name="Baseline", available=True),
    ]
    return ConfigResponse(
        classes=classes, modes=modes, device_profile=profile.name,
        extraction_methods=methods, default_method=resolve_method(config)[0],
        default_attenuation_db=config.extraction.default_attenuation_db)
