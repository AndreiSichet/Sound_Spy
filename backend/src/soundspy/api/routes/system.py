"""System routes: health, app configuration, preloaded models (step 08)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from soundspy import __version__
from soundspy.api.deps import ConfigDep
from soundspy.api.schemas import (
    ClassInfo,
    ConfigResponse,
    HealthResponse,
    MeasuredLatencyResponse,
    MethodInfo,
    ModeInfo,
    ModelsResponse,
)
from soundspy.engine import resources
from soundspy.engine.modes import Mode, requires_array
from soundspy.extraction.offline import (
    default_gate,
    hybrid_available,
    neural_available,
    resolve_method,
)

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(version=__version__)


@router.get("/latency", response_model=MeasuredLatencyResponse)
def measured_latency(config: ConfigDep) -> MeasuredLatencyResponse:
    """The latest acoustic measurement (``experiments/realtime/latency_latest.json``), so the
    page can show a measured delay next to its own estimate."""
    import json
    from pathlib import Path

    path = Path(config.paths["experiments"]) / "realtime" / "latency_latest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return MeasuredLatencyResponse()
    modes = {k: v for k, v in (data.get("modes") or {}).items() if isinstance(v, dict)}
    return MeasuredLatencyResponse(time=data.get("time"), input=data.get("input"),
                                   output=data.get("output"), modes=modes,
                                   settings=data.get("settings"))


@router.get("/models", response_model=ModelsResponse)
def models(request: Request) -> ModelsResponse:
    """The preloader's progress (``off`` when ``api.preload_models`` is false) and the
    process's RAM / graphics-card memory now."""
    preloader = getattr(request.app.state, "preloader", None)
    now = resources.snapshot()
    if preloader is None:
        return ModelsResponse(state="off", resources=now)
    return ModelsResponse(**preloader.status(), resources=now)


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
    hybrid_reason = hybrid_available(config)
    methods = [
        MethodInfo(name="hybrid", display_name="Hybrid", available=hybrid_reason is None,
                   reason=hybrid_reason),
        MethodInfo(name="neural_tse", display_name="Neural", available=reason is None,
                   reason=reason),
        MethodInfo(name="mask_baseline", display_name="Baseline", available=True),
    ]
    default = resolve_method(config)[0]
    return ConfigResponse(
        classes=classes, modes=modes, device_profile=profile.name,
        extraction_methods=methods, default_method=default,
        default_gate=default_gate(config, default),
        default_gates={m.name: default_gate(config, m.name) for m in methods},
        default_attenuation_db=config.extraction.default_attenuation_db)
