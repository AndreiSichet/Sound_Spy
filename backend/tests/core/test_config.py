"""Tests for config loading and path resolution."""

from __future__ import annotations

from soundspy.config import load_device_profile


def test_paths_resolve_to_project_root(config, project_root) -> None:
    assert config.project_root == project_root.resolve()
    assert config.paths["raw"] == project_root.resolve() / "data" / "raw"
    assert config.paths["experiments"] == project_root.resolve() / "experiments"


def test_active_profile_is_laptop(config) -> None:
    profile = config.device_profile
    assert profile.name == "laptop"
    assert profile.device is None and profile.raw_channels is None
    assert not profile.has_geometry


def test_respeaker_profile(configs_dir) -> None:
    profile = load_device_profile("respeaker_xvf3800", configs_dir)
    assert profile.device == "reSpeaker"
    assert profile.raw_channels == (2, 3, 4, 5)


def test_module_configs_loaded(config) -> None:
    assert config.detection["sample_rate"] == 16000
    assert set(config.extraction) == {"mask_baseline", "neural_tse"}
    assert len(config.labels) == 7
    assert config.api.cors_origins == ("http://localhost:5173",)
    assert (config.api.host, config.api.port) == ("127.0.0.1", 8765)
