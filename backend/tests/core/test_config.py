"""Tests for config loading and path resolution."""

from __future__ import annotations

import pytest

from soundspy.config import ConfigError, load_device_profile, parse_device_profile


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
    assert profile.hostapi is None and profile.sample_rate is None


def test_missing_profile_lists_available(configs_dir) -> None:
    with pytest.raises(ConfigError, match="available: \\['laptop', 'respeaker_xvf3800'\\]"):
        load_device_profile("zoom", configs_dir)


def test_full_profile_parses() -> None:
    profile = parse_device_profile(
        "array",
        {
            "device": 3,
            "hostapi": "Windows WASAPI",
            "sample_rate": 16000,
            "raw_channels": [2, 3],
            "geometry": [[0.0, 0.03, 0], [0.0, -0.03, 0]],
        },
    )
    assert profile.device == 3 and profile.sample_rate == 16000
    assert profile.geometry == ((0.0, 0.03, 0.0), (0.0, -0.03, 0.0)) and profile.has_geometry


@pytest.mark.parametrize(
    ("data", "fragment"),
    [
        ({"devise": None}, "unknown field"),
        ({"device": True}, "'device' must be"),
        ({"device": -1}, "index must be >= 0"),
        ({"device": "  "}, "must not be empty"),
        ({"hostapi": 5}, "'hostapi' must be"),
        ({"sample_rate": 0}, "'sample_rate' must be"),
        ({"sample_rate": 44100.0}, "'sample_rate' must be"),
        ({"raw_channels": []}, "'raw_channels' must be"),
        ({"raw_channels": [0, -1]}, "'raw_channels' must be"),
        ({"raw_channels": "0,1"}, "'raw_channels' must be"),
        ({"raw_channels": [1, 1]}, "duplicates"),
        ({"geometry": [[0, 0]]}, "\\[x, y, z\\]"),
        ({"raw_channels": [0, 1], "geometry": [[0, 0, 0]]}, "has 1 positions"),
    ],
)
def test_invalid_profiles_are_clear_errors(data, fragment) -> None:
    with pytest.raises(ConfigError, match=fragment) as exc:
        parse_device_profile("bad", data, source="bad.yaml")
    assert "device profile 'bad' (bad.yaml)" in str(exc.value)


def test_session_sections_loaded(config) -> None:
    assert config.capture.stall_timeout_s == 2.0 and config.capture.queue_chunks == 32
    assert config.engine.chunk_ms == 20 and config.engine.chunk_frames(48000) == 960
    assert config.engine.chunk_frames(16000) == 320
    assert config.output.device is None and config.output.limiter_ceiling_dbfs == -1.0
    assert config.analysis.spectrogram_bands == 128 and config.analysis.spectrogram_fft == 2048


@pytest.mark.parametrize(
    ("section", "key", "value", "fragment"),
    [
        ("capture", "queue_chunks", 0, "capture.queue_chunks' must be an integer >= 1"),
        ("engine", "chunk_ms", 0, "engine.chunk_ms' must be > 0"),
        ("engine", "chunk_ms", "20", "must be a number"),
        ("output", "limiter_ceiling_dbfs", 3, "must be <= 0"),
        ("output", "max_queue_chunks", 1, "max_queue_chunks' must be >= prebuffer"),
        ("output", "device", True, "output.device"),
        ("analysis", "spectrogram_fft", 2000, "power of two"),
        ("analysis", "levels_hz", -5, "levels_hz' must be > 0"),
    ],
)
def test_invalid_session_settings(tmp_path, configs_dir, section, key, value, fragment) -> None:
    import shutil

    import yaml

    from soundspy.config import load_config

    copy = tmp_path / "configs"
    shutil.copytree(configs_dir, copy)
    data = yaml.safe_load((copy / "default.yaml").read_text(encoding="utf-8"))
    data[section][key] = value
    (copy / "default.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match=fragment):
        load_config(copy)


def test_missing_and_unknown_keys(tmp_path, configs_dir) -> None:
    import shutil

    import yaml

    from soundspy.config import load_config

    copy = tmp_path / "configs"
    shutil.copytree(configs_dir, copy)
    data = yaml.safe_load((copy / "default.yaml").read_text(encoding="utf-8"))
    del data["engine"]["chunk_ms"]
    data["analysis"]["colour"] = "blue"
    (copy / "default.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="'engine' is missing \\['chunk_ms'\\]"):
        load_config(copy)
    del data["engine"]
    (copy / "default.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="missing section 'engine'"):
        load_config(copy)


def test_module_configs_loaded(config) -> None:
    assert config.detection["sample_rate"] == 16000
    assert set(config.extraction) == {"mask_baseline", "neural_tse"}
    assert len(config.labels) == 7
    assert config.api.cors_origins == ("http://localhost:5173",)
    assert (config.api.host, config.api.port) == ("127.0.0.1", 8765)
