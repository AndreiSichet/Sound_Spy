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
    assert config.detection.model_sample_rate == 32000
    assert config.detection.backbone == "efficientat_mn10_as"
    assert config.detection.window_s == 1.0 and config.detection.input_context_s == 10.0
    ext = config.extraction
    assert ext.method == "mask_baseline" and ext.offline_max_seconds == 300
    assert (ext.mask_baseline.n_fft, ext.mask_baseline.hop) == (1024, 256)
    assert ext.evaluation.n == 500 and ext.evaluation.sample_rate == 48000
    assert "checkpoint" in ext.neural_tse  # placeholder until step 07
    assert len(config.labels) == 7
    assert config.api.cors_origins == ("http://localhost:5173",)
    assert (config.api.host, config.api.port) == ("127.0.0.1", 8765)


def test_data_paths_and_data_config_loaded(config) -> None:
    for key in ("external", "processed", "mixtures", "model_cache", "custom"):
        assert config.paths[key].is_absolute()
    data = config.data
    assert data.enabled == {"esc50": True, "fsd50k": True, "custom": True}
    assert data.esc50_folds == {"train": (1, 2, 3), "val": (4,), "test": (5,)}
    assert data.mixtures.counts == {"train": 20000, "val": 2000, "test": 2000}
    assert data.mixtures.n_foreground == (1, 3)
    assert data.mixtures.fg_gain_db == (-5.0, 5.0)


@pytest.mark.parametrize(
    ("path", "value", "fragment"),
    [
        (("mixtures", "n_foreground"), [3, 1], r"'mixtures.n_foreground' must be \[low, high\]"),
        (("mixtures", "n_foreground"), [0, 2], "within"),
        (("mixtures", "peak_limit"), 1.5, "'mixtures.peak_limit' must be <= 1"),
        (("mixtures", "event_min_s"), 10.0, "event_min_s' must be <= duration_s"),
        (("mixtures", "counts", "val"), -1, "'mixtures.counts.val' must be an integer >= 0"),
        (("splits", "esc50", "val"), [3], "fold in more than one split"),
        (("splits", "custom", "val_fraction"), 0.9, "fractions must be < 1"),
        (("datasets", "esc50", "enabled"), "yes", "'datasets.esc50.enabled' must be true"),
        (("activity", "hop_s"), 0, "'activity.hop_s' must be > 0"),
    ],
)
def test_invalid_data_config(tmp_path, configs_dir, path, value, fragment) -> None:
    import yaml

    from soundspy.config import load_data_config

    data = yaml.safe_load((configs_dir / "data" / "default.yaml").read_text(encoding="utf-8"))
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    target = tmp_path / "default.yaml"
    target.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match=fragment) as exc:
        load_data_config(target)
    assert str(exc.value).startswith("data/default.yaml")


def test_unknown_data_key_is_rejected(tmp_path, configs_dir) -> None:
    import yaml

    from soundspy.config import load_data_config

    data = yaml.safe_load((configs_dir / "data" / "default.yaml").read_text(encoding="utf-8"))
    data["mixtures"]["reverb"] = True
    target = tmp_path / "default.yaml"
    target.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"'mixtures' has unknown key\(s\) \['reverb'\]"):
        load_data_config(target)


def test_cache_env_points_to_model_cache_only_if_unset(config, tmp_path, monkeypatch) -> None:
    import dataclasses

    from soundspy.config import configure_cache_env

    monkeypatch.setenv("TORCH_HOME", "E:/mine")
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    cfg = dataclasses.replace(config, paths={**config.paths, "model_cache": tmp_path / "m"})
    applied = configure_cache_env(cfg)
    import os

    assert os.environ["TORCH_HOME"] == "E:/mine"  # the user's choice wins
    assert applied == {"HF_HOME": str(tmp_path / "m" / "huggingface"),
                       "XDG_CACHE_HOME": str(tmp_path / "m" / "cache")}
    assert os.environ["HF_HOME"] == str(tmp_path / "m" / "huggingface")


@pytest.mark.parametrize(
    ("file", "path", "value", "fragment"),
    [
        ("mask_baseline.yaml", ["hop"], 300, "'hop' must be a divisor of n_fft"),
        ("mask_baseline.yaml", ["hop"], 1024, "n_fft / hop >= 2"),
        ("mask_baseline.yaml", ["n_fft"], 1023, "'n_fft' must be an even number"),
        ("mask_baseline.yaml", ["dd_alpha"], 1.5, "'dd_alpha' must be <= 0.9999"),
        ("mask_baseline.yaml", ["minstat_bias"], 0.5, "'minstat_bias' must be >= 1"),
        ("default.yaml", ["method"], "neural_tse", "'method' must be one of"),
        ("default.yaml", ["offline", "max_seconds"], 0, "'offline.max_seconds' must be > 0"),
        ("default.yaml", ["evaluation", "n"], 0, "'evaluation.n' must be an integer >= 1"),
    ],
)
def test_invalid_extraction_config(tmp_path, configs_dir, file, path, value, fragment) -> None:
    import shutil

    import yaml

    from soundspy.config import load_config

    copy = tmp_path / "configs"
    shutil.copytree(configs_dir, copy)
    target = copy / "extraction" / file
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    target.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match=fragment):
        load_config(copy)


def test_unknown_extraction_key_is_rejected(tmp_path, configs_dir) -> None:
    import shutil

    from soundspy.config import load_config

    copy = tmp_path / "configs"
    shutil.copytree(configs_dir, copy)
    with (copy / "extraction" / "mask_baseline.yaml").open("a", encoding="utf-8") as fh:
        fh.write("mask_type: wiener\n")
    with pytest.raises(ConfigError, match=r"unknown key\(s\) \['mask_type'\]"):
        load_config(copy)
