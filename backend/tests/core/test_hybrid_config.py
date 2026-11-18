"""``configs/extraction/hybrid.yaml`` (step 07c): loads, validates, and v1 is configured."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from soundspy.config import ConfigError, load_hybrid_config, speech_model_dir


def test_hybrid_config_loads(config) -> None:
    hy = config.extraction.hybrid
    assert hy.speech.labels == ("speech",) and hy.speech.route in ("enhancer", "cascade")
    assert hy.speech.backend == "onnx" and hy.speech.device == "cpu"
    assert hy.speech.atten_lim_db is None  # the session's slider is the only attenuation
    assert hy.gate.enabled and 0 < hy.gate.close_prob_factor <= 1
    assert set(hy.tuning) == {"close_prob_factor", "close_hold_ms", "reopen_ratio_db"}
    assert config.extraction.method == "mask_baseline"  # the test fixture's choice


def test_shipped_defaults(configs_dir) -> None:
    from soundspy.config import load_config

    real = load_config()
    assert real.extraction.method == "hybrid"
    assert real.extraction.neural_tse.checkpoint.parent.parent.name == "tse_waveformer_v1"
    assert real.extraction.neural_tse.gate.enabled is False  # the 07b gate: off by default
    assert speech_model_dir(real).name == "DeepFilterNet3"


def test_bad_values_are_refused(configs_dir, config, tmp_path: Path) -> None:
    source = configs_dir / "extraction" / "hybrid.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["speech"]["labels"] = ["singing"]
    bad = tmp_path / "hybrid.yaml"
    bad.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="speech.labels"):
        load_hybrid_config(bad, config.labels)
    shutil.copy(source, bad)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["gate"]["close_prob_factor"] = 1.5
    bad.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="close_prob_factor"):
        load_hybrid_config(bad, config.labels)
