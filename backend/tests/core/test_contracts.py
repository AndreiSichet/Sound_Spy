"""Tests for core contracts: AudioBuffer, Direction, TargetSpec, events, LabelSet."""

from __future__ import annotations

import numpy as np
import pytest

from soundspy.config import load_label_set
from soundspy.core import (
    AudioBuffer,
    ClassDef,
    Detection,
    Direction,
    EngineStatus,
    LabelSet,
    TargetSpec,
)

# --- AudioBuffer ------------------------------------------------------------------------------


def test_audio_buffer_properties() -> None:
    buf = AudioBuffer(np.zeros((48000, 4), dtype=np.float32), 48000, {"device": "x"})
    assert buf.num_frames == 48000
    assert buf.num_channels == 4
    assert buf.duration_s == pytest.approx(1.0)
    assert buf.metadata == {"device": "x"}


def test_audio_buffer_mono_is_2d() -> None:
    buf = AudioBuffer(np.zeros((10, 1), dtype=np.float32), 16000)
    assert buf.num_channels == 1


@pytest.mark.parametrize(
    ("samples", "sample_rate", "error"),
    [
        (np.zeros(10, dtype=np.float32), 16000, ValueError),  # 1-D
        (np.zeros((10, 1, 1), dtype=np.float32), 16000, ValueError),  # 3-D
        (np.zeros((10, 0), dtype=np.float32), 16000, ValueError),  # no channels
        (np.zeros((10, 1), dtype=np.float64), 16000, TypeError),  # wrong dtype
        (np.zeros((10, 1), dtype=np.int16), 16000, TypeError),
        ([[0.0]], 16000, TypeError),  # not an ndarray
        (np.zeros((10, 1), dtype=np.float32), 0, ValueError),
        (np.zeros((10, 1), dtype=np.float32), -8000, ValueError),
        (np.zeros((10, 1), dtype=np.float32), 16000.0, TypeError),
        (np.zeros((10, 1), dtype=np.float32), True, TypeError),
    ],
)
def test_audio_buffer_validation(samples, sample_rate, error) -> None:
    with pytest.raises(error):
        AudioBuffer(samples, sample_rate)


def test_audio_buffer_is_frozen() -> None:
    buf = AudioBuffer(np.zeros((10, 1), dtype=np.float32), 16000)
    with pytest.raises(AttributeError):
        buf.sample_rate = 8000  # type: ignore[misc]


def test_audio_buffer_has_no_channel_reduction() -> None:
    assert not any(hasattr(AudioBuffer, n) for n in ("mono", "downmix", "to_mono", "select"))


# --- Direction / TargetSpec ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("azimuth", "expected"), [(0, 0.0), (90, 90.0), (360, 0.0), (-90, 270.0), (725, 5.0)]
)
def test_direction_normalizes_azimuth(azimuth: float, expected: float) -> None:
    assert Direction(azimuth, 30).azimuth_deg == pytest.approx(expected)


@pytest.mark.parametrize("width", [0, -10, 360.1, float("nan")])
def test_direction_rejects_bad_width(width: float) -> None:
    with pytest.raises(ValueError):
        Direction(0, width)


def test_direction_rejects_non_finite_azimuth() -> None:
    with pytest.raises(ValueError):
        Direction(float("inf"))


def test_target_spec_defaults() -> None:
    target = TargetSpec()
    assert target.labels == frozenset()
    assert target.passes_everything
    assert target.direction is None and target.source_id is None
    assert target.attenuation_db == 30.0


def test_target_spec_coerces_labels() -> None:
    assert TargetSpec(labels=["speech", "music", "speech"]).labels == {"speech", "music"}
    with pytest.raises(TypeError):
        TargetSpec(labels="speech")  # type: ignore[arg-type]


def test_target_spec_rejects_negative_attenuation() -> None:
    with pytest.raises(ValueError):
        TargetSpec(attenuation_db=-1.0)


def test_target_spec_validate(labels: LabelSet) -> None:
    TargetSpec(labels={"speech", "music"}).validate(labels)
    TargetSpec().validate(labels)
    with pytest.raises(ValueError, match="laughter"):
        TargetSpec(labels={"speech", "laughter"}).validate(labels)


# --- events -------------------------------------------------------------------------------------


def test_events_carry_time() -> None:
    det = Detection(t=1.5, label="speech", probability=0.9, t_start=1.0, t_end=1.5)
    status = EngineStatus(t=0.0, state="idle")
    assert det.t == 1.5 and status.t == 0.0 and status.message == ""
    with pytest.raises(ValueError):
        Detection(t=0, label="speech", probability=1.2, t_start=0, t_end=1)


# --- LabelSet ----------------------------------------------------------------------------------


@pytest.fixture
def labels(configs_dir) -> LabelSet:
    return load_label_set(configs_dir / "classes.yaml")


def test_label_set_from_classes_yaml(labels: LabelSet) -> None:
    assert list(labels) == [
        "speech", "music", "footsteps", "dishes", "chair_scrape", "alarm", "traffic"
    ]
    assert labels.audioset_names("dishes") == ("Dishes, pots, and pans", "Cutlery, silverware")
    assert labels.display_name("dishes") == "Dishes & cutlery"
    assert labels.description("dishes") == "Plates, cups, pots and cutlery clinking"
    for cls in labels.classes:
        assert cls.display_name.strip() and cls.description.strip() and cls.audioset


def test_label_set_from_mapping_validation() -> None:
    ok = LabelSet.from_mapping({"x": {"display_name": "X", "audioset": "Speech"}})
    assert ok.info("x") == ClassDef("x", "X", "", ("Speech",))
    with pytest.raises(ValueError, match="mapping"):
        LabelSet.from_mapping({"x": ["Speech"]})  # old list format
    with pytest.raises(ValueError, match="display_name"):
        LabelSet.from_mapping({"x": {"audioset": ["Speech"]}})
    with pytest.raises(ValueError, match="unknown fields"):
        LabelSet.from_mapping({"x": {"display_name": "X", "label": "oops"}})
    with pytest.raises(ValueError, match="display_name"):
        LabelSet.from_mapping({"x": {"display_name": "  "}})


def test_label_set_index_round_trip(labels: LabelSet) -> None:
    for i, label in enumerate(labels):
        assert labels.index(label) == i
        assert labels.label(i) == label
    with pytest.raises(ValueError):
        labels.index("laughter")
    with pytest.raises(IndexError):
        labels.label(len(labels))


def test_label_set_multi_hot_round_trip(labels: LabelSet) -> None:
    chosen = {"speech", "dishes"}
    vec = labels.to_multi_hot(chosen)
    assert vec.dtype == np.float32 and vec.shape == (7,)
    assert vec.sum() == 2 and vec[labels.index("dishes")] == 1.0
    assert labels.from_multi_hot(vec) == chosen
    assert labels.from_multi_hot(labels.to_multi_hot([])) == frozenset()


def test_label_set_from_multi_hot_threshold(labels: LabelSet) -> None:
    probs = np.array([0.9, 0.4, 0.0, 0.6, 0.0, 0.0, 0.5], dtype=np.float32)
    assert labels.from_multi_hot(probs) == {"speech", "dishes", "traffic"}
    assert labels.from_multi_hot(probs, threshold=0.7) == {"speech"}
    with pytest.raises(ValueError):
        labels.from_multi_hot(np.zeros(3))


def test_label_set_rejects_unknown_and_duplicates(labels: LabelSet) -> None:
    with pytest.raises(ValueError):
        labels.to_multi_hot(["speech", "laughter"])
    with pytest.raises(ValueError):
        LabelSet(["a", "a"])
    with pytest.raises(ValueError):
        LabelSet([])
