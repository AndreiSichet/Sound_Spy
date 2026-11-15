"""Tiny synthetic datasets for the step-04 data tests (no downloads, no drive D:).

``data_env`` builds, in a temp folder:

* a mini AudioSet ontology where "Vehicle horn" has two parents (Alarm and Car);
* a fake ESC-50 tree (meta/esc50.csv + WAVs, 5 folds);
* a fake FSD50K tree (dev.csv / eval.csv with smeared ancestor labels, vocabulary.csv,
  clips_info JSON with uploaders and licenses, WAVs);
* ``.done`` markers, an ESC-50 mapping YAML, and an ``AppConfig`` whose ``paths`` point there.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from soundspy.config import AppConfig
from soundspy.core.labels import ClassDef, LabelSet

SR = 16_000

ONTOLOGY = [
    {"id": "/m/things", "name": "Sounds of things", "child_ids": ["/m/alarm", "/m/veh"]},
    {"id": "/m/hv", "name": "Human voice", "child_ids": ["/m/sp", "/m/shout"]},
    {"id": "/m/sp", "name": "Speech", "child_ids": ["/m/conv"]},
    {"id": "/m/conv", "name": "Conversation", "child_ids": []},
    {"id": "/m/shout", "name": "Shout", "child_ids": []},
    {"id": "/m/mus", "name": "Music", "child_ids": ["/m/gui"]},
    {"id": "/m/gui", "name": "Guitar", "child_ids": []},
    {"id": "/m/alarm", "name": "Alarm", "child_ids": ["/m/horn"]},
    {"id": "/m/veh", "name": "Vehicle", "child_ids": ["/m/car"]},
    {"id": "/m/car", "name": "Car", "child_ids": ["/m/horn"]},
    {"id": "/m/horn", "name": "Vehicle horn, car horn, honking", "child_ids": []},
    {"id": "/m/walk", "name": "Walk, footsteps", "child_ids": []},
    {"id": "/m/dog", "name": "Dog", "child_ids": []},
    {"id": "/m/rain", "name": "Rain", "child_ids": []},
]

LABELS = LabelSet([
    ClassDef("speech", "Speech", "", ("Speech",)),
    ClassDef("music", "Music", "", ("Music",)),
    ClassDef("footsteps", "Footsteps", "", ("Walk, footsteps",)),
    ClassDef("alarm", "Alarms", "", ("Alarm",)),
    ClassDef("traffic", "Traffic", "", ("Vehicle",)),
])

VOCABULARY = [
    ("Speech", "/m/sp"), ("Human_voice", "/m/hv"), ("Conversation", "/m/conv"),
    ("Shout", "/m/shout"), ("Music", "/m/mus"), ("Guitar", "/m/gui"), ("Alarm", "/m/alarm"),
    ("Vehicle", "/m/veh"), ("Car", "/m/car"), ("Vehicle_horn_and_car_horn_and_honking", "/m/horn"),
    ("Walk_and_footsteps", "/m/walk"), ("Dog", "/m/dog"), ("Rain", "/m/rain"),
]
MID = dict(VOCABULARY)

# (fname, labels, part, dev split, uploader) -- labels are smeared like the real FSD50K
FSD_CLIPS = [
    ("101", ["Speech", "Human_voice"], "dev", "train", "anna"),
    ("102", ["Conversation", "Speech", "Human_voice"], "dev", "train", "anna"),
    ("103", ["Guitar", "Music"], "dev", "train", "bob"),
    ("104", ["Music"], "dev", "train", "bob"),
    ("105", ["Walk_and_footsteps"], "dev", "train", "carl"),
    ("106", ["Dog"], "dev", "train", "carl"),
    ("107", ["Rain"], "dev", "train", "dora"),
    ("108", ["Speech", "Human_voice", "Music"], "dev", "train", "dora"),  # mixed -> excluded
    ("109", ["Vehicle_horn_and_car_horn_and_honking", "Car", "Vehicle", "Alarm"],
     "dev", "train", "emil"),  # alarm AND traffic -> excluded (overlap)
    ("110", ["Car", "Vehicle"], "dev", "train", "emil"),
    ("111", ["Alarm"], "dev", "train", "fay"),
    ("112", ["Shout", "Human_voice"], "dev", "train", "fay"),  # bg with an ancestor label
    ("201", ["Speech", "Human_voice"], "dev", "val", "gus"),
    ("202", ["Music"], "dev", "val", "gus"),
    ("203", ["Dog"], "dev", "val", "hal"),
    ("204", ["Walk_and_footsteps"], "dev", "val", "hal"),
    ("205", ["Alarm"], "dev", "val", "ida"),
    ("206", ["Car", "Vehicle"], "dev", "val", "ida"),
    ("207", ["Rain"], "dev", "val", "anna"),  # proposed val, but anna is a train uploader
    ("301", ["Speech", "Human_voice"], "eval", None, "jo"),
    ("302", ["Guitar", "Music"], "eval", None, "jo"),
    ("303", ["Rain"], "eval", None, "kim"),
    ("304", ["Walk_and_footsteps"], "eval", None, "kim"),
    ("305", ["Alarm"], "eval", None, "lee"),
    ("306", ["Car", "Vehicle"], "eval", None, "lee"),
]

ESC_MAPPING = {"dog": ["/m/dog"], "footsteps": ["/m/walk"], "rain": ["/m/rain"],
               "car_horn": ["/m/horn"], "can_opening": "background", "engine": "excluded"}
# (filename, fold, category)
ESC_CLIPS = [
    (f"{fold}-{1000 + fold * 10 + i}-{take}-{t}.wav", fold, cat)
    for fold in range(1, 6)
    for i, (cat, t) in enumerate([("dog", 0), ("footsteps", 25), ("rain", 10),
                                  ("car_horn", 43), ("can_opening", 34), ("engine", 44)])
    for take in ("A", "B")
]


def burst(seconds: float, sr: int, *, start: float, stop: float, freq: float = 440.0,
          amp: float = 0.3, seed: int = 0) -> np.ndarray:
    """Silence with a tone burst in ``[start, stop)`` plus faint noise."""
    rng = np.random.default_rng(seed)
    n = round(seconds * sr)
    x = rng.normal(0.0, 1e-5, n)
    t = np.arange(n) / sr
    on = (t >= start) & (t < stop)
    x[on] += amp * np.sin(2 * np.pi * freq * t[on])
    return x.astype(np.float32)


def write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sr, subtype="PCM_16")


@dataclass
class DataEnv:
    root: Path
    config: AppConfig
    external: Path


def build_tree(root: Path, base: AppConfig) -> DataEnv:
    external = root / "external"
    # ontology
    onto = external / "ontology"
    onto.mkdir(parents=True)
    (onto / "ontology.json").write_text(json.dumps(ONTOLOGY), encoding="utf-8")
    (onto / ".done").write_text("{}", encoding="utf-8")

    # ESC-50
    esc = external / "esc50" / "ESC-50-master"
    (esc / "meta").mkdir(parents=True)
    with open(esc / "meta" / "esc50.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "fold", "target", "category", "esc10", "src_file", "take"])
        for i, (fname, fold, cat) in enumerate(ESC_CLIPS):
            parts = fname[:-4].split("-")
            w.writerow([fname, fold, parts[3], cat, "False", parts[1], parts[2]])
            write_wav(esc / "audio" / fname,
                      burst(2.0, SR, start=0.5, stop=1.5, freq=300 + 20 * i, seed=i), SR)
    (external / "esc50" / ".done").write_text("{}", encoding="utf-8")

    # FSD50K
    fsd = external / "fsd50k"
    truth, meta = fsd / "FSD50K.ground_truth", fsd / "FSD50K.metadata"
    truth.mkdir(parents=True)
    meta.mkdir(parents=True)
    with open(truth / "vocabulary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for i, (label, mid) in enumerate(VOCABULARY):
            w.writerow([i, label, mid])
    info: dict[str, dict] = {"dev": {}, "eval": {}}
    writers = {}
    files = {}
    for part, header in (("dev", ["fname", "labels", "mids", "split"]),
                         ("eval", ["fname", "labels", "mids"])):
        files[part] = open(truth / f"{part}.csv", "w", newline="", encoding="utf-8")  # noqa: SIM115
        writers[part] = csv.writer(files[part])
        writers[part].writerow(header)
    for i, (fname, labels, part, split, uploader) in enumerate(FSD_CLIPS):
        row = [fname, ",".join(labels), ",".join(MID[lab] for lab in labels)]
        writers[part].writerow(row + ([split] if part == "dev" else []))
        info[part][fname] = {"uploader": uploader,
                             "license": "http://creativecommons.org/licenses/by/3.0/"}
        seconds = 3.0 if i % 2 else 1.2
        write_wav(fsd / f"FSD50K.{part}_audio" / f"{fname}.wav",
                  burst(seconds, SR, start=0.2, stop=seconds - 0.2, freq=200 + 30 * i, seed=i),
                  SR)
    for fh in files.values():
        fh.close()
    for part in ("dev", "eval"):
        (meta / f"{part}_clips_info_FSD50K.json").write_text(json.dumps(info[part]),
                                                             encoding="utf-8")
    (fsd / ".done").write_text("{}", encoding="utf-8")

    # configs: ESC-50 mapping in a temp configs dir
    configs = root / "configs"
    (configs / "data").mkdir(parents=True)
    lines = [f"{k}: {json.dumps(v)}" for k, v in ESC_MAPPING.items()]
    (configs / "data" / "esc50_to_audioset.yaml").write_text("\n".join(lines) + "\n",
                                                             encoding="utf-8")

    paths = {**base.paths, "external": external, "processed": root / "processed",
             "mixtures": root / "mixtures", "custom": root / "custom",
             "outputs": root / "outputs", "raw": root / "raw"}
    mix = dataclasses.replace(base.data.mixtures, counts={"train": 50, "val": 20, "test": 20},
                              duration_s=4.0)
    # no validation rebalance by default: the fake tree is far below the real target of 40
    data = dataclasses.replace(base.data, mixtures=mix, fsd50k_min_val_clips_per_class=0)
    config = dataclasses.replace(base, paths=paths, labels=LABELS, configs_dir=configs, data=data)
    return DataEnv(root, config, external)


@pytest.fixture
def data_env(config, tmp_path: Path) -> DataEnv:
    return build_tree(tmp_path / "soundspy-data", config)


@pytest.fixture
def prepared_env(data_env: DataEnv) -> DataEnv:
    from soundspy.data.prepare import prepare

    prepare(data_env.config, workers=1, progress=False)
    return data_env


@pytest.fixture
def kit() -> SimpleNamespace:
    """Constants and helpers of this module (tests cannot import conftest directly)."""
    return SimpleNamespace(SR=SR, ONTOLOGY=ONTOLOGY, LABELS=LABELS, VOCABULARY=VOCABULARY,
                           burst=burst, write_wav=write_wav, build_tree=build_tree)


@pytest.fixture(scope="module")
def shared_prepared_env(tmp_path_factory: pytest.TempPathFactory) -> DataEnv:
    """One prepared tree per test module, for tests that only read it (faster)."""
    from soundspy.config import load_config
    from soundspy.data.prepare import prepare

    env = build_tree(tmp_path_factory.mktemp("shared") / "soundspy-data", load_config())
    prepare(env.config, workers=1, progress=False)
    return env
