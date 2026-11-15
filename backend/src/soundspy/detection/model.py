"""Detection model: front-end + pretrained audio backbone + a new multi-label head.

``DetectorNet(waveform)`` -> one logit per ``LabelSet`` class (sigmoid -> probability):

    waveform (B, samples @ 32 kHz) -> zero-pad to ``context_s`` -> EfficientATMel
        -> backbone.features
        -> backbone.classifier[:5] (pool, Linear 960->1280, Hardswish, Dropout) -> head (1280 -> 7)

**Input context.** EfficientAT was trained on 10 s clips. Fed only 1 s, its last feature map is
4 x 4 cells, the zero padding of the convolutions dominates and the AudioSet head saturates
(zero-shot mAP at chance level for inputs under 5 s; measured in step 05). So the 1 s window is
zero-padded to ``context_s`` (10 s) before the front-end: "a 10 s clip in which only this second
sounds", which is what AudioSet clips look like to the model. Training and live use the same
padding, and the label still describes the 1 s window.

**Head initialisation** (``head_init: audioset``): the backbone already has an AudioSet head with
527 rows, one per AudioSet class. Each new class row starts as the **mean of the AudioSet rows of
the class's own AudioSet names** (``classes.yaml``, mapped through mids), bias included. So before
any training the head already means "speech-ish", "music-ish" … and fine-tuning starts from the
pretrained knowledge instead of from noise. A class with no matching AudioSet row starts random.

**Zero-shot baseline**: the untouched pretrained model, with class ``c``'s probability = the
maximum AudioSet probability over the AudioSet classes inside ``F(c)`` (the class's names and
everything below them in the ontology). ``DetectorNet.zero_shot`` computes it.

The row lists are computed once from the ontology and stored in every checkpoint, so live
inference never needs the ontology.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from soundspy.core.labels import LabelSet
from soundspy.detection.backbones import AUDIOSET_CLASSES, build_backbone
from soundspy.dsp.features import EfficientATMel

CHECKPOINT_VERSION = 1


def load_audioset_labels(path: Path) -> list[tuple[str, str]]:
    """``[(mid, name), ...]`` in output order (527 rows)."""
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    out = [(r[1], r[2]) for r in rows[1:] if len(r) >= 3]
    if len(out) != AUDIOSET_CLASSES:
        raise ValueError(f"{path}: expected {AUDIOSET_CLASSES} AudioSet labels, got {len(out)}")
    return out


@dataclass(frozen=True)
class AudioSetRows:
    """Per class: AudioSet output rows for head init (own names) and zero-shot (all of F(c))."""

    init: dict[str, list[int]]
    zero_shot: dict[str, list[int]]

    def to_json(self) -> dict[str, Any]:
        return {"init": self.init, "zero_shot": self.zero_shot}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> AudioSetRows:
        return cls({k: list(v) for k, v in d["init"].items()},
                   {k: list(v) for k, v in d["zero_shot"].items()})


def audioset_rows(labels: LabelSet, ontology, audioset_labels: list[tuple[str, str]]
                  ) -> AudioSetRows:
    from soundspy.data.ontology import class_mids
    from soundspy.data.roles import RoleRules

    row_of = {mid: i for i, (mid, _name) in enumerate(audioset_labels)}
    own = class_mids(labels, ontology)
    fg = RoleRules.build(labels, ontology).fg
    init = {c: sorted(row_of[m] for m in own[c] if m in row_of) for c in labels}
    zero = {c: sorted(row_of[m] for m in fg[c] if m in row_of) for c in labels}
    return AudioSetRows(init, zero)


@dataclass
class ModelSpec:
    """Everything needed to rebuild a ``DetectorNet`` (stored in checkpoints)."""

    backbone: str
    labels: list[str]
    sample_rate: int
    window_s: float
    rows: AudioSetRows
    augment: dict[str, Any] = field(default_factory=dict)
    context_s: float = 10.0

    def to_json(self) -> dict[str, Any]:
        return {"backbone": self.backbone, "labels": self.labels, "sample_rate": self.sample_rate,
                "window_s": self.window_s, "rows": self.rows.to_json(), "augment": self.augment,
                "context_s": self.context_s}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ModelSpec:
        return cls(d["backbone"], list(d["labels"]), int(d["sample_rate"]), float(d["window_s"]),
                   AudioSetRows.from_json(d["rows"]), dict(d.get("augment", {})),
                   float(d.get("context_s", 10.0)))


class DetectorNet(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        aug = spec.augment
        self.frontend = EfficientATMel(freqm=int(aug.get("freqm", 0)),
                                       timem=int(aug.get("timem", 0)),
                                       mel_range_aug=bool(aug.get("mel_range", False)))
        self.backbone = build_backbone(spec.backbone)
        self.head = nn.Linear(self.backbone.embedding_dim, len(spec.labels))
        self.context_samples = int(round(spec.context_s * spec.sample_rate))

    def pad(self, waveform: torch.Tensor) -> torch.Tensor:
        """Zero-pad (right) to the model's input context."""
        missing = self.context_samples - waveform.shape[-1]
        return nn.functional.pad(waveform, (0, missing)) if missing > 0 else waveform

    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(self.frontend(self.pad(waveform)))
        return self.backbone.classifier[:5](x)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(waveform))

    def audioset_logits(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.backbone.features(self.frontend(self.pad(waveform)))
        return self.backbone.classifier(x)

    def zero_shot(self, waveform: torch.Tensor) -> torch.Tensor:
        """Class probabilities of the **pretrained** AudioSet head (max over F(c) rows)."""
        probs = torch.sigmoid(self.audioset_logits(waveform).float())
        columns = []
        for label in self.spec.labels:
            rows = self.spec.rows.zero_shot.get(label, [])
            if rows:
                idx = torch.tensor(rows, device=probs.device)
                columns.append(probs.index_select(1, idx).max(dim=1).values)
            else:
                columns.append(torch.zeros(probs.shape[0], device=probs.device))
        return torch.stack(columns, dim=1)

    def init_head_from_audioset(self) -> dict[str, int]:
        """Copy the mean AudioSet row(s) of each class into the new head; returns rows used."""
        source = self.backbone.classifier[5]
        used = {}
        with torch.no_grad():
            for j, label in enumerate(self.spec.labels):
                rows = self.spec.rows.init.get(label, [])
                used[label] = len(rows)
                if rows:
                    self.head.weight[j] = source.weight[rows].mean(dim=0)
                    self.head.bias[j] = source.bias[rows].mean()
        return used

    def parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """(backbone params used by the embedding, head params). The 527-row layer is frozen."""
        frozen = set(map(id, self.backbone.classifier[5].parameters()))
        body = [p for p in self.backbone.parameters() if id(p) not in frozen]
        return body, list(self.head.parameters())


def build_model(spec: ModelSpec, *, pretrained_weights: Path | None, head_init: str
                ) -> tuple[DetectorNet, dict[str, int]]:
    """A new ``DetectorNet``: pretrained backbone (if given) and the initialised head."""
    model = DetectorNet(spec)
    used: dict[str, int] = {}
    if pretrained_weights is not None:
        state = torch.load(pretrained_weights, map_location="cpu", weights_only=True)
        model.backbone.load_state_dict(state, strict=True)
    if head_init == "audioset":
        used = model.init_head_from_audioset()
    for p in model.backbone.classifier[5].parameters():
        p.requires_grad_(False)
    return model, used


def save_checkpoint(path: Path, model: DetectorNet, extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": CHECKPOINT_VERSION, "spec": model.spec.to_json(),
               "model": model.state_dict(), **(extra or {})}
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu"
                    ) -> tuple[DetectorNet, dict[str, Any]]:
    """Rebuild the model from a checkpoint; returns ``(model, payload)`` (model in eval mode)."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"{path}: unsupported checkpoint version {payload.get('version')}")
    model = DetectorNet(ModelSpec.from_json(payload["spec"]))
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload
