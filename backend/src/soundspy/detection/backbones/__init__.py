"""Pretrained audio backbones for detection.

Every backbone is an ``nn.Module`` taking normalised log-mel ``(batch, 1, mels, frames)`` and
exposing the same three attributes, so the rest of the code never depends on which one it is:

* ``features``: the convolutional body (``nn.Sequential``);
* ``classifier``: ``[pool, flatten, Linear(c, E), Hardswish, Dropout, Linear(E, 527)]`` — the
  pretrained AudioSet tagger head; layers ``[:5]`` give the ``E``-dim embedding;
* ``embedding_dim``: ``E``.

``efficientat_mn10_as`` (adapted from EfficientAT, MIT) is the real model; ``stub`` is a tiny
random CNN with the same interface for tests (no download, CPU, milliseconds).
"""

from __future__ import annotations

from torch import nn

AUDIOSET_CLASSES = 527


class StubBackbone(nn.Module):
    """Tiny random CNN with the backbone interface (tests only)."""

    def __init__(self, num_classes: int = AUDIOSET_CLASSES, channels: int = 8,
                 embedding_dim: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, channels, 3, stride=2, padding=1), nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim=1),
            nn.Linear(channels, embedding_dim), nn.Hardswish(), nn.Dropout(0.0),
            nn.Linear(embedding_dim, num_classes),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x):
        return self.classifier(self.features(x)), None


def build_backbone(name: str) -> nn.Module:
    """Untrained backbone (weights are loaded by the caller)."""
    if name == "efficientat_mn10_as":
        from soundspy.detection.backbones.efficientat.model import get_model

        return get_model(num_classes=AUDIOSET_CLASSES, width_mult=1.0)
    if name == "stub":
        return StubBackbone()
    raise ValueError(f"unknown backbone {name!r}")
