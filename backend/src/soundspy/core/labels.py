"""``ClassDef`` and ``LabelSet``: the project classes.

Built from ``configs/classes.yaml`` (via ``soundspy.config.load_label_set``). Each class has:

- ``name``: internal ID used by code, models and the database; never changes.
- ``display_name``: what the UI shows.
- ``description``: tooltip / report text.
- ``audioset``: AudioSet ontology names, used only to build datasets.

The order defines the index used in multi-hot label vectors (detector outputs, extraction model
conditioning). Class names are never hard-coded.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ClassDef:
    name: str
    display_name: str
    description: str = ""
    audioset: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"class name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError(f"class {self.name!r}: display_name must be a non-empty string")
        if not isinstance(self.description, str):
            raise ValueError(f"class {self.name!r}: description must be a string")
        audioset = tuple(self.audioset)
        if not all(isinstance(n, str) and n for n in audioset):
            raise ValueError(f"class {self.name!r}: audioset must be a list of names")
        object.__setattr__(self, "audioset", audioset)


class LabelSet:
    def __init__(self, classes: Sequence[ClassDef | str]) -> None:
        defs = [c if isinstance(c, ClassDef) else ClassDef(c, display_name=c) for c in classes]
        if not defs:
            raise ValueError("a LabelSet needs at least one class")
        names = [c.name for c in defs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate class names in {names}")
        self._classes: tuple[ClassDef, ...] = tuple(defs)
        self._labels: tuple[str, ...] = tuple(names)
        self._index = {name: i for i, name in enumerate(names)}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Mapping[str, Any]]) -> LabelSet:
        """Build from the ``classes.yaml`` format.

        ``{name: {display_name, description, audioset}}``; ``display_name`` is required.
        """
        classes = []
        for name, fields in mapping.items():
            if not isinstance(fields, Mapping):
                raise ValueError(
                    f"class {name!r} must be a mapping with display_name, description, audioset"
                )
            unknown = set(fields) - {"display_name", "description", "audioset"}
            if unknown:
                raise ValueError(f"class {name!r}: unknown fields {sorted(unknown)}")
            if "display_name" not in fields:
                raise ValueError(f"class {name!r}: display_name is required")
            audioset = fields.get("audioset", [])
            if isinstance(audioset, str):
                audioset = [audioset]
            classes.append(
                ClassDef(
                    name=str(name),
                    display_name=fields["display_name"],
                    description=fields.get("description", ""),
                    audioset=tuple(audioset),
                )
            )
        return cls(classes)

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def classes(self) -> tuple[ClassDef, ...]:
        return self._classes

    def info(self, label: str) -> ClassDef:
        return self._classes[self.index(label)]

    def display_name(self, label: str) -> str:
        return self.info(label).display_name

    def description(self, label: str) -> str:
        return self.info(label).description

    def audioset_names(self, label: str) -> tuple[str, ...]:
        return self.info(label).audioset

    def index(self, label: str) -> int:
        try:
            return self._index[label]
        except KeyError:
            raise ValueError(f"unknown label {label!r}; known: {list(self._labels)}") from None

    def label(self, i: int) -> str:
        if not 0 <= i < len(self._labels):
            raise IndexError(f"label index {i} out of range 0..{len(self._labels) - 1}")
        return self._labels[i]

    def validate(self, labels: Iterable[str]) -> None:
        unknown = sorted(set(labels) - set(self._labels))
        if unknown:
            raise ValueError(f"unknown labels {unknown}; known: {list(self._labels)}")

    def to_multi_hot(self, labels: Iterable[str]) -> np.ndarray:
        labels = list(labels)
        self.validate(labels)
        vec = np.zeros(len(self._labels), dtype=np.float32)
        for label in labels:
            vec[self._index[label]] = 1.0
        return vec

    def from_multi_hot(
        self, vec: np.ndarray | Sequence[float], threshold: float = 0.5
    ) -> frozenset[str]:
        arr = np.asarray(vec, dtype=np.float32)
        if arr.shape != (len(self._labels),):
            raise ValueError(f"expected a vector of shape ({len(self._labels)},), got {arr.shape}")
        return frozenset(self._labels[i] for i in np.flatnonzero(arr >= threshold))

    def __len__(self) -> int:
        return len(self._labels)

    def __iter__(self) -> Iterator[str]:
        return iter(self._labels)

    def __contains__(self, label: object) -> bool:
        return label in self._index

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LabelSet) and self._classes == other._classes

    def __hash__(self) -> int:
        return hash(self._labels)

    def __repr__(self) -> str:
        return f"LabelSet({list(self._labels)})"
