"""AudioSet ontology graph (``<external>/ontology/ontology.json``).

Every AudioSet class has a permanent id, the **mid** (``/m/09x0r`` = "Speech"), a display name and
``child_ids``. A class may have **several parents** ("Bark" sits under "Dog" and under
"Domestic animals, pets" via "Dog"), so the graph is a DAG, not a tree; every walk keeps a visited
set so it can never loop.

Used to map dataset labels onto the project classes of ``configs/classes.yaml``:

* lookups by mid and by exact name (``Ontology.mid_of("Speech")``);
* :func:`fsd50k_label_map` builds FSD50K's underscore labels (``Walk_and_footsteps``) -> mid
  from the mids FSD50K ships, never by guessing from the spelling;
* :meth:`Ontology.descendants` / :meth:`Ontology.ancestors` feed the role rule
  (``soundspy.data.roles``);
* :func:`validate_classes` checks every ``audioset`` name in ``classes.yaml`` and proposes the
  closest valid names for the ones that do not exist.
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from soundspy.core.labels import LabelSet


class OntologyError(ValueError):
    pass


@dataclass(frozen=True)
class Node:
    mid: str
    name: str
    description: str
    child_ids: tuple[str, ...]
    restrictions: tuple[str, ...] = ()  # "abstract" / "blacklist" in the AudioSet ontology


def normalize_name(name: str) -> str:
    """Spelling-insensitive key for *suggestions only* (never used to assign mids)."""
    text = name.replace("_", " ").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


class Ontology:
    def __init__(self, nodes: Iterable[Mapping[str, Any]]) -> None:
        self._nodes: dict[str, Node] = {}
        for raw in nodes:
            node = Node(
                mid=str(raw["id"]),
                name=str(raw["name"]),
                description=str(raw.get("description", "")),
                child_ids=tuple(raw.get("child_ids", ())),
                restrictions=tuple(raw.get("restrictions", ())),
            )
            if node.mid in self._nodes:
                raise OntologyError(f"duplicate mid {node.mid}")
            self._nodes[node.mid] = node
        self._parents: dict[str, set[str]] = {mid: set() for mid in self._nodes}
        for node in self._nodes.values():
            for child in node.child_ids:
                if child not in self._nodes:
                    raise OntologyError(f"{node.mid} ({node.name}) lists unknown child {child}")
                self._parents[child].add(node.mid)
        self._by_name: dict[str, list[str]] = {}
        for node in self._nodes.values():
            self._by_name.setdefault(node.name, []).append(node.mid)
        self._desc_cache: dict[str, frozenset[str]] = {}
        self._anc_cache: dict[str, frozenset[str]] = {}

    @classmethod
    def load(cls, path: Path) -> Ontology:
        path = Path(path)
        if not path.exists():
            raise OntologyError(
                f"{path} not found: run `soundspy data download ontology` first"
            )
        return cls(json.loads(path.read_text(encoding="utf-8")))

    # -- lookups ------------------------------------------------------------------------------

    def __contains__(self, mid: object) -> bool:
        return mid in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def node(self, mid: str) -> Node:
        try:
            return self._nodes[mid]
        except KeyError:
            raise OntologyError(f"unknown AudioSet mid {mid!r}") from None

    def name(self, mid: str) -> str:
        return self.node(mid).name

    def mids_of(self, name: str) -> tuple[str, ...]:
        """All mids with this exact name (AudioSet names are unique in practice)."""
        return tuple(self._by_name.get(name, ()))

    def mid_of(self, name: str) -> str:
        mids = self.mids_of(name)
        if not mids:
            raise OntologyError(f"no AudioSet class named {name!r}")
        if len(mids) > 1:
            raise OntologyError(f"AudioSet name {name!r} is ambiguous: {mids}")
        return mids[0]

    def names(self) -> list[str]:
        return [n.name for n in self._nodes.values()]

    def parents(self, mid: str) -> frozenset[str]:
        self.node(mid)
        return frozenset(self._parents[mid])

    def children(self, mid: str) -> tuple[str, ...]:
        return self.node(mid).child_ids

    # -- graph walks --------------------------------------------------------------------------

    def descendants(self, mid: str) -> frozenset[str]:
        """``mid`` and everything below it (all paths through multiple parents)."""
        if mid not in self._desc_cache:
            self._desc_cache[mid] = self._walk(mid, lambda m: self._nodes[m].child_ids)
        return self._desc_cache[mid]

    def ancestors(self, mid: str) -> frozenset[str]:
        """Everything above ``mid`` (all parents, grandparents ...), **excluding** ``mid``."""
        if mid not in self._anc_cache:
            self._anc_cache[mid] = self._walk(mid, lambda m: self._parents[m]) - {mid}
        return self._anc_cache[mid]

    def _walk(self, start: str, step) -> frozenset[str]:
        self.node(start)
        seen = {start}
        stack = [start]
        while stack:
            for nxt in step(stack.pop()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return frozenset(seen)

    # -- suggestions --------------------------------------------------------------------------

    def closest(self, name: str, n: int = 5) -> list[str]:
        """Valid names that look most like ``name`` (for error messages and corrections)."""
        key = normalize_name(name)
        words = set(key.split())
        scored = []
        for candidate in self._by_name:
            ckey = normalize_name(candidate)
            ratio = difflib.SequenceMatcher(None, key, ckey).ratio()
            overlap = len(words & set(ckey.split())) / max(1, len(words))
            scored.append((0.6 * ratio + 0.4 * overlap, candidate))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [cand for _score, cand in scored[:n]]


def fsd50k_label_map(
    vocabulary: Iterable[tuple[str, str]], ontology: Ontology
) -> dict[str, str]:
    """FSD50K label -> AudioSet mid, from FSD50K's own ``vocabulary.csv`` (label, mid) pairs.

    The mapping comes from the mids, so ``Walk_and_footsteps`` maps to ``/m/07pbtc8``
    ("Walk, footsteps") without comparing spellings. Unknown mids raise.
    """
    mapping = {}
    for label, mid in vocabulary:
        if mid not in ontology:
            raise OntologyError(f"FSD50K label {label!r} has mid {mid} missing in the ontology")
        mapping[label] = mid
    return mapping


@dataclass(frozen=True)
class ClassNameIssue:
    class_key: str
    name: str
    candidates: tuple[str, ...]


def validate_classes(labels: LabelSet | Path, ontology: Ontology) -> list[ClassNameIssue]:
    """Every ``audioset`` name of every class must exist; returns the ones that do not.

    ``labels`` is a :class:`LabelSet` or the path of ``classes.yaml``.
    """
    if not isinstance(labels, LabelSet):
        from soundspy.config import load_label_set

        labels = load_label_set(Path(labels))
    issues = []
    for key in labels:
        for name in labels.audioset_names(key):
            if not ontology.mids_of(name):
                issues.append(ClassNameIssue(key, name, tuple(ontology.closest(name))))
    return issues


def class_mids(labels: LabelSet, ontology: Ontology) -> dict[str, tuple[str, ...]]:
    """``{class_key: (mid, ...)}`` from the ``audioset`` names (all must be valid)."""
    issues = validate_classes(labels, ontology)
    if issues:
        lines = [f"  {i.class_key}: {i.name!r} not in the AudioSet ontology; closest: "
                 f"{', '.join(i.candidates[:3])}" for i in issues]
        raise OntologyError("classes.yaml has invalid AudioSet names:\n" + "\n".join(lines))
    return {key: tuple(ontology.mid_of(n) for n in labels.audioset_names(key)) for key in labels}
