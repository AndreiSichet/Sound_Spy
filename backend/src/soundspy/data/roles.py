"""Foreground / background / excluded: what a clip may be used for in a mixture.

For each project class ``c`` (``configs/classes.yaml``):

* ``F(c)`` = every AudioSet mid of the class plus all their descendants;
* ``A(c)`` = all ancestors of those mids (the mids themselves excluded).

A clip with AudioSet mids ``M`` is

* **foreground for c** (``fg:<c>``) if every mid of ``M`` is in ``F(c) | A(c)`` and at least one
  is in ``F(c)``. Ancestors are tolerated because FSD50K propagates labels upwards ("smeared"):
  ``Speech, Human_voice`` is still pure speech;
* **background** (``bg``) if no mid of ``M`` is in ``F(c)`` for any class;
* **excluded** otherwise: it contains one of our classes *and* something else (speech + music),
  so it would teach the extractor a wrong target. A clip that is foreground for two classes at
  once (overlapping class definitions, e.g. a car horn is both "Alarm" and "Vehicle") is excluded
  too, and the overlap is reported. Clips without any mid (unmapped ESC-50 classes) are excluded.

Smearing follows **every** parent: a car-horn clip is labelled ``Vehicle_horn, Car, Vehicle`` and
``Alarm``. ``Alarm`` is not an ancestor of the traffic class, so the literal rule would call the
clip "traffic mixed with something else". Therefore the foreground test first drops every label
that is an ancestor of *another label of the same clip* (it is implied by smearing); only the
most specific labels are checked. Elsewhere this gives exactly the rule above; for the horn clip
it correctly reports an alarm + traffic overlap.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from soundspy.core.labels import LabelSet
from soundspy.data.ontology import Ontology, class_mids

BG = "bg"
EXCLUDED = "excluded"
FG_PREFIX = "fg:"


def fg_role(class_key: str) -> str:
    return f"{FG_PREFIX}{class_key}"


def role_class(role: str) -> str | None:
    """``"fg:speech"`` -> ``"speech"``; ``None`` for ``bg`` / ``excluded``."""
    return role[len(FG_PREFIX):] if role.startswith(FG_PREFIX) else None


@dataclass(frozen=True)
class RoleDecision:
    role: str
    overlap: tuple[str, ...] = ()  # classes the clip qualified for when excluded as overlap
    reason: str = ""


@dataclass(frozen=True)
class RoleRules:
    classes: tuple[str, ...]
    fg: dict[str, frozenset[str]]  # F(c)
    allowed: dict[str, frozenset[str]]  # F(c) | A(c)
    ontology: Ontology | None = None  # for dropping smeared labels (see module docstring)

    @classmethod
    def build(cls, labels: LabelSet, ontology: Ontology) -> RoleRules:
        mids = class_mids(labels, ontology)
        fg, allowed = {}, {}
        for key, class_mid_list in mids.items():
            f = frozenset().union(*(ontology.descendants(m) for m in class_mid_list))
            a = frozenset().union(*(ontology.ancestors(m) for m in class_mid_list))
            fg[key], allowed[key] = f, f | a
        return cls(tuple(labels), fg, allowed, ontology)

    def most_specific(self, mids: frozenset[str]) -> frozenset[str]:
        """``mids`` without labels that are ancestors of another label of the same clip."""
        if self.ontology is None:
            return mids
        implied = set()
        for mid in mids:
            if mid in self.ontology:
                implied |= self.ontology.ancestors(mid)
        return frozenset(m for m in mids if m not in implied) or mids

    def overlapping_classes(self) -> dict[tuple[str, str], frozenset[str]]:
        """Pairs of classes whose ``F`` sets share mids (their shared clips get excluded)."""
        out = {}
        for i, a in enumerate(self.classes):
            for b in self.classes[i + 1:]:
                shared = self.fg[a] & self.fg[b]
                if shared:
                    out[(a, b)] = shared
        return out

    def decide(self, mids: Iterable[str]) -> RoleDecision:
        mids = frozenset(mids)
        if not mids:
            return RoleDecision(EXCLUDED, reason="no AudioSet labels")
        specific = self.most_specific(mids)
        candidates = tuple(
            c for c in self.classes
            if specific <= self.allowed[c] and specific & self.fg[c]
        )
        if len(candidates) == 1:
            return RoleDecision(fg_role(candidates[0]))
        if len(candidates) > 1:
            return RoleDecision(EXCLUDED, overlap=candidates,
                                reason="foreground for several classes")
        if not any(mids & self.fg[c] for c in self.classes):
            return RoleDecision(BG)
        return RoleDecision(EXCLUDED, reason="a class mixed with other sounds")
