"""Ontology graph walks, FSD50K label mapping, classes.yaml validation and the role rule."""

from __future__ import annotations

import pytest

from soundspy.core.labels import ClassDef, LabelSet
from soundspy.data.ontology import (
    Ontology,
    OntologyError,
    class_mids,
    fsd50k_label_map,
    validate_classes,
)
from soundspy.data.roles import BG, EXCLUDED, RoleRules


@pytest.fixture
def onto(kit) -> Ontology:
    return Ontology(kit.ONTOLOGY)


def test_lookup_by_mid_and_name(onto: Ontology) -> None:
    assert onto.mid_of("Speech") == "/m/sp"
    assert onto.name("/m/horn") == "Vehicle horn, car horn, honking"
    with pytest.raises(OntologyError):
        onto.mid_of("Chair")


def test_multiple_parents(onto: Ontology) -> None:
    assert onto.parents("/m/horn") == {"/m/alarm", "/m/car"}
    assert onto.ancestors("/m/horn") == {"/m/alarm", "/m/car", "/m/veh", "/m/things"}
    assert "/m/horn" not in onto.ancestors("/m/horn")


def test_descendants_include_self_and_all_paths(onto: Ontology) -> None:
    assert onto.descendants("/m/things") == {
        "/m/things", "/m/alarm", "/m/veh", "/m/car", "/m/horn"}
    assert onto.descendants("/m/walk") == {"/m/walk"}


def test_cycles_do_not_loop() -> None:
    nodes = [{"id": "/a", "name": "A", "child_ids": ["/b"]},
             {"id": "/b", "name": "B", "child_ids": ["/a"]}]
    onto = Ontology(nodes)
    assert onto.descendants("/a") == {"/a", "/b"}
    assert onto.ancestors("/a") == {"/b"}


def test_fsd50k_names_map_through_mids(onto: Ontology, kit) -> None:
    mapping = fsd50k_label_map(kit.VOCABULARY, onto)
    assert mapping["Walk_and_footsteps"] == "/m/walk"
    assert onto.name(mapping["Walk_and_footsteps"]) == "Walk, footsteps"
    with pytest.raises(OntologyError, match="missing"):
        fsd50k_label_map([("Ghost", "/m/nope")], onto)


def test_validate_classes_reports_unknown_names_with_candidates(onto: Ontology) -> None:
    labels = LabelSet([
        ClassDef("speech", "Speech", "", ("Speech",)),
        ClassDef("traffic", "Traffic", "", ("Vehicles",)),
        ClassDef("footsteps", "Footsteps", "", ("Walk footsteps",)),
    ])
    issues = validate_classes(labels, onto)
    assert [(i.class_key, i.name) for i in issues] == [
        ("traffic", "Vehicles"), ("footsteps", "Walk footsteps")]
    assert issues[0].candidates[0] == "Vehicle"
    assert issues[1].candidates[0] == "Walk, footsteps"
    with pytest.raises(OntologyError, match="Vehicles"):
        class_mids(labels, onto)


def test_real_classes_yaml_is_valid_against_the_real_ontology(config) -> None:
    path = config.paths["external"] / "ontology" / "ontology.json"
    if not path.exists():
        pytest.skip("real ontology not downloaded")
    assert validate_classes(config.labels, Ontology.load(path)) == []


@pytest.fixture
def rules(onto: Ontology, kit) -> RoleRules:
    return RoleRules.build(kit.LABELS, onto)


def test_pure_foreground(rules: RoleRules) -> None:
    assert rules.decide(["/m/sp"]).role == "fg:speech"
    assert rules.decide(["/m/conv"]).role == "fg:speech"  # descendant


def test_foreground_with_smeared_ancestor_labels(rules: RoleRules) -> None:
    assert rules.decide(["/m/sp", "/m/hv"]).role == "fg:speech"
    assert rules.decide(["/m/car", "/m/veh"]).role == "fg:traffic"


def test_two_classes_mixed_is_excluded(rules: RoleRules) -> None:
    decision = rules.decide(["/m/sp", "/m/hv", "/m/mus"])
    assert decision.role == EXCLUDED
    assert decision.overlap == ()


def test_foreground_plus_background_label_is_excluded(rules: RoleRules) -> None:
    assert rules.decide(["/m/walk", "/m/dog"]).role == EXCLUDED


def test_background(rules: RoleRules) -> None:
    assert rules.decide(["/m/dog"]).role == BG
    assert rules.decide(["/m/rain", "/m/dog"]).role == BG
    # An ancestor of a class alone is not a class: "Shout, Human voice" is background.
    assert rules.decide(["/m/shout", "/m/hv"]).role == BG


def test_overlapping_class_definitions_are_excluded_and_reported(rules: RoleRules) -> None:
    decision = rules.decide(["/m/horn", "/m/car", "/m/veh", "/m/alarm"])
    assert decision.role == EXCLUDED
    assert decision.overlap == ("alarm", "traffic")
    assert rules.overlapping_classes() == {("alarm", "traffic"): {"/m/horn"}}


def test_no_labels_is_excluded(rules: RoleRules) -> None:
    assert rules.decide([]).role == EXCLUDED
