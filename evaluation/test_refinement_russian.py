"""Tests for Russian refinement parsing fallback behavior."""

from __future__ import annotations

from backend.mock_pipeline import build_refinement_plan


def test_russian_multi_step_refinement_maps_to_ordered_operations() -> None:
    plan = build_refinement_plan("сделай припев грустнее и короче, потом напиши 2 варианта текста")

    assert [operation.target for operation in plan.operations] == [
        "harmonizer",
        "melody_generator",
        "lyric_generator",
    ]
    assert plan.operations[0].params["emotion_valence"] < 0
    assert plan.operations[1].params["length"] == "shorter"
    assert plan.operations[2].params["num_options"] == 2


def test_russian_less_sad_maps_to_positive_valence_adjustment() -> None:
    plan = build_refinement_plan("сделай менее грустно")

    assert [operation.target for operation in plan.operations] == ["harmonizer"]
    assert plan.operations[0].params["emotion_valence"] > 0


def test_russian_jazzier_chords_maps_to_harmonizer() -> None:
    plan = build_refinement_plan("сделай аккорды джазовее")

    assert [operation.target for operation in plan.operations] == ["harmonizer"]
    assert plan.operations[0].params["chord_complexity"] == "high"
