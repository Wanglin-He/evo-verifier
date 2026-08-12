from __future__ import annotations

import math

import pytest

from evo_verifier.asset import Articulation, Asset, Limits, Origin, Part, Shape
from evo_verifier.contract import Contract, ExpectedJoint, Source
from evo_verifier.detectors import (
    AGGREGATE,
    NEAR_MISS,
    check_b7,
    check_b8,
    check_b9,
    check_b10,
)
from evo_verifier.report import Coverage, Prediction

PROMPT = "a cabinet with a door and a drawer"


def part(name: str, size=(0.4, 0.4, 0.4), at=(0.0, 0.0, 0.0)) -> Part:
    return Part(name=name, shapes=(Shape(kind="box", size=size, origin=Origin(xyz=at)),))


def opaque(name: str) -> Part:
    """A part built from a CAD mesh: no analytic bounds."""
    return Part(name=name, shapes=(Shape(kind="mesh", origin=Origin()),))


LIMITED = Limits(lower=0.0, upper=1.5)


def joint(
    name: str,
    kind: str | None,
    parent: str,
    child: str,
    *,
    axis=(0.0, 0.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    limits: Limits | None = LIMITED,
) -> Articulation:
    return Articulation(
        name=name,
        kind=kind,
        parent=parent,
        child=child,
        origin=Origin(xyz=origin),
        axis=axis,
        limits=limits,
    )


def asset(*, parts: list[Part], joints: list[Articulation]) -> Asset:
    return Asset(
        record_id="rec_t",
        parts={p.name: p for p in parts},
        articulations=list(joints),
    )


def contract(*joints: ExpectedJoint) -> Contract:
    return Contract(record_id="rec_t", prompt=PROMPT, joints=joints, extractor="test")


# -- B8 ---------------------------------------------------------------------


def test_b8_passes_when_every_expected_part_moves():
    result = check_b8(
        asset(
            parts=[part("body"), part("door")],
            joints=[joint("hinge", "revolute", "body", "door")],
        ),
        contract(ExpectedJoint(child="door", kind="revolute", source=Source.PRIOR)),
    )
    assert result.score == 1.0
    assert result.prediction is Prediction.PASS
    assert result.coverage is Coverage.FULL


def test_b8_fails_a_part_welded_shut():
    result = check_b8(
        asset(
            parts=[part("body"), part("door")],
            joints=[joint("weld", "fixed", "body", "door", axis=None, limits=None)],
        ),
        contract(ExpectedJoint(child="door", kind="revolute", source=Source.PRIOR)),
    )
    assert result.score == 0.0
    assert result.prediction is Prediction.FAIL
    assert "door" in result.failure_reason


def test_b8_counts_instances_against_the_contracts_count():
    """Four flaps expected, two articulated: half credit, not pass."""
    result = check_b8(
        asset(
            parts=[part("body"), *(part(f"flap_{i}") for i in range(4))],
            joints=[
                joint("h0", "revolute", "body", "flap_0"),
                joint("h1", "revolute", "body", "flap_1"),
            ],
        ),
        contract(ExpectedJoint(child="flap", kind="revolute", count=4, source=Source.PRIOR)),
    )
    assert result.score == 0.5
    assert result.prediction is Prediction.FAIL


def test_b8_does_not_pay_extra_for_more_parts_than_asked():
    result = check_b8(
        asset(
            parts=[part("body"), *(part(f"flap_{i}") for i in range(4))],
            joints=[joint(f"h{i}", "revolute", "body", f"flap_{i}") for i in range(4)],
        ),
        contract(ExpectedJoint(child="flap", kind="revolute", count=2, source=Source.PRIOR)),
    )
    assert result.score == 1.0


def test_b8_a_missing_part_counts_against_but_costs_confidence():
    """The asset cannot articulate a part it never modelled -- but the name may
    also have failed to match, so the verdict is held less firmly."""
    both = check_b8(
        asset(parts=[part("body"), part("door")], joints=[joint("h", "revolute", "body", "door")]),
        contract(
            ExpectedJoint(child="door", kind="revolute", source=Source.PRIOR),
            ExpectedJoint(child="drawer", kind="prismatic", source=Source.PRIOR),
        ),
    )
    assert both.score == 0.0, "B8 aggregates worst-first: one unmet requirement decides"
    assert both.coverage is Coverage.PARTIAL
    assert both.confidence < 1.0
    assert "drawer" in both.failure_reason


def test_b8_narrow_definition_drops_the_missing_part_instead():
    result = check_b8(
        asset(parts=[part("body"), part("door")], joints=[joint("h", "revolute", "body", "door")]),
        contract(
            ExpectedJoint(child="door", kind="revolute", source=Source.PRIOR),
            ExpectedJoint(child="drawer", kind="prismatic", source=Source.PRIOR),
        ),
        missing_is_failure=False,
    )
    assert result.score == 1.0, "the missing part leaves the score under the narrow reading"
    assert result.coverage is Coverage.PARTIAL


def test_b8_a_prior_weighs_less_than_a_quoted_requirement():
    quoted = ExpectedJoint(child="door", kind="revolute", quote=PROMPT)
    guessed = ExpectedJoint(child="drawer", kind="prismatic", source=Source.PRIOR, confidence=0.7)
    built = asset(
        parts=[part("body"), part("door"), part("drawer")],
        joints=[joint("h", "revolute", "body", "door")],
    )
    result = check_b8(built, contract(quoted, guessed), aggregate="mean")
    assert result.score == pytest.approx(1.0 / 1.7, abs=1e-4), "the prior costs only its own weight"
    assert result.raw_measurements["score_explicit_only"] == 1.0


def test_b8_is_not_applicable_when_nothing_should_move():
    result = check_b8(
        asset(parts=[part("body")], joints=[]),
        contract(ExpectedJoint(child="lid", kind="fixed", source=Source.PRIOR)),
    )
    assert result.coverage is Coverage.NOT_APPLICABLE
    assert result.score is None


# -- B7 ---------------------------------------------------------------------


def test_b7_passes_the_right_connection():
    result = check_b7(
        asset(
            parts=[part("cabinet_body"), part("door")],
            joints=[joint("hinge", "revolute", "cabinet_body", "door")],
        ),
        contract(ExpectedJoint(child="door", parent="cabinet body", source=Source.PRIOR)),
    )
    assert result.score == 1.0
    assert result.prediction is Prediction.PASS


def test_b7_fails_a_part_hung_off_the_wrong_thing():
    """The juicer: the pusher's own joint is fine, its other end is not."""
    result = check_b7(
        asset(
            parts=[part("base"), part("chute"), part("lid"), part("pusher")],
            joints=[
                joint("h", "revolute", "base", "lid"),
                joint("slide", "prismatic", "lid", "pusher"),
            ],
        ),
        contract(
            ExpectedJoint(child="lid", parent="base", source=Source.PRIOR),
            ExpectedJoint(child="pusher", parent="chute", source=Source.PRIOR),
        ),
    )
    assert result.prediction is Prediction.FAIL
    assert "pusher" in result.failure_reason and "chute" in result.failure_reason


def test_b7_catches_a_part_riding_on_the_wrong_subtree():
    """The blender: the button's own joint is right, but it travels with the jar."""
    result = check_b7(
        asset(
            parts=[part("base"), part("jar"), part("lid"), part("release_button")],
            joints=[
                joint("collar", "revolute", "base", "jar"),
                joint("snap", "fixed", "jar", "lid", axis=None, limits=None),
                joint("press", "prismatic", "lid", "release_button"),
            ],
        ),
        contract(
            ExpectedJoint(child="jar", parent="base", source=Source.PRIOR),
            ExpectedJoint(child="release button", parent="base", source=Source.PRIOR),
        ),
    )
    rides = [c["rides_along"] for c in result.raw_measurements["connections"]]
    assert any("release_button" in entry for entry in rides)


def test_b7_ignores_a_parent_outside_the_object():
    """A vise clamps to a bench; the bench is not a link."""
    result = check_b7(
        asset(
            parts=[part("base"), part("jaw")],
            joints=[joint("slide", "prismatic", "base", "jaw")],
        ),
        contract(
            ExpectedJoint(child="jaw", parent="base", source=Source.PRIOR),
            ExpectedJoint(child="base", parent="workbench", source=Source.PRIOR),
        ),
    )
    parents = [c["parent_ok"] for c in result.raw_measurements["connections"]]
    assert None in parents, "the external attachment is not judged"


def test_b7_treats_an_unmodelled_base_name_as_unknown_not_wrong():
    """The prompt calls the base "the panel"; the asset calls it cassette. No
    other part answers to panel, and the flap does hang off the root."""
    result = check_b7(
        asset(
            parts=[part("cassette"), part("flap")],
            joints=[joint("hinge", "revolute", "cassette", "flap")],
        ),
        contract(ExpectedJoint(child="flap", parent="panel", source=Source.PRIOR)),
    )
    assert result.raw_measurements["connections"][0]["parent_ok"] is None
    assert result.prediction is Prediction.PASS


def test_b7_still_fails_when_the_wrong_parent_is_not_the_root():
    result = check_b7(
        asset(
            parts=[part("base"), part("lid"), part("pusher")],
            joints=[
                joint("h", "revolute", "base", "lid"),
                joint("slide", "prismatic", "lid", "pusher"),
            ],
        ),
        contract(ExpectedJoint(child="pusher", parent="chute", source=Source.PRIOR)),
    )
    assert result.raw_measurements["connections"][0]["parent_ok"] is False


def test_b7_is_unsupported_when_the_prompt_names_no_parent():
    result = check_b7(
        asset(parts=[part("body"), part("door")], joints=[joint("h", "revolute", "body", "door")]),
        contract(ExpectedJoint(child="door", source=Source.PRIOR)),
    )
    assert result.coverage is Coverage.UNSUPPORTED
    assert result.prediction is Prediction.ABSTAIN


# -- B9 ---------------------------------------------------------------------


def test_b9_passes_the_declared_type():
    result = check_b9(
        asset(
            parts=[part("body"), part("door")],
            joints=[joint("hinge", "revolute", "body", "door")],
        ),
        contract(ExpectedJoint(child="door", kind="revolute", source=Source.PRIOR)),
    )
    assert result.score == 1.0


def test_b9_fails_a_slide_declared_where_a_hinge_was_asked():
    result = check_b9(
        asset(
            parts=[part("body"), part("door")],
            joints=[joint("slide", "prismatic", "body", "door")],
        ),
        contract(ExpectedJoint(child="door", kind="revolute", source=Source.PRIOR)),
    )
    assert result.score == pytest.approx(0.4), "no type credit, self-consistency intact"
    assert result.prediction is Prediction.FAIL


def test_b9_gives_half_credit_between_revolute_and_continuous():
    """Both turn about an axis; only the stop differs. Still a failure."""
    result = check_b9(
        asset(
            parts=[part("body"), part("knob")],
            joints=[joint("spin", "revolute", "body", "knob")],
        ),
        contract(ExpectedJoint(child="knob", kind="continuous", source=Source.PRIOR)),
    )
    assert result.raw_measurements["declared_type_score"] == NEAR_MISS
    assert result.prediction is Prediction.FAIL


def test_b9_catches_a_continuous_joint_wearing_limits():
    result = check_b9(
        asset(
            parts=[part("body"), part("rotor")],
            joints=[
                joint("spin", "continuous", "body", "rotor", limits=Limits(lower=0.0, upper=1.0))
            ],
        ),
        contract(ExpectedJoint(child="rotor", kind="continuous", source=Source.PRIOR)),
    )
    assert result.raw_measurements["consistency_score"] == 0.0
    assert "limited" in result.failure_reason


def test_b9_catches_a_revolute_with_no_limits():
    result = check_b9(
        asset(
            parts=[part("body"), part("door")],
            joints=[joint("hinge", "revolute", "body", "door", limits=None)],
        ),
        contract(ExpectedJoint(child="door", kind="revolute", source=Source.PRIOR)),
    )
    assert result.raw_measurements["consistency_score"] == 0.0


def test_b9_leaves_out_a_part_the_asset_never_modelled():
    """A missing part is B8's finding, not a wrong type."""
    result = check_b9(
        asset(
            parts=[part("body"), part("door")],
            joints=[joint("hinge", "revolute", "body", "door")],
        ),
        contract(
            ExpectedJoint(child="door", kind="revolute", source=Source.PRIOR),
            ExpectedJoint(child="drawer", kind="prismatic", source=Source.PRIOR),
        ),
    )
    assert result.score == 1.0
    assert result.coverage is Coverage.PARTIAL
    assert result.confidence == 0.5
    assert result.raw_measurements["unresolved_names"] == ["drawer"]


# -- B10 --------------------------------------------------------------------


def test_b10_passes_an_axis_pointing_the_way_the_prompt_says():
    result = check_b10(
        asset(
            parts=[opaque("body"), opaque("door")],
            joints=[joint("hinge", "revolute", "body", "door", axis=(0.0, 0.0, 1.0))],
        ),
        contract(
            ExpectedJoint(
                child="door",
                kind="revolute",
                axis_hint="a vertical side hinge",
                source=Source.PRIOR,
            )
        ),
    )
    assert result.score == 1.0
    assert result.raw_measurements["joints"][0]["axis_error_degrees"] == 0.0


def test_b10_fails_a_vertical_hinge_declared_horizontal():
    result = check_b10(
        asset(
            parts=[opaque("body"), opaque("door")],
            joints=[joint("hinge", "revolute", "body", "door", axis=(1.0, 0.0, 0.0))],
        ),
        contract(
            ExpectedJoint(child="door", kind="revolute", axis_hint="vertical", source=Source.PRIOR)
        ),
    )
    assert result.raw_measurements["joints"][0]["axis_error_degrees"] == pytest.approx(90.0)
    assert result.prediction is Prediction.FAIL


def test_b10_reads_horizontal_as_the_plane_not_a_single_direction():
    """Two hinges on opposite panel edges are both horizontal."""
    for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
        result = check_b10(
            asset(
                parts=[opaque("body"), opaque("flap")],
                joints=[joint("hinge", "revolute", "body", "flap", axis=axis)],
            ),
            contract(
                ExpectedJoint(
                    child="flap",
                    kind="revolute",
                    axis_hint="horizontal hinges",
                    source=Source.PRIOR,
                )
            ),
        )
        assert result.score == 1.0


def test_b10_scores_an_anchor_inside_the_part_as_clean():
    result = check_b10(
        asset(
            parts=[part("body"), part("door", size=(0.1, 0.5, 0.8))],
            joints=[joint("hinge", "revolute", "body", "door")],
        ),
        contract(ExpectedJoint(child="door", kind="revolute", source=Source.PRIOR)),
    )
    assert result.raw_measurements["joints"][0]["anchor_over_link"] == 0.0
    assert result.score == 1.0


def test_b10_catches_a_hinge_floating_away_from_what_it_moves():
    """The virtual-hinge signal: the joint is not on the part it turns."""
    result = check_b10(
        asset(
            parts=[part("body"), part("door", size=(0.1, 0.5, 0.8), at=(2.0, 0.0, 0.0))],
            joints=[joint("hinge", "revolute", "body", "door")],
        ),
        contract(ExpectedJoint(child="door", kind="revolute", source=Source.PRIOR)),
    )
    anchor = result.raw_measurements["joints"][0]["anchor_over_link"]
    assert anchor > 1.0
    assert result.score == pytest.approx(0.0, abs=1e-6)
    assert result.prediction is Prediction.FAIL


def test_b10_combines_two_terms_as_their_geometric_mean():
    built = asset(
        parts=[part("body"), part("door", size=(0.1, 0.5, 0.8), at=(0.3, 0.0, 0.0))],
        joints=[joint("hinge", "revolute", "body", "door", axis=(1.0, 0.0, 0.0))],
    )
    asked = contract(
        ExpectedJoint(child="door", kind="revolute", axis_hint="vertical", source=Source.PRIOR)
    )
    result = check_b10(built, asked)
    measured = result.raw_measurements["joints"][0]
    assert result.score == pytest.approx(
        math.sqrt(measured["anchor_score"] * measured["axis_score"]), abs=1e-4
    )


def test_b10_abstains_with_neither_geometry_nor_a_direction():
    result = check_b10(
        asset(
            parts=[opaque("body"), opaque("door")],
            joints=[joint("hinge", "revolute", "body", "door")],
        ),
        contract(ExpectedJoint(child="door", kind="revolute", source=Source.PRIOR)),
    )
    assert result.coverage is Coverage.UNSUPPORTED
    assert result.prediction is Prediction.ABSTAIN
    assert result.score is None


def test_b10_ignores_a_direction_word_that_needs_geometry():
    """ "about its axle" names a direction we cannot resolve in the world frame."""
    result = check_b10(
        asset(
            parts=[opaque("body"), opaque("wheel")],
            joints=[joint("spin", "continuous", "body", "wheel")],
        ),
        contract(
            ExpectedJoint(
                child="wheel", kind="continuous", axis_hint="about its axle", source=Source.PRIOR
            )
        ),
    )
    assert result.coverage is Coverage.UNSUPPORTED


# -- aggregation ------------------------------------------------------------


def _one_of_four_broken():
    return (
        asset(
            parts=[part("body"), *(part(f"flap_{i}") for i in range(4))],
            joints=[
                joint("h0", "revolute", "body", "flap_0"),
                joint("h1", "revolute", "body", "flap_1"),
                joint("h2", "revolute", "body", "flap_2"),
                joint("h3", "prismatic", "body", "flap_3"),  # the wrong one
            ],
        ),
        contract(
            *(
                ExpectedJoint(child=f"flap {i}", kind="revolute", source=Source.PRIOR)
                for i in range(4)
            )
        ),
    )


def test_mean_lets_three_good_joints_hide_a_bad_one():
    """Why the aggregation choice matters: 3 of 4 right averages to 0.85."""
    built, asked = _one_of_four_broken()
    result = check_b9(built, asked, aggregate="mean")
    assert result.score > 0.70
    assert result.prediction is Prediction.PASS


def test_worst_lets_one_bad_joint_decide():
    built, asked = _one_of_four_broken()
    result = check_b9(built, asked, aggregate="worst")
    assert result.score < 0.70
    assert result.prediction is Prediction.FAIL


def test_each_item_has_an_aggregation_chosen_from_evidence():
    assert AGGREGATE == {"B7": "mean", "B8": "worst", "B9": "mean", "B10": "mean"}
