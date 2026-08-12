"""Defects an adversarial review found and confirmed against real data.

Each one produced a confident wrong answer, not a crash, which is the kind that
survives a green test suite. They are kept here rather than spread through the
per-module files so the shape of the mistake stays visible: three root causes,
one section each.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from test_detectors import asset, contract, joint, opaque, part

from evo_verifier.asset import Articulation, Limits, Origin
from evo_verifier.contract import ExpectedJoint, Source
from evo_verifier.detectors import check_b7, check_b8, check_b9
from evo_verifier.matching import assign, similarity
from evo_verifier.report import Coverage, Prediction

# -- a requirement must own its parts ---------------------------------------


def helicopter(*, tail_joint: bool):
    joints = [joint("main_spin", "continuous", "fuselage", "main_rotor", limits=None)]
    if tail_joint:
        joints.append(joint("tail_spin", "continuous", "fuselage", "tail_rotor", limits=None))
    return asset(
        parts=[opaque("fuselage"), opaque("main_rotor"), opaque("tail_rotor")], joints=joints
    )


def rotors():
    return contract(
        ExpectedJoint(
            child="main rotor", parent="fuselage", kind="continuous", source=Source.PRIOR
        ),
        ExpectedJoint(
            child="tail rotor", parent="fuselage", kind="continuous", source=Source.PRIOR
        ),
    )


def test_a_sibling_joint_cannot_answer_for_a_missing_one():
    """The head noun alone scores 0.8, so `tail rotor` matches `main_rotor`.

    Before exclusive assignment this returned 1.0, pass, confidence 1.0, with an
    empty failure reason -- silent on exactly the fault B8 exists to catch.
    """
    assert similarity("tail rotor", "main_rotor") >= 0.8, "the look-alike is real"

    sound = check_b8(helicopter(tail_joint=True), rotors())
    assert sound.prediction is Prediction.PASS

    broken = check_b8(helicopter(tail_joint=False), rotors())
    assert broken.prediction is Prediction.FAIL
    assert broken.score == 0.0
    assert "tail rotor" in broken.failure_reason


def test_b7_does_not_certify_a_connection_from_another_parts_joint():
    result = check_b7(helicopter(tail_joint=False), rotors())
    tail = next(c for c in result.raw_measurements["connections"] if c["child"] == "tail rotor")
    assert tail["part"] != "main_rotor", "the tail rotor is judged on its own part"
    assert tail["actual_parent"] is None


def test_two_requirements_never_share_one_part():
    owned = assign(["temperature dial", "time dial"], ["temperature_dial", "time_dial"])
    assert owned == {0: ["temperature_dial"], 1: ["time_dial"]}
    taken = [name for names in owned.values() for name in names]
    assert len(taken) == len(set(taken))


def test_one_requirement_still_claims_every_instance():
    assert assign(["outlet flap"], ["front_flap", "rear_flap"]) == {0: ["front_flap", "rear_flap"]}


def test_identical_requirements_share_the_instances_out():
    assert assign(["flap", "flap"], ["flap_0", "flap_1"]) == {0: ["flap_0"], 1: ["flap_1"]}


def test_b9_names_the_joint_it_blames():
    """The reason keyed only on the contract name could not say which joint."""
    built = asset(
        parts=[part("body"), part("flap_0"), part("flap_1")],
        joints=[
            joint("good", "revolute", "body", "flap_0"),
            joint("bad", "prismatic", "body", "flap_1"),
        ],
    )
    asked = contract(
        ExpectedJoint(child="flap", kind="revolute", source=Source.PRIOR),
        ExpectedJoint(child="flap", kind="revolute", source=Source.PRIOR),
    )
    result = check_b9(built, asked, aggregate="worst")
    assert "bad" in result.failure_reason
    assert "good" not in result.failure_reason


# -- unreadable is not absent -----------------------------------------------


def unreadable_joint(field: str) -> Articulation:
    """A joint the source declares and the reader could not fold."""
    values = {"parent": "body", "child": "door", "kind": "revolute"}
    values[field] = None  # type: ignore[assignment]
    return Articulation(
        name="hinge",
        kind=values["kind"],
        parent=values["parent"],
        child=values["child"],
        origin=Origin(),
        axis=(0.0, 0.0, 1.0),
        limits=Limits(lower=0.0, upper=1.5, declared_bounds=True),
        unreadable=frozenset({field}),
    )


def test_an_unreadable_child_is_not_a_missing_joint():
    """The door has a joint; the reader just cannot see which part it moves."""
    built = asset(parts=[part("body"), part("door")], joints=[unreadable_joint("child")])
    asked = contract(ExpectedJoint(child="door", kind="revolute", source=Source.PRIOR))
    result = check_b8(built, asked)
    assert result.prediction is not Prediction.FAIL
    assert result.coverage is not Coverage.FULL


def test_an_unreadable_parent_is_not_a_part_connected_to_nothing():
    built = asset(parts=[part("body"), part("door")], joints=[unreadable_joint("parent")])
    asked = contract(
        ExpectedJoint(child="door", parent="body", kind="revolute", source=Source.PRIOR)
    )
    result = check_b7(built, asked)
    assert result.coverage is not Coverage.FULL
    assert "has no joint" not in result.failure_reason


def test_limits_written_but_unreadable_are_not_a_missing_limit():
    """`revolute with no limits` about a joint whose limits are in the file."""
    declared_unreadable = Limits(lower=None, upper=None, declared_bounds=True)
    built = asset(
        parts=[part("body"), part("door")],
        joints=[joint("hinge", "revolute", "body", "door", limits=declared_unreadable)],
    )
    asked = contract(ExpectedJoint(child="door", kind="revolute", source=Source.PRIOR))
    result = check_b9(built, asked)
    assert result.raw_measurements["consistency_score"] == 1.0
    assert "no limits" not in result.failure_reason


def test_limits_genuinely_absent_are_still_a_contradiction():
    built = asset(
        parts=[part("body"), part("door")],
        joints=[joint("hinge", "revolute", "body", "door", limits=None)],
    )
    asked = contract(ExpectedJoint(child="door", kind="revolute", source=Source.PRIOR))
    assert check_b9(built, asked).raw_measurements["consistency_score"] == 0.0


# -- a wrong match is worse than no match -----------------------------------


def test_synonym_groups_stay_disjoint():
    """`grip` in two groups made a handle the same thing as a clamp."""
    assert similarity("handle", "jaw") == 0.0
    assert similarity("handle", "grip") > 0.0
    assert similarity("jaw", "clamp") > 0.0


@pytest.mark.parametrize(
    ("plural", "singular"),
    [("chassis", "chassis"), ("axis", "axis"), ("lens", "lens"), ("flaps", "flap")],
)
def test_a_latin_singular_keeps_its_s(plural, singular):
    from evo_verifier.matching import _singular

    assert _singular(plural) == singular


def test_breaking_an_asset_never_raises_its_score():
    """B7 scored a broken asset above the sound one on a real record.

    Two matcher errors pulled opposite ways: a correctly attached part read as an
    alien rider, and a tie-broken part read as wrongly parented.
    """
    sound = asset(
        parts=[part("post"), part("cover"), part("lock_cover")],
        joints=[
            joint("raise", "prismatic", "post", "cover"),
            joint("open", "revolute", "post", "lock_cover"),
        ],
    )
    asked = contract(
        ExpectedJoint(child="cover", parent="post", source=Source.PRIOR),
        ExpectedJoint(child="lock cover", parent="post", source=Source.PRIOR),
    )
    before = check_b7(sound, asked)
    broken = check_b7(
        replace(
            sound,
            articulations=[
                replace(j, parent="cover") if j.name == "open" else j for j in sound.articulations
            ],
        ),
        asked,
    )
    assert broken.score <= before.score
