from __future__ import annotations

import math

import pytest
from test_detectors import asset, contract, joint, opaque, part

from evo_verifier.contract import ExpectedJoint, Source
from evo_verifier.detectors import check_b8, check_b9
from evo_verifier.inject import ANGLES, Fault, apply, faults_for
from evo_verifier.report import Prediction


def cabinet():
    return asset(
        parts=[part("body"), part("door", size=(0.1, 0.5, 0.8))],
        joints=[joint("hinge", "revolute", "body", "door")],
    )


def angle_between(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    lengths = math.dist((0.0, 0.0, 0.0), a) * math.dist((0.0, 0.0, 0.0), b)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / lengths))))


def test_the_original_asset_is_never_touched():
    original = cabinet()
    before = original.articulations[0]
    apply(original, Fault("B8", "weld", "hinge", "declared fixed"))
    assert original.articulations[0] is before
    assert original.articulations[0].kind == "revolute"


def test_injection_is_deterministic():
    fault = Fault("B10", "tilt", "hinge", "axis rotated", 30.0)
    first, second = apply(cabinet(), fault), apply(cabinet(), fault)
    assert first.articulations[0].axis == second.articulations[0].axis


def test_unjoint_removes_the_joint():
    broken = apply(cabinet(), Fault("B8", "unjoint", "hinge", "joint removed"))
    assert broken.articulations == []
    assert "door" in broken.parts, "the part stays; only its joint goes"


def test_weld_leaves_the_joint_but_stops_it_moving():
    broken = apply(cabinet(), Fault("B8", "weld", "hinge", "declared fixed"))
    assert broken.articulations[0].kind == "fixed"
    assert broken.movable() == []


def test_reparent_changes_only_the_other_end():
    broken = apply(cabinet(), Fault("B7", "reparent", "hinge", "door"))
    assert broken.articulations[0].parent == "door"
    assert broken.articulations[0].child == "door"


def test_retype_changes_the_declared_kind():
    broken = apply(cabinet(), Fault("B9", "retype", "hinge", "prismatic"))
    assert broken.articulations[0].kind == "prismatic"


@pytest.mark.parametrize("degrees", ANGLES)
def test_tilt_turns_the_axis_by_exactly_the_angle_asked(degrees):
    before = cabinet().articulations[0].axis
    after = apply(cabinet(), Fault("B10", "tilt", "hinge", "axis rotated", degrees))
    assert angle_between(before, after.articulations[0].axis) == pytest.approx(degrees, abs=1e-6)


def test_tilt_preserves_the_axis_length():
    for axis in ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.6, 0.8, 0.0)):
        built = asset(
            parts=[opaque("body"), opaque("door")],
            joints=[joint("hinge", "revolute", "body", "door", axis=axis)],
        )
        after = apply(built, Fault("B10", "tilt", "hinge", "axis rotated", 45.0))
        assert math.dist((0.0, 0.0, 0.0), after.articulations[0].axis) == pytest.approx(
            math.dist((0.0, 0.0, 0.0), axis)
        )


def test_detach_moves_the_geometry_not_the_joint():
    """The joint origin is the child frame's origin, so moving it takes the link
    with it and changes nothing. The geometry inside the frame has to move."""
    broken = apply(cabinet(), Fault("B10", "detach", "hinge", "link slid off", 1.0))
    assert broken.articulations[0].origin.xyz == (0.0, 0.0, 0.0), "the joint stayed put"
    box = broken.parts["door"].aabb()
    assert box is not None and box[0][0] > 0.0, "the door is now clear of its own origin"


def test_detach_distance_scales_with_the_link():
    start = cabinet().parts["door"].aabb()[0][0]
    small = apply(cabinet(), Fault("B10", "detach", "hinge", "link slid off", 0.5))
    large = apply(cabinet(), Fault("B10", "detach", "hinge", "link slid off", 1.0))
    moved_small = small.parts["door"].aabb()[0][0] - start
    moved_large = large.parts["door"].aabb()[0][0] - start
    assert moved_large == pytest.approx(2 * moved_small)
    assert moved_large == pytest.approx(math.dist(*cabinet().parts["door"].aabb()))


def test_a_fault_actually_flips_the_detector_it_targets():
    """The point of the whole module: a manufactured positive must read as one."""
    built = cabinet()
    asked = contract(ExpectedJoint(child="door", kind="revolute", source=Source.PRIOR))
    assert check_b8(built, asked).prediction is Prediction.PASS

    welded = apply(built, Fault("B8", "weld", "hinge", "declared fixed"))
    assert check_b8(welded, asked).prediction is Prediction.FAIL

    retyped = apply(built, Fault("B9", "retype", "hinge", "prismatic"))
    assert check_b9(retyped, asked).prediction is Prediction.FAIL


def test_the_fault_list_is_stable_and_covers_every_item():
    built = asset(
        parts=[part("body"), part("door", size=(0.1, 0.5, 0.8)), part("drawer")],
        joints=[joint("hinge", "revolute", "body", "door")],
    )
    first, second = faults_for(built), faults_for(built)
    assert first == second
    assert {fault.item for fault in first} == {"B7", "B8", "B9", "B10"}


def test_reparenting_needs_somewhere_else_to_attach():
    """Two parts leave nowhere wrong to hang the child, so no B7 fault exists."""
    assert not [fault for fault in faults_for(cabinet()) if fault.item == "B7"]


def test_no_fault_is_offered_for_a_joint_that_cannot_carry_it():
    """A link with no analytic bounds cannot be detached by a known amount."""
    built = asset(
        parts=[opaque("body"), opaque("door")],
        joints=[joint("hinge", "revolute", "body", "door")],
    )
    assert not [fault for fault in faults_for(built) if fault.kind == "detach"]
