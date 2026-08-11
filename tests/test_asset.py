from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from evo_verifier.asset import find_model_py, load_asset, read_model_source

HEADER = """
from __future__ import annotations
from sdk import ArticulatedObject, ArticulationType, Box, Cylinder, MotionLimits, Origin
"""


def build(body: str, constants: str = "") -> str:
    return f"{HEADER}\n{constants}\n\ndef build_object_model() -> ArticulatedObject:\n{body}\n"


def test_reads_a_joint_through_module_constants():
    asset = read_model_source(
        build(
            """
    model = ArticulatedObject(name="cabinet")
    body = model.part("cabinet_body")
    body.visual(Box((0.60, 0.40, 0.80)))
    door = model.part("door")
    door.visual(Box((0.02, 0.38, 0.60)))
    model.articulation(
        "door_hinge",
        ArticulationType.REVOLUTE,
        parent=body,
        child=door,
        origin=Origin(xyz=(WIDTH / 2.0, -0.19, 0.0)),
        axis=(0.0, 0.0, 1.0),
        motion_limits=MotionLimits(lower=0.0, upper=1.57, effort=20.0, velocity=1.0),
    )
    return model
""",
            constants="WIDTH = 0.62",
        )
    )
    assert asset.robot_name == "cabinet"
    assert list(asset.parts) == ["cabinet_body", "door"]
    (joint,) = asset.articulations
    assert (joint.name, joint.kind) == ("door_hinge", "revolute")
    assert (joint.parent, joint.child) == ("cabinet_body", "door")
    assert joint.origin.xyz == (0.31, -0.19, 0.0)
    assert joint.axis == (0.0, 0.0, 1.0)
    assert joint.limits is not None and joint.limits.travel == pytest.approx(1.57)
    assert asset.complete


def test_unrolls_loops_and_resolves_f_string_names():
    asset = read_model_source(
        build(
            """
    model = ArticulatedObject(name="panel")
    base = model.part("base")
    for index, x in enumerate(BUTTON_XS):
        button = model.part(f"button_{index}")
        button.visual(Box((0.02, 0.02, 0.01)))
        model.articulation(
            f"button_{index}_press",
            ArticulationType.PRISMATIC,
            parent=base,
            child=button,
            origin=Origin(xyz=(x, 0.0, 0.0)),
            axis=(0.0, 0.0, 1.0),
            motion_limits=MotionLimits(lower=0.0, upper=0.004),
        )
    return model
""",
            constants="BUTTON_XS = (0.334, 0.366)",
        )
    )
    assert [joint.name for joint in asset.articulations] == ["button_0_press", "button_1_press"]
    assert [joint.child for joint in asset.articulations] == ["button_0", "button_1"]
    assert [joint.origin.xyz[0] for joint in asset.articulations] == [0.334, 0.366]  # pyright: ignore[reportOptionalSubscript]


def test_a_fixed_joint_needs_no_axis_and_does_not_move():
    asset = read_model_source(
        build("""
    model = ArticulatedObject(name="lamp")
    base = model.part("base")
    post = model.part("post")
    model.articulation("weld", ArticulationType.FIXED, parent=base, child=post, origin=Origin())
    return model
""")
    )
    (joint,) = asset.articulations
    assert joint.axis is None
    assert not joint.is_movable
    assert asset.movable() == []
    assert asset.complete, "a fixed joint without an axis is not a gap"


def test_an_unresolved_origin_is_recorded_not_guessed():
    """The whole point: a value we cannot read must not become a B10 failure."""
    asset = read_model_source(
        build("""
    model = ArticulatedObject(name="thing")
    base = model.part("base")
    arm = model.part("arm")
    model.articulation(
        "elbow",
        ArticulationType.REVOLUTE,
        parent=base,
        child=arm,
        origin=Origin(xyz=_computed_anchor()),
        axis=(1.0, 0.0, 0.0),
    )
    return model
""")
    )
    (joint,) = asset.articulations
    assert joint.origin.xyz is None
    assert joint.kind == "revolute"
    assert not asset.complete
    assert any("origin unresolved" in note for note in asset.notes)


def test_a_branch_the_reader_cannot_fold_is_not_taken():
    asset = read_model_source(
        build("""
    model = ArticulatedObject(name="thing")
    base = model.part("base")
    if _wants_handle():
        handle = model.part("handle")
    return model
""")
    )
    assert "handle" not in asset.parts
    assert any("branch not taken" in note for note in asset.notes)


def test_a_folded_branch_is_taken():
    asset = read_model_source(
        build(
            """
    model = ArticulatedObject(name="thing")
    base = model.part("base")
    if LEGS > 2:
        model.part("rear_leg")
    else:
        model.part("skid")
    return model
""",
            constants="LEGS = 4",
        )
    )
    assert "rear_leg" in asset.parts
    assert "skid" not in asset.parts


def test_primitive_bounds_give_the_diagonal():
    asset = read_model_source(
        build("""
    model = ArticulatedObject(name="block")
    body = model.part("body")
    body.visual(Box((2.0, 4.0, 4.0)))
    return model
""")
    )
    part = asset.parts["body"]
    assert part.geometry_complete
    assert part.aabb() == ((-1.0, -2.0, -2.0), (1.0, 2.0, 2.0))
    assert asset.diagonal() == pytest.approx(6.0)


def test_a_cad_mesh_leaves_the_diagonal_unknown():
    """86% of visual elements are primitives; the rest need the CAD run."""
    asset = read_model_source(
        build("""
    model = ArticulatedObject(name="shell")
    body = model.part("body")
    body.visual(mesh_from_cadquery(_build_shell(), "shell"))
    return model
""")
    )
    part = asset.parts["body"]
    assert part.shapes[0].kind == "mesh"
    assert part.aabb() is None
    assert not part.geometry_complete
    assert asset.diagonal() is None


def test_a_rotated_shape_reports_no_bounds_rather_than_wrong_ones():
    asset = read_model_source(
        build("""
    model = ArticulatedObject(name="block")
    body = model.part("body")
    body.visual(Box((1.0, 1.0, 1.0)), origin=Origin(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.7854, 0.0)))
    return model
""")
    )
    assert asset.parts["body"].aabb() is None


def test_cylinder_bounds():
    asset = read_model_source(
        build("""
    model = ArticulatedObject(name="rod")
    body = model.part("body")
    body.visual(Cylinder(0.5, 2.0))
    return model
""")
    )
    assert asset.parts["body"].aabb() == ((-0.5, -0.5, -1.0), (0.5, 0.5, 1.0))


def test_graph_helpers():
    asset = read_model_source(
        build("""
    model = ArticulatedObject(name="cabinet")
    body = model.part("cabinet_body")
    door = model.part("door")
    drawer = model.part("drawer")
    model.articulation("hinge", ArticulationType.REVOLUTE, parent=body, child=door,
                       origin=Origin(), axis=(0.0, 0.0, 1.0))
    model.articulation("slide", ArticulationType.PRISMATIC, parent=body, child=drawer,
                       origin=Origin(), axis=(0.0, 1.0, 0.0))
    return model
""")
    )
    assert asset.roots() == ["cabinet_body"]
    assert sorted(asset.children_of("cabinet_body")) == ["door", "drawer"]
    joint = asset.joint_for_child("drawer")
    assert joint is not None and joint.kind == "prismatic"
    assert len(asset.movable()) == 2


def test_math_and_radians_fold():
    asset = read_model_source(
        build("""
    model = ArticulatedObject(name="dial")
    base = model.part("base")
    dial = model.part("dial")
    model.articulation("spin", ArticulationType.CONTINUOUS, parent=base, child=dial,
                       origin=Origin(xyz=(0.0, 0.0, math.pi / 4.0)), axis=(0.0, 0.0, 1.0),
                       motion_limits=MotionLimits(lower=0.0, upper=math.radians(90.0)))
    return model
""")
    )
    (joint,) = asset.articulations
    assert joint.origin.xyz is not None
    assert joint.origin.xyz[2] == pytest.approx(math.pi / 4)
    assert joint.limits is not None and joint.limits.travel == pytest.approx(math.pi / 2)


def test_a_file_without_a_builder_says_so():
    asset = read_model_source("x = 1\n")
    assert asset.notes == ["no build_object_model function"]
    assert not asset.parts


DATA_DIR = os.environ.get("EVO_ARTICRAFT_DATA", "")


@pytest.mark.skipif(not DATA_DIR, reason="set EVO_ARTICRAFT_DATA to the articraft-data clone")
def test_reads_a_real_record():
    record_id = "rec_air_conditioning_machine_7e491607358345d4af18dd84accf03d3"
    asset = load_asset(find_model_py(Path(DATA_DIR), record_id), record_id)
    assert asset.robot_name == "ceiling_cassette_air_conditioner"
    assert len(asset.parts) == 7
    assert len(asset.movable()) == 6
    assert asset.roots() == ["cassette"]
    assert asset.complete
