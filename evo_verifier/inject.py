"""Break a working asset on purpose, in a known way, by a known amount.

B7-B10 have 8, 16, 13 and 10 human failures in the whole annotated set. On that
many positives a per-item F1 moves six points when one case flips, and a
threshold fitted to them is fitted to noise.

So the positives used for tuning are manufactured instead. Take an asset the
humans passed, reparent one joint, or weld it shut, or change its type, or slide
its origin 5% of the moving link away, or tilt its axis 15 degrees -- each one a
defect whose correct verdict is known by construction, and the last two with a
dial on them. That gives a sensitivity curve: how far does a joint have to move
before the detector notices? A threshold picked off that curve has a reason
behind it, and the 47 real failures stay untouched, held back for a single
end-of-line check.

Every injection is deterministic. Same asset and same fault, same result, so a
calibration run can be replayed exactly.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace

from evo_verifier.asset import Articulation, Asset, Origin, Part, Vec3

SHIFTS: tuple[float, ...] = (0.02, 0.05, 0.10, 0.25, 0.50, 1.00)
"""How far a link's geometry slides off its own frame, in link diagonals."""

ANGLES: tuple[float, ...] = (5.0, 15.0, 30.0, 45.0, 90.0)
"""Axis tilts, in degrees."""

RETYPE: dict[str, str] = {
    "revolute": "prismatic",
    "continuous": "prismatic",
    "prismatic": "revolute",
}
"""A wrong type that is wrong in kind, not merely in whether the turn stops."""


@dataclass(frozen=True)
class Fault:
    """One deliberate defect, and the item it should make fail."""

    item: str
    kind: str
    """``reparent``, ``unjoint``, ``weld``, ``retype``, ``detach``, ``tilt``."""
    joint: str
    detail: str
    intensity: float = 1.0
    """Magnitude, where the fault has one. Fractions of a link, or degrees."""

    @property
    def label(self) -> str:
        return f"{self.item}:{self.kind}@{self.intensity:g}"


def apply(asset: Asset, fault: Fault) -> Asset:
    """A copy of the asset with the fault in it. The original is untouched."""
    joints = [
        joint if joint.name != fault.joint else _break(joint, fault)
        for joint in asset.articulations
    ]
    parts = dict(asset.parts)
    if fault.kind == "detach":
        target = next(
            (joint.child for joint in asset.articulations if joint.name == fault.joint), None
        )
        if target and target in parts:
            parts[target] = _moved(parts[target], fault.intensity)
    return Asset(
        record_id=f"{asset.record_id}#{fault.label}",
        robot_name=asset.robot_name,
        source=asset.source,
        parts=parts,
        articulations=[joint for joint in joints if joint is not None],
        notes=list(asset.notes),
    )


def _break(joint: Articulation, fault: Fault) -> Articulation | None:
    if fault.kind == "unjoint":
        return None
    if fault.kind == "weld":
        return replace(joint, kind="fixed", axis=None, limits=None)
    if fault.kind == "reparent":
        return replace(joint, parent=fault.detail)
    if fault.kind == "retype":
        return replace(joint, kind=fault.detail)
    if fault.kind == "tilt":
        return replace(joint, axis=_tilted(joint.axis, fault.intensity))
    if fault.kind == "detach":
        return joint  # the part moves, not the joint -- see _moved
    raise ValueError(f"unknown fault kind {fault.kind!r}")


def _moved(part: Part, fraction: float) -> Part:
    """Slide a link's geometry away from its own frame origin.

    This is how a joint comes off the part it moves, and the only way to
    simulate it: the joint origin *is* the child frame's origin, so translating
    the joint carries the whole link along with it and changes nothing about
    their relationship. What has to move is the geometry inside the frame.
    """
    box = part.aabb()
    if box is None:
        return part
    distance = fraction * math.dist(box[0], box[1])
    shapes = tuple(
        replace(
            shape,
            origin=Origin(
                xyz=None
                if shape.origin.xyz is None
                else (
                    shape.origin.xyz[0] + distance,
                    shape.origin.xyz[1],
                    shape.origin.xyz[2],
                ),
                rpy=shape.origin.rpy,
            ),
        )
        for shape in part.shapes
    )
    return replace(part, shapes=shapes)


def _tilted(axis: Vec3 | None, degrees: float) -> Vec3 | None:
    """Rotate the axis by ``degrees`` about a perpendicular chosen deterministically."""
    if axis is None:
        return None
    length = math.dist((0.0, 0.0, 0.0), axis)
    if length <= 0:
        return axis
    unit = (axis[0] / length, axis[1] / length, axis[2] / length)
    perpendicular = _perpendicular(unit)
    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    # Rodrigues, with unit perpendicular to unit axis so the cross term is the
    # rotation plane and the dot term vanishes.
    cross = (
        perpendicular[1] * unit[2] - perpendicular[2] * unit[1],
        perpendicular[2] * unit[0] - perpendicular[0] * unit[2],
        perpendicular[0] * unit[1] - perpendicular[1] * unit[0],
    )
    turned = tuple(unit[i] * cos + cross[i] * sin for i in range(3))
    return (
        round(turned[0] * length, 9),
        round(turned[1] * length, 9),
        round(turned[2] * length, 9),
    )


def _perpendicular(unit: Vec3) -> Vec3:
    """A unit vector at right angles to this one. Same input, same output."""
    basis = (0.0, 0.0, 1.0) if abs(unit[2]) < 0.9 else (1.0, 0.0, 0.0)
    cross = (
        unit[1] * basis[2] - unit[2] * basis[1],
        unit[2] * basis[0] - unit[0] * basis[2],
        unit[0] * basis[1] - unit[1] * basis[0],
    )
    length = math.dist((0.0, 0.0, 0.0), cross)
    return (cross[0] / length, cross[1] / length, cross[2] / length)


def faults_for(asset: Asset, *, items: Sequence[str] = ("B7", "B8", "B9", "B10")) -> list[Fault]:
    """Every fault this asset can carry, deterministic and in a stable order."""
    return list(_enumerate(asset, items))


def _enumerate(asset: Asset, items: Sequence[str]) -> Iterator[Fault]:
    movable = [joint for joint in asset.movable() if joint.name != "<unnamed>"]
    for joint in movable:
        if "B7" in items and joint.parent:
            for other in sorted(asset.parts):
                if other not in (joint.parent, joint.child):
                    yield Fault("B7", "reparent", joint.name, other)
                    break  # one wrong parent is enough; more is the same defect
        if "B8" in items:
            yield Fault("B8", "unjoint", joint.name, "joint removed")
            yield Fault("B8", "weld", joint.name, "declared fixed")
        if "B9" in items and joint.kind in RETYPE:
            yield Fault("B9", "retype", joint.name, RETYPE[joint.kind])
        if "B10" in items:
            link = asset.parts.get(joint.child or "")
            if link is not None and link.geometry_complete and joint.origin.xyz is not None:
                for shift in SHIFTS:
                    yield Fault("B10", "detach", joint.name, "link slid off its joint", shift)
            if joint.axis is not None:
                for angle in ANGLES:
                    yield Fault("B10", "tilt", joint.name, "axis rotated", angle)
