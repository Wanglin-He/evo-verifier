"""B7-B10: what the declared graph says, checked against what the prompt asked.

None of these four need a simulator. Parent, child, joint type, origin and axis
are structural facts, and the asset declares all of them; the only question is
whether they are the ones the prompt called for.

B8 is here first because it is the most direct: every part the prompt says moves
should carry a joint that moves it.
"""

from __future__ import annotations

from dataclasses import dataclass

from evo_verifier.asset import Asset
from evo_verifier.contract import Contract, ExpectedJoint, Source
from evo_verifier.matching import matches
from evo_verifier.report import Coverage, ItemResult

TAU_B8 = 0.70
"""Protocol placeholder. Calibrate on the dev set, then freeze."""

MARGIN = 0.20
"""Score distance at which the verdict is considered clear of the threshold."""


@dataclass(frozen=True)
class Requirement:
    """One expected motion, and what the asset does about it."""

    child: str
    wanted: int
    """How many parts of this kind should move. The contract's count, or one."""
    parts: tuple[str, ...]
    """Asset parts the name matched."""
    moving: tuple[str, ...]
    """Of those, the ones a non-fixed joint actually moves."""

    @property
    def satisfied(self) -> int:
        return min(len(self.moving), self.wanted)

    @property
    def resolved(self) -> bool:
        """The name found something in the asset, so the verdict is about the asset."""
        return bool(self.parts)


def check_b8(asset: Asset, contract: Contract) -> ItemResult:
    """Does every part the prompt says moves have a non-fixed joint?

    ``S_B8 = matched / expected``.

    An expected part with no counterpart in the asset counts as unmet -- a part
    that is not there certainly has no joint -- but it also means the name never
    resolved, and a name that never resolved might be the matcher's fault rather
    than the asset's. That doubt lands on the confidence, and enough of it makes
    the item abstain instead of accusing the asset.
    """
    wanted = [joint for joint in contract.explicit_joints() if joint.kind != "fixed"]
    if not wanted:
        return ItemResult.not_applicable(
            "B8", "the prompt names no moving part", tools=["contract"]
        )

    moved = {joint.child for joint in asset.movable() if joint.child}
    requirements = [_requirement(joint, asset, moved) for joint in wanted]

    expected = sum(requirement.wanted for requirement in requirements)
    satisfied = sum(requirement.satisfied for requirement in requirements)
    score = satisfied / expected if expected else 0.0

    unresolved = [r for r in requirements if not r.resolved]
    still = [r for r in requirements if r.resolved and r.satisfied < r.wanted]

    quality = 1.0 - len(unresolved) / len(requirements)
    stated = [j.confidence for j in wanted]
    margin = min(1.0, abs(score - TAU_B8) / MARGIN)
    confidence = round(quality * (sum(stated) / len(stated)) * margin, 4)

    return ItemResult.scored(
        "B8",
        round(score, 4),
        threshold=TAU_B8,
        confidence=confidence,
        coverage=Coverage.FULL if not unresolved else Coverage.PARTIAL,
        tools=["contract", "parser", "matching"],
        raw_measurements={
            "expected": expected,
            "satisfied": satisfied,
            "requirements": [
                {
                    "child": r.child,
                    "wanted": r.wanted,
                    "matched_parts": list(r.parts),
                    "moving_parts": list(r.moving),
                }
                for r in requirements
            ],
            "unresolved_names": [r.child for r in unresolved],
            "priors_ignored": [
                joint.child for joint in contract.joints if joint.source is not Source.EXPLICIT
            ],
        },
        failure_reason=_reason(still, unresolved),
        repair_hint=_repair(still, unresolved, contract),
    )


def _requirement(joint: ExpectedJoint, asset: Asset, moved: set[str]) -> Requirement:
    found = tuple(match.name for match in matches(joint.child, asset.parts))
    return Requirement(
        child=joint.child,
        wanted=max(1, joint.count or 1),
        parts=found,
        moving=tuple(name for name in found if name in moved),
    )


def _reason(still: list[Requirement], unresolved: list[Requirement]) -> str:
    parts = []
    for requirement in still:
        fixed = [name for name in requirement.parts if name not in requirement.moving]
        parts.append(
            f"{requirement.child}: {len(requirement.moving)}/{requirement.wanted} moving"
            + (f" ({', '.join(fixed)} has no joint)" if fixed else "")
        )
    if unresolved:
        parts.append("no part found for " + ", ".join(r.child for r in unresolved))
    return "; ".join(parts)


def _repair(
    still: list[Requirement], unresolved: list[Requirement], contract: Contract
) -> str:
    attachment = {joint.child: joint.parent for joint in contract.joints}
    hints = []
    for requirement in (*still, *unresolved):
        target = attachment.get(requirement.child) or "its supporting part"
        verb = "add" if requirement.resolved else "model"
        hints.append(f"{verb} {requirement.child} with a non-fixed joint to {target}")
    return "; ".join(hints)
