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
    weight: float = 1.0
    """1.0 for a quoted requirement, the prior's own confidence otherwise."""
    explicit: bool = True

    @property
    def satisfied(self) -> int:
        return min(len(self.moving), self.wanted)

    @property
    def resolved(self) -> bool:
        """The name found something in the asset, so the verdict is about the asset."""
        return bool(self.parts)


def check_b8(asset: Asset, contract: Contract, *, use_priors: bool = True) -> ItemResult:
    """Does every part that should move have a non-fixed joint?

    ``S_B8 = satisfied / expected``, weighted: a quoted requirement counts fully,
    a category prior counts at its own confidence.

    Priors are in the sum because the humans put them there. Asked why they
    failed B8 on assets whose every stated motion had a joint, the annotator
    said the parts a real object needs were never modelled at all -- a quad bike
    with no suspension. That judgement is not in the prompt, so a contract
    holding only quoted requirements cannot reach it. ``use_priors=False`` scores
    the prompt-grounded subset instead, and the difference between the two is
    itself a result worth reporting: it measures how much of the human standard
    the prompt actually carries.

    An expected part with no counterpart in the asset counts as unmet -- a part
    that is not there certainly has no joint -- but a name that never resolved
    might be the matcher's fault rather than the asset's. That doubt lands on the
    confidence, and enough of it makes the item abstain rather than accuse.
    """
    wanted = [
        joint
        for joint in contract.joints
        if joint.kind != "fixed" and (use_priors or joint.source is Source.EXPLICIT)
    ]
    if not wanted:
        return ItemResult.not_applicable(
            "B8", "nothing in the contract is required to move", tools=["contract"]
        )

    moved = {joint.child for joint in asset.movable() if joint.child}
    requirements = [_requirement(joint, asset, moved) for joint in wanted]
    score = _weighted(requirements)

    unresolved = [r for r in requirements if not r.resolved]
    still = [r for r in requirements if r.resolved and r.satisfied < r.wanted]

    quality = 1.0 - len(unresolved) / len(requirements)
    stated = sum(r.weight for r in requirements) / len(requirements)
    margin = min(1.0, abs(score - TAU_B8) / MARGIN)

    return ItemResult.scored(
        "B8",
        round(score, 4),
        threshold=TAU_B8,
        confidence=round(quality * stated * margin, 4),
        coverage=Coverage.FULL if not unresolved else Coverage.PARTIAL,
        tools=["contract", "parser", "matching"],
        raw_measurements={
            "expected": round(sum(r.weight * r.wanted for r in requirements), 4),
            "satisfied": round(sum(r.weight * r.satisfied for r in requirements), 4),
            "score_explicit_only": round(_weighted([r for r in requirements if r.explicit]), 4),
            "used_priors": use_priors,
            "requirements": [
                {
                    "child": r.child,
                    "wanted": r.wanted,
                    "source": "explicit" if r.explicit else "prior",
                    "matched_parts": list(r.parts),
                    "moving_parts": list(r.moving),
                }
                for r in requirements
            ],
            "unresolved_names": [r.child for r in unresolved],
        },
        failure_reason=_reason(still, unresolved),
        repair_hint=_repair(still, unresolved, contract),
    )


def _weighted(requirements: list[Requirement]) -> float:
    expected = sum(r.weight * r.wanted for r in requirements)
    if not expected:
        return 0.0
    return sum(r.weight * r.satisfied for r in requirements) / expected


def _requirement(joint: ExpectedJoint, asset: Asset, moved: set[str]) -> Requirement:
    found = tuple(match.name for match in matches(joint.child, asset.parts))
    explicit = joint.source is Source.EXPLICIT
    return Requirement(
        child=joint.child,
        wanted=max(1, joint.count or 1),
        parts=found,
        moving=tuple(name for name in found if name in moved),
        weight=1.0 if explicit else min(joint.confidence, 0.7),
        explicit=explicit,
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


def _repair(still: list[Requirement], unresolved: list[Requirement], contract: Contract) -> str:
    attachment = {joint.child: joint.parent for joint in contract.joints}
    hints = []
    for requirement in (*still, *unresolved):
        target = attachment.get(requirement.child) or "its supporting part"
        verb = "add" if requirement.resolved else "model"
        hints.append(f"{verb} {requirement.child} with a non-fixed joint to {target}")
    return "; ".join(hints)
