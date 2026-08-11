"""B7-B10: what the declared graph says, checked against what the prompt asked.

None of these four need a simulator. Parent, child, joint type, origin and axis
are structural facts, and the asset declares all of them; the only question is
whether they are the ones the prompt called for.

B8 is here first because it is the most direct: every part the prompt says moves
should carry a joint that moves it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from evo_verifier.asset import Articulation, Asset
from evo_verifier.contract import Contract, ExpectedJoint, Source
from evo_verifier.matching import STRONG, matches, similarity
from evo_verifier.report import Coverage, ItemResult

TAU_B8 = 0.70
"""Protocol placeholder. Calibrate on the dev set, then freeze."""

MARGIN = 0.20
"""Score distance at which a verdict is considered clear of the threshold.

Reported, not used to abstain. The protocol multiplies confidence by the margin,
which suits a continuous score; these scores are coarse -- a handful of weighted
requirements per asset -- and the commonest failure pattern lands exactly on
0.70, where the margin is zero and every such asset abstains. Confidence here
measures whether the evidence was readable, which is what abstention should be
about. A borderline score is a decision, not an absence of one.
"""


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


def check_b8(
    asset: Asset,
    contract: Contract,
    *,
    use_priors: bool = True,
    missing_is_failure: bool = True,
) -> ItemResult:
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

    unresolved = [r for r in requirements if not r.resolved]
    still = [r for r in requirements if r.resolved and r.satisfied < r.wanted]
    scored = requirements if missing_is_failure else [r for r in requirements if r.resolved]
    if not scored:
        return ItemResult.unsupported(
            "B8", "no expected part was found in the asset", tools=["contract", "matching"]
        )
    score = _weighted(scored)

    quality = 1.0 - len(unresolved) / len(requirements)
    stated = sum(r.weight for r in requirements) / len(requirements)

    return ItemResult.scored(
        "B8",
        round(score, 4),
        threshold=TAU_B8,
        confidence=round(quality * stated, 4),
        coverage=Coverage.FULL if not unresolved else Coverage.PARTIAL,
        tools=["contract", "parser", "matching"],
        raw_measurements={
            "expected": round(sum(r.weight * r.wanted for r in scored), 4),
            "satisfied": round(sum(r.weight * r.satisfied for r in scored), 4),
            "missing_is_failure": missing_is_failure,
            "margin": round(min(1.0, abs(score - TAU_B8) / MARGIN), 4),
            "score_explicit_only": round(_weighted([r for r in scored if r.explicit]), 4),
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


TAU_B7 = 0.70

EXTERNAL = frozenset(
    {
        "ground",
        "floor",
        "wall",
        "ceiling",
        "world",
        "bench",
        "workbench",
        "table",
        "desk",
        "surface",
        "storefront",
    }
)
"""Things a prompt attaches an object to that are not part of the object. A vise
clamps to a bench; the bench is not a link, so its absence is not a defect."""


@dataclass(frozen=True)
class Connection:
    """One expected attachment, and what the asset actually connected."""

    child: str
    expected_parent: str
    part: str | None
    """Asset part the moving name matched."""
    actual_parent: str | None
    parent_ok: bool | None
    """``None`` when the prompt named no parent, or named something external."""
    parent_exists: bool
    """Whether the expected parent is modelled at all."""
    intruders: tuple[str, ...] = ()
    """Parts that ride on this joint but were declared to attach elsewhere."""


def check_b7(asset: Asset, contract: Contract) -> ItemResult:
    """Does each joint connect the parts the prompt said it connects?

    ``S_B7 = 0.3 * child + 0.4 * parent + 0.3 * subtree purity``, renormalised
    over the terms that apply.

    The protocol weights these 0.4 / 0.3 / 0.3, putting the child first. That
    cannot be right for this item: with a 0.70 threshold, an asset that hangs a
    part off entirely the wrong thing scores exactly 0.70 and passes. The item is
    named for the connection object, so the parent carries the most weight here,
    which is enough for one wrong connection to fail. A deviation from the
    written formula, recorded rather than hidden.

    The connection object is the whole item. Every B7 failure in the annotated
    set is one: a juicer pusher hung off the lid instead of the chute, so it
    swings away when the lid opens; a revolving door whose wings turn against the
    storefront because the central post was never modelled; a spray bottle whose
    pump rod reaches the head instead of the trigger it is supposed to be driven
    by. Each of those has a joint, of the right type, on the right part. Only the
    other end is wrong.

    Subtree purity is what catches the juicer: the pusher's own joint looks
    correct until you notice it is riding on the lid's subtree.
    """
    stated = [joint for joint in contract.joints if joint.child]
    usable = [joint for joint in stated if joint.parent]
    if not usable:
        return ItemResult.unsupported(
            "B7", "the prompt never says what anything attaches to", tools=["contract"]
        )

    by_child = {joint.child: joint for joint in asset.articulations if joint.child}
    connections = [_connection(joint, asset, by_child, stated) for joint in usable]

    child_scores = [1.0 if c.part and c.actual_parent is not None else 0.0 for c in connections]
    parent_scores = [float(c.parent_ok) for c in connections if c.parent_ok is not None]
    purity = [
        1.0 if not c.intruders else 0.0
        for c in connections
        if c.part and c.actual_parent is not None
    ]

    terms = [(0.3, child_scores), (0.4, parent_scores), (0.3, purity)]
    live = [(weight, values) for weight, values in terms if values]
    total = sum(weight for weight, _ in live)
    score = sum(weight * sum(values) / len(values) for weight, values in live) / total

    unresolved = [c.child for c in connections if not c.part]
    unjointed = [c for c in connections if c.part and c.actual_parent is None]
    wrong = [c for c in connections if c.parent_ok is False or c.intruders]
    quality = 1.0 - len(unresolved) / len(connections)

    return ItemResult.scored(
        "B7",
        round(score, 4),
        threshold=TAU_B7,
        confidence=round(quality, 4),
        coverage=Coverage.FULL if not unresolved else Coverage.PARTIAL,
        tools=["contract", "parser", "matching", "graph"],
        raw_measurements={
            "child_score": round(sum(child_scores) / len(child_scores), 4),
            "parent_score": (
                round(sum(parent_scores) / len(parent_scores), 4) if parent_scores else None
            ),
            "subtree_score": round(sum(purity) / len(purity), 4) if purity else None,
            "connections": [
                {
                    "child": c.child,
                    "part": c.part,
                    "expected_parent": c.expected_parent,
                    "actual_parent": c.actual_parent,
                    "parent_ok": c.parent_ok,
                    "expected_parent_modelled": c.parent_exists,
                    "rides_along": list(c.intruders),
                }
                for c in connections
            ],
            "unresolved_names": unresolved,
            "margin": round(min(1.0, abs(score - TAU_B7) / MARGIN), 4),
        },
        failure_reason="; ".join(
            [_connection_reason(c) for c in wrong]
            + [f"{c.part} has no joint connecting it to anything" for c in unjointed]
            + ([f"no part found for {', '.join(unresolved)}"] if unresolved else [])
        ),
        repair_hint="; ".join(
            f"reconnect {c.part or c.child} to {c.expected_parent}"
            + ("" if c.parent_exists else ", which the asset does not model yet")
            for c in wrong
            if c.parent_ok is False
        ),
    )


def _connection(
    expected: ExpectedJoint,
    asset: Asset,
    by_child: dict[str, Articulation],
    stated: list[ExpectedJoint],
) -> Connection:
    part = next(
        (match.name for match in matches(expected.child, asset.parts) if match.name in by_child),
        None,
    ) or next((match.name for match in matches(expected.child, asset.parts)), None)
    joint = by_child.get(part or "")
    actual_parent = joint.parent if joint else None

    external = tokens_are_external(expected.parent)
    parent_exists = bool(matches(expected.parent, asset.parts))
    roots = asset.roots()
    if external or actual_parent is None:
        parent_ok = None
    elif similarity(expected.parent, actual_parent) >= STRONG:
        parent_ok = True
    elif not parent_exists and actual_parent in roots:
        # The prompt names the base by one of its aspects -- "the panel edges",
        # "the frame", "the top deck" -- and the asset calls the whole base
        # something else. Attaching to the root is what those phrases mean, and
        # no other part answers to the name, so this is a naming gap rather than
        # a wrong connection. Unknown, not wrong.
        parent_ok = None
    else:
        parent_ok = False

    return Connection(
        child=expected.child,
        expected_parent=expected.parent,
        part=part,
        actual_parent=actual_parent,
        parent_ok=parent_ok,
        parent_exists=parent_exists or external,
        intruders=_intruders(part, asset, stated, expected),
    )


def tokens_are_external(name: str) -> bool:
    words = set(name.lower().replace("_", " ").split())
    return bool(words & EXTERNAL)


def _intruders(
    part: str | None, asset: Asset, stated: list[ExpectedJoint], expected: ExpectedJoint
) -> tuple[str, ...]:
    """Other declared-independent parts that ride on this one's subtree."""
    if not part:
        return ()
    riding = asset.descendants(part)
    if not riding:
        return ()
    found: list[str] = []
    for other in stated:
        if other is expected or not other.parent:
            continue
        if similarity(other.parent, expected.child) >= STRONG:
            continue  # the prompt does attach it here
        if similarity(other.parent, part) >= STRONG:
            continue
        found.extend(match.name for match in matches(other.child, riding))
    return tuple(dict.fromkeys(found))


def _connection_reason(connection: Connection) -> str:
    if connection.parent_ok is False:
        missing = "" if connection.parent_exists else ", which is not modelled"
        return (
            f"{connection.part or connection.child} attaches to "
            f"{connection.actual_parent}, expected {connection.expected_parent}{missing}"
        )
    return f"{connection.part} carries {', '.join(connection.intruders)}, declared elsewhere"


TAU_B9 = 0.70

TYPE_AGREEMENT: dict[tuple[str, str], float] = {
    ("revolute", "continuous"): 0.5,
    ("continuous", "revolute"): 0.5,
}
"""Near misses. Both turn about an axis and differ only in whether it stops,
which is a different mistake from turning where it should slide. Still below
the threshold -- a knob that should spin freely but hits a stop is wrong."""


@dataclass(frozen=True)
class TypeCheck:
    """One expected joint compared with the one the asset declares."""

    child: str
    expected: str
    declared: str | None
    joint: str
    agreement: float
    consistency: float
    contradiction: str = ""

    @property
    def score(self) -> float:
        return 0.6 * self.agreement + 0.4 * self.consistency


def check_b9(asset: Asset, contract: Contract) -> ItemResult:
    """Is each joint the kind of joint the prompt asked for?

    ``S_B9 = 0.6 * declared type + 0.4 * self-consistency``.

    The protocol's second term is a trajectory classifier -- run FK, watch the
    child move, decide whether that was a rotation or a translation. Against a
    declared graph with no meshes that term is circular: FK replays exactly what
    the joint type says, so it can only ever agree. What is genuinely independent
    is whether the declaration contradicts itself, and it often does: a
    ``continuous`` joint carrying finite limits is not the free spin the word
    promises, and a ``revolute`` with no limits is a continuous joint wearing the
    wrong name. That is what the 0.4 measures here, and it is a deviation from
    the written formula -- recorded, not hidden.

    A part the asset never modelled is B8's finding, not a wrong type, so it
    leaves this score and costs confidence instead.
    """
    wanted = [joint for joint in contract.joints if joint.kind]
    if not wanted:
        return ItemResult.not_applicable("B9", "the prompt names no joint type", tools=["contract"])

    by_child = {joint.child: joint for joint in asset.articulations if joint.child}
    checks, unresolved = [], []
    for expected in wanted:
        found = [match.name for match in matches(expected.child, asset.parts)]
        declared = next((by_child[name] for name in found if name in by_child), None)
        if declared is None:
            unresolved.append(expected.child)
            continue
        checks.append(_type_check(expected.child, expected.kind or "", declared))

    if not checks:
        return ItemResult.unsupported(
            "B9", "no expected joint was found in the asset", tools=["contract", "matching"]
        )

    score = sum(check.score for check in checks) / len(checks)
    quality = len(checks) / len(wanted)
    wrong = [check for check in checks if check.score < 1.0]

    return ItemResult.scored(
        "B9",
        round(score, 4),
        threshold=TAU_B9,
        confidence=round(quality, 4),
        coverage=Coverage.FULL if not unresolved else Coverage.PARTIAL,
        tools=["contract", "parser", "matching"],
        raw_measurements={
            "declared_type_score": round(sum(check.agreement for check in checks) / len(checks), 4),
            "consistency_score": round(sum(check.consistency for check in checks) / len(checks), 4),
            "joints": [
                {
                    "child": check.child,
                    "joint": check.joint,
                    "expected": check.expected,
                    "declared": check.declared,
                    "agreement": check.agreement,
                    "contradiction": check.contradiction,
                }
                for check in checks
            ],
            "unresolved_names": unresolved,
            "margin": round(min(1.0, abs(score - TAU_B9) / MARGIN), 4),
        },
        failure_reason="; ".join(
            f"{check.child}: expected {check.expected}, declared {check.declared}"
            + (f" ({check.contradiction})" if check.contradiction else "")
            for check in wrong
        ),
        repair_hint="; ".join(
            f"declare {check.child} as {check.expected} and revisit its axis, origin and limits"
            for check in wrong
            if check.agreement < 1.0
        ),
    )


def _type_check(child: str, expected: str, declared: Articulation) -> TypeCheck:
    kind = declared.kind or ""
    agreement = 1.0 if kind == expected else TYPE_AGREEMENT.get((expected, kind), 0.0)
    contradiction = _contradiction(declared)
    return TypeCheck(
        child=child,
        expected=expected,
        declared=declared.kind,
        joint=declared.name,
        agreement=agreement,
        consistency=0.0 if contradiction else 1.0,
        contradiction=contradiction,
    )


def _contradiction(joint: Articulation) -> str:
    """Ways a declaration disagrees with itself. Empty when it holds together."""
    travel = joint.limits.travel if joint.limits else None
    if joint.kind == "continuous" and travel is not None:
        return f"continuous but limited to {travel:.3f}"
    if joint.kind == "revolute" and travel is None:
        return "revolute with no limits"
    if joint.kind == "revolute" and travel is not None and travel > 2 * math.pi:
        return f"revolute travelling {travel:.3f} rad, past a full turn"
    if joint.kind == "prismatic" and travel is not None and travel <= 0:
        return "prismatic with no travel"
    if joint.kind == "fixed" and joint.axis is not None:
        return "fixed but carrying an axis"
    if joint.axis is not None and not any(abs(component) > 1e-9 for component in joint.axis):
        return "zero-length axis"
    return ""


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
