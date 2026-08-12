"""B7-B10: what the declared graph says, checked against what the prompt asked.

None of these four need a simulator. Parent, child, joint type, origin and axis
are structural facts, and the asset declares all of them; the only question is
whether they are the ones the prompt called for.

B8 is here first because it is the most direct: every part the prompt says moves
should carry a joint that moves it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from evo_verifier.asset import Articulation, Asset
from evo_verifier.contract import Contract, ExpectedJoint, Source
from evo_verifier.matching import STRONG, assign, matches, similarity
from evo_verifier.report import Coverage, ItemResult

TAU_B8 = 0.70
"""Protocol placeholder. Calibrate on the dev set, then freeze."""

AGGREGATE: dict[str, str] = {"B7": "worst", "B8": "worst", "B9": "worst", "B10": "worst"}
"""How per-joint scores become one asset score, per item: ``worst`` or ``mean``.

The protocol gives per-joint formulas and never says how to combine them, and
the choice decides more than the formulas do. Under ``mean`` with tau at 0.70 an
asset must break more than 30% of its joints before the average falls far
enough, so one wrong hinge among four is a pass.

Measured both ways, against injected faults whose answer is known and against
the human labels. Injection prefers ``worst`` on every item and not narrowly:
B7 31% to 47%, B8 32% to 63%, B9 7% to 97%, B10 12% to 34%. Human labels agree
on B8 (F1 0.40 to 0.56) and B10 (0 to 0.67); on B7 they cannot adjudicate, with
six positives and kappa below chance either way.

B9 is the one that looks like a loss and is not. Under ``mean`` it scores kappa
0.000 by never predicting failure at all -- silence, not agreement. Under
``worst`` it answers, catches 97% of injected type errors, and disagrees with
the human labels. That disagreement is a result rather than a defect: six
records in the annotated set declare exactly the type the prompt asked for and
were failed anyway, because the annotators were watching a mesh move rather than
reading a type string. Tuning the detector to agree with a label that measures
something else would only hide that.

An earlier reading of this table said mean for B7 and B9. It was taken before
the matcher was fixed, when a requirement could be answered by a sibling's part,
and it did not survive the fix. Revisit it again on a dev set that is not 31
already-failing records.
"""


def _aggregate_for(item: str, override: str | None) -> str:
    return override or AGGREGATE[item]


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
    uncertain: bool = False
    """A shortfall this reader cannot attribute to the asset."""

    @property
    def satisfied(self) -> int:
        return min(len(self.moving), self.wanted)

    @property
    def resolved(self) -> bool:
        """The name found something we can hold the asset responsible for."""
        return bool(self.parts) and not self.uncertain


def check_b8(
    asset: Asset,
    contract: Contract,
    *,
    use_priors: bool = True,
    missing_is_failure: bool = True,
    aggregate: str | None = None,
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
    aggregate = _aggregate_for("B8", aggregate)
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
    blind = any(not joint.readable("child") for joint in asset.movable())
    owned = assign([joint.child for joint in wanted], asset.parts)
    requirements = [
        _requirement(joint, owned[index], moved, blind) for index, joint in enumerate(wanted)
    ]

    uncertain = [r for r in requirements if r.uncertain]
    unresolved = [r for r in requirements if not r.parts]
    still = [r for r in requirements if r.resolved and r.satisfied < r.wanted]
    # An unattributable shortfall never reaches the score, whichever reading of
    # the item is in force: it is doubt about the reader, not about the asset.
    attributable = [r for r in requirements if not r.uncertain]
    scored = attributable if missing_is_failure else [r for r in attributable if r.parts]
    if not scored:
        return ItemResult.unsupported(
            "B8",
            "no expected part could be pinned on the asset",
            tools=["contract", "matching"],
        )
    score = _weighted(scored, aggregate)

    # Doubt from three sources: a name that found nothing (missing part, or a
    # matcher failure -- indistinguishable), a shortfall we cannot attribute,
    # and requirements resting on category priors rather than the prompt.
    quality = (len(requirements) - len(unresolved) - len(uncertain)) / len(requirements)
    stated = sum(r.weight for r in requirements) / len(requirements)

    return ItemResult.scored(
        "B8",
        round(score, 4),
        threshold=TAU_B8,
        confidence=round(quality * stated, 4),
        coverage=Coverage.FULL if not (unresolved or uncertain) else Coverage.PARTIAL,
        tools=["contract", "parser", "matching"],
        raw_measurements={
            "expected": round(sum(r.weight * r.wanted for r in scored), 4),
            "satisfied": round(sum(r.weight * r.satisfied for r in scored), 4),
            "missing_is_failure": missing_is_failure,
            "margin": round(min(1.0, abs(score - TAU_B8) / MARGIN), 4),
            "score_explicit_only": round(
                _weighted([r for r in scored if r.explicit], aggregate), 4
            ),
            "used_priors": use_priors,
            "aggregate": aggregate,
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
            "unattributable_names": [r.child for r in uncertain],
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
    unreadable: bool = False
    """The source declares this connection and the reader could not fold it.

    Kept apart from "the part has no joint": both leave ``actual_parent`` empty,
    and only one of them is the asset's doing. Collapsing them made the item
    report that a part connects to nothing, at full confidence, about a joint
    whose parent argument was simply an expression this reader will not
    evaluate."""


def check_b7(asset: Asset, contract: Contract, *, aggregate: str | None = None) -> ItemResult:
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
    aggregate = _aggregate_for("B7", aggregate)
    stated = [joint for joint in contract.joints if joint.child]
    usable = [joint for joint in stated if joint.parent]
    if not usable:
        return ItemResult.unsupported(
            "B7", "the prompt never says what anything attaches to", tools=["contract"]
        )

    by_child = {joint.child: joint for joint in asset.articulations if joint.child}
    blind = any(not joint.readable("child") for joint in asset.articulations)
    owned = assign([joint.child for joint in stated], asset.parts)
    mine = {id(joint): owned[index] for index, joint in enumerate(stated)}
    connections = [
        _connection(joint, mine[id(joint)], by_child, mine, stated, blind, asset)
        for joint in usable
    ]

    decided = [c for c in connections if not c.unreadable]
    child_scores = [1.0 if c.actual_parent is not None else 0.0 for c in decided if c.part]
    parent_scores = [float(c.parent_ok) for c in decided if c.parent_ok is not None]
    purity = [
        1.0 if not c.intruders else 0.0 for c in decided if c.part and c.actual_parent is not None
    ]

    terms = [(0.3, child_scores), (0.4, parent_scores), (0.3, purity)]
    live = [(weight, values) for weight, values in terms if values]
    if not live:
        return ItemResult.unsupported(
            "B7",
            "no connection could be read well enough to judge",
            tools=["contract", "parser", "matching"],
        )
    total = sum(weight for weight, _ in live)
    score = sum(weight * _combine(values, aggregate) for weight, values in live) / total

    unresolved = [c.child for c in connections if not c.part and not c.unreadable]
    unreadable = [c.child for c in connections if c.unreadable]
    unjointed = [c for c in decided if c.part and c.actual_parent is None]
    wrong = [c for c in decided if c.parent_ok is False or c.intruders]
    quality = len(decided) / len(connections)

    return ItemResult.scored(
        "B7",
        round(score, 4),
        threshold=TAU_B7,
        confidence=round(quality, 4),
        coverage=Coverage.FULL if len(decided) == len(connections) else Coverage.PARTIAL,
        tools=["contract", "parser", "matching", "graph"],
        raw_measurements={
            "child_score": (
                round(sum(child_scores) / len(child_scores), 4) if child_scores else None
            ),
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
            "unreadable_names": unreadable,
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
    owned: list[str],
    by_child: dict[str, Articulation],
    mine: dict[int, list[str]],
    stated: list[ExpectedJoint],
    blind: bool,
    asset: Asset,
) -> Connection:
    """One expected attachment, judged only against the parts assigned to it.

    ``owned`` comes from the exclusive assignment, so a requirement can no
    longer be answered by a sibling's part -- which is how a helicopter with no
    tail-rotor joint used to certify that its tail rotor was attached to the
    fuselage, by reading the main rotor's joint.
    """
    part = next((name for name in owned if name in by_child), None) or (owned[0] if owned else None)
    joint = by_child.get(part or "")
    actual_parent = joint.parent if joint else None
    unreadable = bool(joint and not joint.readable("parent")) or (blind and part is None)

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
        parent_ok=None if unreadable else parent_ok,
        parent_exists=parent_exists or external,
        intruders=_intruders(part, asset, mine, stated, expected),
        unreadable=unreadable,
    )


def tokens_are_external(name: str) -> bool:
    words = set(name.lower().replace("_", " ").split())
    return bool(words & EXTERNAL)


def _intruders(
    part: str | None,
    asset: Asset,
    mine: dict[int, list[str]],
    stated: list[ExpectedJoint],
    expected: ExpectedJoint,
) -> tuple[str, ...]:
    """Other declared-independent parts that ride on this one's subtree.

    Read off the exclusive assignment rather than matched afresh: re-matching
    every other requirement's name against the subtree let a correctly attached
    part be reported as an alien rider, which on one record made a broken asset
    score higher than the sound one.
    """
    if not part:
        return ()
    riding = asset.descendants(part)
    if not riding:
        return ()
    found: list[str] = []
    for index, other in enumerate(stated):
        if other is expected or not other.parent:
            continue
        if similarity(other.parent, expected.child) >= STRONG:
            continue  # the prompt does attach it here
        if similarity(other.parent, part) >= STRONG:
            continue
        found.extend(name for name in mine.get(id(stated[index]), ()) if name in riding)
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

NEAR_MISS = 0.4
"""Credit for confusing revolute with continuous.

Both turn about an axis and differ only in whether the turn stops, which is a
milder mistake than turning where the asset should slide -- but still a mistake:
a wheel that should spin freely but hits a stop does not roll. So the value has
to satisfy ``0.6 * NEAR_MISS + 0.4 < tau``, or a near miss would pass. At 0.5 it
lands exactly on 0.70 and passes, which a unit test caught.
"""

TYPE_AGREEMENT: dict[tuple[str, str], float] = {
    ("revolute", "continuous"): NEAR_MISS,
    ("continuous", "revolute"): NEAR_MISS,
}


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


def check_b9(asset: Asset, contract: Contract, *, aggregate: str | None = None) -> ItemResult:
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
    aggregate = _aggregate_for("B9", aggregate)
    wanted = [joint for joint in contract.joints if joint.kind]
    if not wanted:
        return ItemResult.not_applicable("B9", "the prompt names no joint type", tools=["contract"])

    by_child = {joint.child: joint for joint in asset.articulations if joint.child}
    owned = assign([joint.child for joint in wanted], asset.parts)
    checks, unresolved = [], []
    for index, expected in enumerate(wanted):
        found = [name for name in owned[index] if name in by_child]
        if not found:
            unresolved.append(expected.child)
            continue
        # Every instance assigned to this requirement, not just the first: "four
        # flaps rotate" is four joints, and checking one cannot see the fourth
        # declared as a slide. Assigned, so a sibling's joint cannot answer here.
        checks.extend(
            _type_check(expected.child, expected.kind or "", by_child[name]) for name in found
        )

    if not checks:
        return ItemResult.unsupported(
            "B9", "no expected joint was found in the asset", tools=["contract", "matching"]
        )

    score = _combine([check.score for check in checks], aggregate)
    quality = 1.0 - len(unresolved) / len(wanted)
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
            f"{check.joint} ({check.child}): expected {check.expected}, "
            f"declared {check.declared}"
            + (f" ({check.contradiction})" if check.contradiction else "")
            for check in wrong
        ),
        repair_hint="; ".join(
            f"declare {check.joint} as {check.expected} and revisit its axis, origin and limits"
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
    """Ways a declaration disagrees with itself. Empty when it holds together.

    A bound the source writes and this reader cannot fold is not a contradiction
    -- saying "revolute with no limits" about a joint whose limits are right
    there in the file is a confident claim the source refutes.
    """
    if joint.limits is not None and joint.limits.bounds_unreadable:
        return ""
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


TAU_B10 = 0.70

ANCHOR_TOLERANCE = 0.05
"""Anchor distance, as a fraction of the moving link, at which the score falls to 1/e."""

AXIS_TOLERANCE = 15.0
"""Degrees of axis error at which the score falls to 1/e."""

_VERTICAL = re.compile(r"vertical|upright|up.?down|plumb", re.IGNORECASE)
_HORIZONTAL = re.compile(r"horizontal|transverse|lateral|side.?to.?side", re.IGNORECASE)


@dataclass(frozen=True)
class Placement:
    """Where one joint sits and which way it points, against what was asked."""

    child: str
    joint: str
    anchor: float | None
    """Distance from the joint to the link it moves, over that link's diagonal."""
    anchor_score: float | None
    axis_error: float | None
    """Degrees between the declared axis and the direction the prompt named."""
    axis_score: float | None
    wanted_axis: str = ""

    @property
    def score(self) -> float | None:
        """Geometric mean of the terms that could be computed.

        The protocol multiplies its terms. With one of them missing a product
        would silently become the other term alone, which reads as full evidence;
        a geometric mean over what was available keeps the multiplicative
        character and stays honest about how many terms it rests on.
        """
        parts = [term for term in (self.anchor_score, self.axis_score) if term is not None]
        if not parts:
            return None
        product = 1.0
        for term in parts:
            product *= term
        return product ** (1 / len(parts))


def check_b10(asset: Asset, contract: Contract, *, aggregate: str | None = None) -> ItemResult:
    """Is each joint in the right place, pointing the right way?

    ``S_B10`` combines an anchor term and an axis term, geometric mean over
    whichever could be computed.

    **Anchor.** The protocol divides the anchor distance by ``D``, the whole
    asset's diagonal, which needs every part to have analytic geometry -- true
    for 19 of 607 records. But the joint origin *is* the moving link's own frame
    origin, so the distance that matters is from the joint to the link it moves,
    over that link's own diagonal. That needs geometry for one part instead of
    all of them, and it measures the thing the item is about: a hinge sitting
    outside the door it swings is not attached to it. Deviation from ``D``,
    recorded.

    **Axis.** Compared against the direction the prompt named -- "a vertical side
    hinge", "horizontal hinges at the panel edges". Only direction words that
    resolve in the world frame are used; "about its axle" and "coaxial" name a
    direction relative to geometry we do not have, and are left uncomputed.

    The protocol's third term, trajectory direction, is the same circular
    quantity as B9's classifier against a declared graph, and is omitted.

    On the annotated set this decides a minority of joints and abstains on the
    rest. That is the honest state of the item without meshes, and the shape is
    ready for them: filling in the anchor term is a geometry problem, not a
    redesign.
    """
    aggregate = _aggregate_for("B10", aggregate)
    wanted = [joint for joint in contract.joints if joint.child]
    if not wanted:
        return ItemResult.not_applicable("B10", "the prompt names no joint", tools=["contract"])

    by_child = {joint.child: joint for joint in asset.articulations if joint.child}
    owned = assign([joint.child for joint in wanted], asset.parts)
    placements, unresolved = [], []
    for index, expected in enumerate(wanted):
        found = [name for name in owned[index] if name in by_child]
        if not found:
            unresolved.append(expected.child)
            continue
        placements.extend(_placement(name, by_child[name], expected, asset) for name in found)

    scored = [p for p in placements if p.score is not None]
    if not scored:
        return ItemResult.unsupported(
            "B10",
            "no joint has analytic geometry or a directional hint to check",
            tools=["contract", "parser", "geometry"],
        )

    score = _combine([p.score for p in scored if p.score is not None], aggregate)
    complete = [p for p in scored if p.anchor_score is not None and p.axis_score is not None]
    wrong = [p for p in scored if (p.score or 1.0) < TAU_B10]

    return ItemResult.scored(
        "B10",
        round(score, 4),
        threshold=TAU_B10,
        confidence=round(
            (1.0 - len(unresolved) / len(wanted)) * (len(scored) / max(1, len(placements))), 4
        ),
        coverage=(Coverage.FULL if len(complete) == len(wanted) else Coverage.PARTIAL),
        tools=["contract", "parser", "matching", "geometry"],
        raw_measurements={
            "checked": len(scored),
            "of_expected": len(wanted),
            "with_both_terms": len(complete),
            "anchor_tolerance": ANCHOR_TOLERANCE,
            "axis_tolerance_degrees": AXIS_TOLERANCE,
            "margin": round(min(1.0, abs(score - TAU_B10) / MARGIN), 4),
            "joints": [
                {
                    "child": p.child,
                    "joint": p.joint,
                    "anchor_over_link": p.anchor,
                    "anchor_score": p.anchor_score,
                    "wanted_axis": p.wanted_axis,
                    "axis_error_degrees": p.axis_error,
                    "axis_score": p.axis_score,
                    "score": p.score,
                }
                for p in placements
            ],
            "unresolved_names": unresolved,
        },
        failure_reason="; ".join(_placement_reason(p) for p in wrong),
        repair_hint="; ".join(
            f"move {p.joint} onto the {p.child} interface"
            if p.anchor_score is not None and p.anchor_score < 0.7
            else f"align {p.joint} with the {p.wanted_axis} direction the prompt names"
            for p in wrong
        ),
    )


def _placement(part: str, joint: Articulation, expected: ExpectedJoint, asset: Asset) -> Placement:
    anchor = _anchor(part, joint, asset)
    axis_error = _axis_error(joint, expected.axis_hint)
    return Placement(
        child=part,
        joint=joint.name,
        anchor=anchor,
        anchor_score=None if anchor is None else math.exp(-anchor / ANCHOR_TOLERANCE),
        axis_error=axis_error,
        axis_score=None if axis_error is None else math.exp(-axis_error / AXIS_TOLERANCE),
        wanted_axis=_wanted_axis(expected.axis_hint),
    )


def _anchor(part: str, joint: Articulation, asset: Asset) -> float | None:
    """Distance from the joint to the link it moves, over that link's diagonal.

    The joint origin is the child frame's own origin, so the child's shapes are
    already positioned relative to it: the distance is from the origin to the
    child's own bounding box, zero when the joint sits inside the part.
    """
    link = asset.parts.get(part)
    if link is None or not link.geometry_complete or joint.origin.xyz is None:
        return None
    box = link.aabb()
    if box is None:
        return None
    low, high = box
    gap = math.dist(
        (0.0, 0.0, 0.0),
        tuple(max(low[axis], 0.0) + min(high[axis], 0.0) for axis in range(3)),
    )
    diagonal = math.dist(low, high)
    return gap / diagonal if diagonal > 0 else None


def _wanted_axis(hint: str) -> str:
    if not hint:
        return ""
    if _VERTICAL.search(hint):
        return "vertical"
    if _HORIZONTAL.search(hint):
        return "horizontal"
    return ""


def _axis_error(joint: Articulation, hint: str) -> float | None:
    """Degrees between the declared axis and the direction the prompt named."""
    wanted = _wanted_axis(hint)
    if not wanted or joint.axis is None:
        return None
    length = math.dist((0.0, 0.0, 0.0), joint.axis)
    if length <= 0:
        return None
    from_vertical = math.degrees(math.acos(min(1.0, abs(joint.axis[2]) / length)))
    return from_vertical if wanted == "vertical" else 90.0 - from_vertical


def _placement_reason(placement: Placement) -> str:
    reasons = []
    if placement.anchor_score is not None and placement.anchor_score < 0.7:
        reasons.append(
            f"{placement.joint} sits {placement.anchor:.3f} link-diagonals off {placement.child}"
        )
    if placement.axis_score is not None and placement.axis_score < 0.7:
        reasons.append(
            f"{placement.joint} axis is {placement.axis_error:.1f} deg from {placement.wanted_axis}"
        )
    return "; ".join(reasons) or f"{placement.joint} placement below threshold"


def _combine(scores: list[float], how: str) -> float:
    """One number for the asset from one number per joint."""
    if not scores:
        return 0.0
    return min(scores) if how == "worst" else sum(scores) / len(scores)


def _weighted(requirements: list[Requirement], how: str = "mean") -> float:
    if how == "worst":
        return _combine([r.satisfied / r.wanted for r in requirements if r.wanted], how)
    expected = sum(r.weight * r.wanted for r in requirements)
    if not expected:
        return 0.0
    return sum(r.weight * r.satisfied for r in requirements) / expected


def _requirement(
    joint: ExpectedJoint, found: list[str], moved: set[str], blind: bool
) -> Requirement:
    """One requirement against the parts assigned to it, and nobody else's.

    ``blind`` when the asset has a movable joint whose child would not fold: a
    part that looks jointless might be the one that joint moves, so the absence
    cannot be pinned on the asset.
    """
    explicit = joint.source is Source.EXPLICIT
    moving = tuple(name for name in found if name in moved)
    return Requirement(
        child=joint.child,
        wanted=max(1, joint.count or 1),
        parts=tuple(found),
        moving=moving,
        weight=1.0 if explicit else min(joint.confidence, 0.7),
        explicit=explicit,
        uncertain=blind and len(moving) < max(1, joint.count or 1),
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
