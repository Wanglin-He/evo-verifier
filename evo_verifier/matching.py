"""Match a prompt's words to an asset's identifiers.

A contract says ``outlet flap``, ``feed pusher``, ``central steel post``. The
asset says ``front_flap``, ``pusher``, ``storefront``. Every B7-B10 check has to
cross that gap first, and how it fails matters: a matcher that quietly gives up
looks exactly like a missing part, which is a real B8 failure. So matching
returns a score, callers decide with it, and "no match" travels as its own
outcome rather than as evidence of a defect.

Deterministic on purpose. A frozen verifier cannot have a model in this path
deciding that ``rotor`` is or is not a ``wing assembly`` differently each run.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

STRONG = 0.60
"""At or above this, treat it as the same thing. Tuned on the 31-contract set."""

STOPWORDS = frozenset(
    {"a", "an", "the", "of", "and", "with", "its", "their", "for", "on", "in", "at", "to"}
)

DESCRIPTORS = frozenset(
    {
        "big",
        "broad",
        "clear",
        "compact",
        "deep",
        "full",
        "large",
        "long",
        "main",
        "narrow",
        "short",
        "single",
        "slim",
        "small",
        "tall",
        "thin",
        "wide",
    }
)
"""Size words. They rarely distinguish two parts of the same object."""

FILLERS = frozenset({"assembly", "unit", "pack", "module", "block", "component", "part", "piece"})
"""Words that name no part on their own. ``seat_assembly`` is a seat."""

SYNONYMS: tuple[frozenset[str], ...] = (
    frozenset({"lid", "cover", "cap"}),
    frozenset({"base", "body", "housing", "chassis", "shell", "case"}),
    frozenset({"post", "column", "pillar", "mast", "shaft"}),
    frozenset({"door", "hatch", "panel"}),
    frozenset({"knob", "dial", "selector"}),
    frozenset({"button", "actuator", "switch"}),
    frozenset({"arm", "boom", "lever"}),
    frozenset({"handle", "grip"}),
    frozenset({"drawer", "tray", "basket", "tub", "drum"}),
    frozenset({"jaw", "clamp", "grip"}),
    frozenset({"wheel", "roller", "rotor", "disc", "flywheel"}),
    frozenset({"hinge", "pivot", "joint"}),
    frozenset({"leg", "foot", "stand"}),
    frozenset({"head", "top"}),
    frozenset({"rod", "pin", "axle", "spindle"}),
)
"""Words this dataset uses interchangeably. Small and hand-checked, not a thesaurus."""

_WORD = re.compile(r"[a-z0-9]+")


def normalise(name: str) -> str:
    return " ".join(tokens(name)) or name.strip().lower()


def tokens(name: str) -> tuple[str, ...]:
    """Content words of a part name, singularised, in order.

    Trailing digits go: the SDK writes ``button_0``, ``button_1`` for repeated
    parts, and the index says nothing about what the part is.
    """
    words = _WORD.findall(name.lower().replace("_", " ").replace("-", " "))
    while words and words[-1].isdigit():
        words.pop()
    kept = [_singular(word) for word in words if word not in STOPWORDS]
    content = [word for word in kept if word not in DESCRIPTORS and word not in FILLERS]
    return tuple(content or kept)


def _singular(word: str) -> str:
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("es") and word[-3] in "sxzh":
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _synonym_class(word: str) -> frozenset[str]:
    for group in SYNONYMS:
        if word in group:
            return group
    return frozenset({word})


def _shares(left: str, right: str) -> bool:
    return left == right or bool(_synonym_class(left) & _synonym_class(right))


def similarity(wanted: str, candidate: str) -> float:
    """How much a contract name and an asset name are the same thing, in [0, 1].

    Head nouns carry the weight: ``outlet flap`` and ``front_flap`` are the same
    kind of part with different modifiers, while ``flap motor`` is not.
    """
    left, right = tokens(wanted), tokens(candidate)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    shared = sum(1 for word in left if any(_shares(word, other) for other in right))
    if shared == min(len(left), len(right)):
        return 0.8  # one name is the other plus modifiers: rear_lock, rear lock lever
    head = 1.0 if _shares(left[-1], right[-1]) else 0.0
    if not head and not shared:
        return 0.0
    overlap = shared / min(len(left), len(right))
    return round(0.6 * head + 0.4 * min(1.0, overlap), 4)


@dataclass(frozen=True)
class Match:
    """One candidate and how well it fits."""

    name: str
    score: float

    @property
    def strong(self) -> bool:
        return self.score >= STRONG


def rank(wanted: str, candidates: Iterable[str]) -> list[Match]:
    """Every candidate that shares anything, best first."""
    scored = [Match(name, similarity(wanted, name)) for name in candidates]
    return sorted((m for m in scored if m.score > 0), key=lambda m: (-m.score, m.name))


def matches(wanted: str, candidates: Iterable[str]) -> list[Match]:
    """Only the candidates good enough to act on.

    Plural: ``outlet flap`` legitimately matches four separate flap parts, and a
    count in the contract is checked against how many came back.
    """
    return [match for match in rank(wanted, candidates) if match.strong]


def best(wanted: str, candidates: Iterable[str]) -> Match | None:
    ranked = rank(wanted, candidates)
    return ranked[0] if ranked and ranked[0].strong else None


def unmatched(wanted: Sequence[str], candidates: Iterable[str]) -> list[str]:
    candidates = list(candidates)
    return [name for name in wanted if best(name, candidates) is None]
