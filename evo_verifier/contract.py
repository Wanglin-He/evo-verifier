"""What the prompt asked for, in a form a checker can compare against.

There is no reference asset -- a folding chair has many correct designs -- so the
requirements have to come from the prompt. That is what a contract is: the prose
turned into structured claims, each one pointing back at the words it came from.

Two fields carry most of the weight.

``parent`` on an expected joint is the connection object, and B7 is almost
entirely about it: the real failures in the annotated set are a juicer pusher
hung off the lid instead of the chute, a revolving door whose wings turn against
the storefront because the central post was never modelled, a spray bottle whose
pump rod never reaches the trigger. "This part moves" would catch none of them.

``quote`` is the span of the prompt a requirement was read from, and
``source="explicit"`` is only honest if that span is really there. So the check
is mechanical: an explicit requirement whose quote is not in the prompt is
rejected. An extractor that invents a requirement cannot also invent the
evidence for it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

CONTRACT_VERSION = "1"

JOINT_KINDS = frozenset({"revolute", "continuous", "prismatic", "fixed"})
"""The four ArticulationType values the SDK declares."""


class Source(StrEnum):
    """Where a requirement came from. It decides how hard it may be scored."""

    EXPLICIT = "explicit"
    """The prompt says so. Scored at full weight."""
    PRIOR = "prior"
    """Category common sense, not written down. Scored at reduced confidence."""


class ContractError(ValueError):
    """A contract that would let an unsupported requirement score an asset."""


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass(frozen=True)
class ExpectedPart:
    """A part the prompt calls for, named the way the prompt names it."""

    name: str
    count: int | None = None
    """How many, when the prompt says. ``None`` means unspecified, not one."""
    source: Source = Source.EXPLICIT
    confidence: float = 1.0
    quote: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "source": self.source.value,
            "confidence": self.confidence,
            "quote": self.quote,
        }


@dataclass(frozen=True)
class ExpectedJoint:
    """A motion the prompt calls for, and what it moves against."""

    child: str
    """The part that moves."""
    parent: str = ""
    """What it is attached to. Empty when the prompt does not say."""
    kind: str | None = None
    axis_hint: str = ""
    """``vertical``, ``horizontal``, ``along the column`` -- free prose, B10 reads it."""
    location_hint: str = ""
    """Where the joint sits: ``at the panel edge``, ``rear``. B10 reads it."""
    count: int | None = None
    source: Source = Source.EXPLICIT
    confidence: float = 1.0
    quote: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "child": self.child,
            "parent": self.parent,
            "kind": self.kind,
            "axis_hint": self.axis_hint,
            "location_hint": self.location_hint,
            "count": self.count,
            "source": self.source.value,
            "confidence": self.confidence,
            "quote": self.quote,
        }


@dataclass
class Contract:
    """One record's requirements, extracted once and then frozen."""

    record_id: str
    prompt: str
    parts: tuple[ExpectedPart, ...] = ()
    joints: tuple[ExpectedJoint, ...] = ()
    extractor: str = ""
    """Model and version that produced this. Provenance, not decoration."""
    version: str = CONTRACT_VERSION
    notes: list[str] = field(default_factory=list)

    def movable(self) -> tuple[ExpectedJoint, ...]:
        return tuple(joint for joint in self.joints if joint.kind != "fixed")

    def explicit_joints(self) -> tuple[ExpectedJoint, ...]:
        return tuple(joint for joint in self.joints if joint.source is Source.EXPLICIT)

    def unsupported_quotes(self) -> list[str]:
        """Explicit requirements whose quote is not in the prompt.

        Empty is the only acceptable answer for a contract that scores anything.
        """
        prompt = _normalise(self.prompt)
        missing = []
        for requirement in (*self.parts, *self.joints):
            if requirement.source is not Source.EXPLICIT:
                continue
            quote = _normalise(requirement.quote)
            if not quote or quote not in prompt:
                label = getattr(requirement, "name", None) or getattr(requirement, "child", "?")
                missing.append(f"{label}: {requirement.quote!r}")
        return missing

    def validate(self) -> None:
        if not self.record_id:
            raise ContractError("contract has no record id")
        if not self.prompt.strip():
            raise ContractError(f"{self.record_id}: contract has no prompt")
        for joint in self.joints:
            if not joint.child:
                raise ContractError(f"{self.record_id}: a joint with no moving part")
            if joint.kind is not None and joint.kind not in JOINT_KINDS:
                raise ContractError(f"{self.record_id}: unknown joint kind {joint.kind!r}")
        for requirement in (*self.parts, *self.joints):
            if not 0.0 <= requirement.confidence <= 1.0:
                raise ContractError(f"{self.record_id}: confidence outside [0, 1]")
        unsupported = self.unsupported_quotes()
        if unsupported:
            raise ContractError(
                f"{self.record_id}: explicit requirements not found in the prompt: {unsupported}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "record_id": self.record_id,
            "prompt": self.prompt,
            "extractor": self.extractor,
            "expected_parts": [part.to_dict() for part in self.parts],
            "expected_joints": [joint.to_dict() for joint in self.joints],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Contract:
        return cls(
            record_id=str(payload["record_id"]),
            prompt=str(payload.get("prompt", "")),
            parts=tuple(_part(item) for item in payload.get("expected_parts", ())),
            joints=tuple(_joint(item) for item in payload.get("expected_joints", ())),
            extractor=str(payload.get("extractor", "")),
            version=str(payload.get("version", CONTRACT_VERSION)),
            notes=list(payload.get("notes", ())),
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path | str) -> Contract:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _source(value: Any) -> Source:
    try:
        return Source(str(value or "explicit"))
    except ValueError:
        raise ContractError(f"unknown source {value!r}") from None


def _count(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ContractError(f"count is not a number: {value!r}") from None


def _part(payload: Mapping[str, Any]) -> ExpectedPart:
    return ExpectedPart(
        name=str(payload.get("name", "")).strip(),
        count=_count(payload.get("count")),
        source=_source(payload.get("source")),
        confidence=float(payload.get("confidence", 1.0)),
        quote=str(payload.get("quote", "")),
    )


def _joint(payload: Mapping[str, Any]) -> ExpectedJoint:
    kind = payload.get("kind")
    return ExpectedJoint(
        child=str(payload.get("child", "")).strip(),
        parent=str(payload.get("parent", "") or "").strip(),
        kind=str(kind).strip().lower() if kind else None,
        axis_hint=str(payload.get("axis_hint", "") or "").strip(),
        location_hint=str(payload.get("location_hint", "") or "").strip(),
        count=_count(payload.get("count")),
        source=_source(payload.get("source")),
        confidence=float(payload.get("confidence", 1.0)),
        quote=str(payload.get("quote", "")),
    )


def load_contracts(directory: Path | str) -> dict[str, Contract]:
    """Load ``<record_id>.json`` contracts, keyed by record id."""
    contracts: dict[str, Contract] = {}
    for path in sorted(Path(directory).glob("*.json")):
        contract = Contract.load(path)
        contract.validate()
        contracts[contract.record_id] = contract
    return contracts


def save_contracts(contracts: Iterable[Contract], directory: Path | str) -> int:
    directory = Path(directory)
    written = 0
    for contract in contracts:
        contract.save(directory / f"{contract.record_id}.json")
        written += 1
    return written


def summarise(contracts: Sequence[Contract]) -> str:
    """One line of shape, for eyeballing a batch before trusting it."""
    joints = [joint for contract in contracts for joint in contract.joints]
    explicit = sum(1 for joint in joints if joint.source is Source.EXPLICIT)
    with_parent = sum(1 for joint in joints if joint.parent)
    typed = sum(1 for joint in joints if joint.kind)
    return (
        f"{len(contracts)} contracts, {len(joints)} expected joints "
        f"({explicit} explicit, {with_parent} with a connection object, {typed} typed)"
    )
