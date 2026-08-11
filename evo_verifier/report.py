"""The verifier's output contract.

Three people write detectors for different items and their results have to add up
to one report, so the shape of a single item's result is the integration point.
It is enforced here rather than agreed in a document.

What the validation is protecting:

- A score always arrives with the threshold it was compared against, so a frozen
  report can be re-read after the threshold moves.
- ``not-applicable`` (the contract asked for nothing) and ``abstain`` (the tools
  could not answer) are different states. N/A leaves the aggregate entirely;
  abstain is a refusal to predict and shows up in selective-risk reporting.
- No score is silently 0 or 1. An item with no evidence has ``score = None``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from evo_verifier.items import BY_ID, FAMILY_WEIGHTS, ITEM_IDS, Family

DEFAULT_THRESHOLD = 0.70
"""Protocol placeholder. Calibrate per item on the dev set, then freeze."""

ABSTAIN_BELOW_CONFIDENCE = 0.50
"""Below this the verifier declines to predict instead of guessing."""


class Coverage(StrEnum):
    """How much of what the item asks for the evidence actually covers."""

    FULL = "full"
    ESTIMATED_PHYSICS = "estimated-physics"
    """Ran, but on inferred or default mass/friction/damping."""
    PARTIAL = "partial"
    """Some atomic evidence only -- geometry without the dynamics, say."""
    UNSUPPORTED = "unsupported"
    """The tools cannot answer this item's definition at all."""
    NOT_APPLICABLE = "not-applicable"
    """The contract asks for nothing here. Leaves the aggregate."""


class Prediction(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    ABSTAIN = "abstain"


_SCORELESS = {Coverage.UNSUPPORTED, Coverage.NOT_APPLICABLE}


class ReportError(ValueError):
    """A result that would corrupt an aggregate or an alignment table."""


@dataclass(frozen=True)
class ToolFailure:
    """Infrastructure broke. Not the asset's fault, and never scored as zero."""

    tool: str
    message: str
    item_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "message": self.message, "items": list(self.item_ids)}


@dataclass(frozen=True)
class ItemResult:
    """One item's verdict, with the evidence needed to re-check it."""

    item_id: str
    score: float | None
    prediction: Prediction | None
    threshold: float
    confidence: float
    coverage: Coverage
    tools: tuple[str, ...] = ()
    raw_measurements: Mapping[str, Any] = field(default_factory=dict)
    failure_reason: str = ""
    repair_hint: str = ""

    def __post_init__(self) -> None:
        if self.item_id not in BY_ID:
            raise ReportError(f"unknown item id {self.item_id!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ReportError(f"{self.item_id}: confidence {self.confidence} outside [0, 1]")
        if not 0.0 <= self.threshold <= 1.0:
            raise ReportError(f"{self.item_id}: threshold {self.threshold} outside [0, 1]")
        if self.coverage in _SCORELESS:
            if self.score is not None:
                raise ReportError(f"{self.item_id}: {self.coverage} must not carry a score")
        elif self.score is None:
            raise ReportError(f"{self.item_id}: {self.coverage} needs a score")
        elif not 0.0 <= self.score <= 1.0:
            raise ReportError(f"{self.item_id}: score {self.score} outside [0, 1]")
        if self.coverage is Coverage.NOT_APPLICABLE and self.prediction is not None:
            raise ReportError(f"{self.item_id}: not-applicable must not predict")
        if self.coverage is Coverage.UNSUPPORTED and self.prediction is not Prediction.ABSTAIN:
            raise ReportError(f"{self.item_id}: unsupported must abstain")
        if self.score is not None and self.prediction is None:
            raise ReportError(f"{self.item_id}: a scored item must predict")
        if self.prediction is Prediction.FAIL and not self.failure_reason:
            raise ReportError(f"{self.item_id}: a failure must say what failed")

    @classmethod
    def scored(
        cls,
        item_id: str,
        score: float,
        *,
        coverage: Coverage = Coverage.FULL,
        threshold: float = DEFAULT_THRESHOLD,
        confidence: float = 1.0,
        tools: Sequence[str] = (),
        raw_measurements: Mapping[str, Any] | None = None,
        failure_reason: str = "",
        repair_hint: str = "",
    ) -> ItemResult:
        """Build a result and derive the prediction from score, threshold, confidence."""
        if coverage in _SCORELESS:
            raise ReportError(f"{item_id}: {coverage} cannot be scored")
        if confidence < ABSTAIN_BELOW_CONFIDENCE:
            prediction = Prediction.ABSTAIN
        else:
            prediction = Prediction.PASS if score >= threshold else Prediction.FAIL
        return cls(
            item_id=item_id,
            score=score,
            prediction=prediction,
            threshold=threshold,
            confidence=confidence,
            coverage=coverage,
            tools=tuple(tools),
            raw_measurements=dict(raw_measurements or {}),
            failure_reason=failure_reason,
            repair_hint=repair_hint,
        )

    @classmethod
    def not_applicable(cls, item_id: str, reason: str, *, tools: Sequence[str] = ()) -> ItemResult:
        """The contract asks for nothing here. Not a pass, not a zero."""
        return cls(
            item_id=item_id,
            score=None,
            prediction=None,
            threshold=DEFAULT_THRESHOLD,
            confidence=1.0,
            coverage=Coverage.NOT_APPLICABLE,
            tools=tuple(tools),
            failure_reason="",
            repair_hint="",
            raw_measurements={"reason": reason},
        )

    @classmethod
    def unsupported(cls, item_id: str, reason: str, *, tools: Sequence[str] = ()) -> ItemResult:
        """The tools cannot answer this item. An honest abstention."""
        return cls(
            item_id=item_id,
            score=None,
            prediction=Prediction.ABSTAIN,
            threshold=DEFAULT_THRESHOLD,
            confidence=0.0,
            coverage=Coverage.UNSUPPORTED,
            tools=tuple(tools),
            raw_measurements={"reason": reason},
        )

    @property
    def counts_toward_score(self) -> bool:
        """Only decided items reach the aggregate. N/A and abstain do not."""
        return self.score is not None and self.prediction is not Prediction.ABSTAIN

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "score": self.score,
            "prediction": self.prediction.value if self.prediction else None,
            "threshold": self.threshold,
            "confidence": self.confidence,
            "coverage": self.coverage.value,
            "tools": list(self.tools),
            "raw_measurements": dict(self.raw_measurements),
        }
        if self.failure_reason:
            payload["failure_reason"] = self.failure_reason
        if self.repair_hint:
            payload["repair_hint"] = self.repair_hint
        return payload

    @classmethod
    def from_dict(cls, item_id: str, payload: Mapping[str, Any]) -> ItemResult:
        prediction = payload.get("prediction")
        return cls(
            item_id=item_id,
            score=payload.get("score"),
            prediction=Prediction(prediction) if prediction else None,
            threshold=float(payload.get("threshold", DEFAULT_THRESHOLD)),
            confidence=float(payload.get("confidence", 1.0)),
            coverage=Coverage(payload["coverage"]),
            tools=tuple(payload.get("tools", ())),
            raw_measurements=dict(payload.get("raw_measurements", {})),
            failure_reason=payload.get("failure_reason", ""),
            repair_hint=payload.get("repair_hint", ""),
        )


@dataclass
class VerificationReport:
    """Everything the verifier says about one asset."""

    asset_id: str
    verifier_version: str
    contract_version: str = ""
    items: dict[str, ItemResult] = field(default_factory=dict)
    tool_failures: list[ToolFailure] = field(default_factory=list)
    repair_queue: list[str] = field(default_factory=list)

    def add(self, result: ItemResult) -> None:
        if result.item_id in self.items:
            raise ReportError(f"{result.item_id} reported twice for {self.asset_id}")
        self.items[result.item_id] = result

    def missing_items(self) -> tuple[str, ...]:
        """Items nobody reported. A partial report is legal; a silent one is not."""
        return tuple(item_id for item_id in ITEM_IDS if item_id not in self.items)

    def family_scores(self) -> dict[Family, float | None]:
        """Mean of the decided items in each family. ``None`` when none decided."""
        totals: dict[Family, list[float]] = {family: [] for family in FAMILY_WEIGHTS}
        for result in self.items.values():
            if result.counts_toward_score and result.score is not None:
                totals[BY_ID[result.item_id].family].append(result.score)
        return {
            family: (sum(scores) / len(scores) if scores else None)
            for family, scores in totals.items()
        }

    def score_full(self) -> float | None:
        """``100 * sum(w_f S_f) / sum(w_f)`` over families that have evidence.

        A family with nothing decided leaves the numerator and the denominator
        together. It is never a zero.
        """
        families = self.family_scores()
        numerator = sum(
            FAMILY_WEIGHTS[family] * score
            for family, score in families.items()
            if score is not None
        )
        denominator = sum(
            FAMILY_WEIGHTS[family] for family, score in families.items() if score is not None
        )
        return 100.0 * numerator / denominator if denominator else None

    def human_coverage(self) -> float | None:
        """Share of applicable items the verifier actually decided."""
        applicable = [
            result
            for result in self.items.values()
            if result.coverage is not Coverage.NOT_APPLICABLE
        ]
        if not applicable:
            return None
        return sum(1 for result in applicable if result.counts_toward_score) / len(applicable)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verifier_version": self.verifier_version,
            "asset_id": self.asset_id,
            "contract_version": self.contract_version,
            "human_metrics": {
                item_id: self.items[item_id].to_dict()
                for item_id in ITEM_IDS
                if item_id in self.items
            },
            "families": {
                family: score for family, score in self.family_scores().items() if score is not None
            },
            "scores": {
                "Score_full": self.score_full(),
                "human_coverage": self.human_coverage(),
            },
            "tool_failures": [failure.to_dict() for failure in self.tool_failures],
            "repair_queue": list(self.repair_queue),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VerificationReport:
        report = cls(
            asset_id=str(payload["asset_id"]),
            verifier_version=str(payload.get("verifier_version", "")),
            contract_version=str(payload.get("contract_version", "")),
            repair_queue=list(payload.get("repair_queue", [])),
        )
        for item_id, item_payload in (payload.get("human_metrics") or {}).items():
            report.add(ItemResult.from_dict(item_id, item_payload))
        for failure in payload.get("tool_failures", []):
            report.tool_failures.append(
                ToolFailure(
                    tool=str(failure.get("tool", "")),
                    message=str(failure.get("message", "")),
                    item_ids=tuple(failure.get("items", ())),
                )
            )
        return report

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path | str) -> VerificationReport:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_reports(directory: Path | str) -> dict[str, VerificationReport]:
    """Load ``*.json`` reports from a directory, keyed by asset id."""
    reports: dict[str, VerificationReport] = {}
    for path in sorted(Path(directory).glob("*.json")):
        report = VerificationReport.load(path)
        if report.asset_id in reports:
            raise ReportError(f"two reports for asset {report.asset_id}")
        reports[report.asset_id] = report
    return reports


def collect(
    asset_id: str,
    verifier_version: str,
    results: Iterable[ItemResult],
    *,
    contract_version: str = "",
) -> VerificationReport:
    """Merge one person's item results into a report."""
    report = VerificationReport(
        asset_id=asset_id, verifier_version=verifier_version, contract_version=contract_version
    )
    for result in results:
        report.add(result)
    return report
