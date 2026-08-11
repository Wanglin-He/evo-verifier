"""Per-item human/verifier agreement.

The positive class is FAIL. The verifier exists to find broken assets, so recall
is over failures and precision is over the failures it claims.

What leaves the table, and why:

- human MISSING or N/A: there is nothing to agree with.
- verifier abstain or not-applicable: a refusal is not a wrong answer. It is
  counted as lost coverage instead, which is the honest place for it.

Both exclusions are reported as counts. A verifier can always look accurate by
abstaining, so accuracy without coverage means nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from evo_verifier.items import BY_ID, ITEMS, Item
from evo_verifier.labels import AnnotatedCase, Label
from evo_verifier.report import Coverage, Prediction, VerificationReport


@dataclass(frozen=True)
class ItemEvaluation:
    """One item's agreement with the human labels, over a set of cases."""

    item_id: str
    true_positive: int
    """Human failed it, verifier failed it."""
    false_positive: int
    """Human passed it, verifier failed it."""
    false_negative: int
    """Human failed it, verifier passed it. The one that matters most."""
    true_negative: int
    abstained: int
    """Human answered, verifier declined."""
    not_reported: int
    """Human answered, no result for this item at all."""
    verifier_na: int
    """Human answered, verifier called it not-applicable."""
    human_na: int
    """Human called it not-applicable."""
    human_missing: int
    """No human answer."""

    @property
    def decided(self) -> int:
        return self.true_positive + self.false_positive + self.false_negative + self.true_negative

    @property
    def eligible(self) -> int:
        """Cases with a human pass/fail answer -- the most we could have covered."""
        return self.decided + self.abstained + self.not_reported + self.verifier_na

    @property
    def human_failures(self) -> int:
        """Positives available. Small numbers here make every metric below noisy."""
        return self.true_positive + self.false_negative

    @property
    def coverage(self) -> float | None:
        return self.decided / self.eligible if self.eligible else None

    @property
    def precision(self) -> float | None:
        claimed = self.true_positive + self.false_positive
        return self.true_positive / claimed if claimed else None

    @property
    def recall(self) -> float | None:
        actual = self.true_positive + self.false_negative
        return self.true_positive / actual if actual else None

    @property
    def specificity(self) -> float | None:
        actual = self.true_negative + self.false_positive
        return self.true_negative / actual if actual else None

    @property
    def f1(self) -> float | None:
        precision, recall = self.precision, self.recall
        if precision is None or recall is None or precision + recall == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    @property
    def balanced_accuracy(self) -> float | None:
        recall, specificity = self.recall, self.specificity
        if recall is None or specificity is None:
            return None
        return (recall + specificity) / 2

    @property
    def kappa(self) -> float | None:
        """Cohen's kappa on the 2x2 table.

        Guards against the trap of a 98%-pass item: agreeing by always predicting
        pass scores near zero here.
        """
        total = self.decided
        if total == 0:
            return None
        observed = (self.true_positive + self.true_negative) / total
        predicted_fail = (self.true_positive + self.false_positive) / total
        actual_fail = (self.true_positive + self.false_negative) / total
        expected = predicted_fail * actual_fail + (1 - predicted_fail) * (1 - actual_fail)
        if expected >= 1.0:
            return None
        return (observed - expected) / (1 - expected)


def evaluate_item(
    cases: Iterable[AnnotatedCase],
    reports: dict[str, VerificationReport],
    item: Item | str,
) -> ItemEvaluation:
    item_id = item if isinstance(item, str) else item.item_id
    if item_id not in BY_ID:
        raise KeyError(f"unknown item id {item_id!r}")
    counts = dict.fromkeys(
        (
            "tp",
            "fp",
            "fn",
            "tn",
            "abstained",
            "not_reported",
            "verifier_na",
            "human_na",
            "human_missing",
        ),
        0,
    )
    for case in cases:
        human = case.labels[item_id]
        if human is Label.MISSING:
            counts["human_missing"] += 1
            continue
        if human is Label.NA:
            counts["human_na"] += 1
            continue
        result = reports.get(case.record_id)
        result = result.items.get(item_id) if result else None
        if result is None:
            counts["not_reported"] += 1
        elif result.coverage is Coverage.NOT_APPLICABLE:
            counts["verifier_na"] += 1
        elif result.prediction is Prediction.ABSTAIN:
            counts["abstained"] += 1
        elif result.prediction is Prediction.FAIL:
            counts["tp" if human is Label.FAIL else "fp"] += 1
        else:
            counts["fn" if human is Label.FAIL else "tn"] += 1
    return ItemEvaluation(
        item_id=item_id,
        true_positive=counts["tp"],
        false_positive=counts["fp"],
        false_negative=counts["fn"],
        true_negative=counts["tn"],
        abstained=counts["abstained"],
        not_reported=counts["not_reported"],
        verifier_na=counts["verifier_na"],
        human_na=counts["human_na"],
        human_missing=counts["human_missing"],
    )


def evaluate(
    cases: Iterable[AnnotatedCase],
    reports: dict[str, VerificationReport],
    items: Sequence[Item | str] = ITEMS,
) -> list[ItemEvaluation]:
    cases = list(cases)
    return [evaluate_item(cases, reports, item) for item in items]


_HEADERS = (
    ("item", 5),
    ("n+", 5),
    ("TP", 4),
    ("FP", 4),
    ("FN", 4),
    ("TN", 5),
    ("prec", 6),
    ("rec", 6),
    ("F1", 6),
    ("bal-acc", 8),
    ("kappa", 7),
    ("cover", 7),
    ("abst", 5),
    ("n/r", 5),
)


def format_table(evaluations: Sequence[ItemEvaluation]) -> str:
    """A fixed-width table. ``n+`` is the number of human failures available."""
    header = "  ".join(name.rjust(width) for name, width in _HEADERS)
    rows = (
        "  ".join(
            value.rjust(width) for value, (_, width) in zip(_row(evaluation), _HEADERS, strict=True)
        )
        for evaluation in evaluations
    )
    return "\n".join([header, "-" * len(header), *rows])


def _row(evaluation: ItemEvaluation) -> tuple[str, ...]:
    return (
        evaluation.item_id,
        str(evaluation.human_failures),
        str(evaluation.true_positive),
        str(evaluation.false_positive),
        str(evaluation.false_negative),
        str(evaluation.true_negative),
        _ratio(evaluation.precision),
        _ratio(evaluation.recall),
        _ratio(evaluation.f1),
        _ratio(evaluation.balanced_accuracy),
        _ratio(evaluation.kappa),
        _ratio(evaluation.coverage),
        str(evaluation.abstained),
        str(evaluation.not_reported),
    )


def _ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"
