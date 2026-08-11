from __future__ import annotations

import pytest
from conftest import write_annotations

from evo_verifier.evaluate import evaluate, evaluate_item, format_table
from evo_verifier.items import BY_ID
from evo_verifier.labels import load_annotations
from evo_verifier.report import ItemResult, VerificationReport, collect

B7 = BY_ID["B7"].column


def _report(asset_id: str, result: ItemResult | None) -> VerificationReport:
    return collect(asset_id, "v0", [result] if result else [])


@pytest.fixture
def confusion(tmp_path):
    """One case per cell, plus the three states that leave the table."""
    path = write_annotations(
        tmp_path / "a.csv",
        [
            {B7: "不满足"},  # rec_0 verifier fail -> TP
            {B7: "满足"},  # rec_1 verifier fail -> FP
            {B7: "不满足"},  # rec_2 verifier pass -> FN
            {B7: "满足"},  # rec_3 verifier pass -> TN
            {B7: "不满足"},  # rec_4 verifier abstains
            {B7: "不涉及"},  # rec_5 human N/A
            {B7: ""},  # rec_6 human missing
            {B7: "满足"},  # rec_7 no report
        ],
    )
    fail = ItemResult.scored("B7", 0.1, failure_reason="wrong parent")
    reports = {
        "rec_0": _report("rec_0", fail),
        "rec_1": _report("rec_1", fail),
        "rec_2": _report("rec_2", ItemResult.scored("B7", 0.9)),
        "rec_3": _report("rec_3", ItemResult.scored("B7", 0.9)),
        "rec_4": _report("rec_4", ItemResult.scored("B7", 0.1, confidence=0.2)),
        "rec_5": _report("rec_5", ItemResult.scored("B7", 0.9)),
        "rec_6": _report("rec_6", ItemResult.scored("B7", 0.9)),
    }
    return load_annotations(path), reports


def test_confusion_cells(confusion):
    cases, reports = confusion
    result = evaluate_item(cases, reports, "B7")
    assert (result.true_positive, result.false_positive) == (1, 1)
    assert (result.false_negative, result.true_negative) == (1, 1)


def test_human_na_and_missing_leave_the_table(confusion):
    cases, reports = confusion
    result = evaluate_item(cases, reports, "B7")
    assert result.human_na == 1
    assert result.human_missing == 1
    assert result.decided == 4


def test_an_abstention_is_lost_coverage_not_a_false_negative(confusion):
    cases, reports = confusion
    result = evaluate_item(cases, reports, "B7")
    assert result.abstained == 1
    assert result.false_negative == 1  # rec_2 only
    assert result.not_reported == 1
    assert result.eligible == 6
    assert result.coverage == pytest.approx(4 / 6)


def test_metrics(confusion):
    cases, reports = confusion
    result = evaluate_item(cases, reports, "B7")
    assert result.human_failures == 2
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.f1 == pytest.approx(0.5)
    assert result.balanced_accuracy == pytest.approx(0.5)
    assert result.kappa == pytest.approx(0.0)


def test_kappa_punishes_always_predicting_pass(tmp_path):
    """The B7 trap: 98% of labels are 满足, so accuracy alone looks excellent."""
    rows = [{B7: "满足"} for _ in range(49)] + [{B7: "不满足"}]
    cases = load_annotations(write_annotations(tmp_path / "a.csv", rows))
    reports = {
        case.record_id: _report(case.record_id, ItemResult.scored("B7", 0.9)) for case in cases
    }
    result = evaluate_item(cases, reports, "B7")
    accuracy = (result.true_positive + result.true_negative) / result.decided
    assert accuracy == pytest.approx(0.98)
    assert result.kappa == pytest.approx(0.0)
    assert result.f1 is None  # nothing claimed, nothing found


def test_table_lists_every_requested_item(confusion):
    cases, reports = confusion
    table = format_table(evaluate(cases, reports, ["B7", "B8"]))
    assert table.splitlines()[2].startswith("   B7")
    assert table.splitlines()[3].startswith("   B8")
