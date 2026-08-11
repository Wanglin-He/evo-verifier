from __future__ import annotations

import pytest

from evo_verifier.report import (
    Coverage,
    ItemResult,
    Prediction,
    ReportError,
    VerificationReport,
    collect,
)


def test_prediction_follows_the_threshold():
    assert ItemResult.scored("B7", 0.70).prediction is Prediction.PASS
    assert ItemResult.scored("B7", 0.699, failure_reason="below").prediction is Prediction.FAIL


def test_low_confidence_abstains_instead_of_guessing():
    result = ItemResult.scored("B10", 0.2, confidence=0.4)
    assert result.prediction is Prediction.ABSTAIN
    assert not result.counts_toward_score


def test_not_applicable_carries_no_score_and_no_prediction():
    result = ItemResult.not_applicable("B14", "no coupling in the contract")
    assert result.score is None
    assert result.prediction is None
    assert result.coverage is Coverage.NOT_APPLICABLE


def test_unsupported_abstains():
    result = ItemResult.unsupported("A4", "VLM not calibrated")
    assert result.prediction is Prediction.ABSTAIN
    assert not result.counts_toward_score


def test_a_scoreless_coverage_cannot_be_scored():
    with pytest.raises(ReportError, match="cannot be scored"):
        ItemResult.scored("B14", 0.5, coverage=Coverage.NOT_APPLICABLE)


def test_a_failure_must_say_what_failed():
    with pytest.raises(ReportError, match="say what failed"):
        ItemResult("B7", 0.1, Prediction.FAIL, 0.7, 1.0, Coverage.FULL)


def test_scores_stay_in_range():
    with pytest.raises(ReportError, match="outside"):
        ItemResult("B7", 1.4, Prediction.PASS, 0.7, 1.0, Coverage.FULL)


def test_unknown_item_id_raises():
    with pytest.raises(ReportError, match="unknown item"):
        ItemResult.scored("B15", 0.9)


def test_reporting_an_item_twice_raises():
    report = VerificationReport(asset_id="rec_x", verifier_version="v0")
    report.add(ItemResult.scored("B7", 0.9))
    with pytest.raises(ReportError, match="twice"):
        report.add(ItemResult.scored("B7", 0.8))


def _four_families() -> VerificationReport:
    return collect(
        "rec_x",
        "v0",
        [
            ItemResult.scored("B7", 0.8),  # semantic
            ItemResult.scored("B10", 0.6, failure_reason="axis off"),  # static
            ItemResult.scored("B11", 0.4, failure_reason="blocked"),  # motion
            ItemResult.scored("B12", 0.2, failure_reason="no hinge"),  # physics
        ],
    )


def test_score_full_weights_the_families():
    report = _four_families()
    assert report.score_full() == pytest.approx(48.0)
    assert report.human_coverage() == pytest.approx(1.0)


def test_an_empty_family_leaves_numerator_and_denominator_together():
    """Dropping physics must not act like scoring physics at zero."""
    report = _four_families()
    del report.items["B12"]
    report.add(ItemResult.not_applicable("B12", "no mount required"))
    assert report.score_full() == pytest.approx(100 * 0.43 / 0.75)


def test_abstained_items_lower_coverage_without_touching_the_score():
    report = _four_families()
    del report.items["B11"]
    report.add(ItemResult.unsupported("B11", "no simulator"))
    assert report.score_full() == pytest.approx(100 * 0.36 / 0.70)
    assert report.human_coverage() == pytest.approx(0.75)


def test_missing_items_are_visible():
    report = collect("rec_x", "v0", [ItemResult.scored("B7", 0.9)])
    assert report.missing_items() == (
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "B8",
        "B9",
        "B10",
        "B11",
        "B12",
        "B13",
        "B14",
    )


def test_round_trip_through_json(tmp_path):
    report = _four_families()
    report.repair_queue.append("add hinge geometry")
    path = tmp_path / "rec_x.json"
    report.save(path)
    loaded = VerificationReport.load(path)
    assert loaded.to_dict() == report.to_dict()
    assert loaded.items["B10"].failure_reason == "axis off"
