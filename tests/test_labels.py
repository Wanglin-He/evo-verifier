from __future__ import annotations

import pytest
from conftest import write_annotations

from evo_verifier.items import BY_ID
from evo_verifier.labels import (
    AnnotationError,
    Label,
    count_labels,
    failures,
    group_failures,
    load_annotations,
)

B9 = BY_ID["B9"].column
B7 = BY_ID["B7"].column


def test_blank_cell_is_missing_not_pass(tmp_path):
    """The 334 legacy rows leave six columns empty. Empty must not become 满足."""
    path = write_annotations(
        tmp_path / "a.csv", [{B9: "", "标注版本": "old", "检查是否完成": "否"}]
    )
    (case,) = load_annotations(path)
    assert case.labels["B9"] is Label.MISSING
    assert case.labels["B7"] is Label.PASS


def test_three_answers_map_to_three_labels(tmp_path):
    path = write_annotations(
        tmp_path / "a.csv",
        [{B7: "满足"}, {B7: "不满足"}, {B7: "不涉及"}],
    )
    cases = load_annotations(path)
    assert [case.labels["B7"] for case in cases] == [Label.PASS, Label.FAIL, Label.NA]


def test_unknown_label_raises(tmp_path):
    path = write_annotations(tmp_path / "a.csv", [{B7: "部分满足"}])
    with pytest.raises(AnnotationError, match="unknown label"):
        load_annotations(path)


def test_duplicate_record_id_raises(tmp_path):
    path = write_annotations(tmp_path / "a.csv", [{"Record ID": "rec_x"}, {"Record ID": "rec_x"}])
    with pytest.raises(AnnotationError, match="duplicate record id"):
        load_annotations(path)


def test_missing_column_raises(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("Record ID,Category\nrec_0,chair\n", encoding="utf-8")
    with pytest.raises(AnnotationError, match="missing columns"):
        load_annotations(path)


def test_counts_exclude_missing_from_the_failure_rate(tmp_path):
    path = write_annotations(
        tmp_path / "a.csv",
        [{B9: "满足"}, {B9: "不满足"}, {B9: ""}, {B9: "不涉及"}],
    )
    counts = count_labels(load_annotations(path), "B9")
    assert (counts.passed, counts.failed, counts.not_applicable, counts.missing) == (1, 1, 1, 1)
    assert counts.scored == 2
    assert counts.failure_rate == 0.5


def test_group_failures_yields_each_case_once(tmp_path):
    path = write_annotations(
        tmp_path / "a.csv",
        [{B7: "不满足", B9: "不满足"}, {B7: "满足"}],
    )
    cases = load_annotations(path)
    assert [case.record_id for case in group_failures(cases, "B7-B10")] == ["rec_0"]


def test_snapshot_shape(snapshot_cases):
    """Guards the numbers the plan was built on. Re-export changes this on purpose."""
    assert len(snapshot_cases) == 607
    assert sum(1 for case in snapshot_cases if case.review_complete) == 273
    assert {case.annotation_version for case in snapshot_cases} == {"new", "old"}


def test_snapshot_b7_b10_positives_are_scarce(snapshot_cases):
    """47 positives across four items is the constraint the whole plan works around."""
    counts = {item: len(failures(snapshot_cases, item)) for item in ("B7", "B8", "B9", "B10")}
    assert counts == {"B7": 8, "B8": 16, "B9": 13, "B10": 10}


def test_snapshot_legacy_rows_leave_six_items_unanswered(snapshot_cases):
    old = [case for case in snapshot_cases if case.annotation_version == "old"]
    blank = {
        item_id for item_id in BY_ID if all(case.labels[item_id] is Label.MISSING for case in old)
    }
    assert blank == {"A4", "A5", "A6", "B9", "B13", "B14"}
