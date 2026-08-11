"""Read the annotation platform's export into typed labels.

One rule matters more than the rest: an empty cell is not a pass. The 334
legacy-converted rows leave six columns empty, and reading those as 满足 inflates
every denominator that touches A4, A5, A6, B9, B13 or B14. MISSING is its own
label and never reaches a metric.

Anything this file does not recognise raises. A silent coercion here is a wrong
number in a paper.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from evo_verifier.items import ITEMS, Group, Item, items_in_group


class Label(StrEnum):
    """What a human said about one item on one asset."""

    PASS = "pass"
    FAIL = "fail"
    NA = "not-applicable"
    MISSING = "missing"
    """No answer recorded. Excluded everywhere; never treated as a pass."""


_LABEL_TEXT: dict[str, Label] = {
    "满足": Label.PASS,
    "不满足": Label.FAIL,
    "不涉及": Label.NA,
    "": Label.MISSING,
}

_YES_NO: dict[str, bool | None] = {"是": True, "否": False, "": None}

RECORD_ID = "Record ID"
CATEGORY = "Category"
PROMPT_PREVIEW = "Prompt Preview"
REVIEW_COMPLETE = "检查是否完成"
HAS_VIEWER_FAILURE = "是否存在Viewer失败项"
NEEDS_SIMULATION = "是否待仿真"
ANNOTATION_VERSION = "标注版本"
NOTES = "Notes"
ANNOTATOR = "Annotator"
LEGACY_SOURCE = "Legacy Source"
CREATED_AT = "Created At"
UPDATED_AT = "Updated At"


class AnnotationError(ValueError):
    """The export does not match the schema this module was written against."""


@dataclass(frozen=True)
class AnnotatedCase:
    """One row of the export: an asset and its fourteen human labels."""

    record_id: str
    category: str
    prompt_preview: str
    annotation_version: str
    """``new`` for the 14-item schema, ``old`` for legacy-converted rows."""
    review_complete: bool
    has_viewer_failure: bool | None
    needs_simulation: bool | None
    annotator: str
    legacy_source: str
    notes: str
    created_at: str
    updated_at: str
    labels: dict[str, Label]
    """Keyed by item id (``A1`` ... ``B14``). Always all fourteen keys."""

    def label(self, item: Item | str) -> Label:
        return self.labels[item if isinstance(item, str) else item.item_id]

    def is_failure(self, item: Item | str) -> bool:
        return self.label(item) is Label.FAIL


def load_annotations(path: Path | str) -> list[AnnotatedCase]:
    """Parse the platform's ``/api/export`` CSV.

    Raises on an unknown label, a missing column, a blank or duplicate record id.
    """
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AnnotationError(f"{path} has no header row")
        _check_columns(reader.fieldnames, path)
        cases = [_case(row, path, line) for line, row in enumerate(reader, start=2)]

    seen: set[str] = set()
    for case in cases:
        if case.record_id in seen:
            raise AnnotationError(f"{path}: duplicate record id {case.record_id}")
        seen.add(case.record_id)
    return cases


def _check_columns(fieldnames: Sequence[str], path: Path) -> None:
    present = set(fieldnames)
    required = {
        RECORD_ID,
        CATEGORY,
        PROMPT_PREVIEW,
        REVIEW_COMPLETE,
        HAS_VIEWER_FAILURE,
        NEEDS_SIMULATION,
        ANNOTATION_VERSION,
        ANNOTATOR,
        *(item.column for item in ITEMS),
    }
    missing = sorted(required - present)
    if missing:
        raise AnnotationError(f"{path} is missing columns: {missing}")


def _case(row: dict[str, str], path: Path, line: int) -> AnnotatedCase:
    record_id = (row.get(RECORD_ID) or "").strip()
    if not record_id:
        raise AnnotationError(f"{path}:{line} has a blank record id")
    labels = {item.item_id: _label(row[item.column], item, path, line) for item in ITEMS}
    return AnnotatedCase(
        record_id=record_id,
        category=(row.get(CATEGORY) or "").strip(),
        prompt_preview=row.get(PROMPT_PREVIEW) or "",
        annotation_version=(row.get(ANNOTATION_VERSION) or "").strip(),
        review_complete=_yes_no(row.get(REVIEW_COMPLETE), REVIEW_COMPLETE, path, line) is True,
        has_viewer_failure=_yes_no(row.get(HAS_VIEWER_FAILURE), HAS_VIEWER_FAILURE, path, line),
        needs_simulation=_yes_no(row.get(NEEDS_SIMULATION), NEEDS_SIMULATION, path, line),
        annotator=(row.get(ANNOTATOR) or "").strip(),
        legacy_source=(row.get(LEGACY_SOURCE) or "").strip(),
        notes=row.get(NOTES) or "",
        created_at=(row.get(CREATED_AT) or "").strip(),
        updated_at=(row.get(UPDATED_AT) or "").strip(),
        labels=labels,
    )


def _label(value: str, item: Item, path: Path, line: int) -> Label:
    try:
        return _LABEL_TEXT[(value or "").strip()]
    except KeyError:
        raise AnnotationError(
            f"{path}:{line} column {item.column!r} has unknown label {value!r}"
        ) from None


def _yes_no(value: str | None, column: str, path: Path, line: int) -> bool | None:
    try:
        return _YES_NO[(value or "").strip()]
    except KeyError:
        raise AnnotationError(
            f"{path}:{line} column {column!r} has unknown value {value!r}"
        ) from None


@dataclass(frozen=True)
class ItemCounts:
    """How one item is distributed over a set of cases."""

    item_id: str
    passed: int
    failed: int
    not_applicable: int
    missing: int

    @property
    def scored(self) -> int:
        """Cases where the item was answered pass or fail."""
        return self.passed + self.failed

    @property
    def failure_rate(self) -> float | None:
        """Share of answered cases a human called a failure."""
        return self.failed / self.scored if self.scored else None


def count_labels(cases: Iterable[AnnotatedCase], item: Item | str) -> ItemCounts:
    item_id = item if isinstance(item, str) else item.item_id
    tally = dict.fromkeys(Label, 0)
    for case in cases:
        tally[case.labels[item_id]] += 1
    return ItemCounts(
        item_id=item_id,
        passed=tally[Label.PASS],
        failed=tally[Label.FAIL],
        not_applicable=tally[Label.NA],
        missing=tally[Label.MISSING],
    )


def count_all(cases: Iterable[AnnotatedCase]) -> list[ItemCounts]:
    cases = list(cases)
    return [count_labels(cases, item) for item in ITEMS]


def failures(cases: Iterable[AnnotatedCase], item: Item | str) -> list[AnnotatedCase]:
    """The cases a human marked as failing this item.

    These are the positives. There are not many of them for B7-B10, which is why
    threshold calibration has to lean on injected faults instead.
    """
    item_id = item if isinstance(item, str) else item.item_id
    return [case for case in cases if case.labels[item_id] is Label.FAIL]


def group_failures(cases: Iterable[AnnotatedCase], group: Group) -> Iterator[AnnotatedCase]:
    """Cases failing at least one item in the group, each yielded once."""
    owned = items_in_group(group)
    for case in cases:
        if any(case.labels[item.item_id] is Label.FAIL for item in owned):
            yield case
