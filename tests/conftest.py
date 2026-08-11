from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from evo_verifier.items import ITEMS

SNAPSHOT = Path(__file__).resolve().parent.parent / "data" / "annotations-2026-08-11.csv"

_META = [
    "Record ID",
    "Category",
    "Prompt Preview",
    "检查是否完成",
    "是否存在Viewer失败项",
    "是否待仿真",
    "标注版本",
]
_TAIL = ["Notes", "Annotator", "Legacy Source", "Created At", "Updated At"]


def write_annotations(path: Path, rows: Sequence[Mapping[str, str]]) -> Path:
    """Write an export-shaped CSV. Unset item columns default to 满足."""
    columns = [*_META, *(item.column for item in ITEMS), *_TAIL]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, row in enumerate(rows):
            record = dict.fromkeys(columns, "")
            record.update({item.column: "满足" for item in ITEMS})
            record["Record ID"] = f"rec_{index}"
            record["检查是否完成"] = "是"
            record["标注版本"] = "new"
            record.update(row)
            writer.writerow(record)
    return path


@pytest.fixture
def snapshot_cases():
    """The real export, as downloaded on 2026-08-11."""
    from evo_verifier.labels import load_annotations

    return load_annotations(SNAPSHOT)
