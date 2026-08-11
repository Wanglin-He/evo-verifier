"""``python -m evo_verifier`` -- look at the labels, or score a run of reports."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from evo_verifier.evaluate import evaluate, format_table
from evo_verifier.items import ITEMS, items_in_group
from evo_verifier.labels import AnnotatedCase, count_all, load_annotations
from evo_verifier.report import load_reports

DEFAULT_ANNOTATIONS = Path(__file__).resolve().parent.parent / "data" / "annotations-2026-08-11.csv"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evo_verifier", description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help="annotation export CSV (default: the frozen snapshot in data/)",
    )
    parser.add_argument(
        "--group",
        choices=("A1-A6", "B7-B10", "B11-B14"),
        help="restrict to the items one person owns",
    )
    parser.add_argument(
        "--complete-only",
        action="store_true",
        help="only rows annotated under the 14-item schema",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("stats", help="label distribution per item")
    scored = subparsers.add_parser("evaluate", help="agreement between reports and humans")
    scored.add_argument("reports", type=Path, help="directory of report JSON files")

    args = parser.parse_args(argv)
    cases = load_annotations(args.annotations)
    if args.complete_only:
        cases = [case for case in cases if case.review_complete]
    items = items_in_group(args.group) if args.group else ITEMS

    if args.command == "stats":
        _print_stats(cases, items)
        return 0

    reports = load_reports(args.reports)
    matched = sum(1 for case in cases if case.record_id in reports)
    print(f"{len(cases)} cases, {len(reports)} reports, {matched} matched\n")
    print(format_table(evaluate(cases, reports, items)))
    return 0


def _print_stats(cases: list[AnnotatedCase], items: Sequence[object]) -> None:
    wanted = {item.item_id for item in items}
    versions = sorted({case.annotation_version for case in cases})
    print(f"{len(cases)} cases, versions: {', '.join(versions)}")
    print(f"{sum(1 for case in cases if case.review_complete)} marked complete\n")
    header = f"{'item':5} {'pass':>6} {'fail':>6} {'n/a':>5} {'missing':>8} {'fail rate':>10}"
    print(header)
    print("-" * len(header))
    for counts in count_all(cases):
        if counts.item_id not in wanted:
            continue
        rate = "-" if counts.failure_rate is None else f"{counts.failure_rate:.1%}"
        print(
            f"{counts.item_id:5} {counts.passed:6} {counts.failed:6} "
            f"{counts.not_applicable:5} {counts.missing:8} {rate:>10}"
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
