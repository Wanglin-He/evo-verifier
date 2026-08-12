"""``python -m evo_verifier`` -- look at the labels, or score a run of reports."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from evo_verifier.asset import find_model_py, load_asset, read_prompt
from evo_verifier.contract import ContractError
from evo_verifier.evaluate import evaluate, format_table
from evo_verifier.extract import (
    DEFAULT_MODEL,
    ENDPOINT,
    INSTRUCTION,
    ExtractionError,
    extract,
    read_api_key,
)
from evo_verifier.items import ITEMS, Item, items_in_group
from evo_verifier.labels import AnnotatedCase, Label, count_all, load_annotations
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
    parsed = subparsers.add_parser("parse", help="how much of each model.py resolves statically")
    parsed.add_argument("data_dir", type=Path, help="an articraft-data clone")
    extracted = subparsers.add_parser("extract", help="prompt -> contract, once, via a model")
    extracted.add_argument("data_dir", type=Path, help="an articraft-data clone")
    extracted.add_argument("--out", type=Path, required=True, help="directory for contract JSON")
    extracted.add_argument("--env-file", type=Path, help=".env holding GEMINI_API_KEY")
    extracted.add_argument("--model", default=DEFAULT_MODEL)
    extracted.add_argument("--limit", type=int, help="stop after this many records")
    extracted.add_argument(
        "--failures-only",
        action="store_true",
        help="only records a human failed on one of the selected items",
    )
    extracted.add_argument(
        "--sample",
        type=int,
        help="keep every human-failed record, then fill to this many with an even "
        "deterministic stride over the rest",
    )
    extracted.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be sent and to whom, call nothing",
    )
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

    if args.command == "parse":
        _print_parse(cases, args.data_dir)
        return 0

    if args.command == "extract":
        return _extract(cases, items, args)

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


def _extract(cases: list[AnnotatedCase], items: Sequence[Item], args: argparse.Namespace) -> int:
    """Send prompts to a model, once, and write the contracts to disk.

    Re-running skips records already written, so a interrupted batch resumes
    instead of paying twice.
    """
    wanted = {item.item_id for item in items}
    if args.failures_only:
        cases = [c for c in cases if any(c.labels[i] is Label.FAIL for i in wanted)]
    if args.sample:
        cases = _stratified(cases, wanted, args.sample)
    if args.limit:
        cases = cases[: args.limit]
    todo = [case for case in cases if not (args.out / f"{case.record_id}.json").exists()]

    print(f"{len(todo)} records to extract ({len(cases) - len(todo)} already on disk)")
    print(f"model: {args.model}   endpoint: {ENDPOINT.format(model=args.model)}")
    if args.dry_run:
        if todo:
            prompt = read_prompt(args.data_dir, todo[0].record_id)
            print(f"\nfirst prompt ({todo[0].record_id}):\n{prompt}\n")
            print(f"instruction is {len(INSTRUCTION)} characters, sent with every request")
        print("dry run: nothing was sent")
        return 0

    api_key = read_api_key(args.env_file)
    written, failed = 0, []
    for index, case in enumerate(todo, start=1):
        prompt = read_prompt(args.data_dir, case.record_id)
        try:
            contract = extract(case.record_id, prompt, api_key=api_key, model=args.model)
        except (ExtractionError, ContractError) as error:
            failed.append((case.record_id, str(error)[:160]))
            print(f"  [{index}/{len(todo)}] {case.record_id}: FAILED")
            continue
        contract.save(args.out / f"{case.record_id}.json")
        written += 1
        print(f"  [{index}/{len(todo)}] {case.record_id}: {len(contract.joints)} joints")

    print(f"\nwrote {written}, failed {len(failed)}")
    for record_id, error in failed:
        print(f"  {record_id}: {error}")
    return 0 if written else 1


def _stratified(cases: list[AnnotatedCase], items: set[str], size: int) -> list[AnnotatedCase]:
    """Every human failure, then an even stride over the rest up to ``size``.

    Failures are the scarce half of the data -- 47 across B7-B10 -- so all of
    them are kept and recall is measured on the lot. The rest are sampled at a
    known rate, which leaves the false-alarm rate unbiased and precision
    enriched by a factor the reporting can correct for. The stride is over
    record ids in sorted order, so the same request returns the same sample.
    """
    failed = [c for c in cases if any(c.labels[i] is Label.FAIL for i in items)]
    passed = sorted((c for c in cases if c not in failed), key=lambda case: case.record_id)
    room = max(0, size - len(failed))
    if room >= len(passed):
        return failed + passed
    stride = len(passed) / room if room else 0
    picked = [passed[int(index * stride)] for index in range(room)]
    return failed + picked


def _print_parse(cases: list[AnnotatedCase], data_dir: Path) -> None:
    """Report what the static reader recovers, and what it openly does not."""
    assets, failures = [], []
    for case in cases:
        try:
            assets.append(load_asset(find_model_py(data_dir, case.record_id), case.record_id))
        except (OSError, SyntaxError, ValueError) as error:
            failures.append((case.record_id, str(error)))
    joints = [joint for asset in assets for joint in asset.articulations]
    print(f"{len(assets)}/{len(cases)} records read, {len(failures)} unreadable")
    for record_id, error in failures[:5]:
        print(f"  {record_id}: {error}")
    if not joints:
        return

    checks = (
        ("name", lambda j: j.name != "<unnamed>"),
        ("kind", lambda j: j.kind is not None),
        ("parent", lambda j: j.parent is not None),
        ("child", lambda j: j.child is not None),
        ("origin", lambda j: j.origin.xyz is not None),
        ("axis", lambda j: j.axis is not None or j.kind == "fixed"),
    )
    print(f"\n{len(joints)} joints")
    for label, resolved in checks:
        count = sum(1 for joint in joints if resolved(joint))
        print(f"  {label:8} {count:5}  {count / len(joints):6.1%}")
    ready = sum(
        1
        for joint in joints
        if all(resolved(joint) for _, resolved in checks[1:])  # a name is cosmetic
    )
    print(f"  {'B7-B10':8} {ready:5}  {ready / len(joints):6.1%}  (every field those items read)")

    clean = sum(1 for asset in assets if asset.complete)
    scaled = sum(1 for asset in assets if asset.diagonal() is not None)
    print(f"\n{clean}/{len(assets)} records read with nothing unresolved")
    print(f"{scaled}/{len(assets)} records have a diagonal D -- the rest need the CAD run")

    reasons = Counter(note.split(":")[0] for asset in assets for note in asset.notes)
    if reasons:
        print("\nunresolved, by cause")
        for reason, count in reasons.most_common(8):
            print(f"  {count:5}  {reason}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
