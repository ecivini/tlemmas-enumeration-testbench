"""Precompute T-lemma size statistics into standardized logs."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pysmt.environment
from enumerators.util.pysmt import SuspendTypeChecking
from pysmt.fnode import FNode
from pysmt.shortcuts import read_smtlib

TLEMMAS_NUM_KEY = "Lemmas"
TLEMMAS_AVG_SIZE_KEY = "Average T-lemma size"
TLEMMAS_MEDIAN_SIZE_KEY = "Median T-lemma size"


@dataclass
class Report:
    seen: int = 0
    computed: int = 0
    unchanged: int = 0
    written: int = 0
    skipped: int = 0
    warnings: list[str] = field(default_factory=list)

    def warn(self, path: Path, message: str) -> None:
        self.warnings.append(f"{path}: {message}")


def get_clause_size(formula: FNode) -> int:
    stack = [formula]
    result = 0
    while stack:
        clause = stack.pop()
        if clause.is_or():
            stack.extend(clause.args())
        else:
            result += 1
    return result


def compute_tlemmas_stats(tlemmas: list[FNode]) -> tuple[float, float]:
    literal_counts = [get_clause_size(lemma) for lemma in tlemmas]
    if not literal_counts:
        return 0.0, 0.0
    return statistics.mean(literal_counts), statistics.median(literal_counts)


def find_tlemmas_path(logs_path: Path) -> Path | None:
    tlemmas_path = logs_path.parent / "tlemmas.smt2"
    return tlemmas_path if tlemmas_path.is_file() else None


def get_tlemmas_from_path(tlemmas_path: Path) -> list[FNode]:
    with SuspendTypeChecking():
        tlemmas_and = read_smtlib(str(tlemmas_path))
    if tlemmas_and.is_and():
        return list(tlemmas_and.args())
    if tlemmas_and.is_or():
        return [tlemmas_and]
    raise ValueError("Unexpected T-lemmas format")


def compute_stats_for_file(tlemmas_path: Path) -> tuple[int, float, float]:
    pysmt.environment.push_env()
    try:
        tlemmas = get_tlemmas_from_path(tlemmas_path)
        avg, median = compute_tlemmas_stats(tlemmas)
    finally:
        pysmt.environment.pop_env()
    return len(tlemmas), avg, median


def has_stats(logs: dict[str, Any]) -> bool:
    return TLEMMAS_AVG_SIZE_KEY in logs and TLEMMAS_MEDIAN_SIZE_KEY in logs


def process_logs_path(
    logs_path: Path, write: bool, force: bool, report: Report
) -> None:
    report.seen += 1
    try:
        logs = json.loads(logs_path.read_text())
    except json.JSONDecodeError as err:
        report.skipped += 1
        report.warn(logs_path, f"could not parse JSON: {err}")
        return

    if has_stats(logs) and not force:
        report.unchanged += 1
        return

    tlemmas_path = find_tlemmas_path(logs_path)
    if tlemmas_path is None:
        report.skipped += 1
        report.warn(logs_path, "missing tlemmas.smt2 next to logs.json")
        return

    try:
        count, avg, median = compute_stats_for_file(tlemmas_path)
    except Exception as err:
        report.skipped += 1
        report.warn(logs_path, f"could not compute stats: {err}")
        return

    expected_count = logs.get(TLEMMAS_NUM_KEY)
    if expected_count is not None and count != expected_count:
        report.warn(
            logs_path,
            f"computed {count} lemmas but logs.json reports {expected_count}",
        )

    report.computed += 1
    if not write:
        return

    logs[TLEMMAS_AVG_SIZE_KEY] = avg
    logs[TLEMMAS_MEDIAN_SIZE_KEY] = median
    logs_path.write_text(json.dumps(logs, indent=4) + "\n")
    report.written += 1


def process_results_root(
    results_root: Path,
    write: bool,
    force: bool,
    include_hidden_runs: bool,
) -> Report:
    report = Report()
    for logs_path in sorted(results_root.rglob("logs.json")):
        rel = logs_path.relative_to(results_root)
        if not include_hidden_runs and rel.parts and rel.parts[0].startswith("_"):
            continue
        process_logs_path(logs_path, write=write, force=force, report=report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse generated T-lemma SMT-LIB files and write size statistics "
            "to each neighboring logs.json."
        )
    )
    parser.add_argument(
        "results_root",
        nargs="?",
        type=Path,
        default=Path("results_standardized"),
        help="Root containing standardized result directories.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Update logs.json files. Without this, only print a dry-run report.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute stats even when logs.json already contains them.",
    )
    parser.add_argument(
        "--include-hidden-runs",
        action="store_true",
        help="Also process result runs whose top-level directory starts with '_'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = process_results_root(
        results_root=args.results_root,
        write=args.write,
        force=args.force,
        include_hidden_runs=args.include_hidden_runs,
    )

    mode = "write" if args.write else "dry-run"
    print(f"Mode: {mode}")
    print(f"Logs seen: {report.seen}")
    print(f"Computed: {report.computed}")
    print(f"Unchanged: {report.unchanged}")
    print(f"Written: {report.written}")
    print(f"Skipped: {report.skipped}")
    if report.warnings:
        print(f"Warnings: {len(report.warnings)}")
        for warning in report.warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
