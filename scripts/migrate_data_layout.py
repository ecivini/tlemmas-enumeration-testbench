"""Migrate benchmark data to per-instance problem.smt2 files.

This is a one-time helper for Phase 1 of the data layout migration. It is a
dry run by default; pass ``--write`` to apply the planned moves and deletions.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path


DATA_ROOT = Path("data/benchmark")


@dataclass(frozen=True)
class Operation:
    kind: str
    src: Path
    dst: Path | None
    tracked: bool


class MigrationError(RuntimeError):
    """Raised when the data tree cannot be migrated safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate data/benchmark formulas to problem.smt2 layout."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Apply the migration. Without this flag, only print a dry run.",
    )
    return parser.parse_args()


def run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise MigrationError(f"git {' '.join(args)} failed: {message}")
    return result.stdout


def tracked_files(root: Path) -> set[Path]:
    output = run_git(["ls-files", "--", root.as_posix()])
    return {Path(line) for line in output.splitlines()}


def is_under_queries(path: Path) -> bool:
    try:
        relative = path.relative_to(DATA_ROOT)
    except ValueError:
        return False
    return "queries" in relative.parts


def is_transition_file(path: Path) -> bool:
    return path.name.startswith("transition") and path.suffix == ".smt2"


def is_synthetic_file(path: Path) -> bool:
    try:
        relative = path.relative_to(DATA_ROOT)
    except ValueError:
        return False
    return len(relative.parts) > 1 and relative.parts[0] == "synthetic"


def is_synthetic_nested_problem(path: Path) -> bool:
    if path.name != "problem.smt2" or not is_synthetic_file(path):
        return False
    return path.parent.parent.name.isdigit() and not path.parent.name.isdigit()


def build_operations(root: Path, tracked: set[Path]) -> list[Operation]:
    if not root.is_dir():
        raise MigrationError(f"{root} does not exist or is not a directory")

    operations: list[Operation] = []
    target_paths: dict[Path, Path] = {}

    for src in sorted(root.rglob("problem.smt2")):
        if not is_synthetic_nested_problem(src):
            continue
        dst = src.parent.parent / "problem.smt2"
        if dst.exists():
            raise MigrationError(f"target already exists: {dst}")
        if dst in target_paths:
            raise MigrationError(
                f"multiple sources target {dst}: {target_paths[dst]} and {src}"
            )
        target_paths[dst] = src
        operations.append(
            Operation(
                "synthetic_problem_repair",
                src,
                dst,
                Path(src.as_posix()) in tracked,
            )
        )

    for src in sorted(root.rglob("*.smt2")):
        if is_under_queries(src) or src.name == "problem.smt2":
            continue

        relative_src = Path(src.as_posix())
        is_tracked = relative_src in tracked

        if is_transition_file(src):
            operations.append(
                Operation("transition_delete", src, None, is_tracked)
            )
            continue

        if src.name == "encoding.smt2":
            dst = src.with_name("problem.smt2")
            kind = "encoding_rename"
        elif is_synthetic_file(src):
            dst = src.parent / "problem.smt2"
            kind = "synthetic_flat_move"
        else:
            target_dir = src.with_suffix("")
            if target_dir.exists():
                raise MigrationError(
                    "flat formula target directory already exists: "
                    f"{target_dir}"
                )
            dst = target_dir / "problem.smt2"
            kind = "flat_move"

        if dst.exists():
            raise MigrationError(f"target already exists: {dst}")
        if dst in target_paths:
            raise MigrationError(
                f"multiple sources target {dst}: {target_paths[dst]} and {src}"
            )
        target_paths[dst] = src
        operations.append(Operation(kind, src, dst, is_tracked))

    return operations


def summarize(operations: list[Operation]) -> dict[str, int]:
    summary = {
        "flat_moves": 0,
        "synthetic_flat_moves": 0,
        "synthetic_problem_repairs": 0,
        "encoding_renames": 0,
        "transition_deletes": 0,
        "tracked_operations": 0,
        "filesystem_operations": 0,
    }
    for operation in operations:
        if operation.kind == "flat_move":
            summary["flat_moves"] += 1
        elif operation.kind == "synthetic_flat_move":
            summary["synthetic_flat_moves"] += 1
        elif operation.kind == "synthetic_problem_repair":
            summary["synthetic_problem_repairs"] += 1
        elif operation.kind == "encoding_rename":
            summary["encoding_renames"] += 1
        elif operation.kind == "transition_delete":
            summary["transition_deletes"] += 1

        if operation.tracked:
            summary["tracked_operations"] += 1
        else:
            summary["filesystem_operations"] += 1

    return summary


def print_summary(operations: list[Operation], write: bool) -> None:
    mode = "write" if write else "dry-run"
    print(f"mode: {mode}")
    for key, value in summarize(operations).items():
        print(f"{key}: {value}")


def apply_operation(operation: Operation) -> None:
    if operation.kind in {
        "flat_move",
        "synthetic_flat_move",
        "synthetic_problem_repair",
        "encoding_rename",
    }:
        if operation.dst is None:
            raise MigrationError(f"missing destination for {operation.src}")
        if operation.kind == "flat_move":
            operation.dst.parent.mkdir(parents=True, exist_ok=False)
        if operation.tracked:
            run_git(["mv", operation.src.as_posix(), operation.dst.as_posix()])
        else:
            operation.src.rename(operation.dst)
        if operation.kind == "synthetic_problem_repair":
            operation.src.parent.rmdir()
        return

    if operation.kind == "transition_delete":
        if operation.tracked:
            run_git(["rm", operation.src.as_posix()])
        else:
            operation.src.unlink()
        return

    raise MigrationError(f"unknown operation kind: {operation.kind}")


def validate_no_transitions(root: Path) -> None:
    transitions = [
        path for path in root.rglob("transition*.smt2") if not is_under_queries(path)
    ]
    if transitions:
        raise MigrationError(
            "transition files remain after migration: "
            + ", ".join(path.as_posix() for path in transitions[:10])
        )


def main() -> int:
    args = parse_args()
    try:
        tracked = tracked_files(DATA_ROOT)
        operations = build_operations(DATA_ROOT, tracked)
        print_summary(operations, args.write)

        if not args.write:
            return 0

        for operation in operations:
            apply_operation(operation)
        validate_no_transitions(DATA_ROOT)
    except MigrationError as error:
        print(f"error: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
