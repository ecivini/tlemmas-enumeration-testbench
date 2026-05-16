"""Generate T-lemmas for a single SMT formula."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Literal, cast, get_args

from enumerators.formula import get_normalized
from enumerators.solvers.mathsat_partial_extended import (
    DivideByPartialAllSMTStrategy,
    DivideByProjectedEnumerationStrategy,
    MathSATExtendedPartialEnumerator,
)
from enumerators.solvers.mathsat_total import MathSATTotalEnumerator
from enumerators.solvers.solver import SMTEnumerator
from enumerators.solvers.with_partitioning import WithPartitioningWrapper
from pysmt.fnode import FNode
from pysmt.shortcuts import And, read_smtlib, write_smtlib

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIVIDE_STRATEGIES: dict[str, type] = {
    "partial": DivideByPartialAllSMTStrategy,
    "projection": DivideByProjectedEnumerationStrategy,
}

SOLVER = Literal["sequential", "parallel"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_formula(path: Path) -> FNode:
    """Read an SMT-LIB formula from a file."""
    try:
        phi = cast(FNode, read_smtlib(str(path)))
        return phi
    except Exception as e:
        print(f"[-] Failed to read formula {path}: {e}")
        raise SystemExit(1) from e


def create_solver(
    solver_type: SOLVER,
    procs: int,
    projection: bool,
    partition: bool,
    divide_strategy: str = "partial",
    logger: dict[str, Any] | None = None,
) -> SMTEnumerator:
    """Create and configure an SMT solver for T-lemma enumeration."""
    if solver_type == "sequential":
        solver: SMTEnumerator = MathSATTotalEnumerator(
            project_on_theory_atoms=projection,
            computation_logger=logger,
        )
    else:
        solver = MathSATExtendedPartialEnumerator(
            project_on_theory_atoms=projection,
            computation_logger=logger,
            parallel_procs=procs,
            divide_strategy=DIVIDE_STRATEGIES[divide_strategy],
        )

    if partition:
        solver = WithPartitioningWrapper(solver, computation_logger=logger)

    return solver


def run_enumeration(
    formula: FNode,
    atoms: list[FNode],
    solver: SMTEnumerator,
    output_dir: Path,
    logger: dict[str, Any],
) -> bool:
    """Run T-lemma enumeration and save results to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = solver.get_converter()
    formula = get_normalized(formula, converter)
    atoms = [
        natom
        for atom in atoms
        if not (natom := get_normalized(atom, converter)).is_bool_constant()
    ]

    if logger is not None:
        logger["Atoms count"] = len(atoms)

    start_time = time.time()
    sat = solver.check_all_sat(formula, atoms=atoms)
    total_time = time.time() - start_time

    tlemmas = solver.get_theory_lemmas()
    write_smtlib(And(tlemmas), str(output_dir / "tlemmas.smt2"))

    logger.update(
        {
            "T-Lemmas number": len(tlemmas),
            "Satisfiable": sat,
            "Total time": total_time,
        }
    )

    with (output_dir / "logs.json").open("w") as log_file:
        json.dump(logger, log_file, indent=4)

    return sat


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate T-lemmas for an SMT formula."
    )

    parser.add_argument("formula", type=Path, help="Path to the input SMT-LIB formula")
    parser.add_argument("output_dir", type=Path, help="Base directory for output files")
    parser.add_argument("procs", type=int, help="Number of parallel processes")
    parser.add_argument("solver", choices=get_args(SOLVER), help="Base solver type")

    parser.add_argument(
        "--projection", action="store_true", help="Enable projection on theory atoms"
    )
    parser.add_argument(
        "--partition", action="store_true", help="Enable partitioning wrapper"
    )
    parser.add_argument(
        "--parallel-divide-strategy",
        choices=DIVIDE_STRATEGIES.keys(),
        default="partial",
        help="Divide strategy for parallel enumeration",
    )
    parser.add_argument(
        "--queries-dir",
        type=Path,
        default=None,
        help="Directory of query formulas to add atoms from",
    )

    args = parser.parse_args()

    logger: dict = {}
    formula = read_formula(args.formula)
    atoms = list(formula.get_atoms())

    if args.queries_dir is not None:
        for query_file in sorted(args.queries_dir.glob("*.smt2")):
            query = read_formula(query_file)
            atoms.extend(query.get_atoms())

    solver = create_solver(
        solver_type=args.solver,
        procs=args.procs,
        projection=args.projection,
        partition=args.partition,
        divide_strategy=args.parallel_divide_strategy,
        logger=logger,
    )

    run_enumeration(
        formula=formula,
        atoms=atoms,
        solver=solver,
        output_dir=args.output_dir,
        logger=logger,
    )


if __name__ == "__main__":
    main()
