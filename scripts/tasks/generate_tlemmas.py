"""Generate T-lemmas for a single SMT formula."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Literal, get_args

from enumerators.solvers import (
    DivideByPartialAllSMTStrategy,
    DivideByProjectedEnumerationStrategy,
    DivideStrategy,
    MathSATDivideAndConquerEnumerator,
    MathSATTotalEnumerator,
    SMTEnumerator,
    WithPartitioningWrapper,
    WithProjectionWrapper,
)
from enumerators.util.pysmt import SuspendTypeChecking
from enumerators.walkers.normalizer import NormalizerWalker
from pysmt.fnode import FNode
from pysmt.shortcuts import And, read_smtlib, write_smtlib

DIVIDE_STRATEGIES: dict[str, DivideStrategy] = {
    "partial": DivideByPartialAllSMTStrategy(),
    "projection": DivideByProjectedEnumerationStrategy(),
}

SOLVER = Literal["sequential", "parallel"]


def create_solver(
    solver_type: SOLVER,
    procs: int,
    projection: bool,
    partition: bool,
    divide_strategy: str,
    partition_on_formula_components: bool,
    share_tlemmas_between_partitions: bool,
    logger: dict[str, Any] | None = None,
) -> SMTEnumerator:
    """Create and configure an SMT solver for T-lemma enumeration."""
    if solver_type == "sequential":
        solver: SMTEnumerator = MathSATTotalEnumerator(
            computation_logger=logger,
        )
    else:
        solver = MathSATDivideAndConquerEnumerator(
            computation_logger=logger,
            parallel_procs=procs,
            divide_strategy=DIVIDE_STRATEGIES[divide_strategy],
        )

    if projection and not partition:
        solver = WithProjectionWrapper(solver)

    if partition:
        solver = WithPartitioningWrapper(
            solver,
            partition_on_formula_components=partition_on_formula_components,
            share_tlemmas_between_partitions=share_tlemmas_between_partitions,
            computation_logger=logger,
        )

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
    norm = NormalizerWalker(solver.get_converter())

    formula = norm.normalize(formula)
    atoms = [
        natom.arg(0) if natom.is_not() else natom
        for atom in atoms
        if not (natom := norm.normalize(atom)).is_bool_constant()
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
            # "T-Lemmas number": len(tlemmas),
            "Satisfiable": sat,
            "Total time": total_time,
        }
    )

    with (output_dir / "logs.json").open("w") as log_file:
        json.dump(logger, log_file, indent=4)

    return sat


def add_gen_args(parser: argparse.ArgumentParser) -> None:
    """Add common T-lemma generation arguments to a subparser."""
    parser.add_argument(
        "--solver",
        choices=get_args(SOLVER),
        default="parallel",
        help="Solver mode: sequential or parallel (default: parallel)",
    )
    parser.add_argument(
        "--projection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable projection on theory atoms (default: disabled)",
    )
    parser.add_argument(
        "--partition",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable partitiong wrapper (default: disabled)",
    )
    parser.add_argument(
        "--parallel-divide-strategy",
        choices=DIVIDE_STRATEGIES.keys(),
        default="partial",
        help="Divide strategy for parallel enumeration",
    )
    parser.add_argument(
        "--partition-find-components",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Find relevant formula components when partitioning (default: disabled)",
    )
    parser.add_argument(
        "--partition-share-tlemmas",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Share learned T-lemmas across partitions (default: disabled)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate T-lemmas for an SMT formula."
    )

    parser.add_argument("formula", type=Path, help="Path to the input SMT-LIB formula")
    parser.add_argument("output_dir", type=Path, help="Base directory for output files")
    parser.add_argument("procs", type=int, help="Number of parallel processes")
    parser.add_argument(
        "--queries-dir",
        type=Path,
        default=None,
        help="Directory of query formulas to add atoms from",
    )
    add_gen_args(parser)

    args = parser.parse_args()

    logger: dict = {}
    with SuspendTypeChecking():
        formula = read_smtlib(str(args.formula))
    atoms = list(formula.get_atoms())

    with SuspendTypeChecking():
        if args.queries_dir is not None:
            for query_file in sorted(args.queries_dir.glob("*.smt2")):
                query = read_smtlib(query_file)
                atoms.extend(query.get_atoms())

    solver = create_solver(
        solver_type=args.solver,
        procs=args.procs,
        projection=args.projection,
        partition=args.partition,
        divide_strategy=args.parallel_divide_strategy,
        partition_on_formula_components=args.partition_find_components,
        share_tlemmas_between_partitions=args.partition_share_tlemmas,
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
