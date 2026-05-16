"""Generate T-lemmas for a single SMT formula."""

import argparse
from pathlib import Path
from typing import get_args

from tasks.tlemma_utils import (
    DIVIDE_STRATEGIES,
    SOLVER,
    create_solver,
    read_formula,
    run_enumeration,
)


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
