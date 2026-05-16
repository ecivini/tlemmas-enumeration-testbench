import argparse
import json
import sys
import time
from pathlib import Path

from enumerators.formula import get_normalized
from enumerators.solvers.mathsat_partial_extended import (
    DivideByPartialAllSMTStrategy,
    DivideByProjectedEnumerationStrategy,
    MathSATExtendedPartialEnumerator,
)
from enumerators.solvers.mathsat_total import MathSATTotalEnumerator
from enumerators.solvers.with_partitioning import WithPartitioningWrapper
from pysmt.shortcuts import And, read_smtlib, write_smtlib

DIVIDE_STRATEGIES = {
    "partial": DivideByPartialAllSMTStrategy,
    "projection": DivideByProjectedEnumerationStrategy,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate T-lemmas for an SMT formula."
    )

    # Positional arguments
    parser.add_argument("formula", type=Path, help="Path to the input SMT-LIB formula")
    parser.add_argument("output_dir", type=Path, help="Base directory for output files")
    parser.add_argument("procs", type=int, help="Number of parallel processes")
    parser.add_argument(
        "solver", choices=["sequential", "parallel"], help="Base solver type"
    )

    # Optional flags
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

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = {}

    try:
        phi = read_smtlib(str(args.formula))
    except Exception as e:
        print(f"[-] Failed to read formula {args.formula}: {e}")
        sys.exit(1)

    if args.solver == "sequential":
        solver = MathSATTotalEnumerator(
            project_on_theory_atoms=args.projection, computation_logger=logger
        )
    else:  # parallel
        solver = MathSATExtendedPartialEnumerator(
            project_on_theory_atoms=args.projection,
            computation_logger=logger,
            parallel_procs=args.procs,
            divide_strategy=DIVIDE_STRATEGIES[args.parallel_divide_strategy],
        )

    if args.partition:
        solver = WithPartitioningWrapper(solver, computation_logger=logger)

    phi = get_normalized(phi, solver.get_converter())

    start_time = time.time()
    try:
        sat = solver.check_all_sat(phi)
    except Exception as e:
        print(f"[-] Exception during compilation of {args.formula}: {e}")
        sys.exit(1)

    total_time = time.time() - start_time

    tlemmas = solver.get_theory_lemmas()
    write_smtlib(And(tlemmas), str(args.output_dir / "tlemmas.smt2"))

    logger.update(
        {
            "T-Lemmas number": len(tlemmas),
            "Satisfiable": sat,
            "Total time": total_time,
        }
    )

    with (args.output_dir / "logs.json").open("w") as log_file:
        json.dump(logger, log_file, indent=4)


if __name__ == "__main__":
    main()
