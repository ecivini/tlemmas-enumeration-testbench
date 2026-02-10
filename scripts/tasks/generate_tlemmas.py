from theorydd.solvers.mathsat_total import MathSATTotalEnumerator
from theorydd.solvers.mathsat_partial_extended import MathSATExtendedPartialEnumerator
from theorydd.solvers.with_partitioning import WithPartitioningWrapper
from theorydd.formula import get_normalized
from pysmt.shortcuts import read_smtlib, And, write_smtlib

import sys
import os
import json
import time


def main():
    if len(sys.argv) != 6:
        print(
            "Usage: python3 scripts/tasks/compile_tasks.py <input formula> "
            "<base output path> <allsmt_processes> <solver> <project t-atoms>"
        )
        sys.exit(1)

    # Check base output path exists, otherwise create it
    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2])

    logger = {}
    phi = read_smtlib(sys.argv[1])

    # Find project_on_theory_atoms
    project_tatoms = sys.argv[5].lower() == "true"

    # Create the appropriate solver
    solver_name = sys.argv[4].lower().strip()
    if solver_name == "sequential":
        solver = MathSATTotalEnumerator(
            project_on_theory_atoms=project_tatoms, computation_logger=logger
        )
    elif solver_name == "parallel":
        solver = MathSATExtendedPartialEnumerator(
            project_on_theory_atoms=project_tatoms,
            computation_logger=logger,
            parallel_procs=int(sys.argv[3]),
        )
    elif solver_name == "partition":
        solver = WithPartitioningWrapper(
            MathSATExtendedPartialEnumerator(
                project_on_theory_atoms=project_tatoms,
                computation_logger=logger,
                parallel_procs=int(sys.argv[3]),
            ),
            computation_logger=logger,
        )
    else:
        raise ValueError("Invalid solver")

    # Normalize phi
    phi = get_normalized(phi, solver.get_converter())

    start = time.time()
    sat = "unknown"
    try:
        sat = solver.check_all_sat(phi)
    except Exception:
        print(f"[-] Exception during compilation of {sys.argv[1]}")
        sys.exit(1)
    total_time = time.time() - start

    # Store T-lemmas
    tlemmas = solver.get_theory_lemmas()
    tlemmas_and = And(tlemmas)

    tlemmas_path = os.path.join(sys.argv[2], "tlemmas.smt2")
    write_smtlib(tlemmas_and, tlemmas_path)

    logger["T-Lemmas number"] = len(tlemmas)
    logger["Satisfiable"] = sat
    logger["Total time"] = total_time

    log_path = os.path.join(sys.argv[2], "logs.json")
    with open(log_path, "w") as log_file:
        json.dump(logger, log_file, indent=4)


if __name__ == "__main__":
    main()
