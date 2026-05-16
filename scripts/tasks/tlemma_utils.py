"""Shared enumeration logic for T-lemma generation tasks."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal, cast

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

DIVIDE_STRATEGIES: dict[str, type] = {
    "partial": DivideByPartialAllSMTStrategy,
    "projection": DivideByProjectedEnumerationStrategy,
}

SOLVER = Literal["sequential", "parallel"]


def create_solver(
    solver_type: SOLVER,
    procs: int,
    projection: bool,
    partition: bool,
    divide_strategy: str = "partial",
    logger: dict[str, Any] | None = None,
) -> Any:
    """Create and configure an SMT solver for T-lemma enumeration.

    Args:
        solver_type: "sequential" or "parallel"
        procs: Number of parallel processes (used when solver_type is "parallel")
        projection: Enable projection on theory atoms
        partition: Enable partitioning wrapper
        divide_strategy: Strategy for parallel division ("partial" or "projection")
        logger: Optional computation logger dict

    Returns:
        Configured SMTEnumerator instance
    """
    if solver_type == "sequential":
        solver: Any = MathSATTotalEnumerator(
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
    """Run T-lemma enumeration and save T-lemmas in output_dir.

    Args:
        formula: The SMT formula to enumerate
        atoms: List of atoms to consider for All-SMT
        solver: Configured SMTEnumerator instance
        output_dir: Directory to write tlemmas.smt2 and logs.json
        logger: Computation logger dict (mutated in place)

    Returns:
        True if satisfiable, False otherwise
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = solver.get_converter()
    formula = get_normalized(formula, converter)
    atoms = [
        natom
        for atom in atoms
        if not (natom := get_normalized(atom, converter)).is_bool_constant()
    ]
    print(f"{len(atoms)} normalized atoms")

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


def read_formula(path: Path) -> FNode:
    """Read an SMT-LIB formula from a file.

    Args:
        path: Path to the .smt2 file

    Returns:
        The parsed FNode formula

    Raises:
        SystemExit: If the file cannot be read
    """
    try:
        phi = cast(FNode, read_smtlib(str(path)))
        return phi
    except Exception as e:
        print(f"[-] Failed to read formula {path}: {e}")
        raise SystemExit(1) from e
