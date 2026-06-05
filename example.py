from typing import cast

from enumerators.formula import get_normalized
from enumerators.solvers.mathsat_divide_and_conquer import (
    MathSATDivideAndConquerEnumerator,
)
from enumerators.solvers.with_partitioning import WithPartitioningWrapper
from pysmt.fnode import FNode
from pysmt.shortcuts import TRUE, Solver, read_smtlib

# Create variables
phi = cast(FNode, read_smtlib("data/benchmark/planning/h4/Painter/10_3.smt2"))
with Solver("msat") as msat:
    phi = get_normalized(phi, msat.converter)

print("Enumerating with phi")
# Enumerate using Divide & Conquer enumeration
enumerator = WithPartitioningWrapper(
    MathSATDivideAndConquerEnumerator(parallel_procs=8)
)
result = enumerator.check_all_sat(phi)

# Get the theory lemmas
lemmas = enumerator.get_theory_lemmas()
print(f"Found {len(lemmas)} theory lemmas")
print(f"Model count: {enumerator.get_models_count()}")

print("Enumerating with phi")
# Enumerate using Divide & Conquer enumeration
enumerator = WithPartitioningWrapper(
    MathSATDivideAndConquerEnumerator(parallel_procs=8)
)
result = enumerator.check_all_sat(TRUE(), atoms=phi.get_atoms())

# Get the theory lemmas
lemmas = enumerator.get_theory_lemmas()
print(f"Found {len(lemmas)} theory lemmas")
print(f"Model count: {enumerator.get_models_count()}")
