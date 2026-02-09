import json
import os
import sys
import time
from typing import Iterable

from pysmt.fnode import FNode
from pysmt.oracles import get_logic
from pysmt.shortcuts import And, Iff, Not, Solver, read_smtlib
from theorydd.formula import get_normalized
from theorydd.walkers.walker_bool_abstraction import BooleanAbstractionWalker
from theorydd.walkers.walker_refinement import RefinementWalker

from tabularallsat import TabularAllSATInterface


def assert_models_are_tsat(phi: FNode, models: list[Iterable[FNode]]) -> None:
    with Solver() as check_solver:
        check_solver.add_assertion(phi)
        for model in models:
            check_solver.push()
            check_solver.add_assertions(model)
            sat = check_solver.solve()
            assert sat, "T-UNSAT model found: {}".format(model)
            check_solver.pop()


def assert_lemmas_are_tvalid(lemmas: list[FNode]):
    with Solver("msat") as check_solver:
        for lemma in lemmas:
            assert check_solver.is_valid(lemma), "Lemma {} is not valid".format(
                lemma.serialize()
            )


def assert_phi_equiv_phi_and_lemmas(phi: FNode, phi_and_lemmas):
    with Solver("msat") as check_solver:
        assert check_solver.is_valid(
            Iff(phi, phi_and_lemmas)
        ), "Phi and Phi & lemmas are not theory-equivalent"


def process_raw_tlemmas(raw_tlemmas: FNode) -> list[FNode]:
    if raw_tlemmas.is_and():
        return list(raw_tlemmas.args())
    elif raw_tlemmas.is_or():
        return [raw_tlemmas]
    else:
        raise ValueError("Unexpected T-lemmas format")


def gt_model_count(logs: dict) -> int:
    return logs["T-DDNNF"]["Total models"]


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: python3 scripts/tasks/tlemmas_check.py <input formula> "
            "<base output path> <tlemmas to check path> <ground truth logs path | optional>"
        )
        sys.exit(1)

    tabularallsat_path = os.getenv("TABULARALLSAT_PATH")
    if tabularallsat_path is None:
        raise ValueError("TABULARALLSAT_PATH environment variable not set")

    # Check base output path exists, otherwise create it
    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2])

    phi = read_smtlib(sys.argv[1])
    # Process T-lemmas to check
    tlemmas_path = sys.argv[3]
    if not os.path.isfile(tlemmas_path):
        print("[-] Invalid tlemmas path")
        sys.exit(1)
    tlemmas = process_raw_tlemmas(read_smtlib(tlemmas_path))

    # Read logs file
    gt_logs = None
    if len(sys.argv) >= 5 and os.path.isfile(sys.argv[4]):
        with open(sys.argv[4], "r") as logs_file:
            gt_logs = json.load(logs_file)

    logger = {}

    start_time = time.time()

    normalize_solver = Solver("msat")
    phi = get_normalized(phi, normalize_solver.converter)

    # ---- Generate lemmas ----
    print("Generating T-lemmas...")
    phi_atoms = list(phi.get_atoms())

    # ---- Build Boolean abstraction of phi & lemmas ----
    print("Normalizing T-lemmas...")
    lemmas = [get_normalized(lemma, normalize_solver.converter) for lemma in tlemmas]

    print("building boolean abstraction...")
    phi_and_lemmas = And(phi, And(lemmas))
    phi_and_lemmas_atoms = phi_and_lemmas.get_atoms()
    assert set(phi_atoms) <= phi_and_lemmas_atoms
    bool_walker = BooleanAbstractionWalker(atoms=phi_and_lemmas_atoms)
    phi_and_lemmas_abstr = bool_walker.walk(phi_and_lemmas)
    phi_abstr = bool_walker.walk(phi)
    assert len(phi_abstr.get_atoms()) == len(
        phi_atoms
    ), "Abstraction should preserve atoms of phi"

    # NOTE: Some lemmas introduce fresh Skolem variables, which should be existentially quantified for the lemma to
    # be t-valid.
    # However, MathSAT does not support quantifiers, and will flag these lemmas as non t-valid.
    # Anyway, these new variables only appear in fresh atoms, which are later existentially quantified, so that
    # correctness is preserved.
    # It seems the only case this happens is with arrays (e.g. extensionality lemma), so we skip the following checks
    # in that case.
    if not get_logic(phi).theory.arrays:
        assert_lemmas_are_tvalid(lemmas)
        # assert_phi_equiv_phi_and_lemmas(phi, phi_and_lemmas)

    print("Running AllSAT on Boolean abstraction ...")
    solver_abstr = TabularAllSATInterface(tabularallsat_path)

    # Check phi_and_lemmas is t-reduced
    print("Refining Boolean abstraction ...")
    # enumerate projected on theory atoms, as T-satisfiability only depends on them
    theory_atoms = [atom for atom in phi_atoms if atom.is_theory_relation()]
    theory_atoms_abstr = [bool_walker.walk(atom) for atom in theory_atoms]
    refinement_walker = RefinementWalker(abstraction=bool_walker.abstraction)
    refined_models = [
        [
            refinement_walker.walk(lit)
            for lit in [atom if value else Not(atom) for atom, value in model.items()]
        ]
        for model in solver_abstr.projected_allsat(
            phi_and_lemmas_abstr, theory_atoms_abstr, total=True
        )
    ]

    assert_models_are_tsat(phi, refined_models)

    if gt_logs is not None:
        assert len(refined_models) == gt_model_count(
            gt_logs
        ), "Refined models number should match ground truth"

    logger["T-LEMMAS CHECK"] = {}
    logger["T-LEMMAS CHECK"]["Total time"] = time.time() - start_time

    log_path = os.path.join(sys.argv[2], "logs.json")
    with open(log_path, "w") as log_file:
        json.dump(logger, log_file, indent=4)


if __name__ == "__main__":
    main()
