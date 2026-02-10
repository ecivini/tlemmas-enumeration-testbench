import multiprocessing
import os
import subprocess
from tempfile import TemporaryDirectory
from typing import Callable, Iterator, Protocol

import numpy as np
from allsat_cnf.label_cnfizer import LabelCNFizer
from allsat_cnf.utils import is_cnf
from dimacs import DimacsInterface
from dimacs import read_models as read_models_cython
from pysmt.fnode import FNode
from pysmt.formula import FormulaManager
from pysmt.shortcuts import get_env


def _run_cmd(cmd: list[str], cwd: str) -> None:
    proc = subprocess.Popen(
        args=cmd,
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # stdout=subprocess.PIPE,
        # stderr=subprocess.STDOUT,
        # text=True,
    )
    proc.wait()
    # print("STDOUT of TabularAllSAT:", file=sys.stderr)
    #
    # print("".join(proc.stdout.readlines()), file=sys.stderr)


class ProjectedModelEnumerator(Protocol):
    def projected_allsat(
        self, formula: FNode, projected_vars: list[FNode], total: bool = True
    ) -> Iterator[dict[FNode, bool]]: ...


class TabularAllSATInterface(ProjectedModelEnumerator):
    """Adapter for TabularAllSAT"""

    def __init__(self, ta_bin: str):
        # check that the binary exists and is executable
        if not (os.path.isfile(ta_bin) and os.access(ta_bin, os.X_OK)):
            raise ValueError(
                f"TabularAllSAT binary not found or not executable: {ta_bin}"
            )

        self.ta_bin = ta_bin

    def projected_allsat(
        self, formula: FNode, projected_vars: list[FNode], total: bool = True
    ) -> Iterator[dict[FNode, bool]]:
        """
        Enumerates projected models using TabularAllSAT.

        >>> from pysmt.shortcuts import Symbol
        >>> from scripts.tasks.tabularallsat import TabularAllSATInterface
        >>> import os, sys
        >>> this_module = sys.modules['scripts.tasks.tabularallsat']
        >>> project_root = os.path.join(os.path.dirname(os.path.abspath(this_module.__file__)), "../../")
        >>> solver_path = os.path.join(project_root, "tabularAllSAT/cdcl-vsads/solver")
        >>> ta_interface = TabularAllSATInterface(solver_path)
        >>> x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        >>> list(ta_interface.projected_allsat(x & y & z, [x, y]))
        [{x: True, y: True}]

        >>> list(ta_interface.projected_allsat(x & ~y, [x, y]))
        [{x: True, y: False}]

        >>> assert len(list(ta_interface.projected_allsat(x | y, [x]))) == 2

        >>> assert len(list(ta_interface.projected_allsat(x | y, [x, y]))) == 3

        >>> assert len(list(ta_interface.projected_allsat((x & y) | (y & z) | (x & z), [x, y, z]))) == 4
        """
        if not is_cnf(formula):
            formula = LabelCNFizer().convert_as_formula(formula)
        yield from self._invoke_solver(formula, projected_vars, total)

    def _invoke_solver(
        self, formula: FNode, projected_vars: list[FNode], total: bool
    ) -> Iterator[dict[FNode, bool]]:
        with TemporaryDirectory() as tmpdir:
            dimacs_file = f"{tmpdir}/input.dimacs"

            dimacs = DimacsInterface()

            with open(dimacs_file, "w") as fw:
                fw.writelines(dimacs.pysmt_to_dimacs(formula, projected_vars))

            solver_options_cmd = ["--enum_total"] if total else []
            cmd = [self.ta_bin] + solver_options_cmd + ["--output-file=1"]
            cmd += [dimacs_file]
            _run_cmd(cmd, tmpdir)

            yield from self._read_models(
                f"{tmpdir}/output.txt", dimacs.int_list_to_model
            )

    @staticmethod
    def _read_models(
        model_file: str,
        int_list_to_model: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    ) -> Iterator[dict[FNode, bool]]:
        yield from read_models_cython(model_file, int_list_to_model)


_PARTIAL_MODELS: list[dict[FNode, bool]] | None = None
_PHI = None
_PHI_ATOMS = []


class ParallelWrapper(ProjectedModelEnumerator):
    """Wrapper to enumerate models in parallel using multiple workers."""

    def __init__(
        self,
        ta_interface: TabularAllSATInterface,
        num_workers: int,
        num_vars_per_partial_model: int = 10,
    ):
        self.ta_interface = ta_interface
        self.num_workers = num_workers
        self.num_vars_per_partial_model = num_vars_per_partial_model

    def _initialize_worker(
        self,
        partial_models: list[dict[FNode, bool]],
        phi: FNode,
        phi_atoms: list[FNode],
    ) -> None:
        global _PARTIAL_MODELS, _PHI, _PHI_ATOMS

        _PARTIAL_MODELS = partial_models
        mgr: FormulaManager = get_env().formula_manager
        _PHI = mgr.normalize(phi)

        _PHI_ATOMS = [mgr.normalize(a) for a in phi_atoms]

    def _parallel_worker(self, args: tuple) -> list[dict[FNode, bool]]:
        """Worker function for parallel all-smt extension

        Args:
            args: tuple of (model_id,) where model_id is the index of the partial model to process

        Returns:
            list of models found by the worker
        """
        global _PHI, _PHI_ATOMS, _PARTIAL_MODELS

        (model_id,) = args
        mgr = get_env().formula_manager

        model = [
            atom if value else mgr.Not(atom)
            for atom, value in (
                (mgr.normalize(a), v) for a, v in _PARTIAL_MODELS[model_id].items()
            )
        ]

        found_models = self.ta_interface.projected_allsat(
            mgr.And(_PHI, *model), projected_vars=_PHI_ATOMS, total=True
        )

        return list(found_models)

    def projected_allsat(
        self, formula: FNode, projected_vars: list[FNode], total: bool = True
    ) -> Iterator[dict[FNode, bool]]:
        print("Generating partial models for parallel enumeration...", flush=True)
        if self.num_vars_per_partial_model > 0:
            partial_projected_vars = projected_vars[: self.num_vars_per_partial_model]
        else:
            partial_projected_vars = projected_vars
        partial_models = list(
            self.ta_interface.projected_allsat(
                formula, partial_projected_vars, total=False
            )
        )
        worker_args = [(i,) for i in range(len(partial_models))]

        # Use a process pool to maintain constant number of workers
        pool = multiprocessing.Pool(
            processes=self.num_workers,
            initializer=self._initialize_worker,
            initargs=(partial_models, formula, projected_vars),
        )
        mgr = get_env().formula_manager
        model_count = 0
        with pool:
            # Use imap_unordered to process results as they complete
            for models in pool.imap_unordered(self._parallel_worker, worker_args):
                model_count += len(models)
                yield from (
                    {mgr.normalize(var): value for var, value in model.items()}
                    for model in models
                )
        print(f"Total models found: {model_count}", flush=True)


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
