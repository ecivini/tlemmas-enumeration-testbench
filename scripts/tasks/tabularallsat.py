import os
import subprocess
from tempfile import TemporaryDirectory
from typing import Callable, Iterator

import numpy as np
from allsat_cnf.label_cnfizer import LabelCNFizer
from allsat_cnf.utils import is_cnf
from pysmt.fnode import FNode

from dimacs import DimacsInterface, read_models as read_models_cython


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


class TabularAllSATInterface:
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


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
