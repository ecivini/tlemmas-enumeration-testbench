import itertools
from typing import Any, Callable, Generator, Iterator

import cython
import numpy as np
from allsat_cnf.utils import get_clauses, get_literals, is_cnf
from pysmt.environment import Environment
from pysmt.fnode import FNode
from pysmt.formula import FormulaManager
from pysmt.shortcuts import get_env, get_free_variables


@cython.cclass
class CustomIndexArray:
    """
    A numpy array that supports custom indexing.
    """
    _arr: np.ndarray
    _start_index: cython.int

    def __init__(self, arr: np.ndarray, start_index: int = 0):
        self._arr = arr
        self._start_index = start_index

    @cython.cfunc
    def _shift_slice(self, index: slice) -> slice:
        start: cython.int = index.start - self._start_index if index.start is not None else 0
        stop: cython.int = index.stop - self._start_index if index.stop is not None else len(self)
        step: cython.int = index.step if index.step is not None else 1
        return slice(start, stop, step)

    def __getitem__(self, index: int | np.ndarray | slice) -> Any | np.ndarray:
        if isinstance(index, slice):
            return self._arr[self._shift_slice(index)]
        else:
            return self._arr[index - self._start_index]

    def __setitem__(self, index: int | np.ndarray | slice, value: Any | np.ndarray) -> None:
        if isinstance(index, slice):
            self._arr[self._shift_slice(index)] = value
        else:
            self._arr[index - self._start_index] = value

    def __len__(self) -> int:
        return len(self._arr)


class DimacsInterface:
    _env: Environment
    _mgr: FormulaManager
    _header_mode: cython.int
    _int_to_lit: CustomIndexArray
    _lit_to_int: dict[FNode, int]

    def __init__(self, environment=None):
        self._env = environment or get_env()
        self._mgr = self._env.formula_manager
        self._int_to_lit = CustomIndexArray(np.empty(0, dtype=object), start_index=0)
        self._lit_to_int = {}

    # @cython.cfunc
    def _init_lit_map(self, variables: list[FNode]) -> None:
        self._int_to_lit: CustomIndexArray = CustomIndexArray(np.empty((len(variables) + 1) * 2 + 1, dtype=object),
                                                              start_index=-len(variables))
        # fill from 1 to n with sorted variables, and from -1 to -n with their negations
        variables.sort(key=str)
        not_fn = self._mgr.Not
        not_variables = [not_fn(var) for var in reversed(variables)]
        self._int_to_lit[1:len(variables) + 1] = variables
        self._int_to_lit[-len(variables):0] = not_variables
        self._lit_to_int = dict(itertools.chain(zip(not_variables, range(-len(variables), 0)),
                                                zip(variables, range(1, len(variables) + 1))))

    def pysmt_to_dimacs(self, formula: FNode, projected_vars: list[FNode]) -> Generator[str, None, None]:
        """Converts a CNF formula to the DIMACS format.
        Yields lines of the DIMACS format.
        """
        assert is_cnf(formula)
        variables = list(get_free_variables(formula) | set(projected_vars))
        self._init_lit_map(variables)

        n_vars = len(self._lit_to_int) // 2
        clauses = get_clauses(formula)
        n_clauses = len(clauses)

        pv = [self._lit_to_int[var] for var in projected_vars]
        yield f"p cnf {n_vars} {n_clauses} {len(pv)}\n"
        yield f"c p show {' '.join(map(str, pv))}\n"
        yield from map(self._clause_to_str, clauses)

    def _clause_to_str(self, clause: FNode) -> str:
        return "{} 0\n".format(' '.join(str(self._lit_to_int[lit]) for lit in get_literals(clause)))

    def int_list_to_model(self, int_list: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Converts a list of integers to a model over FNodes."""
        return self._int_to_lit[abs(int_list)], int_list > 0


@cython.boundscheck(False)
@cython.wraparound(False)
def read_models(model_file: str, int_list_to_model: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]) -> Iterator[
    dict[FNode, bool]]:
    """
    Cythonized model reading using fast line-by-line parsing and memory views.
    This version is simpler and more readable than the low-level C approach.
    """

    model_tuple: tuple
    current_model: dict[FNode, bool]

    with open(model_file, "r") as f:
        for line in f:
            stripped_line = line.strip()
            current_row_np = np.fromstring(
                stripped_line,
                dtype=np.int32,
                sep=' '
            )
            model_tuple = int_list_to_model(current_row_np)
            # map from numpy bools to Python bools
            current_model = dict(zip(model_tuple[0], map(bool, model_tuple[1])))
            yield current_model
