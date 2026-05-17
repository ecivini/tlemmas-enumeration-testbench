from __future__ import annotations

import argparse
import json
import os
import random
import resource
import signal
import subprocess
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

import yaml
from tasks.generate_tlemmas import DIVIDE_STRATEGIES

# Track active subprocesses per worker for cleanup on SIGTERM
_active_procs: list[subprocess.Popen] = []


def _worker_sigterm_handler(signum: int, frame) -> None:
    """Kill all active subprocesses when the pool is terminated."""
    for proc in _active_procs:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def init_worker() -> None:
    """Initialize a worker process: ignore SIGINT, cleanup on SIGTERM."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _worker_sigterm_handler)


class Config:
    def __init__(self, path: Path | str = "config.yaml"):
        raw = yaml.safe_load(Path(path).read_text())
        self.benchmarks: list[str] = raw["benchmarks"]
        self.results = Path(raw["results"])
        self.processes = int(raw["processes"])
        self.allsmt_processes = str(raw["allsmt_processes"])
        self.timeout = int(raw["timeout"])
        self.memory_bytes = int(raw["memory"]) * 1024 * 1024
        self.tlemmas_dir = Path(raw["tlemmas_dir"])
        self.gt_tlemmas_dir = Path(raw["gt_tlemmas_dir"])
        self.check_workers = str(raw["tlemmas_check_parallel_workers"])
        self.check_proj_vars = str(
            raw["tlemmas_check_num_projected_vars_per_partial_model"]
        )


def run_cmd(command: list[str], timeout: int, mem_bytes: int) -> tuple[int, str]:
    """Runs a subprocess with memory limits and timeout."""

    def preexec():
        os.setsid()
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, hard))

    print(f"\t[+] {' '.join(command)}")
    proc = subprocess.Popen(
        command, preexec_fn=preexec, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    _active_procs.append(proc)

    try:
        _, stderr = proc.communicate(timeout=timeout + 2)
        return proc.returncode, stderr.decode()
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        return -1, "timeout"
    except Exception as e:
        os.killpg(proc.pid, signal.SIGKILL)
        return -1, f"exception: {e}"
    finally:
        _active_procs.remove(proc)


def execute_task(
    command: list[str], formula: Path, config: Config
) -> tuple[Path, str | None]:
    """Shared wrapper to execute a command and handle errors safely."""
    try:
        rc, stderr = run_cmd(command, config.timeout, config.memory_bytes)
        if rc != 0 or stderr:
            print(f"[-] Failed ({formula}): {stderr.strip()}")
            return formula, stderr.strip()

        print(f"[+] Done: {formula}")
        return formula, None
    except Exception as e:
        print(f"[-] Exception ({formula}): {e}")
        return formula, str(e)


def task_gen(
    item: tuple[Path, Path, Path | None],
    config: Config,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, str | None]:
    """Execute T-lemma generation for a formula, optionally with queries."""
    formula, benchmark_root, queries_dir = item
    print(f"[+] Generating T-lemmas: {formula}")

    if queries_dir is not None:
        rel = formula.parent.relative_to(benchmark_root)
    else:
        rel = formula.relative_to(benchmark_root).with_suffix("")
    cmd = [
        "python3",
        "scripts/tasks/generate_tlemmas.py",
        str(formula),
        str(output_dir / rel),
        config.allsmt_processes,
        args.solver,
    ]

    if queries_dir is not None:
        cmd.extend(["--queries-dir", str(queries_dir)])
    if args.projection:
        cmd.append("--projection")
    if args.partition:
        cmd.append("--partition")
    if args.solver == "parallel":
        cmd.extend(["--parallel-divide-strategy", args.parallel_divide_strategy])

    return execute_task(cmd, formula, config)


def task_check(
    formula: Path, config: Config, output_dir: Path, tlemmas: dict, gt_tlemmas: dict
) -> tuple[Path, str | None]:
    print(f"[+] Checking T-lemmas: {formula}")
    f_str = str(formula)

    t_path = next((p for k, p in tlemmas.items() if k in f_str), None)
    gt_path = next((p for k, p in gt_tlemmas.items() if k in f_str), None)
    gt_logs = str(gt_path.parent / "logs.json") if gt_path else ""

    cmd = [
        "python3",
        "scripts/tasks/tlemmas_check.py",
        str(formula),
        str(output_dir / formula.with_suffix("")),
        str(t_path),
        config.check_workers,
        config.check_proj_vars,
        gt_logs,
    ]
    return execute_task(cmd, formula, config)


def _default_file_filter(_: Path) -> bool:
    return True


def _default_output_path(f: Path, p: Path, out: Path) -> Path:
    return out / f.relative_to(p).with_suffix("")


def get_pending_items(
    paths: list[str],
    output_dir: Path,
    file_filter: Callable[[Path], bool] = _default_file_filter,
    output_path_fn: Callable[[Path, Path, Path], Path] = _default_output_path,
) -> list[tuple[Path, Path]]:
    """Discover pending .smt2 files under benchmark paths.

    Args:
        paths: Benchmark directories to scan.
        output_dir: Output directory for checking already-computed items.
        file_filter: Predicate applied to each .smt2 file; only matching files are kept.
        output_path_fn: Computes the output path for skip checking. Defaults to
            output_dir / f.relative_to(root).with_suffix("").

    Returns:
        List of (formula, benchmark_root) tuples.
    """
    skip = {p.parent for p in output_dir.rglob("logs.json")}
    items: list[tuple[Path, Path]] = []

    for root in paths:
        p = Path(root)
        for f in p.rglob("*.smt2"):
            if not file_filter(f):
                continue
            out_path = output_path_fn(f, p, output_dir)
            if out_path in skip:
                print(f"[-] Skipping already computed: {f}")
                continue
            items.append((f, p))

    random.shuffle(items)
    return items


def index_tlemmas(base: Path) -> dict[str, Path]:
    """Creates a normalized index of t-lemmas for fast matching."""
    if not base.is_dir():
        return {}

    index = {}
    for p in base.rglob("*.smt2"):
        key = str(p.parent).replace(str(base), "").replace("data/benchmark/", "")
        key = key.replace("/randgen", "").replace("/ldd_randgen", "")
        index[key] = p
        print(f"[+] Indexed tlemma: {p}")
    return index


def _add_gen_args(parser: argparse.ArgumentParser) -> None:
    """Add common T-lemma generation arguments to a subparser."""
    parser.add_argument(
        "--solver", choices=["sequential", "parallel"], default="parallel"
    )
    parser.add_argument("--projection", action="store_true")
    parser.add_argument("--partition", action="store_true")
    parser.add_argument(
        "--parallel-divide-strategy",
        choices=DIVIDE_STRATEGIES.keys(),
        default="partial",
        help="Divide strategy for parallel enumeration",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="benchmark_controller.py")
    parser.add_argument("test_name", help="Output subdirectory name.")
    sub = parser.add_subparsers(dest="task", required=True)

    gen = sub.add_parser("tlemmas_gen", help="Generate T-lemmas.")
    _add_gen_args(gen)

    sub.add_parser("tlemmas_check", help="Check T-lemma correctness.")

    queries_gen = sub.add_parser(
        "tlemmas_gen_queries",
        help="Generate T-lemmas for instances with query formulas.",
    )
    _add_gen_args(queries_gen)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()

    output_dir = config.results / args.test_name
    output_dir.mkdir(parents=True, exist_ok=True)

    items: list[tuple[Path, Path, Path | None]] | list[Path]
    worker: Callable

    if args.task == "tlemmas_gen_queries":
        raw = get_pending_items(
            config.benchmarks,
            output_dir,
            lambda f: f.name == "encoding.smt2",
            lambda f, p, out: out / f.parent.relative_to(p),
        )
        items = [(f, root, f.parent / "queries") for f, root in raw]
        worker = partial(task_gen, config=config, output_dir=output_dir, args=args)
    elif args.task == "tlemmas_gen":
        raw = get_pending_items(config.benchmarks, output_dir)
        items = [(f, root, None) for f, root in raw]
        worker = partial(task_gen, config=config, output_dir=output_dir, args=args)
    else:
        tlemmas = index_tlemmas(config.tlemmas_dir)
        gt_tlemmas = index_tlemmas(config.gt_tlemmas_dir)
        items = [f for f, _ in get_pending_items(config.benchmarks, output_dir)]
        worker = partial(
            task_check,
            config=config,
            output_dir=output_dir,
            tlemmas=tlemmas,
            gt_tlemmas=gt_tlemmas,
        )

    if not items:
        print("[+] All benchmarks already computed.")
        return

    start_time = time.time()
    errors = {}

    with Pool(processes=config.processes, initializer=init_worker) as pool:
        try:
            for item, error in pool.imap_unordered(worker, items):  # type: ignore[arg-type]
                if error:
                    errors[str(item)] = error
        except KeyboardInterrupt:
            print("\n[-] Interrupted, terminating workers...")
            pool.terminate()
            pool.join()
            return

    if errors:
        (output_dir / "errors.json").write_text(json.dumps(errors, indent=4))

    total_time = time.time() - start_time
    n_for = len(items)
    n_err = len(errors)
    print(f"\n[+] Completed {n_for} jobs in {total_time:.2f}s ({n_err} errors)")


if __name__ == "__main__":
    main()
