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

import yaml


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

    try:
        _, stderr = proc.communicate(timeout=timeout + 2)
        return proc.returncode, stderr.decode()
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)  # Safely kills the whole process tree
        return -1, "timeout"


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
    formula: Path, config: Config, output_dir: Path, args: argparse.Namespace
) -> tuple[Path, str | None]:
    print(f"[+] Generating T-lemmas: {formula}")

    cmd = [
        "python3",
        "scripts/tasks/generate_tlemmas.py",
        str(formula),
        str(output_dir / formula.with_suffix("")),
        config.allsmt_processes,
        args.solver,
    ]

    if args.projection:
        cmd.append("--projection")
    if args.partition:
        cmd.append("--partition")

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


def get_pending_formulas(paths: list[str], output_dir: Path) -> list[Path]:
    """Returns shuffled list of .smt2 files that haven't been computed yet."""
    skip = {p.parent.name + ".smt2" for p in output_dir.rglob("logs.json")}
    formulas = []

    for root in paths:
        p = Path(root)
        candidates = [p] if p.is_file() else p.rglob("*.smt2")
        for f in candidates:
            if f.suffix == ".smt2" and f.name not in skip:
                formulas.append(f)
            elif f.suffix == ".smt2":
                print(f"[-] Skipping already computed: {f}")

    random.shuffle(formulas)
    return formulas


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="benchmark_controller.py")
    parser.add_argument("test_name", help="Output subdirectory name.")
    sub = parser.add_subparsers(dest="task", required=True)

    gen = sub.add_parser("tlemmas_gen", help="Generate T-lemmas.")
    gen.add_argument("--solver", choices=["sequential", "parallel"], default="parallel")
    gen.add_argument("--projection", action="store_true")
    gen.add_argument("--partition", action="store_true")

    sub.add_parser("tlemmas_check", help="Check T-lemma correctness.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config()

    output_dir = config.results / args.test_name
    output_dir.mkdir(parents=True, exist_ok=True)
    formulas = get_pending_formulas(config.benchmarks, output_dir)

    if not formulas:
        print("[+] All benchmarks already computed.")
        return

    if args.task == "tlemmas_gen":
        worker = partial(task_gen, config=config, output_dir=output_dir, args=args)
    else:
        tlemmas = index_tlemmas(config.tlemmas_dir)
        gt_tlemmas = index_tlemmas(config.gt_tlemmas_dir)
        worker = partial(
            task_check,
            config=config,
            output_dir=output_dir,
            tlemmas=tlemmas,
            gt_tlemmas=gt_tlemmas,
        )

    start_time = time.time()
    errors = {}

    with Pool(processes=config.processes) as pool:
        for formula, error in pool.imap_unordered(worker, formulas):
            if error:
                errors[str(formula)] = error

    if errors:
        (output_dir / "errors.json").write_text(json.dumps(errors, indent=4))

    total_time = time.time() - start_time
    n_for = len(formulas)
    n_err = len(errors)
    print(f"\n[+] Completed {n_for} jobs in {total_time:.2f}s ({n_err} errors)")


if __name__ == "__main__":
    main()
