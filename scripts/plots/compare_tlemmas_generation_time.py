import argparse
import json
import os
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pysmt
import pysmt.environment
from pysmt.fnode import FNode
from tasks.generate_tlemmas import read_formula

RESULTS_TIME_KEY = "Total time"
RESULTS_TLEMMAS_NUM_KEY = "Lemmas"


def get_current_results_times(
    err_file: str | None,
    paths: list[str],
    timeout: float,
    benchmark_paths: list[str] | None = None,
) -> tuple[dict[str, float], dict[str, int], dict[str, float], dict[str, float]]:
    times: dict[str, float] = {}
    tlemmas: dict[str, int] = {}
    avgs: dict[str, float] = {}
    medians: dict[str, float] = {}

    for base_dir in paths:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file != "logs.json":
                    continue

                file_path = os.path.join(root, file)
                print("parsing file", file_path)
                with open(file_path, "r") as f:
                    data = json.load(f)

                problem_name = os.path.relpath(os.path.dirname(file_path), base_dir)
                times[problem_name] = data[RESULTS_TIME_KEY]
                tlemmas[problem_name] = data[RESULTS_TLEMMAS_NUM_KEY]

                pysmt.environment.push_env()
                tlemmas_fnode = get_tlemmas_from_logs(file_path)
                avg, med = compute_tlemmas_stats(tlemmas_fnode)
                pysmt.environment.pop_env()
                assert len(tlemmas_fnode) == tlemmas[problem_name]

                avgs[problem_name] = avg
                medians[problem_name] = med

    if err_file:
        with open(err_file, "r") as f:
            errors = json.load(f)

        for problem, reason in errors.items():
            if reason != "timeout":
                raise ValueError("Unexpected error reason in data:", reason)

            found = None
            for bp in benchmark_paths or []:
                prefix = bp.rstrip("/\\") + os.sep
                if problem.startswith(prefix):
                    found = problem[len(prefix) :].replace(".smt2", "")
                    break

            if found is not None and found not in times:
                times[found] = timeout
            elif found is None:
                times[problem.split(os.sep)[-1].replace(".smt2", "")] = timeout

    return times, tlemmas, avgs, medians


def get_clause_size(formula: FNode) -> int:
    stack = [formula]
    result = 0
    while stack:
        clause = stack.pop()
        if clause.is_or():
            stack += clause.args()
        else:
            result += 1
    return result


def compute_tlemmas_stats(tlemmas: list[FNode]) -> tuple[float, float]:
    literals_num_list = [get_clause_size(lemma) for lemma in tlemmas]

    avg_lemma_size = statistics.mean(literals_num_list)
    median_lemma_size = statistics.median(literals_num_list)

    return avg_lemma_size, median_lemma_size


def get_tlemmas_from_logs(logs_path: str) -> list[FNode]:
    dir_name = os.path.dirname(logs_path)
    files = [
        os.path.join(dir_name, f)
        for f in os.listdir(dir_name)
        if f.endswith(".smt2") and os.path.isfile(os.path.join(dir_name, f))
    ]
    # there should be only one file
    assert len(files) == 1, "multiple .smt2 files in {}: {}".format(dir_name, files)
    tlemmas_path = files[0]

    tlemmas_and = read_formula(tlemmas_path)
    if tlemmas_and.is_and():
        return list(tlemmas_and.args())
    elif tlemmas_and.is_or():
        return [tlemmas_and]
    else:
        raise ValueError("Unexpected T-lemmas format")


def create_cactus_plot(
    *datasets: tuple[dict[str, int] | dict[str, float], str],
    show_vbs: bool = False,
    timeout: float = 3600.0,
    out_path: str = "cactus.pdf",
) -> None:
    MARKERS = ["o", "^", "s", "D", "v", "<", ">", "p", "*", "h"]

    assert len(datasets) >= 2, "Need at least 2 datasets"

    # Verify all dicts share same keys
    keys = list(datasets[0][0].keys())
    for data, _ in datasets[1:]:
        assert data.keys() == datasets[0][0].keys(), (
            "All data dicts must share same keys"
        )

    # Clamp & sort per dataset
    sorted_data = []
    for data, label in datasets:
        times = sorted(min(data[k], timeout) for k in keys)
        sorted_data.append((times, label))

    # VBS
    vbs_times = None
    if show_vbs:
        raw = [min(data[k] for data, _ in datasets) for k in keys]
        vbs_times = sorted(min(t, timeout) for t in raw)

    # Plot
    plt.figure(figsize=(9, 6))
    for i, (times, label) in enumerate(sorted_data):
        x = np.arange(1, len(times) + 1)
        plt.plot(x, times, label=label, marker=MARKERS[i % len(MARKERS)], markersize=2)

    if show_vbs and vbs_times:
        x = np.arange(1, len(vbs_times) + 1)
        plt.plot(x, vbs_times, label="Virtual Best", marker="s", markersize=1)

    plt.xlabel("Number of problems solved", fontsize=24)
    plt.ylabel("Time (s)", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(out_path)


def create_scatter_plot(
    x_data: dict,
    x_label: str,
    y_data: dict,
    y_label: str,
    lower_threshold: float = 1.0,
    timeout: float | None = None,
    label_suffix: str = "",
    log_scale: bool = True,
    out_path: str = "scatter.pdf",
) -> None:
    common_keys = sorted(set(x_data.keys()) & set(y_data.keys()))
    if not common_keys:
        print("No data for plot:", out_path)
        return

    completed_x, completed_y = [], []
    timeout_x, timeout_y = [], []
    x_timeouts = 0
    y_timeouts = 0
    x_below = 0
    y_below = 0

    for key in common_keys:
        xv = x_data[key]
        yv = y_data[key]

        if timeout is not None:
            x_is_timeout = xv >= timeout
            y_is_timeout = yv >= timeout
            if x_is_timeout or y_is_timeout:
                if x_is_timeout:
                    x_timeouts += 1
                if y_is_timeout:
                    y_timeouts += 1
                timeout_x.append(timeout if x_is_timeout else xv)
                timeout_y.append(timeout if y_is_timeout else yv)
                continue
            if xv <= lower_threshold:
                x_below += 1
            if yv <= lower_threshold:
                y_below += 1
        completed_x.append(xv)
        completed_y.append(yv)

    plot_max = (
        timeout if timeout is not None else max(max(completed_x), max(completed_y))
    )

    _, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        x=completed_x,
        y=completed_y,
        color="lightskyblue",
        edgecolors="black",
        s=100,
        zorder=4,
        alpha=0.5,
        marker="X",
    )

    if timeout is not None and timeout_x:
        ax.scatter(
            x=timeout_x,
            y=timeout_y,
            color="red",
            edgecolors="black",
            s=100,
            zorder=4,
            alpha=0.5,
            marker="X",
        )

    ax.plot(
        [1e-2, plot_max],
        [1e-2, plot_max],
        label="y = x",
        zorder=2,
        color="gray",
        linestyle="--",
    )

    if timeout is not None:
        ax.axvline(timeout, linestyle="--", color="gray")
        ax.axhline(timeout, linestyle="--", color="gray")

    if timeout is not None:
        print(
            f"\n{out_path}\n"
            f"{x_label} timeouts: {x_timeouts}"
            f"| below {lower_threshold} sec: {x_below}"
        )
        print(
            f"{y_label} timeouts: {y_timeouts} | below {lower_threshold} sec: {y_below}"
        )
        print(f"Timed out problems: {len(timeout_x)}")

    if log_scale:
        ax.set_xscale("symlog", linthresh=10)
        ax.set_yscale("symlog", linthresh=10)
    else:
        ax.set_xscale("linear")
        ax.set_yscale("linear")
    ax.set_aspect("equal")

    ax.set_xlim(left=1e-2, right=plot_max * 1.1)
    ax.set_ylim(bottom=1e-2, top=plot_max * 1.1)

    ax.set_xlabel(f"{x_label}{label_suffix}", fontsize=24)
    ax.set_ylabel(f"{y_label}{label_suffix}", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(out_path)


def save_legend_plot(
    handles: list,
    labels: list,
    out_path: str,
) -> None:
    fig = plt.figure(figsize=(8, 2))
    fig.legend(handles, labels, fontsize=14, ncol=3, loc="center")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _load_run_data(
    run_dir: str,
    timeout: float,
    benchmark_paths: list[str] | None = None,
) -> tuple[dict[str, float], dict[str, int], dict[str, float], dict[str, float]]:
    """Load benchmark metrics from one result directory."""
    err_file = os.path.join(run_dir, "errors.json")
    if not os.path.exists(err_file):
        err_file = None
    return get_current_results_times(
        err_file, [run_dir], timeout=timeout, benchmark_paths=benchmark_paths
    )


def _align_common_keys(*datasets: dict) -> list[dict]:
    """Keep only problems present in every dataset."""
    if not datasets:
        return []

    common_keys = set(datasets[0].keys())
    for data in datasets[1:]:
        common_keys &= set(data.keys())

    return [{key: data[key] for key in sorted(common_keys)} for data in datasets]


def _plot_all_scatter_pairs(
    i: int,
    j: int,
    labels: list[str],
    all_times: list[dict[str, float]],
    all_tlemmas: list[dict[str, int]],
    all_medians: list[dict[str, float]],
    timeout: float,
    out_dir: str,
) -> None:
    def get_items(
        idx: int,
    ) -> tuple[str, dict[str, float], dict[str, int], dict[str, float]]:
        return labels[idx], all_times[idx], all_tlemmas[idx], all_medians[idx]

    labi, timei, lemi, medi = get_items(i)
    labj, timej, lemj, medj = get_items(j)
    pair_tag = f"{labi}_vs_{labj}"
    scatter_configs = [
        (timei, labi, timej, labj, " (time)", True, "gen_time", timeout),
        (lemi, labi, lemj, labj, " (# T-lemmas)", True, "num", None),
        (medi, labi, medj, labj, " (median T-lemma size)", False, "median_size", None),
    ]
    for x_data, x_label, y_data, y_label, suffix, log, suffix_fn, to in scatter_configs:
        create_scatter_plot(
            x_data,
            x_label,
            y_data,
            y_label,
            timeout=to,
            label_suffix=suffix,
            log_scale=log,
            out_path=os.path.join(
                out_dir,
                f"{pair_tag}_tlemmas_{suffix_fn}.pdf",
            ),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare T-lemma generation results across multiple runs."
    )
    parser.add_argument(
        "--data",
        nargs=2,
        action="append",
        required=True,
        metavar=("DIR", "LABEL"),
        help="A results directory and its label (repeatable, at least 2)",
    )
    parser.add_argument("--out-dir", default=".", help="Directory for generated plots")
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="Timeout in seconds for detecting timed-out benchmarks (default: 3600)",
    )
    parser.add_argument(
        "--benchmark-dir",
        action="append",
        dest="benchmark_dirs",
        default=None,
        help="Benchmark input directories (used to match error keys to log keys)",
    )
    args = parser.parse_args()
    if len(args.data) < 2:
        parser.error("At least 2 datasets are required")
    return args


def main() -> None:
    args = _parse_args()
    timeout = args.timeout
    out_dir = args.out_dir

    labels = []
    all_times = []
    all_tlemmas = []
    all_medians = []
    for dir_path, label in args.data:
        labels.append(label)
        times, tlemmas, _, medians = _load_run_data(
            dir_path, timeout=timeout, benchmark_paths=args.benchmark_dirs
        )
        all_times.append(times)
        all_tlemmas.append(tlemmas)
        all_medians.append(medians)

    all_times = _align_common_keys(*all_times)
    all_tlemmas = _align_common_keys(*all_tlemmas)
    all_medians = _align_common_keys(*all_medians)

    os.makedirs(out_dir, exist_ok=True)

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            _plot_all_scatter_pairs(
                i, j, labels, all_times, all_tlemmas, all_medians, timeout, out_dir
            )

    cactus_tag = "_vs_".join(labels)
    create_cactus_plot(
        *[(all_times[i], labels[i]) for i in range(len(labels))],
        timeout=timeout,
        out_path=os.path.join(out_dir, f"cactus_{cactus_tag}.pdf"),
    )
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    save_legend_plot(
        handles,
        legend_labels,
        out_path=os.path.join(out_dir, f"{cactus_tag}_legend.pdf"),
    )


if __name__ == "__main__":
    main()
