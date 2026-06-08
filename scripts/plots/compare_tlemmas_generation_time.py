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
) -> tuple[dict, dict, dict, dict]:
    times = {}
    tlemmas = {}
    avgs = {}
    medians = {}

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

                # extracts stats from the lemmas
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
    *datasets: tuple[dict, str],
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
    first: dict,
    current: dict,
    x_label: str,
    y_label: str,
    lower_threshold: float = 1.0,
    timeout: float = 3600.0,
    out_path: str = "scatter.pdf",
):
    neither_x, neither_y = [], []
    one_timeout_x, one_timeout_y = [], []
    both_timeout_x, both_timeout_y = [], []
    first_timeouts = 0
    current_timeouts = 0
    first_under_lower_threshold = 0
    current_under_lower_threshold = 0

    for problem in current.keys():
        first_val = first[problem]
        current_val = current[problem]

        first_is_timeout = first_val >= timeout
        current_is_timeout = current_val >= timeout

        if first_is_timeout:
            first_timeouts += 1
        elif first_val <= lower_threshold:
            first_under_lower_threshold += 1

        if current_is_timeout:
            current_timeouts += 1
        elif current_val <= lower_threshold:
            current_under_lower_threshold += 1

        if first_is_timeout and current_is_timeout:
            both_timeout_x.append(timeout)
            both_timeout_y.append(timeout)
        elif first_is_timeout or current_is_timeout:
            one_timeout_x.append(current_val)
            one_timeout_y.append(first_val)
        else:
            neither_x.append(current_val)
            neither_y.append(first_val)

    linthresh = 10

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 7))

    # Scatter plot - non-timeouts
    ax.scatter(
        x=neither_x,
        y=neither_y,
        color="lightskyblue",
        edgecolors="black",
        s=100,
        zorder=4,
        alpha=1,
        marker="X",
        label="Completed",
    )

    # Scatter plot - one timeout
    ax.scatter(
        x=one_timeout_x,
        y=one_timeout_y,
        color="orange",
        edgecolors="black",
        s=100,
        zorder=4,
        alpha=1,
        marker="^",
        label="One timed out",
    )

    # Scatter plot - both timed out
    ax.scatter(
        x=both_timeout_x,
        y=both_timeout_y,
        color="red",
        edgecolors="black",
        s=100,
        zorder=4,
        alpha=1,
        marker="s",
        label="Both timed out",
    )

    # Reference line y = x
    ax.plot(
        [1e-2, timeout],
        [1e-2, timeout],
        label="y = x",
        zorder=2,
        color="gray",
        linestyle="--",
    )

    # Timeout lines (dashed)
    ax.axvline(timeout, linestyle="--", color="gray")

    print(
        f"\n{out_path}\n"
        f"{x_label} timeouts: {current_timeouts}"
        f"| below {lower_threshold} sec: {current_under_lower_threshold}"
    )

    ax.axhline(timeout, linestyle="--", color="gray")

    both_timeouts = len(both_timeout_x)
    one_timeout_count = len(one_timeout_x)
    unique_timeouts = both_timeouts + one_timeout_count

    print(
        f"{y_label} timeouts: {first_timeouts} "
        f"| below {lower_threshold} sec: {first_under_lower_threshold}"
    )
    print(
        f"Both timed out: {both_timeouts} | "
        f"Exactly one timed out: {one_timeout_count} | "
        f"Unique problems with at least one timeout: {unique_timeouts}"
    )

    # Set symlog scale
    ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_aspect("equal")

    # Set limits
    ax.set_xlim(left=1e-2, right=timeout * 1.1)
    ax.set_ylim(bottom=1e-2, top=timeout * 1.1)

    # Labels
    ax.set_xlabel(f"{x_label} times", fontsize=24)
    ax.set_ylabel(f"{y_label} times", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Grid
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    # Collect legend handles/labels for external legend plot
    handles, labels = ax.get_legend_handles_labels()

    # Show plot (without legend)
    plt.tight_layout()
    plt.savefig(out_path)

    return handles, labels


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


def create_tlemmas_scatter_plot(
    first: dict,
    current: dict,
    first_label: str,
    curr_label: str,
    out_path: str = "scatter_num.pdf",
    log_scale: bool = True,
):
    first_times = []
    current_times = []

    for problem in current.keys():
        if problem not in first:
            continue

        first_times.append(first[problem])
        current_times.append(current[problem])

    if not first_times or not current_times:
        print("No data for plot:", out_path)
        return

    timeout = max(max(first_times), max(current_times))

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 7))

    # Scatter plot
    ax.scatter(
        x=current_times,
        y=first_times,
        color="lightskyblue",
        edgecolors="black",
        s=100,
        zorder=4,
        alpha=1,
        marker="X",
    )

    # Reference line y = x
    ax.plot(
        [1e-2, timeout],
        [1e-2, timeout],
        label="y = x",
        zorder=2,
        color="gray",
        linestyle="--",
    )

    # Timeout lines (dashed)
    # ax.axvline(timeout, linestyle="--", color="gray")
    # ax.axhline(timeout, linestyle="--", color="gray")

    # Set symlog scale
    if log_scale:
        ax.set_xscale("symlog")
        ax.set_yscale("symlog")
    else:
        ax.set_xscale("linear")
        ax.set_yscale("linear")
    ax.set_aspect("equal")

    # Set limits
    ax.set_xlim(left=1e-2, right=timeout * 1.2)
    ax.set_ylim(bottom=1e-2, top=timeout * 1.2)

    # Labels
    ax.set_xlabel(f"{curr_label}", fontsize=24)
    ax.set_ylabel(f"{first_label}", fontsize=24)

    # Grid
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    # Legend
    # ax.legend()
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Show plot
    plt.tight_layout()
    plt.savefig(out_path)


def linearize_data(h3: dict, h4: dict) -> dict:
    # rename all fields in h3 from x_y to h3_x_y:
    result = {}
    for key in h3:
        result[f"h3_{key}"] = h3[key]

    # Add h4 with the same adjusted format
    for key in h4:
        result[f"h4_{key}"] = h4[key]

    return result


def _load_run_data(
    run_dir: str,
    timeout: float,
    benchmark_paths: list[str] | None = None,
) -> tuple[dict, dict, dict, dict]:
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare T-lemma generation results between two runs."
    )
    parser.add_argument("first_dir", help="First results/data directory")
    parser.add_argument("second_dir", help="Second results/data directory")
    parser.add_argument(
        "--first-label",
        default=None,
        help="Label for first run (default: directory name)",
    )
    parser.add_argument(
        "--second-label",
        default=None,
        help="Label for second run (default: directory name)",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory for generated plots",
    )
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    first_label = args.first_label or os.path.basename(os.path.normpath(args.first_dir))
    second_label = args.second_label or os.path.basename(
        os.path.normpath(args.second_dir)
    )

    (
        first_times,
        first_tlemmas,
        _,
        first_median_tlemmas_sizes,
    ) = _load_run_data(
        args.first_dir, timeout=args.timeout, benchmark_paths=args.benchmark_dirs
    )
    (
        second_times,
        second_tlemmas,
        _,
        second_median_tlemmas_sizes,
    ) = _load_run_data(
        args.second_dir, timeout=args.timeout, benchmark_paths=args.benchmark_dirs
    )

    first_times, second_times = _align_common_keys(first_times, second_times)
    first_tlemmas, second_tlemmas = _align_common_keys(first_tlemmas, second_tlemmas)
    first_median_tlemmas_sizes, second_median_tlemmas_sizes = _align_common_keys(
        first_median_tlemmas_sizes,
        second_median_tlemmas_sizes,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    handles, labels = create_scatter_plot(
        first_times,
        second_times,
        x_label=second_label,
        y_label=first_label,
        timeout=args.timeout,
        out_path=os.path.join(
            args.out_dir, f"{first_label}_vs_{second_label}_tlemmas_gen_time.pdf"
        ),
    )
    save_legend_plot(
        handles,
        labels,
        out_path=os.path.join(
            args.out_dir, f"{first_label}_vs_{second_label}_legend.pdf"
        ),
    )
    create_tlemmas_scatter_plot(
        first_tlemmas,
        second_tlemmas,
        first_label,
        second_label,
        out_path=os.path.join(
            args.out_dir, f"{first_label}_vs_{second_label}_tlemmas_num.pdf"
        ),
    )
    create_tlemmas_scatter_plot(
        first_median_tlemmas_sizes,
        second_median_tlemmas_sizes,
        first_label,
        second_label,
        out_path=os.path.join(
            args.out_dir, f"{first_label}_vs_{second_label}_tlemmas_median_size.pdf"
        ),
        log_scale=False,
    )
    create_cactus_plot(
        (first_times, first_label),
        (second_times, second_label),
        timeout=args.timeout,
        out_path=os.path.join(
            args.out_dir, f"cactus_{first_label}_vs_{second_label}.pdf"
        ),
    )


if __name__ == "__main__":
    main()
