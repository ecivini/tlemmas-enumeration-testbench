import json
import os

from pysmt.shortcuts import read_smtlib
from pysmt.fnode import FNode
import pysmt

import matplotlib.pyplot as plt
import numpy as np

import statistics

DEFAULT_TIMEOUT = 3600.0  # seconds
RESULTS_TIME_KEY = "Total time"
RESULTS_TLEMMAS_NUM_KEY = "T-Lemmas number"


def extract_tlemmas_from_smt2(path: str) -> int:
    try:
        lemmas = read_smtlib(path)
        if lemmas.is_or():
            return 1

        assert lemmas.is_and()

        return len(lemmas.args())
    except:
        return None


def get_current_results_times(
    err_file: str | None, paths: list[str], solver: str | None = None
) -> tuple[dict, dict, dict, dict]:
    times = {}
    tlemmas = {}
    avgs = {}
    medians = {}
    literals = {}

    if err_file:
        with open(err_file, "r") as f:
            errors = json.load(f)
            for problem, reason in errors.items():
                if reason == "timeout":
                    key_name = problem.split(os.sep)[-1].replace(".smt2", "")
                    times[key_name] = DEFAULT_TIMEOUT
                else:
                    raise ValueError("Unexpected error reason in data:", reason)

    for base_dir in paths:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file != "logs.json":
                    continue

                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    data = json.load(f)

                problem_name = os.path.dirname(file_path).split(os.sep)[-1]
                times[problem_name] = data[RESULTS_TIME_KEY]
                tlemmas[problem_name] = data[RESULTS_TLEMMAS_NUM_KEY]

                # extracts stats from the lemmas
                tlemmas_fnode = get_tlemmas_from_logs(file_path)
                avg, med, lits = compute_tlemmas_stats(tlemmas_fnode, solver)

                avgs[problem_name] = avg
                medians[problem_name] = med
                literals[problem_name] = lits

    return times, tlemmas, avgs, medians


def get_all_or_children(formula):
    if formula.is_or():
        children = []
        for arg in formula.args():
            children.extend(get_all_or_children(arg))
        return children
    else:
        return [formula]


def compute_tlemmas_stats(
    tlemmas: list[FNode], solver: str | None = None
) -> tuple[float, float, list]:
    literals_num_list = []
    for lemma in tlemmas:
        # lemma is an or clause
        lemma_size = len(get_all_or_children(lemma))
        literals_num_list.append(lemma_size)

    literals_num_list.sort()

    avg_lemma_size = statistics.mean(literals_num_list)
    median_lemma_size = statistics.median(literals_num_list)

    return (avg_lemma_size, median_lemma_size, literals_num_list)


def get_tlemmas_from_logs(logs_path: str) -> list[FNode]:
    dir_name = os.path.dirname(logs_path)
    files = [
        os.path.join(dir_name, f)
        for f in os.listdir(dir_name)
        if f.endswith(".smt2") and os.path.isfile(os.path.join(dir_name, f))
    ]
    # there should be only one file
    assert len(files) == 1
    tlemmas_path = files[0]

    pysmt.environment.reset_env()
    tlemmas_and = read_smtlib(tlemmas_path)
    if tlemmas_and.is_and():
        return list(tlemmas_and.args())
    elif tlemmas_and.is_or():
        return [tlemmas_and]
    else:
        raise ValueError("Unexpected T-lemmas format")


def create_cactus_plot(
    first: dict,
    current: dict,
    x1_label: str,
    x2_label: str,
    third: dict | None = None,  # Eventual third solver data
    x3_label: str | None = None,  # Eventual third solver label
    fourth: dict | None = None,
    x4_label: str | None = None,
    show_vbs: bool = False,
    out_path: str = "cactus.pdf",
) -> None:
    first_times = []
    current_times = []
    third_times = []
    fourth_times = []
    vbs_times = []
    for problem in current:
        first_time = (
            first[problem] if first[problem] <= DEFAULT_TIMEOUT else DEFAULT_TIMEOUT
        )
        current_time = (
            current[problem] if current[problem] <= DEFAULT_TIMEOUT else DEFAULT_TIMEOUT
        )

        third_time = None
        if third is not None:
            third_time = (
                third[problem] if third[problem] <= DEFAULT_TIMEOUT else DEFAULT_TIMEOUT
            )

        fourth_time = None
        if fourth is not None:
            fourth_time = (
                fourth[problem]
                if fourth[problem] <= DEFAULT_TIMEOUT
                else DEFAULT_TIMEOUT
            )

        vbs_time = min(first_time, current_time)
        if third_time:
            vbs_time = min(vbs_time, third_time)
        if fourth_time:
            vbs_time = min(vbs_time, fourth_time)

        first_times.append(first_time)
        current_times.append(current_time)
        if third_time:
            third_times.append(third_time)
        if fourth_time:
            fourth_times.append(fourth_time)

        vbs_times.append(vbs_time)

    first_times.sort()
    current_times.sort()
    third_times.sort()
    fourth_times.sort()
    vbs_times.sort()

    x1 = np.arange(1, len(first_times) + 1)
    x2 = np.arange(1, len(current_times) + 1)
    x3 = np.arange(1, len(third_times) + 1)
    x4 = np.arange(1, len(fourth_times) + 1)
    x5 = np.arange(1, len(vbs_times) + 1)

    # Plot
    plt.figure(figsize=(9, 6))
    plt.plot(x1, first_times, label=x1_label, marker="o", markersize=2)
    plt.plot(x2, current_times, label=x2_label, marker="^", markersize=2)
    if len(x3) > 0:
        plt.plot(x3, third_times, label=x3_label, marker="+", markersize=2)
    if len(x4) > 0:
        plt.plot(x4, fourth_times, label=x4_label, marker="+", markersize=2)
    if show_vbs:
        plt.plot(x5, vbs_times, label="Virtual Best", marker="s", markersize=1)

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
    out_path: str = "scatter.pdf",
):
    first_times = []
    current_times = []
    first_timeouts = 0
    current_timeouts = 0

    first_under_lower_threshold = 0
    current_under_lower_threshold = 0

    for problem in current.keys():
        first_times.append(first[problem])
        current_times.append(current[problem])

        if first[problem] >= DEFAULT_TIMEOUT:
            first_timeouts += 1
        elif first[problem] <= lower_threshold:
            first_under_lower_threshold += 1

        if current[problem] >= DEFAULT_TIMEOUT:
            current_timeouts += 1
        elif current[problem] <= lower_threshold:
            current_under_lower_threshold += 1

    timeout = DEFAULT_TIMEOUT
    linthresh = 10  # Linear region until 1

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
    ax.axvline(
        timeout,
        linestyle="--",
        color="gray",
        # label=(
        #     f"{x_label} timeouts: {current_timeouts} "  # noqa
        #     f"| below {lower_threshold} sec: {current_under_lower_threshold}"
        # ),
    )

    print(
        f"\n{out_path}\n"
        f"{x_label} timeouts: {current_timeouts}"  # noqa
        f"| below {lower_threshold} sec: {current_under_lower_threshold}"
    )

    ax.axhline(
        timeout,
        linestyle="--",
        color="gray",
        label=(
            f"{y_label} timeouts: {first_timeouts} "  # noqa
            f"| below {lower_threshold} sec: {first_under_lower_threshold}"
        ),
    )

    print(
        f"{y_label} timeouts: {first_timeouts} "  # noqa
        f"| below {lower_threshold} sec: {first_under_lower_threshold}"
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

    # Legend
    # ax.legend(loc="lower right")

    # Show plot
    plt.tight_layout()
    plt.savefig(out_path)


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


if __name__ == "__main__":
    solver_x1 = "Baseline"
    solver_x2 = "D&C"
    solver_x3 = "D&C+Proj"
    solver_x4 = "D&C+Proj+Part"

    ###########################################################################
    # RAND PROBLEMS
    x1_times, x1_tlemmas, _, x1_median_tlemmas_sizes = get_current_results_times(
        None,
        [
            "results/test_baseline/data",
        ],
    )

    (
        x2_times,
        x2_tlemmas,
        _,
        x2_median_tlemmas_sizes,
    ) = get_current_results_times(
        None,
        [
            "results/test_divconq/data",
        ],
    )

    (
        x3_times,
        x3_tlemmas,
        _,
        x3_median_tlemmas_sizes,
    ) = get_current_results_times(
        None,
        [
            "results/test_divconq_proj/data",
        ],
    )

    (
        x4_times,
        x4_tlemmas,
        _,
        x4_median_tlemmas_sizes,
    ) = get_current_results_times(
        None,
        [
            "results/test_divconq_proj_part/data",
        ],
    )

    ###################################################################
    ###################################################################
    # BEGINNING OF THE PLOTS

    # Scatter plots
    create_scatter_plot(
        x1_times,
        x2_times,
        x_label=solver_x2,
        y_label=solver_x1,
        out_path="seq_vs_par45_tlemmas_gen_time.pdf",
    )
    create_scatter_plot(
        x1_times,
        x3_times,
        x_label=solver_x3,
        y_label=solver_x1,
        out_path="seq_vs_par45_proj_atoms_tlemmas_gen_time.pdf",
    )
    create_scatter_plot(
        x1_times,
        x4_times,
        x_label=solver_x4,
        y_label=solver_x1,
        out_path="seq_vs_partition_tlemmas_gen_time.pdf",
    )

    create_scatter_plot(
        x2_times,
        x3_times,
        x_label=solver_x3,
        y_label=solver_x2,
        out_path="par45_vs_par45_proj_atoms_tlemmas_gen_time.pdf",
    )
    create_scatter_plot(
        x2_times,
        x4_times,
        x_label=solver_x4,
        y_label=solver_x2,
        out_path="par45_vs_partition_tlemmas_gen_time.pdf",
    )

    create_scatter_plot(
        x3_times,
        x4_times,
        x_label=solver_x4,
        y_label=solver_x3,
        out_path="par45_proj_vs_partition_tlemmas_gen_time.pdf",
    )

    # T-lemmas number
    create_tlemmas_scatter_plot(
        x1_tlemmas,
        x2_tlemmas,
        solver_x1,
        solver_x2,
        out_path="seq_vs_par45_tlemmas_num.pdf",
    )
    create_tlemmas_scatter_plot(
        x1_tlemmas,
        x3_tlemmas,
        solver_x1,
        solver_x3,
        out_path="seq_vs_par45_proj_atoms_tlemmas_num.pdf",
    )
    create_tlemmas_scatter_plot(
        x1_tlemmas,
        x4_tlemmas,
        solver_x1,
        solver_x4,
        out_path="seq_vs_partition_tlemmas_num.pdf",
    )

    create_tlemmas_scatter_plot(
        x2_tlemmas,
        x3_tlemmas,
        solver_x2,
        solver_x3,
        out_path="par45_vs_par45_proj_atoms_tlemmas_num.pdf",
    )
    create_tlemmas_scatter_plot(
        x2_tlemmas,
        x4_tlemmas,
        solver_x2,
        solver_x4,
        out_path="par45_vs_partition_tlemmas_num.pdf",
    )

    create_tlemmas_scatter_plot(
        x3_tlemmas,
        x4_tlemmas,
        solver_x3,
        solver_x4,
        out_path="par45_proj_vs_partition_tlemmas_num.pdf",
    )

    # Tlemmas average sizes
    # create_tlemmas_scatter_plot(
    #     prev_avg_tlemmas_sizes,
    #     current_avg_tlemmas_sizes,
    #     solver_prev,
    #     solver_curr,
    #     out_path="seq_vs_par45_tlemmas_avg_size.pdf",
    # )

    # create_tlemmas_scatter_plot(
    #     prev_avg_tlemmas_sizes,
    #     x3_avg_tlemmas_sizes,
    #     solver_prev,
    #     solver_x3,
    #     out_path="seq_vs_par45_proj_atoms_tlemmas_avg_size.pdf",
    # )

    # create_tlemmas_scatter_plot(
    #     current_avg_tlemmas_sizes,
    #     x3_avg_tlemmas_sizes,
    #     solver_curr,
    #     solver_x3,
    #     out_path="par45_vs_par45_proj_atoms_tlemmas_avg_size.pdf",
    # )

    # Tlemmas median sizes
    create_tlemmas_scatter_plot(
        x1_median_tlemmas_sizes,
        x2_median_tlemmas_sizes,
        solver_x1,
        solver_x2,
        out_path="seq_vs_par45_tlemmas_median_size.pdf",
        log_scale=False,
    )
    create_tlemmas_scatter_plot(
        x1_median_tlemmas_sizes,
        x3_median_tlemmas_sizes,
        solver_x1,
        solver_x3,
        out_path="seq_vs_par45_proj_atoms_tlemmas_median_size.pdf",
        log_scale=False,
    )
    create_tlemmas_scatter_plot(
        x1_median_tlemmas_sizes,
        x4_median_tlemmas_sizes,
        solver_x1,
        solver_x4,
        out_path="seq_vs_partition_tlemmas_median_size.pdf",
        log_scale=False,
    )

    create_tlemmas_scatter_plot(
        x2_median_tlemmas_sizes,
        x3_median_tlemmas_sizes,
        solver_x2,
        solver_x3,
        out_path="par45_vs_par45_proj_atoms_tlemmas_median_size.pdf",
        log_scale=False,
    )
    create_tlemmas_scatter_plot(
        x2_median_tlemmas_sizes,
        x4_median_tlemmas_sizes,
        solver_x2,
        solver_x4,
        out_path="par45_vs_partition_tlemmas_median_size.pdf",
        log_scale=False,
    )

    create_tlemmas_scatter_plot(
        x3_median_tlemmas_sizes,
        x4_median_tlemmas_sizes,
        solver_x3,
        solver_x4,
        out_path="par45_proj_vs_partition_tlemmas_median_size.pdf",
        log_scale=False,
    )

    # Cactus plots
    create_cactus_plot(
        x1_times,
        x2_times,
        x1_label=solver_x1,
        x2_label=solver_x2,
        third=x3_times,
        x3_label=solver_x3,
        fourth=x4_times,
        x4_label=solver_x4,
        out_path="cactus_seq_vs_par45_vs_par45_proj_atoms_vs_partition_tlemmas_gen_time.pdf",
    )
