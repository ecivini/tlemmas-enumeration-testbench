import argparse
import itertools
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import matplotlib.axes as pltaxes
import numpy as np
from matplotlib import ticker

matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)

import matplotlib.pyplot as plt  # noqa: E402

RESULTS_TIME_KEY = "Total time"
RESULTS_TLEMMAS_NUM_KEY = "Lemmas"
RESULTS_TLEMMAS_MEDIAN_SIZE_KEY = "Median T-lemma size"
TICK_FONTSIZE = 22


RunData = tuple[dict[str, float], dict[str, float], dict[str, float]]


def _latex_escape(label: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in label)


def _method_label(label: str) -> str:
    return rf"\textsf{{{_latex_escape(label)}}}"


def _normalize_problem_name(problem: str, run_dir: Path) -> str:
    path = Path(os.path.normpath(problem))
    if path.name in {"logs.json", "tlemmas.smt2"}:
        path = path.parent

    if path.suffix == ".smt2" and path.stem == "encoding":
        raise ValueError(f"Unsupported legacy formula error key: {problem}")
    if path.name == "problem.smt2":
        path = path.parent
    elif path.suffix == ".smt2":
        path = path.with_suffix("")

    parts = path.parts
    benchmark_name = run_dir.name

    if parts[:2] == ("data", "benchmark"):
        parts = parts[2:]
        if parts and parts[0] == benchmark_name:
            parts = parts[1:]

    return str(Path(*parts)) if parts else path.name


def _load_run_data(
    run_dir: Path,
    timeout: float,
    read_lemma_stats: bool,
) -> RunData:
    times: dict[str, float] = {}
    lemma_counts: dict[str, float] = {}
    median_sizes: dict[str, float] = {}

    for logs_path in sorted(run_dir.rglob("logs.json")):
        data: dict[str, Any] = json.loads(logs_path.read_text())
        problem_name = os.path.relpath(logs_path.parent, run_dir)

        times[problem_name] = float(data[RESULTS_TIME_KEY])
        lemma_counts[problem_name] = float(data[RESULTS_TLEMMAS_NUM_KEY])

        if read_lemma_stats and RESULTS_TLEMMAS_MEDIAN_SIZE_KEY in data:
            median_sizes[problem_name] = float(data[RESULTS_TLEMMAS_MEDIAN_SIZE_KEY])

    errors_path = run_dir / "errors.json"
    if errors_path.exists():
        errors: dict[str, str] = json.loads(errors_path.read_text())
        for problem, reason in errors.items():
            if reason != "timeout":
                raise ValueError(f"Unexpected error reason in {errors_path}: {reason}")

            problem_name = _normalize_problem_name(problem, run_dir)
            times.setdefault(problem_name, timeout)

    return times, lemma_counts, median_sizes


def _align_common_keys(
    datasets: Sequence[dict[str, float]],
) -> list[dict[str, float]]:
    if not datasets:
        return []

    common_keys = set(datasets[0])
    for dataset in datasets[1:]:
        common_keys &= set(dataset)

    return [
        {problem: dataset[problem] for problem in sorted(common_keys)}
        for dataset in datasets
    ]


def _pairwise_common_value_max(datasets: Sequence[dict[str, float]]) -> float | None:
    value_max: float | None = None
    for i, j in itertools.combinations(range(len(datasets)), 2):
        common_keys = set(datasets[i]) & set(datasets[j])
        for dataset_index in (i, j):
            for key in common_keys:
                value = datasets[dataset_index][key]
                if value_max is None or value > value_max:
                    value_max = value

    return value_max


def _file_label(label: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", label)).strip("_").lower()


def _add_diagonal_guides(
    ax: pltaxes.Axes,
    axis_max: float,
    log_scale: bool,
) -> None:
    main_line_style = {
        "color": "black",
        "linestyle": ":",
        "linewidth": 1.8,
        "alpha": 0.85,
        "zorder": 3,
    }
    factor_line_style = {
        "color": "black",
        "linestyle": ":",
        "linewidth": 1.2,
        "alpha": 0.45,
        "zorder": 2,
    }

    def to_axes_x(value: float) -> float:
        return ax.transAxes.inverted().transform(
            ax.transData.transform((value, 0.0))
        )[0]

    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        transform=ax.transAxes,
        clip_on=True,
        **main_line_style,
    )

    if not log_scale:
        return

    max_factor_exponent = min(10, int(math.log10(axis_max)))
    one_coord = to_axes_x(1.0)
    for exponent in range(1, max_factor_exponent + 1):
        factor = 10**exponent
        if factor > axis_max:
            break

        offset = to_axes_x(float(factor)) - one_coord
        if offset >= 1.0:
            break

        for x_values, y_values in (
            ([offset, 1.0], [0.0, 1.0 - offset]),
            ([0.0, 1.0 - offset], [offset, 1.0]),
        ):
            ax.plot(
                x_values,
                y_values,
                transform=ax.transAxes,
                clip_on=True,
                **factor_line_style,
            )


def _set_matching_axis_limits_and_ticks(
    ax: pltaxes.Axes,
    plot_min: float,
    axis_max: float,
    log_scale: bool,
    integer_ticks: bool = False,
    omit_upper_tick: bool = False,
) -> None:
    ax.set_xlim(left=plot_min, right=axis_max)
    ax.set_ylim(bottom=plot_min, top=axis_max)

    def omit_top_tick_if_too_high(ticks: list[float]) -> list[float]:
        if not omit_upper_tick or len(ticks) <= 2:
            return ticks

        tick_step = ticks[-1] - ticks[-2]
        if axis_max - ticks[-1] <= tick_step * 0.25:
            return ticks[:-1]
        return ticks

    if log_scale:
        max_exponent = math.floor(math.log10(axis_max))
        exponents = range(0, max_exponent + 1)
        ticks = [0.0] + [10**exponent for exponent in exponents]
        ticks = omit_top_tick_if_too_high(ticks)
        tick_labels = ["0"] + [rf"$10^{{{exponent}}}$" for exponent in exponents]
        tick_labels = tick_labels[: len(ticks)]
    elif integer_ticks:
        locator = ticker.MaxNLocator(integer=True)
        ticks = [
            tick
            for tick in locator.tick_values(plot_min, axis_max)
            if plot_min <= tick <= axis_max
        ]
        ticks = omit_top_tick_if_too_high(ticks)
        tick_labels = [f"{tick:g}" for tick in ticks]
    else:
        locator = ticker.MaxNLocator(nbins="auto")
        ticks = [
            tick
            for tick in locator.tick_values(plot_min, axis_max)
            if plot_min <= tick <= axis_max
        ]
        tick_labels = [f"{tick:g}" for tick in ticks]

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_box_aspect(1)


def _set_tick_fontsize(ax: pltaxes.Axes) -> None:
    ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="both", which="minor", labelsize=TICK_FONTSIZE)


def create_cactus_plot(
    datasets: Sequence[tuple[dict[str, float], str]],
    timeout: float,
    out_path: Path,
    legend_loc: str = "center left",
    legend_bbox_to_anchor: tuple[float, float] | None = None,
) -> None:
    markers = ["o", "^", "s", "D", "v", "<", ">", "p", "*", "h"]

    if len(datasets) < 2:
        raise ValueError("Need at least 2 datasets")

    keys = set(datasets[0][0])
    if not keys:
        raise ValueError("No common benchmark data to plot")
    for data, label in datasets[1:]:
        if set(data) != keys:
            raise ValueError(f"{label} does not have the same problem keys")

    _, ax = plt.subplots(figsize=(6, 5))
    for idx, (data, label) in enumerate(datasets):
        sorted_times = sorted(min(data[problem], timeout) for problem in keys)
        x_values = np.arange(1, len(sorted_times) + 1)
        ax.plot(
            x_values,
            sorted_times,
            label=_method_label(label),
            marker=markers[idx % len(markers)],
            markersize=2,
        )

    ax.axhline(timeout, linestyle="--", color="black", alpha=0.5)
    ax.set_xlabel("Number of problems solved", fontsize=24)
    ax.set_ylabel("Time (s)", fontsize=24)
    _set_tick_fontsize(ax)
    ax.grid(True)
    legend_kwargs: dict[str, Any] = {"fontsize": 18, "loc": legend_loc}
    if legend_bbox_to_anchor is not None:
        legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
    ax.legend(**legend_kwargs)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def create_scatter_plot(
    x_data: dict[str, float],
    x_label: str,
    y_data: dict[str, float],
    y_label: str,
    timeout: float | None = None,
    label_suffix: str = "",
    log_scale: bool = True,
    axis_max: float | None = None,
    integer_ticks: bool = False,
    omit_upper_tick: bool = False,
    out_path: Path = Path("scatter.pdf"),
) -> None:
    common_keys = sorted(set(x_data) & set(y_data))
    if not common_keys:
        print("No data for plot:", out_path)
        return

    completed_x, completed_y = [], []
    timeout_x, timeout_y = [], []
    for key in common_keys:
        xv = x_data[key]
        yv = y_data[key]

        if timeout is not None:
            x_is_timeout = xv >= timeout
            y_is_timeout = yv >= timeout
            if x_is_timeout or y_is_timeout:
                timeout_x.append(timeout if x_is_timeout else xv)
                timeout_y.append(timeout if y_is_timeout else yv)
                continue
        completed_x.append(xv)
        completed_y.append(yv)

    data_values = completed_x + completed_y + timeout_x + timeout_y
    plot_max = max(
        timeout if timeout is not None else max(data_values, default=1.0),
        1.0,
    )
    if axis_max is None:
        axis_max = plot_max * 1.1
    plot_min = 0.0

    _, ax = plt.subplots(figsize=(5, 5))

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

    if timeout is not None:
        ax.axvline(timeout, linestyle="--", color="black", alpha=0.5)
        ax.axhline(timeout, linestyle="--", color="black", alpha=0.5)

    if log_scale:
        ax.set_xscale("symlog", linthresh=1.0, base=10)
        ax.set_yscale("symlog", linthresh=1.0, base=10)
    else:
        ax.set_xscale("linear")
        ax.set_yscale("linear")
    _set_matching_axis_limits_and_ticks(
        ax,
        plot_min,
        axis_max,
        log_scale,
        integer_ticks=integer_ticks,
        omit_upper_tick=omit_upper_tick,
    )

    _add_diagonal_guides(ax, axis_max, log_scale)

    ax.set_xlabel(f"{_method_label(x_label)}{label_suffix}", fontsize=24)
    ax.set_ylabel(f"{_method_label(y_label)}{label_suffix}", fontsize=24)
    _set_tick_fontsize(ax)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_all_pairs(
    labels: list[str],
    times: list[dict[str, float]],
    lemma_counts: list[dict[str, float]],
    median_sizes: list[dict[str, float]],
    timeout: float,
    out_dir: Path,
) -> None:
    lemma_count_max = _pairwise_common_value_max(lemma_counts)
    lemma_count_axis_max = max(lemma_count_max or 1.0, 1.0) * 1.1

    median_size_max = _pairwise_common_value_max(median_sizes)
    median_size_axis_max = (
        max(median_size_max, 1.0) * 1.1 if median_size_max is not None else None
    )

    for i, j in itertools.combinations(range(len(labels)), 2):
        left_label = labels[i]
        right_label = labels[j]
        pair_tag = f"{_file_label(left_label)}_vs_{_file_label(right_label)}"

        create_scatter_plot(
            times[i],
            left_label,
            times[j],
            right_label,
            timeout=timeout,
            out_path=out_dir / f"{pair_tag}_tlemmas_gen_time.pdf",
        )
        create_scatter_plot(
            lemma_counts[i],
            left_label,
            lemma_counts[j],
            right_label,
            timeout=None,
            axis_max=lemma_count_axis_max,
            omit_upper_tick=True,
            out_path=out_dir / f"{pair_tag}_tlemmas_num.pdf",
        )

        if median_sizes[i] and median_sizes[j]:
            create_scatter_plot(
                median_sizes[i],
                left_label,
                median_sizes[j],
                right_label,
                timeout=None,
                log_scale=False,
                axis_max=median_size_axis_max,
                integer_ticks=True,
                omit_upper_tick=True,
                out_path=out_dir / f"{pair_tag}_tlemmas_median_size.pdf",
            )


def print_method_stats(
    labels: Sequence[str],
    times: Sequence[dict[str, float]],
    timeout: float,
    lower_threshold: float = 1.0,
) -> None:
    aligned_times = _align_common_keys(times)
    if not aligned_times:
        return

    total = len(aligned_times[0])
    print("\nMethod stats:")
    for label, data in zip(labels, aligned_times, strict=True):
        timeouts = sum(value >= timeout for value in data.values())
        below = sum(
            value <= lower_threshold for value in data.values() if value < timeout
        )
        print(
            f"{label}: problems: {total} | timeouts: {timeouts} "
            f"| below {lower_threshold} sec: {below}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare standardized T-lemma generation results across runs."
    )
    parser.add_argument(
        "--data",
        nargs=2,
        action="append",
        required=True,
        metavar=("DIR", "LABEL"),
        help="A standardized results directory and its label (repeatable).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="Timeout in seconds for timed-out benchmarks.",
    )
    parser.add_argument(
        "--lemma-stats",
        action="store_true",
        help=(
            "Plot precomputed median T-lemma size from logs.json. "
            "Run scripts/plots/precompute_tlemma_stats.py first."
        ),
    )
    args = parser.parse_args()
    if len(args.data) < 2:
        parser.error("At least 2 datasets are required")
    return args


def main() -> None:
    args = _parse_args()
    labels: list[str] = []
    times: list[dict[str, float]] = []
    lemma_counts: list[dict[str, float]] = []
    median_sizes: list[dict[str, float]] = []

    for dir_path, label in args.data:
        run_times, run_lemma_counts, run_median_sizes = _load_run_data(
            Path(dir_path),
            timeout=args.timeout,
            read_lemma_stats=args.lemma_stats,
        )
        labels.append(label)
        times.append(run_times)
        lemma_counts.append(run_lemma_counts)
        median_sizes.append(run_median_sizes)

    cactus_times = _align_common_keys(times)
    cactus_legend_loc = "center left"
    cactus_legend_bbox_to_anchor = None
    if args.out_dir.name == "planning":
        cactus_legend_loc = "center right"
        cactus_legend_bbox_to_anchor = (1.0, 0.6)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_all_pairs(
        labels=labels,
        times=times,
        lemma_counts=lemma_counts,
        median_sizes=median_sizes,
        timeout=args.timeout,
        out_dir=args.out_dir,
    )
    create_cactus_plot(
        [(cactus_times[i], labels[i]) for i in range(len(labels))],
        timeout=args.timeout,
        out_path=args.out_dir / "cactus_all_methods.pdf",
        legend_loc=cactus_legend_loc,
        legend_bbox_to_anchor=cactus_legend_bbox_to_anchor,
    )
    print_method_stats(labels, times, timeout=args.timeout)


if __name__ == "__main__":
    main()
