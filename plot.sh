#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON:-python3}"
results_dir="${RESULTS_DIR:-results}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tlemmas-matplotlib}"
mkdir -p "$MPLCONFIGDIR"

"$python_bin" scripts/plots/precompute_tlemma_stats.py "$results_dir" --write

configs=(
  "test_baseline"
  "test_divconq"
  "test_divconq_proj"
  "test_divconq_proj_part"
  "test_divconq_proj_part_divbyproj"
)

labels=(
  "Baseline"
  "C&C"
  "C&C+Proj."
  "C&C+Proj.+Part."
  "CP&C+Proj.+Part."
)

benchmarks=(
  "synthetic:3600"
  "planning:3600"
)

extra_args=()
if [[ "${PLOT_LEMMA_STATS:-0}" == "1" ]]; then
  extra_args+=(--lemma-stats)
fi

for benchmark_timeout in "${benchmarks[@]}"; do
  IFS=: read -r benchmark timeout <<<"$benchmark_timeout"
  out_dir="plots/${benchmark}"
  mkdir -p "$out_dir"

  data_args=()
  for ((i = 0; i < ${#configs[@]}; i++)); do
    data_args+=(--data "${results_dir}/${configs[$i]}/${benchmark}" "${labels[$i]}")
  done

  "$python_bin" scripts/plots/compare_tlemmas_generation_time.py \
    "${data_args[@]}" \
    --timeout "$timeout" \
    --out-dir "$out_dir" \
    "${extra_args[@]}"
done

numeric_benchmark="numeric-planning-canonical"
numeric_out_dir="plots/${numeric_benchmark}"
mkdir -p "$numeric_out_dir"

"$python_bin" scripts/plots/compare_tlemmas_generation_time.py \
  --data "${results_dir}/test_divconq_proj_part/${numeric_benchmark}" \
  "C&C+Proj.+Part." \
  --data "${results_dir}/test_divconq_proj_part_divbyproj/${numeric_benchmark}" \
  "CP&C+Proj.+Part." \
  --timeout 14400 \
  --out-dir "$numeric_out_dir" \
  "${extra_args[@]}"
