#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

device="${1:-0}"
output_root="${2:-outputs/table1}"
python_bin="${PYTHON_BIN:-python}"

if [[ ! -f train.py ]]; then
  echo "train.py not found under ${repo_root}" >&2
  exit 2
fi

run_one() {
  local name="$1"
  local config="$2"
  local prior_weight="$3"
  local support_ratio="$4"
  local output_dir="${output_root}/${name}"

  if [[ -e "${output_dir}" ]]; then
    echo "Refusing to reuse existing output directory: ${output_dir}" >&2
    return 2
  fi

  "${python_bin}" train.py \
    --config "${config}" \
    --device "${device}" \
    --setting ZSL \
    --output-dir "${output_dir}" \
    --prior-fusion-weight "${prior_weight}" \
    --gpr-support-ratio "${support_ratio}" \
    --no-diagnostic-heads
}

run_one ntu60_55_5 configs/table1/ntu60_xsub_unseen5.yaml 1.0 0.4
run_one ntu60_48_12 configs/table1/ntu60_xsub_unseen12.yaml 0.1 0.5
run_one ntu120_110_10 configs/table1/ntu120_xsub_unseen10.yaml 1.0 0.4
run_one ntu120_96_24 configs/table1/ntu120_xsub_unseen24.yaml 0.1 0.4

echo "Table 1 runs completed under ${output_root}"
