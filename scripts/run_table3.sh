#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

device="${1:-0}"
output_root="${2:-outputs/table3}"
python_bin="${PYTHON_BIN:-python}"

if [[ ! -f train.py ]]; then
  echo "train.py not found under ${repo_root}" >&2
  exit 2
fi
if [[ ! -f scripts/evaluate_gzsl.py ]]; then
  echo "scripts/evaluate_gzsl.py not found under ${repo_root}" >&2
  exit 2
fi

run_one() {
  local name="$1"
  local config="$2"
  local train_dir="${output_root}/${name}/train"
  local eval_dir="${output_root}/${name}/eval"

  if [[ -e "${output_root}/${name}" ]]; then
    echo "Refusing to reuse existing output directory: ${output_root}/${name}" >&2
    return 2
  fi

  "${python_bin}" train.py \
    --config "${config}" \
    --device "${device}" \
    --output-dir "${train_dir}"

  "${python_bin}" scripts/evaluate_gzsl.py \
    --config "${config}" \
    --checkpoint "${train_dir}/best_gzsl.pt" \
    --device "${device}" \
    --output-dir "${eval_dir}" \
    --summary-log "${eval_dir}/summary.log"
}

run_one ntu60_55_5 configs/table3/ntu60_xsub_unseen5.yaml
run_one ntu60_48_12 configs/table3/ntu60_xsub_unseen12.yaml
run_one ntu120_110_10 configs/table3/ntu120_xsub_unseen10.yaml
run_one ntu120_96_24 configs/table3/ntu120_xsub_unseen24.yaml

echo "Table 3 runs completed under ${output_root}"
