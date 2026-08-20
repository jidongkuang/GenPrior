#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

device="${1:-0}"
output_root="${2:-outputs/evaluation/table1}"
checkpoint_root="${CHECKPOINT_ROOT:-checkpoints/table1}"
python_bin="${PYTHON_BIN:-python}"

names=(
  ntu60_55_5
  ntu60_48_12
  ntu120_110_10
  ntu120_96_24
)
configs=(
  configs/table1/ntu60_xsub_unseen5.yaml
  configs/table1/ntu60_xsub_unseen12.yaml
  configs/table1/ntu120_xsub_unseen10.yaml
  configs/table1/ntu120_xsub_unseen24.yaml
)

for index in "${!names[@]}"; do
  checkpoint="${checkpoint_root}/${names[index]}/best_gpr.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 2
  fi
  if [[ ! -f "${configs[index]}" ]]; then
    echo "Missing config: ${configs[index]}" >&2
    exit 2
  fi
done

for index in "${!names[@]}"; do
  name="${names[index]}"
  "${python_bin}" scripts/evaluate_zsl.py \
    --config "${configs[index]}" \
    --checkpoint "${checkpoint_root}/${name}/best_gpr.pt" \
    --device "${device}" \
    --output-dir "${output_root}/${name}"
done

echo "Table 1 evaluation completed under ${output_root}"
