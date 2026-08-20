#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

device="${1:-0}"
output_root="${2:-outputs/table2}"
python_bin="${PYTHON_BIN:-python}"

if [[ ! -f train.py ]]; then
  echo "train.py not found under ${repo_root}" >&2
  exit 2
fi

configs=(
  ntu60_xsub_unseen5_split1
  ntu60_xsub_unseen5_split2
  ntu60_xsub_unseen5_split3
  ntu120_xsub_unseen10_split1
  ntu120_xsub_unseen10_split2
  ntu120_xsub_unseen10_split3
  pku51_xsub_unseen5_split1
  pku51_xsub_unseen5_split2
  pku51_xsub_unseen5_split3
)

for name in "${configs[@]}"; do
  config="configs/table2/${name}.yaml"
  output_dir="${output_root}/${name}"
  if [[ -e "${output_dir}" ]]; then
    echo "Refusing to reuse existing output directory: ${output_dir}" >&2
    exit 2
  fi

  "${python_bin}" train.py \
    --config "${config}" \
    --device "${device}" \
    --setting ZSL \
    --output-dir "${output_dir}" \
    --gpr-support-ratio 0.4 \
    --no-diagnostic-heads
done

echo "Table 2 runs completed under ${output_root}"
