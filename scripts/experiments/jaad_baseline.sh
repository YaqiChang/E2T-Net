#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/meta/anaconda3/envs/3dhuman/bin/python"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
DATASET_ROOT="$("${PYTHON_BIN}" - <<'PY'
from path_config import get_path_value
print(get_path_value('JAAD_pn_root', '/media/meta/File/datasets/Intention/JAAD_dataset/PN_ego'))
PY
)"
OUTPUT_ROOT="${REPO_ROOT}/output/stage5_ablation"
LOG_NAME="jaad_baseline"

mkdir -p "${OUTPUT_ROOT}"

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/train.py"
  "--dataset" "jaad"
  "--data_dir" "${DATASET_ROOT}"
  "--out_dir" "${OUTPUT_ROOT}"
  "--log_name" "${LOG_NAME}"
  "--input" "16"
  "--output" "32"
  "--stride" "16"
  "--skip" "1"
  "--use_image" "false"
  "--use_attribute" "true"
  "--use_opticalflow" "false"
  "--use_pose" "false"
  "--use_fused_decoder_input" "false"
)

echo "Running JAAD baseline experiment"
echo "Final command:"
printf ' %q' "${CMD[@]}"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

exec "${CMD[@]}"
