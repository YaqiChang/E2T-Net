#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/cyq/anaconda3/envs/py38/bin/python"
REPO_ROOT="/media/cyq/Data/project/PIP/E2T-Net"
cd "${REPO_ROOT}"
DATASET_ROOT="$("${PYTHON_BIN}" - <<'PY'
from path_config import get_path_value
print(get_path_value('JAAD_pn_root', '/media/cyq/Data/dataset/Intention/JAAD_dataset/PN_ego'))
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
