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
POSE_FILE="$("${PYTHON_BIN}" - <<'PY'
from path_config import get_path_value
print(get_path_value('JAAD_pose_npz_fixed', '/media/meta/File/datasets/Intention/JAAD_dataset/PN_ego/jaad_pose_annotations_fixed.npz'))
PY
)"
OUTPUT_ROOT="${REPO_ROOT}/output/stage5_ablation"
LOG_NAME="jaad_pose_no_fused"

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
  "--use_pose" "true"
  "--pose_file" "${POSE_FILE}"
  "--pose_format" "jaad_hrnet_npz"
  "--use_fused_decoder_input" "false"
)

echo "Running JAAD pose experiment without fused decoder input"
echo "Final command:"
printf ' %q' "${CMD[@]}"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

exec "${CMD[@]}"
