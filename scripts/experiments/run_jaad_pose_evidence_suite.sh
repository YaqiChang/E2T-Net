#!/usr/bin/env bash
set -euo pipefail

# Stop rules:
# 1) 参数或命名不符预期时立即停
# 2) baseline 失败不继续
# 3) pose_direct_last 失败不继续 pose_accumulator

PYTHON_BIN="${PYTHON_BIN:-/home/meta/anaconda3/envs/3dhuman/bin/python}"
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

DATE_PREFIX="$(date +%m%d)"

run_exp() {
  local log_name="$1"
  shift

  echo "============================================================"
  echo "Running experiment: ${log_name}"
  echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
  echo "============================================================"

  "${PYTHON_BIN}" train.py \
    --log_name "${log_name}" \
    "$@"

  echo "Finished experiment: ${log_name}"
  echo "Finish time: $(date '+%Y-%m-%d %H:%M:%S')"

  echo "Releasing GPU cache..."
  "${PYTHON_BIN}" - <<'PY'
import gc
import torch

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
print("GPU cache released.")
PY

  echo
}

run_exp \
  "${DATE_PREFIX}_baseline" \
  --use_pose False \
  --use_decision_accumulator False

run_exp \
  "${DATE_PREFIX}_pose_direct_last" \
  --use_pose True \
  --use_decision_accumulator False \
  --belief_dim 64 \
  --belief_readout last

run_exp \
  "${DATE_PREFIX}_pose_accumulator" \
  --use_pose True \
  --use_decision_accumulator True \
  --belief_dim 64 \
  --belief_readout last

echo "All experiments finished."
