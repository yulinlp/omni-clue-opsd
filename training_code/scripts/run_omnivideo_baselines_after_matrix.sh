#!/usr/bin/env bash
set -euo pipefail

# Admit a baseline pair only after the coverage merge gate has produced and
# validated the four-arm matrix. This is intentionally a watcher: it keeps
# the allocated worker useful overnight without starting a stale or coarse
# dataset by accident.

project_root="${OMNI_OPSD_PROJECT_ROOT:?set OMNI_OPSD_PROJECT_ROOT}"
runtime_root="${OMNI_OPSD_RUNTIME_ROOT:?set OMNI_OPSD_RUNTIME_ROOT}"
matrix_root="${OMNI_OPSD_MATRIX_DATASET_ROOT:?set OMNI_OPSD_MATRIX_DATASET_ROOT}"
persistent_matrix_root="${matrix_root}"
model="${OMNI_OPSD_MODEL:?set OMNI_OPSD_MODEL}"
ms_swift_root="${OMNI_OPSD_MS_SWIFT_ROOT:?set OMNI_OPSD_MS_SWIFT_ROOT}"
python_deps="${OMNI_OPSD_PYTHON_DEPS:?set OMNI_OPSD_PYTHON_DEPS}"
python_bin="${OMNI_OPSD_PYTHON_BIN:?set OMNI_OPSD_PYTHON_BIN}"
swift_bin="${OMNI_OPSD_SWIFT_BIN:?set OMNI_OPSD_SWIFT_BIN}"
queue_root="${OMNI_OPSD_QUEUE_ROOT:?set OMNI_OPSD_QUEUE_ROOT}"
queue_arms="${OMNI_OPSD_QUEUE_ARMS:?set OMNI_OPSD_QUEUE_ARMS}"
gate_marker="${OMNI_OPSD_MATRIX_GATE_MARKER:-${runtime_root}/omnivideo_score_20260905/COVERAGE_MATRIX_SUCCESS}"
log_path="${OMNI_OPSD_BASELINE_WATCH_LOG:-${queue_root}/wait_and_run.log}"
poll_seconds="${OMNI_OPSD_BASELINE_POLL_SECONDS:-60}"

mkdir -p "${queue_root}"
log() { printf '%s %s\n' "$(date '+%F %T %Z')" "$*" >> "${log_path}"; }

while [[ ! -e "${gate_marker}" ]]; do
  log waiting_for_matrix_gate
  sleep "${poll_seconds}"
done

for arm in sft grpo opsd clue_opsd; do
  [[ -s "${matrix_root}/omnivideo_100k_train.${arm}.jsonl" ]] || {
    log missing_matrix_dataset_${arm}
    exit 2
  }
done
[[ -s "${matrix_root}/training_matrix_summary.json" ]] || { log missing_matrix_summary; exit 2; }

nproc="${OMNI_OPSD_NPROC_PER_NODE:-8}"
per_device_batch="${OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
gradient_accumulation="${OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS:-4}"
full_max_steps="${OMNI_OPSD_FULL_MAX_STEPS:-300}"
if [[ "${full_max_steps}" == auto ]]; then
  target_epochs="${OMNI_OPSD_TARGET_EPOCHS:-1.92}"
  full_max_steps="$("${python_bin}" - \
    "${matrix_root}/training_matrix_summary.json" \
    "${nproc}" "${per_device_batch}" "${gradient_accumulation}" "${target_epochs}" <<'PY'
import json
import math
import sys

summary_path, nproc, per_device, accumulation, epochs = sys.argv[1:]
rows = int(json.load(open(summary_path, encoding="utf-8"))["rows"])
effective_batch = int(nproc) * int(per_device) * int(accumulation)
print(max(1, math.ceil(rows * float(epochs) / effective_batch)))
PY
  )"
  log auto_training_steps_rows_$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["rows"])' "${matrix_root}/training_matrix_summary.json")_steps_${full_max_steps}
fi
full_save_steps="${OMNI_OPSD_FULL_SAVE_STEPS:-25}"
if [[ "${full_save_steps}" == auto ]]; then
  full_save_steps=$(( full_max_steps < 100 ? (full_max_steps + 3) / 4 : 25 ))
  (( full_save_steps > 0 )) || full_save_steps=1
fi

local_video_cache="${OMNI_OPSD_LOCAL_VIDEO_CACHE:-}"
if [[ -n "${local_video_cache}" ]]; then
  scratch_root="${OMNI_OPSD_SCRATCH_ROOT:-$(dirname "$(dirname "${model}")")}" 
  local_matrix_root="${OMNI_OPSD_LOCAL_MATRIX_ROOT:-${scratch_root}/training_matrix_exact_flips}"
  log localizing_training_matrix_to_${local_matrix_root}
  "${python_bin}" "${project_root}/scripts/materialize_local_training_matrix.py" \
    --source-root "${persistent_matrix_root}" \
    --output-root "${local_matrix_root}" \
    --video-cache-root "${local_video_cache}" \
    --arms sft,grpo,opsd,clue_opsd >> "${log_path}" 2>&1
  matrix_root="${local_matrix_root}"
  log local_training_matrix_success
fi

log matrix_gate_passed_starting_${queue_arms}
exec env \
  OMNI_OPSD_PROJECT_ROOT="${project_root}" \
  OMNI_OPSD_QUEUE_ROOT="${queue_root}" \
  OMNI_OPSD_MATRIX_DATASET_ROOT="${matrix_root}" \
  OMNI_OPSD_PERSISTENT_MATRIX_DATASET_ROOT="${persistent_matrix_root}" \
  OMNI_OPSD_MODEL="${model}" \
  OMNI_OPSD_MS_SWIFT_ROOT="${ms_swift_root}" \
  OMNI_OPSD_PYTHON_DEPS="${python_deps}" \
  OMNI_OPSD_PYTHON_BIN="${python_bin}" \
  OMNI_OPSD_SWIFT_BIN="${swift_bin}" \
  OMNI_OPSD_QUEUE_ARMS="${queue_arms}" \
  OMNI_OPSD_VISIBLE_DEVICES="${OMNI_OPSD_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
  OMNI_OPSD_NPROC_PER_NODE="${nproc}" \
  OMNI_OPSD_FULL_MAX_STEPS="${full_max_steps}" \
  OMNI_OPSD_FULL_SAVE_STEPS="${full_save_steps}" \
  OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS="${gradient_accumulation}" \
  OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE="${per_device_batch}" \
  OMNI_OPSD_MAX_LENGTH="${OMNI_OPSD_MAX_LENGTH:-32768}" \
  OMNI_OPSD_MIN_PIXELS="${OMNI_OPSD_MIN_PIXELS:-3136}" \
  OMNI_OPSD_MAX_GRAD_NORM="${OMNI_OPSD_MAX_GRAD_NORM:-1.0}" \
  OMNI_OPSD_GKD_MAX_GRAD_NORM="${OMNI_OPSD_GKD_MAX_GRAD_NORM:-0}" \
  OMNI_OPSD_ATTN_IMPL="${OMNI_OPSD_ATTN_IMPL:-sdpa}" \
  OMNI_OPSD_SEED="${OMNI_OPSD_SEED:-20260904}" \
  USE_AUDIO_IN_VIDEO="${USE_AUDIO_IN_VIDEO:-1}" \
  HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}" \
  HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1800}" \
  bash "${project_root}/scripts/run_omnivideo_baseline_queue.sh"
