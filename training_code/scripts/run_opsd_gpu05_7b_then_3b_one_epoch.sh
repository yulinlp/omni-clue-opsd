#!/usr/bin/env bash
set -Eeuo pipefail

# Persistent sequential OPSD experiment for the four A100 cards in xfu's
# gpu05 allocation.  The caller should start this script with nohup+setsid;
# this script itself does not depend on an interactive terminal.
#
# The two models are deliberately run one after the other.  Each run uses
# 4 ranks x 1 sample x 16 gradient-accumulation steps = 64 samples/update.
# With 5,000 rows, 78 optimizer steps consume 4,992 rows, which is the
# project's one-epoch convention (the final eight rows do not form a full
# synchronized update).

project_root="${OMNI_OPSD_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
sequence_root="${OMNI_OPSD_SEQUENCE_ROOT:-${project_root}/output/xfu_gpu05_opsd_formal_1epoch_$(date +%Y%m%d_%H%M%S)}"
dataset="${OMNI_OPSD_DATASET:-${project_root}/data/gap5000/training_matrix/omnivideo_100k_train.opsd.cuda_safe_f256_p7840.jsonl}"
env_root="${OMNI_OPSD_ENV:-/share/home/ylhu/.conda/envs/vllm}"
swift_root="${OMNI_OPSD_MS_SWIFT_ROOT:-${project_root}/ms-swift}"
ffmpeg_bin="${OMNI_OPSD_FFMPEG_BIN:-/share/home/ylhu/.conda/envs/omniagent_gyh/bin/ffmpeg}"
devices="${OMNI_OPSD_CUDA_DEVICES:-0,1,2,3}"
nproc="${OMNI_OPSD_NPROC_PER_NODE:-4}"

mkdir -p "${sequence_root}"

if [[ ! -f "${dataset}" ]]; then
  echo "Dataset does not exist: ${dataset}" >&2
  exit 2
fi
if [[ ! -d "${swift_root}" ]]; then
  echo "ms-swift checkout does not exist: ${swift_root}" >&2
  exit 2
fi
if [[ ! -x "${ffmpeg_bin}" ]]; then
  echo "ffmpeg is not executable: ${ffmpeg_bin}" >&2
  exit 2
fi

dataset_rows="$(awk 'NF {count++} END {print count+0}' "${dataset}")"
dataset_sha256="$(sha256sum "${dataset}" | awk '{print $1}')"
printf '%s\n' \
  "sequence_started_at=$(date --iso-8601=seconds)" \
  "project_root=${project_root}" \
  "dataset=${dataset}" \
  "dataset_rows=${dataset_rows}" \
  "dataset_sha256=${dataset_sha256}" \
  "models=Qwen2.5-Omni-7B then Qwen2.5-Omni-3B" \
  "devices=${devices}" \
  "nproc=${nproc}" \
  "per_device_train_batch_size=1" \
  "gradient_accumulation_steps=16" \
  "effective_batch_size=64" \
  "max_steps=78" \
  "save_steps=13" \
  "one_epoch_samples=4992_of_5000" \
  "model_training_order=7B_then_3B" \
  "use_vllm=true" \
  "vllm_mode=colocate" \
  "vllm_gpu_memory_utilization=0.30" \
  "vllm_sleep_level=1" \
  "vllm_rollout_audio=disabled_for_vllm_0.11_overlapping_modality_limit" \
  "train_forward_audio_video=true" \
  "gkd_logits_topk=100_with_tail_mass" \
  "rollout_top_p=1.0" \
  "rollout_top_k=20" \
  "video_reader=decord" \
  "max_frames_per_view=256" \
  "max_pixels=7840" \
  "started_by=nohup_setsid" \
  > "${sequence_root}/SEQUENCE_CONFIG.txt"

printf '%s\n' "${BASHPID}" > "${sequence_root}/sequence_script.pid"
touch "${sequence_root}/SEQUENCE_RUNNING"

run_model() {
  local label="$1"
  local model="$2"
  local port="$3"
  local model_root="${sequence_root}/${label}"
  local rc

  mkdir -p "${model_root}"
  if [[ -f "${model_root}/COMPLETED" ]]; then
    echo "SKIP ${label}: COMPLETED already exists"
    return 0
  fi
  if [[ ! -d "${model}" ]]; then
    echo "Model does not exist: ${model}" >&2
    return 2
  fi

  printf '%s\n' \
    "label=${label}" \
    "model=${model}" \
    "root=${model_root}" \
    "started_at=$(date --iso-8601=seconds)" \
    "status=RUNNING" \
    > "${model_root}/STATUS.txt"
  touch "${model_root}/RUNNING"

  export OMNI_OPSD_PROJECT_ROOT="${project_root}"
  export OMNI_OPSD_MS_SWIFT_ROOT="${swift_root}"
  export OMNI_OPSD_ENV="${env_root}"
  export OMNI_OPSD_MODEL="${model}"
  export OMNI_OPSD_DATASET="${dataset}"
  export OMNI_OPSD_OPSD_ROOT="${model_root}"
  export OMNI_OPSD_OPSD_TAG="xfu_gpu05_opsd_formal_1epoch_${label}_20260909"
  export OMNI_OPSD_EXPERIMENT_LABEL="xfu_gpu05_opsd_formal_1epoch_${label}"
  export OMNI_OPSD_CUDA_DEVICES="${devices}"
  export OMNI_OPSD_NPROC_PER_NODE="${nproc}"
  export OMNI_OPSD_MAX_STEPS="${OMNI_OPSD_FORMAL_MAX_STEPS:-78}"
  export OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS="${OMNI_OPSD_FORMAL_GRADIENT_ACCUMULATION_STEPS:-16}"
  export OMNI_OPSD_SAVE_STEPS="${OMNI_OPSD_FORMAL_SAVE_STEPS:-13}"
  export OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE="${OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
  export OMNI_OPSD_MAX_LENGTH="${OMNI_OPSD_MAX_LENGTH:-32768}"
  export OMNI_OPSD_MIN_PIXELS="${OMNI_OPSD_MIN_PIXELS:-3136}"
  export OMNI_OPSD_MAX_COMPLETION_LENGTH="${OMNI_OPSD_MAX_COMPLETION_LENGTH:-8}"
  export OMNI_OPSD_GKD_LOGITS_TOPK="${OMNI_OPSD_GKD_LOGITS_TOPK:-100}"
  export OMNI_OPSD_ROLLOUT_TOP_P="${OMNI_OPSD_ROLLOUT_TOP_P:-1.0}"
  export OMNI_OPSD_ROLLOUT_TOP_K="${OMNI_OPSD_ROLLOUT_TOP_K:-20}"
  export OMNI_OPSD_USE_VLLM="true"
  export OMNI_OPSD_VLLM_MODE="colocate"
  export OMNI_OPSD_VLLM_GPU_MEMORY_UTILIZATION="${OMNI_OPSD_VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
  export OMNI_OPSD_VLLM_SLEEP_LEVEL="${OMNI_OPSD_VLLM_SLEEP_LEVEL:-1}"
  export OMNI_OPSD_VLLM_DROP_AUDIO="1"
  export OMNI_OPSD_GKD_SAFE_MODE="1"
  export OMNI_OPSD_GKD_MAX_GRAD_NORM="${OMNI_OPSD_GKD_MAX_GRAD_NORM:-1.0}"
  export OMNI_OPSD_MAX_GRAD_NORM="${OMNI_OPSD_MAX_GRAD_NORM:-1.0}"
  export OMNI_OPSD_ALLOC_CONF="${OMNI_OPSD_ALLOC_CONF:-expandable_segments:True}"
  export OMNI_OPSD_ATTN_IMPL="${OMNI_OPSD_ATTN_IMPL:-sdpa}"
  export OMNI_OPSD_ALLOW_SPARSE_ENGINEERING_PILOT="0"
  export USE_AUDIO_IN_VIDEO="1"
  export FORCE_QWENVL_VIDEO_READER="decord"
  export MAX_NUM_WORKERS_FETCH_VIDEO="1"
  export OMNI_OPSD_FFMPEG_BIN="${ffmpeg_bin}"
  export OMNI_OPSD_DEBUG_VLLM_INPUT="0"
  export MASTER_ADDR="127.0.0.1"
  export MASTER_PORT="${port}"

  {
    printf 'launch_started_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'model=%s\nmodel_root=%s\nmaster_port=%s\n' "${model}" "${model_root}" "${port}"
    printf 'max_steps=%s\ngradient_accumulation_steps=%s\nsave_steps=%s\n' \
      "${OMNI_OPSD_MAX_STEPS}" "${OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS}" "${OMNI_OPSD_SAVE_STEPS}"
    printf 'use_vllm=%s\nvllm_drop_audio=%s\ngkd_safe_mode=%s\n' \
      "${OMNI_OPSD_USE_VLLM}" "${OMNI_OPSD_VLLM_DROP_AUDIO}" "${OMNI_OPSD_GKD_SAFE_MODE}"
  } > "${model_root}/MODEL_LAUNCH_CONFIG.txt"

  echo "START ${label} $(date --iso-8601=seconds)"
  set +e
  bash "${project_root}/scripts/run_gap5000_opsd_cuda.sh" formal \
    > "${model_root}/driver_console.log" 2>&1
  rc=$?
  set -e

  printf '%s\n' \
    "label=${label}" \
    "model=${model}" \
    "ended_at=$(date --iso-8601=seconds)" \
    "exit_code=${rc}" \
    "status=$([[ "${rc}" -eq 0 ]] && printf COMPLETED || printf FAILED)" \
    >> "${model_root}/STATUS.txt"
  printf '%s\n' "${rc}" > "${model_root}/exit_code"
  if [[ "${rc}" -eq 0 ]]; then
    touch "${model_root}/COMPLETED"
    echo "END ${label} $(date --iso-8601=seconds) rc=0"
    return 0
  fi
  touch "${model_root}/FAILED"
  echo "END ${label} $(date --iso-8601=seconds) rc=${rc}" >&2
  return "${rc}"
}

model7b="${OMNI_OPSD_MODEL_7B:-/share/home/ylhu/models/Qwen2.5-Omni-7B}"
model3b="${OMNI_OPSD_MODEL_3B:-/share/home/ylhu/models/Qwen2.5-Omni-3B}"

if ! run_model qwen25_omni_7b "${model7b}" "${OMNI_OPSD_MASTER_PORT_7B:-29961}"; then
  printf '%s\n' "sequence_stopped_at=7b" "stopped_at=$(date --iso-8601=seconds)" > "${sequence_root}/SEQUENCE_FAILED.txt"
  rm -f "${sequence_root}/SEQUENCE_RUNNING"
  exit 1
fi

if ! run_model qwen25_omni_3b "${model3b}" "${OMNI_OPSD_MASTER_PORT_3B:-29962}"; then
  printf '%s\n' "sequence_stopped_at=3b" "stopped_at=$(date --iso-8601=seconds)" > "${sequence_root}/SEQUENCE_FAILED.txt"
  rm -f "${sequence_root}/SEQUENCE_RUNNING"
  exit 1
fi

printf '%s\n' \
  "sequence_completed_at=$(date --iso-8601=seconds)" \
  "qwen25_omni_7b=COMPLETED" \
  "qwen25_omni_3b=COMPLETED" \
  > "${sequence_root}/SEQUENCE_COMPLETED.txt"
rm -f "${sequence_root}/SEQUENCE_RUNNING"
echo "SEQUENCE_COMPLETED ${sequence_root} $(date --iso-8601=seconds)"
