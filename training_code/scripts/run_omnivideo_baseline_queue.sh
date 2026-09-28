#!/usr/bin/env bash
set -euo pipefail

# Run a pair of OmniVideo-100K baseline arms on one complete 8-NPU worker.
# The gate starts one queue on worker2 and one on worker3, so the four arms
# occupy all sixteen NPUs while keeping each distributed job on one host.

project_root="${OMNI_OPSD_PROJECT_ROOT:?set OMNI_OPSD_PROJECT_ROOT}"
queue_root="${OMNI_OPSD_QUEUE_ROOT:?set OMNI_OPSD_QUEUE_ROOT}"
dataset_root="${OMNI_OPSD_MATRIX_DATASET_ROOT:?set OMNI_OPSD_MATRIX_DATASET_ROOT}"
model="${OMNI_OPSD_MODEL:?set OMNI_OPSD_MODEL}"
ms_swift_root="${OMNI_OPSD_MS_SWIFT_ROOT:?set OMNI_OPSD_MS_SWIFT_ROOT}"
python_deps="${OMNI_OPSD_PYTHON_DEPS:?set OMNI_OPSD_PYTHON_DEPS}"
swift_bin="${OMNI_OPSD_SWIFT_BIN:-$(command -v swift)}"
python_bin="${OMNI_OPSD_PYTHON_BIN:-$(command -v python)}"
queue_arms_csv="${OMNI_OPSD_QUEUE_ARMS:?set OMNI_OPSD_QUEUE_ARMS (for example sft,opsd)}"
arm_launcher="${project_root}/scripts/run_video_odyssey_training_arm.sh"
nproc="${OMNI_OPSD_NPROC_PER_NODE:-8}"
devices="${OMNI_OPSD_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  arm_launcher="${project_root}/scripts/run_training_arm_a100.sh"
  devices="${OMNI_OPSD_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
  if [[ "${devices}" != "${CUDA_VISIBLE_DEVICES}" ]]; then
    echo "OMNI_OPSD_VISIBLE_DEVICES must match CUDA_VISIBLE_DEVICES" >&2
    exit 2
  fi
  IFS=',' read -r -a cuda_devices <<< "${devices}"
  nproc="${OMNI_OPSD_NPROC_PER_NODE:-${#cuda_devices[@]}}"
  export OMNI_OPSD_GKD_SAFE_MODE="${OMNI_OPSD_GKD_SAFE_MODE:-0}"
  export OMNI_OPSD_GKD_MAX_GRAD_NORM="${OMNI_OPSD_GKD_MAX_GRAD_NORM:-1.0}"
fi
max_steps="${OMNI_OPSD_FULL_MAX_STEPS:-300}"
save_steps="${OMNI_OPSD_FULL_SAVE_STEPS:-25}"
gradient_accumulation="${OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS:-4}"
per_device_batch="${OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
max_length="${OMNI_OPSD_MAX_LENGTH:-32768}"
allocator_conf="${OMNI_OPSD_ALLOC_CONF:-}"
min_pixels="${OMNI_OPSD_MIN_PIXELS:-3136}"
max_grad_norm="${OMNI_OPSD_MAX_GRAD_NORM:-1.0}"
# Ascend 910B1 can OOM in aclnnLinalgVectorNorm while clipping the large
# multimodal GKD gradient.  Keep clipping for SFT/GRPO, but make the GKD
# workaround explicit and overridable so a future platform can restore it.
gkd_max_grad_norm="${OMNI_OPSD_GKD_MAX_GRAD_NORM:-0}"
attn_impl="${OMNI_OPSD_ATTN_IMPL:-sdpa}"
audio_in_video="${USE_AUDIO_IN_VIDEO:-1}"
connect_timeout="${HCCL_CONNECT_TIMEOUT:-1800}"
exec_timeout="${HCCL_EXEC_TIMEOUT:-1800}"
seed="${OMNI_OPSD_SEED:-20260904}"
resume_checkpoint="${OMNI_OPSD_RESUME_FROM_CHECKPOINT:-}"

IFS=',' read -r -a arms <<< "${queue_arms_csv}"
if [[ "${#arms[@]}" -eq 0 ]]; then
  echo "The baseline queue is empty" >&2
  exit 2
fi
IFS=',' read -r -a visible_devices <<< "${devices}"
if [[ "${#visible_devices[@]}" -ne "${nproc}" ]]; then
  echo "Expected ${nproc} visible NPUs, got ${#visible_devices[@]} (${devices})" >&2
  exit 2
fi
for arm in "${arms[@]}"; do
  case "${arm}" in
    sft|grpo|opsd|clue_opsd) ;;
    *) echo "Unsupported baseline arm: ${arm}" >&2; exit 2 ;;
  esac
  dataset="${dataset_root}/omnivideo_100k_train.${arm}.jsonl"
  [[ -s "${dataset}" ]] || { echo "Missing dataset: ${dataset}" >&2; exit 2; }
done
[[ -x "${arm_launcher}" ]] || { echo "Training arm launcher is not executable: ${arm_launcher}" >&2; exit 2; }
if [[ -n "${resume_checkpoint}" && ! -d "${resume_checkpoint}" ]]; then
  echo "Resume checkpoint does not exist: ${resume_checkpoint}" >&2
  exit 2
fi
for required in "${model}" "${ms_swift_root}/swift/rl_core/data.py" "${python_bin}" "${swift_bin}"; do
  [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 2; }
done

mkdir -p "${queue_root}/logs" "${queue_root}/pids"
printf 'timestamp\tphase\tarm\tstatus\texit_code\tdevices\n' > "${queue_root}/completion_manifest.tsv"
printf '%s\n' \
  "dataset_root=${dataset_root}" \
  "persistent_dataset_root=${OMNI_OPSD_PERSISTENT_MATRIX_DATASET_ROOT:-${dataset_root}}" \
  "model=${model}" \
  "ms_swift_root=${ms_swift_root}" \
  "nproc=${nproc}" \
  "visible_devices=${devices}" \
  "audio_in_video=${audio_in_video}" \
  "attn_impl=${attn_impl}" \
  "max_steps=${max_steps}" \
  "allocator_conf=${allocator_conf:-default}" \
  "min_pixels=${min_pixels}" \
  "max_grad_norm=${max_grad_norm}" \
  "gkd_max_grad_norm=${gkd_max_grad_norm}" \
  "per_device_train_batch_size=${per_device_batch}" \
  "gradient_accumulation_steps=${gradient_accumulation}" \
  "effective_batch_size=$((nproc * per_device_batch * gradient_accumulation))" \
  "resume_from_checkpoint=${resume_checkpoint}" \
  "seed=${seed}" \
  > "${queue_root}/QUEUE_CLASSIFICATION.txt"

declare -A smoke_ok=()
overall_failed=0

monitor_npu() {
  local output="$1"
  while [[ -e "${output}/RUNNING" ]]; do
    printf 'sample_unix=%s\n' "$(date +%s)"
    if command -v npu-smi >/dev/null 2>&1; then
      npu-smi info
    elif command -v nvidia-smi >/dev/null 2>&1; then
      # The training launcher is shared by Ascend and CUDA nodes.  Keep the
      # historical function name for compatibility, but monitor A100s too.
      nvidia-smi
    else
      printf '%s\n' 'No npu-smi or nvidia-smi found; stopping accelerator monitor.'
      break
    fi
    sleep 5
  done
}

port_for() {
  case "$1" in
    sft) printf '29911' ;;
    grpo) printf '29912' ;;
    opsd) printf '29913' ;;
    clue_opsd) printf '29914' ;;
  esac
}

run_arm() {
  local phase="$1"
  local arm="$2"
  local output="${queue_root}/${phase}/${arm}"
  local log="${queue_root}/logs/${phase}.${arm}.log"
  local dataset="${dataset_root}/omnivideo_100k_train.${arm}.jsonl"
  local port steps phase_save_steps arm_grad_norm
  local resume_for_run=""
  port="$(port_for "${arm}")"
  arm_grad_norm="${max_grad_norm}"
  case "${arm}" in
    opsd|clue_opsd) arm_grad_norm="${gkd_max_grad_norm}" ;;
  esac
  if [[ "${phase}" == smoke ]]; then
    steps=1
    phase_save_steps=1
  else
    steps="${max_steps}"
    phase_save_steps="${save_steps}"
    resume_for_run="${resume_checkpoint}"
  fi
  mkdir -p "${output}"
  if [[ -e "${output}/SUCCESS" || -e "${output}/FAILED" || -e "${output}/SKIPPED" ]]; then
    echo "Refusing to reuse terminal output: ${output}" >&2
    return 2
  fi
  touch "${output}/RUNNING"
  monitor_npu "${output}" > "${output}/npu_samples.log" 2>&1 &
  local monitor_pid=$!
  printf '%s\n' "$$" > "${queue_root}/pids/${phase}.${arm}.queue.pid"
  printf '%s\n' \
    "phase=${phase}" \
    "arm=${arm}" \
    "dataset=${dataset}" \
    "devices=${devices}" \
    "max_grad_norm=${arm_grad_norm}" \
    "started_at=$(date '+%F %T %Z')" \
    > "${output}/RUN_CLASSIFICATION.txt"
  set +e
  env \
    PATH="$(dirname "${python_bin}"):${PATH}" \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT="${port}" \
    ASCEND_RT_VISIBLE_DEVICES="${devices}" \
    OMNI_OPSD_MODEL="${model}" \
    OMNI_OPSD_MS_SWIFT_ROOT="${ms_swift_root}" \
    OMNI_OPSD_PYTHON_DEPS="${python_deps}" \
    OMNI_OPSD_PYTHON_BIN="${python_bin}" \
    OMNI_OPSD_SWIFT_BIN="${swift_bin}" \
    OMNI_OPSD_PROJECT_ROOT="${project_root}" \
    OMNI_OPSD_ARM="${arm}" \
    OMNI_OPSD_DATASET="${dataset}" \
    OMNI_OPSD_OUTPUT_DIR="${output}" \
    OMNI_OPSD_NPROC_PER_NODE="${nproc}" \
    OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS="${gradient_accumulation}" \
    OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE="${per_device_batch}" \
    OMNI_OPSD_MAX_LENGTH="${max_length}" \
    OMNI_OPSD_ALLOC_CONF="${allocator_conf}" \
    OMNI_OPSD_MIN_PIXELS="${min_pixels}" \
    OMNI_OPSD_MAX_GRAD_NORM="${arm_grad_norm}" \
    OMNI_OPSD_MAX_STEPS="${steps}" \
    OMNI_OPSD_SAVE_STEPS="${phase_save_steps}" \
    OMNI_OPSD_RESUME_FROM_CHECKPOINT="${resume_for_run}" \
    OMNI_OPSD_CLUE_EMA_ALPHA="${OMNI_OPSD_CLUE_EMA_ALPHA:-0.05}" \
    OMNI_OPSD_ALLOW_NON_EMA_CLUE="${OMNI_OPSD_ALLOW_NON_EMA_CLUE:-0}" \
    OMNI_OPSD_EXPERIMENT_LABEL="omnivideo_100k_f1_${phase}_${arm}_${nproc}devices" \
    OMNI_OPSD_ATTN_IMPL="${attn_impl}" \
    OMNI_OPSD_SEED="${seed}" \
    USE_AUDIO_IN_VIDEO="${audio_in_video}" \
    HCCL_CONNECT_TIMEOUT="${connect_timeout}" \
    HCCL_EXEC_TIMEOUT="${exec_timeout}" \
    MODEL_SEQ_LEN="${OMNI_OPSD_MODEL_SEQ_LEN:-32768}" \
    MAX_NUM_WORKERS_FETCH_VIDEO="${OMNI_OPSD_MAX_NUM_WORKERS_FETCH_VIDEO:-1}" \
    bash "${arm_launcher}" "${arm}" > "${log}" 2>&1
  local status=$?
  set -e
  rm -f "${output}/RUNNING"
  wait "${monitor_pid}" || true
  printf '%s\n' "finished_at=$(date '+%F %T %Z')" >> "${output}/RUN_CLASSIFICATION.txt"
  printf '%s\n' "${status}" > "${output}/exit_code.txt"
  if [[ "${status}" -eq 0 ]]; then
    touch "${output}/SUCCESS"
    printf '%s\t%s\t%s\tSUCCESS\t0\t%s\n' "$(date --iso-8601=seconds)" "${phase}" "${arm}" "${devices}" >> "${queue_root}/completion_manifest.tsv"
  else
    touch "${output}/FAILED"
    printf '%s\t%s\t%s\tFAILED\t%s\t%s\n' "$(date --iso-8601=seconds)" "${phase}" "${arm}" "${status}" "${devices}" >> "${queue_root}/completion_manifest.tsv"
  fi
  return "${status}"
}

for arm in "${arms[@]}"; do
  if run_arm smoke "${arm}"; then
    smoke_ok["${arm}"]=1
  else
    smoke_ok["${arm}"]=0
    overall_failed=1
    echo "smoke failed for ${arm}; its full run will be skipped" >&2
  fi
done

for arm in "${arms[@]}"; do
  if [[ "${smoke_ok[${arm}]:-0}" -ne 1 ]]; then
    output="${queue_root}/full/${arm}"
    mkdir -p "${output}"
    printf '%s\n' "smoke_failed" > "${output}/SKIPPED"
    printf '%s\tfull\t%s\tSKIPPED\tsmoke_failed\t%s\n' "$(date --iso-8601=seconds)" "${arm}" "${devices}" >> "${queue_root}/completion_manifest.tsv"
    continue
  fi
  if ! run_arm full "${arm}"; then
    overall_failed=1
    echo "full run failed for ${arm}; continuing to the next arm" >&2
  fi
done

exit "${overall_failed}"
