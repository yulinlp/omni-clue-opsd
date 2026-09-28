#!/usr/bin/env bash
set -euo pipefail

# Sequential formal CUDA runner for the dynamic-budget 5,000-row OmniVideo
# corpus.  It uses all four cards in the selected CUDA slice and waits for SFT
# to finish before starting OPSD, so the two runs never contend for memory.

project_root="${OMNI_OPSD_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_env="${OMNI_OPSD_ENV:-/share/home/ylhu/.conda/envs/vllm}"
swift_root="${OMNI_OPSD_MS_SWIFT_ROOT:-${project_root}/ms-swift}"
model="${OMNI_OPSD_MODEL:-/share/home/ylhu/models/Qwen2.5-Omni-7B}"
devices="${OMNI_OPSD_CUDA_DEVICES:-0,1,2,3}"
tag="${OMNI_OPSD_FORMAL_TAG:-$(date +%Y%m%d_%H%M%S)}"
root="${OMNI_OPSD_FORMAL_ROOT:-${project_root}/output/gap5000_formal_${tag}}"
ffmpeg_bin="${OMNI_OPSD_FFMPEG_BIN:-}"
if [[ -z "${ffmpeg_bin}" && -x "/share/home/ylhu/.conda/envs/omniagent_gyh/bin/ffmpeg" ]]; then
  ffmpeg_bin="/share/home/ylhu/.conda/envs/omniagent_gyh/bin/ffmpeg"
fi
if [[ -n "${ffmpeg_bin}" ]]; then
  [[ -x "${ffmpeg_bin}" ]] || { echo "ffmpeg is not executable: ${ffmpeg_bin}" >&2; exit 2; }
  export PATH="$(dirname "${ffmpeg_bin}"):${PATH}"
fi

mkdir -p "${root}/environment" "${root}/logs"

export CUDA_VISIBLE_DEVICES="${devices}"
export ASCEND_RT_VISIBLE_DEVICES="${devices}"
IFS=',' read -r -a cuda_devices <<< "${devices}"
if [[ "${#cuda_devices[@]}" -ne 4 ]]; then
  echo "The dynamic formal arm requires four CUDA devices; got ${devices}" >&2
  exit 2
fi
export OMNI_OPSD_NPROC_PER_NODE="${OMNI_OPSD_NPROC_PER_NODE:-${#cuda_devices[@]}}"
if [[ "${OMNI_OPSD_NPROC_PER_NODE}" -ne "${#cuda_devices[@]}" ]]; then
  echo "OMNI_OPSD_NPROC_PER_NODE must match CUDA_VISIBLE_DEVICES=${devices}" >&2
  exit 2
fi
export OMNI_OPSD_PROJECT_ROOT="${project_root}"
export OMNI_OPSD_MS_SWIFT_ROOT="${swift_root}"
export OMNI_OPSD_PYTHON_DEPS="${project_root}"
export OMNI_OPSD_PYTHON_BIN="${OMNI_OPSD_PYTHON_BIN:-${python_env}/bin/python}"
export OMNI_OPSD_SWIFT_BIN="${OMNI_OPSD_SWIFT_BIN:-${python_env}/bin/swift}"
export OMNI_OPSD_MODEL="${model}"
export OMNI_OPSD_MAX_STEPS="${OMNI_OPSD_MAX_STEPS:-157}"
export OMNI_OPSD_NUM_TRAIN_EPOCHS="${OMNI_OPSD_NUM_TRAIN_EPOCHS:-1}"
export OMNI_OPSD_SAVE_STEPS="${OMNI_OPSD_SAVE_STEPS:-25}"
export OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE="${OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS="${OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS:-4}"
export OMNI_OPSD_LORA_RANK="${OMNI_OPSD_LORA_RANK:-64}"
export OMNI_OPSD_LORA_ALPHA="${OMNI_OPSD_LORA_ALPHA:-128}"
export OMNI_OPSD_MAX_LENGTH="${OMNI_OPSD_MAX_LENGTH:-32768}"
export OMNI_OPSD_MIN_PIXELS="${OMNI_OPSD_MIN_PIXELS:-3136}"
export OMNI_OPSD_MAX_COMPLETION_LENGTH="${OMNI_OPSD_MAX_COMPLETION_LENGTH:-512}"
export OMNI_OPSD_GKD_LOGITS_TOPK="${OMNI_OPSD_GKD_LOGITS_TOPK:-20}"
export OMNI_OPSD_ROLLOUT_TOP_P="${OMNI_OPSD_ROLLOUT_TOP_P:-0.95}"
export OMNI_OPSD_ROLLOUT_TOP_K="${OMNI_OPSD_ROLLOUT_TOP_K:-20}"
export OMNI_OPSD_USE_VLLM="${OMNI_OPSD_USE_VLLM:-true}"
export OMNI_OPSD_VLLM_MODE="${OMNI_OPSD_VLLM_MODE:-colocate}"
export OMNI_OPSD_VLLM_TENSOR_PARALLEL_SIZE="${OMNI_OPSD_VLLM_TENSOR_PARALLEL_SIZE:-2}"
export OMNI_OPSD_VLLM_GPU_MEMORY_UTILIZATION="${OMNI_OPSD_VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
export OMNI_OPSD_VLLM_SLEEP_LEVEL="${OMNI_OPSD_VLLM_SLEEP_LEVEL:-1}"
export OMNI_OPSD_ATTN_IMPL="${OMNI_OPSD_ATTN_IMPL:-sdpa}"
export OMNI_OPSD_GKD_SAFE_MODE="${OMNI_OPSD_GKD_SAFE_MODE:-0}"
export OMNI_OPSD_GKD_MAX_GRAD_NORM="${OMNI_OPSD_GKD_MAX_GRAD_NORM:-0}"
export OMNI_OPSD_MAX_GRAD_NORM="${OMNI_OPSD_MAX_GRAD_NORM:-0}"
export OMNI_OPSD_OPSD_EMA_ALPHA="${OMNI_OPSD_OPSD_EMA_ALPHA:-0.05}"
formal_opsd_ema_alpha="${OMNI_OPSD_OPSD_EMA_ALPHA}"
export USE_AUDIO_IN_VIDEO="${USE_AUDIO_IN_VIDEO:-1}"
export MAX_NUM_WORKERS_FETCH_VIDEO="${MAX_NUM_WORKERS_FETCH_VIDEO:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED=1
export PATH="${python_env}/bin:${PATH}"
export PYTHONPATH="${project_root}/src:${project_root}:${swift_root}${PYTHONPATH:+:${PYTHONPATH}}"

{
  printf 'project_root=%s\nmodel=%s\ndevices=%s\nnproc=%s\nmax_steps=%s\nnum_train_epochs=%s\nsave_steps=%s\nper_device_train_batch_size=%s\ngradient_accumulation_steps=%s\nlora_rank=%s\nlora_alpha=%s\ngkd_logits_topk=%s\nrollout_top_p=%s\nrollout_top_k=%s\nopsd_ema_alpha=%s\n' \
    "${project_root}" "${model}" "${devices}" "${OMNI_OPSD_NPROC_PER_NODE}" "${OMNI_OPSD_MAX_STEPS}" \
    "${OMNI_OPSD_NUM_TRAIN_EPOCHS}" "${OMNI_OPSD_SAVE_STEPS}" "${OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE}" \
    "${OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS}" "${OMNI_OPSD_LORA_RANK}" "${OMNI_OPSD_LORA_ALPHA}" \
    "${OMNI_OPSD_GKD_LOGITS_TOPK}" "${OMNI_OPSD_ROLLOUT_TOP_P}" "${OMNI_OPSD_ROLLOUT_TOP_K}" \
    "${OMNI_OPSD_OPSD_EMA_ALPHA}"
  printf 'python=%s\nswift=%s\nffmpeg_bin=%s\nuse_audio_in_video=%s\nuse_vllm=%s\nvllm_mode=%s\nvllm_tensor_parallel_size=%s\nvllm_gpu_memory_utilization=%s\nvllm_sleep_level=%s\nstarted_at=%s\n' \
    "${OMNI_OPSD_PYTHON_BIN}" "${OMNI_OPSD_SWIFT_BIN}" "${ffmpeg_bin:-PATH lookup}" \
    "${USE_AUDIO_IN_VIDEO}" "${OMNI_OPSD_USE_VLLM}" "${OMNI_OPSD_VLLM_MODE}" \
    "${OMNI_OPSD_VLLM_TENSOR_PARALLEL_SIZE}" "${OMNI_OPSD_VLLM_GPU_MEMORY_UTILIZATION}" \
    "${OMNI_OPSD_VLLM_SLEEP_LEVEL}" "$(date --iso-8601=seconds)"
} > "${root}/environment/launch_config.txt"
"${OMNI_OPSD_PYTHON_BIN}" -m pip freeze > "${root}/environment/pip-freeze.txt"
git -C "${swift_root}" rev-parse HEAD > "${root}/environment/ms-swift-commit.txt"
git -C "${swift_root}" diff > "${root}/environment/ms-swift-local.patch"
git -C "${project_root}" rev-parse HEAD > "${root}/environment/project-commit.txt" 2>/dev/null || true
cp "${project_root}/data/gap5000/selection_audit.json" "${root}/environment/"
cp "${project_root}/data/gap5000/dynamic_budget_v1/dynamic_budget_summary.json" "${root}/environment/"

run_arm() {
  local arm="$1"
  local dataset="${project_root}/data/gap5000/dynamic_budget_v1/${arm}/formal/data/"
  if [[ "${arm}" == "sft" ]]; then
    dataset+="sft.jsonl"
  else
    dataset+="reasoning.jsonl"
  fi
  local output="${root}/${arm}"
  local log="${root}/logs/${arm}.log"
  local port="$2"
  mkdir -p "${output}"
  export OMNI_OPSD_ARM="${arm}"
  export OMNI_OPSD_DATASET="${dataset}"
  export OMNI_OPSD_OUTPUT_DIR="${output}"
  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT="${port}"
  export OMNI_OPSD_EXPERIMENT_LABEL="gap5000_formal_${arm}"
  if [[ "${arm}" == "sft" ]]; then
    export OMNI_OPSD_LEARNING_RATE="${OMNI_OPSD_SFT_LEARNING_RATE:-1e-5}"
    export OMNI_OPSD_OPSD_EMA_ALPHA="0.0"
  else
    export OMNI_OPSD_LEARNING_RATE="${OMNI_OPSD_OPSD_LEARNING_RATE:-2e-6}"
    export OMNI_OPSD_OPSD_EMA_ALPHA="${OMNI_OPSD_OPSD_EMA_ALPHA_OVERRIDE:-${formal_opsd_ema_alpha}}"
  fi
  echo "START ${arm} $(date --iso-8601=seconds)" | tee "${root}/${arm}.status"
  set +e
  bash "${project_root}/scripts/run_training_arm_a100.sh" "${arm}" > "${log}" 2>&1
  local rc=$?
  set -e
  printf 'END %s %s rc=%s\n' "${arm}" "$(date --iso-8601=seconds)" "${rc}" | tee -a "${root}/${arm}.status"
  printf '%s\n' "${rc}" > "${output}/exit_code.txt"
  if [[ "${rc}" -eq 0 ]]; then
    touch "${output}/SUCCESS"
  else
    touch "${output}/FAILED"
  fi
  return "${rc}"
}

cd "${project_root}"
if run_arm sft 29901; then
  run_arm opsd 29902
else
  echo "SFT failed; OPSD was not started" | tee "${root}/opsd.status"
  exit 1
fi
