#!/usr/bin/env bash
set -euo pipefail

# Standalone CUDA OPSD runner for the dynamic-budget formal arm.  The default
# video backend is decord because torchvision/PyAV can exhaust the small CPU
# allocation when four ranks decode full videos concurrently.  GKD uses the
# section-6/7 top-20 JSD setup and a colocated vLLM rollout.

project_root="${OMNI_OPSD_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_env="${OMNI_OPSD_ENV:-/share/home/ylhu/.conda/envs/vllm}"
swift_root="${OMNI_OPSD_MS_SWIFT_ROOT:-${project_root}/ms-swift}"
model="${OMNI_OPSD_MODEL:-/share/home/ylhu/models/Qwen2.5-Omni-7B}"
devices="${OMNI_OPSD_CUDA_DEVICES:-0,1,2,3}"
mode="${1:-formal}"

case "${mode}" in
  smoke)
    tag="${OMNI_OPSD_OPSD_TAG:-$(date +%Y%m%d_%H%M%S)}"
    max_steps="${OMNI_OPSD_MAX_STEPS:-1}"
    gradient_accumulation="${OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS:-4}"
    save_steps="${OMNI_OPSD_SAVE_STEPS:-1}"
    label="gap5000_opsd_cuda_smoke"
    ;;
  formal)
    tag="${OMNI_OPSD_OPSD_TAG:-$(date +%Y%m%d_%H%M%S)}"
    max_steps="${OMNI_OPSD_MAX_STEPS:-157}"
    gradient_accumulation="${OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS:-4}"
    save_steps="${OMNI_OPSD_SAVE_STEPS:-25}"
    label="gap5000_opsd_cuda_formal"
    ;;
  *)
    echo "usage: $0 smoke|formal" >&2
    exit 2
    ;;
esac

root="${OMNI_OPSD_OPSD_ROOT:-${project_root}/output/${label}_${tag}}"
if [[ "${mode}" == "smoke" ]]; then
  default_dataset="${project_root}/data/gap5000/dynamic_budget_v1/opsd/gate/data/reasoning.jsonl"
else
  default_dataset="${project_root}/data/gap5000/dynamic_budget_v1/opsd/formal/data/reasoning.jsonl"
fi
dataset="${OMNI_OPSD_DATASET:-${default_dataset}}"
allocator_conf="${OMNI_OPSD_ALLOC_CONF:-expandable_segments:True}"
# Qwen-Omni's bounded audio loader uses audioread, which requires an ffmpeg
# compatible executable even when video frames are decoded by decord.  The
# cluster's vLLM environment intentionally does not bundle ffmpeg; use an
# existing shared static build when present, while allowing callers to provide
# another binary through OMNI_OPSD_FFMPEG_BIN.
ffmpeg_bin="${OMNI_OPSD_FFMPEG_BIN:-}"
if [[ -z "${ffmpeg_bin}" && -x "/share/home/ylhu/.conda/envs/omniagent_gyh/bin/ffmpeg" ]]; then
  ffmpeg_bin="/share/home/ylhu/.conda/envs/omniagent_gyh/bin/ffmpeg"
fi
if [[ -n "${ffmpeg_bin}" ]]; then
  if [[ ! -x "${ffmpeg_bin}" ]]; then
    echo "OMNI_OPSD_FFMPEG_BIN is not executable: ${ffmpeg_bin}" >&2
    exit 2
  fi
  export PATH="$(dirname "${ffmpeg_bin}"):${PATH}"
fi
mkdir -p "${root}"

IFS=',' read -r -a cuda_devices <<< "${devices}"
nproc="${OMNI_OPSD_NPROC_PER_NODE:-${#cuda_devices[@]}}"
if [[ "${nproc}" != "${#cuda_devices[@]}" ]]; then
  echo "OMNI_OPSD_NPROC_PER_NODE=${nproc} must match OMNI_OPSD_CUDA_DEVICES=${devices}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${devices}"
export ASCEND_RT_VISIBLE_DEVICES="${devices}"
export OMNI_OPSD_ARM=opsd
export OMNI_OPSD_NPROC_PER_NODE="${nproc}"
export OMNI_OPSD_PROJECT_ROOT="${project_root}"
export OMNI_OPSD_MS_SWIFT_ROOT="${swift_root}"
export OMNI_OPSD_PYTHON_DEPS="${project_root}"
export OMNI_OPSD_PYTHON_BIN="${OMNI_OPSD_PYTHON_BIN:-${python_env}/bin/python}"
export OMNI_OPSD_SWIFT_BIN="${OMNI_OPSD_SWIFT_BIN:-${python_env}/bin/swift}"
export OMNI_OPSD_MODEL="${model}"
export OMNI_OPSD_DATASET="${dataset}"
export OMNI_OPSD_ALLOC_CONF="${allocator_conf}"
export OMNI_OPSD_OUTPUT_DIR="${root}/opsd"
export OMNI_OPSD_MAX_STEPS="${max_steps}"
export OMNI_OPSD_NUM_TRAIN_EPOCHS="${OMNI_OPSD_NUM_TRAIN_EPOCHS:-1}"
export OMNI_OPSD_SAVE_STEPS="${save_steps}"
export OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE="${OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS="${gradient_accumulation}"
export OMNI_OPSD_LORA_RANK="${OMNI_OPSD_LORA_RANK:-64}"
export OMNI_OPSD_LORA_ALPHA="${OMNI_OPSD_LORA_ALPHA:-128}"
export OMNI_OPSD_LEARNING_RATE="${OMNI_OPSD_LEARNING_RATE:-2e-6}"
export OMNI_OPSD_MAX_LENGTH="${OMNI_OPSD_MAX_LENGTH:-32768}"
export OMNI_OPSD_MIN_PIXELS="${OMNI_OPSD_MIN_PIXELS:-3136}"
export OMNI_OPSD_MAX_COMPLETION_LENGTH="${OMNI_OPSD_MAX_COMPLETION_LENGTH:-512}"
export OMNI_OPSD_GKD_LOGITS_TOPK="${OMNI_OPSD_GKD_LOGITS_TOPK:-20}"
export OMNI_OPSD_DIAG_ENABLED="${OMNI_OPSD_DIAG_ENABLED:-true}"
export OMNI_OPSD_DIAG_TOP_K="${OMNI_OPSD_DIAG_TOP_K:-20}"
export OMNI_OPSD_DIAG_TEMPERATURE="${OMNI_OPSD_DIAG_TEMPERATURE:-1.0}"
export OMNI_OPSD_DIAG_FREQUENCY="${OMNI_OPSD_DIAG_FREQUENCY:-1}"
export OMNI_OPSD_DIAG_CHUNK_SIZE="${OMNI_OPSD_DIAG_CHUNK_SIZE:-256}"
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
export USE_AUDIO_IN_VIDEO="${USE_AUDIO_IN_VIDEO:-1}"
export MAX_NUM_WORKERS_FETCH_VIDEO="${MAX_NUM_WORKERS_FETCH_VIDEO:-1}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${allocator_conf}"
export PATH="${python_env}/bin:${PATH}"
export PYTHONPATH="${project_root}/src:${project_root}:${swift_root}${PYTHONPATH:+:${PYTHONPATH}}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29931}"
export OMNI_OPSD_EXPERIMENT_LABEL="${OMNI_OPSD_EXPERIMENT_LABEL:-${label}}"

{
  printf 'mode=%s\nroot=%s\ndataset=%s\ndevices=%s\nnproc=%s\nmax_steps=%s\nnum_train_epochs=%s\nper_device_train_batch_size=%s\ngradient_accumulation_steps=%s\nglobal_batch_size=%s\n' \
    "${mode}" "${root}" "${dataset}" "${devices}" "${nproc}" "${max_steps}" \
    "${OMNI_OPSD_NUM_TRAIN_EPOCHS}" "${OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE}" \
    "${gradient_accumulation}" "$((nproc * OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE * gradient_accumulation))"
  printf 'video_reader=%s\nuse_audio_in_video=%s\ngkd_logits_topk=%s\ndiag_enabled=%s\ndiag_top_k=%s\ndiag_temperature=%s\ndiag_frequency=%s\ndiag_chunk_size=%s\nrollout_top_p=%s\nrollout_top_k=%s\nlora_rank=%s\nlora_alpha=%s\nlearning_rate=%s\nopsd_ema_alpha=%s\nuse_vllm=%s\nvllm_mode=%s\nvllm_tensor_parallel_size=%s\nvllm_gpu_memory_utilization=%s\nvllm_sleep_level=%s\nallocator_conf=%s\nffmpeg_bin=%s\nstarted_at=%s\n' \
    "${FORCE_QWENVL_VIDEO_READER}" "${USE_AUDIO_IN_VIDEO}" "${OMNI_OPSD_GKD_LOGITS_TOPK}" \
    "${OMNI_OPSD_DIAG_ENABLED}" "${OMNI_OPSD_DIAG_TOP_K}" "${OMNI_OPSD_DIAG_TEMPERATURE}" \
    "${OMNI_OPSD_DIAG_FREQUENCY}" "${OMNI_OPSD_DIAG_CHUNK_SIZE}" \
    "${OMNI_OPSD_ROLLOUT_TOP_P}" "${OMNI_OPSD_ROLLOUT_TOP_K}" "${OMNI_OPSD_LORA_RANK}" \
    "${OMNI_OPSD_LORA_ALPHA}" "${OMNI_OPSD_LEARNING_RATE}" "${OMNI_OPSD_OPSD_EMA_ALPHA}" \
    "${OMNI_OPSD_USE_VLLM}" "${OMNI_OPSD_VLLM_MODE}" "${OMNI_OPSD_VLLM_TENSOR_PARALLEL_SIZE}" \
    "${OMNI_OPSD_VLLM_GPU_MEMORY_UTILIZATION}" \
    "${OMNI_OPSD_VLLM_SLEEP_LEVEL}" "${allocator_conf}" "${ffmpeg_bin:-PATH lookup}" "$(date --iso-8601=seconds)"
} > "${root}/launch_config.txt"

exec bash "${project_root}/scripts/run_training_arm_a100.sh" opsd \
  > "${root}/opsd.log" 2>&1
