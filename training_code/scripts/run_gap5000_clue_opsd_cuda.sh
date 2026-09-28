#!/usr/bin/env bash
set -euo pipefail

# CUDA/A100 launcher for the section-6/7 dynamic-budget CLUE-OPSD arm.
# Student media are the same full-video inputs as SFT/standard OPSD; only the
# on-policy teacher receives the dataset-provided evidence intervals.

project_root="${OMNI_OPSD_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_env="${OMNI_OPSD_ENV:-/share/home/ylhu/.conda/envs/vllm}"
swift_root="${OMNI_OPSD_MS_SWIFT_ROOT:-${project_root}/ms-swift}"
model="${OMNI_OPSD_MODEL:-/share/home/ylhu/models/Qwen2.5-Omni-7B}"
devices="${OMNI_OPSD_CUDA_DEVICES:-0,1,2,3}"
mode="${1:-formal}"

case "${mode}" in
  smoke)
    tag="${OMNI_OPSD_CLUE_OPSD_TAG:-$(date +%Y%m%d_%H%M%S)}"
    max_steps="${OMNI_OPSD_MAX_STEPS:-1}"
    save_steps="${OMNI_OPSD_SAVE_STEPS:-1}"
    label="gap5000_clue_opsd_cuda_smoke"
    ;;
  formal)
    tag="${OMNI_OPSD_CLUE_OPSD_TAG:-$(date +%Y%m%d_%H%M%S)}"
    max_steps="${OMNI_OPSD_MAX_STEPS:-157}"
    save_steps="${OMNI_OPSD_SAVE_STEPS:-25}"
    label="gap5000_clue_opsd_cuda_formal"
    ;;
  *)
    echo "usage: $0 smoke|formal" >&2
    exit 2
    ;;
esac

root="${OMNI_OPSD_CLUE_OPSD_ROOT:-${project_root}/output/${label}_${tag}}"
if [[ "${mode}" == "smoke" ]]; then
  default_dataset="${project_root}/data/gap5000/dynamic_budget_v1/clue_opsd/gate/data/reasoning.jsonl"
else
  default_dataset="${project_root}/data/gap5000/dynamic_budget_v1/clue_opsd/formal/data/reasoning.jsonl"
fi
dataset="${OMNI_OPSD_DATASET:-${default_dataset}}"
allocator_conf="${OMNI_OPSD_ALLOC_CONF:-expandable_segments:True}"
ffmpeg_bin="${OMNI_OPSD_FFMPEG_BIN:-}"
if [[ -z "${ffmpeg_bin}" && -x "/share/home/ylhu/.conda/envs/omniagent_gyh/bin/ffmpeg" ]]; then
  ffmpeg_bin="/share/home/ylhu/.conda/envs/omniagent_gyh/bin/ffmpeg"
fi
if [[ -n "${ffmpeg_bin}" ]]; then
  [[ -x "${ffmpeg_bin}" ]] || { echo "ffmpeg is not executable: ${ffmpeg_bin}" >&2; exit 2; }
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
export OMNI_OPSD_ARM=clue_opsd
export OMNI_OPSD_NPROC_PER_NODE="${nproc}"
export OMNI_OPSD_PROJECT_ROOT="${project_root}"
export OMNI_OPSD_MS_SWIFT_ROOT="${swift_root}"
export OMNI_OPSD_PYTHON_DEPS="${project_root}"
export OMNI_OPSD_PYTHON_BIN="${OMNI_OPSD_PYTHON_BIN:-${python_env}/bin/python}"
export OMNI_OPSD_SWIFT_BIN="${OMNI_OPSD_SWIFT_BIN:-${python_env}/bin/swift}"
export OMNI_OPSD_MODEL="${model}"
export OMNI_OPSD_DATASET="${dataset}"
export OMNI_OPSD_ALLOC_CONF="${allocator_conf}"
export OMNI_OPSD_OUTPUT_DIR="${root}/clue_opsd"
export OMNI_OPSD_MAX_STEPS="${max_steps}"
export OMNI_OPSD_NUM_TRAIN_EPOCHS="${OMNI_OPSD_NUM_TRAIN_EPOCHS:-1}"
export OMNI_OPSD_SAVE_STEPS="${save_steps}"
# On A100-80G, a 240-frame full-video sample plus the colocated vLLM rollout
# exceeds memory at batch=2.  Use batch=1 and accumulate 8 steps to retain the
# section-6/7 effective global batch of 32 on four ranks.
export OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE="${OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS="${OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS:-8}"
export OMNI_OPSD_LORA_RANK="${OMNI_OPSD_LORA_RANK:-64}"
export OMNI_OPSD_LORA_ALPHA="${OMNI_OPSD_LORA_ALPHA:-128}"
export OMNI_OPSD_LEARNING_RATE="${OMNI_OPSD_LEARNING_RATE:-2e-6}"
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
export OMNI_OPSD_STEPS_PER_GENERATION="${OMNI_OPSD_STEPS_PER_GENERATION:-1}"
export OMNI_OPSD_DDP_FIND_UNUSED_PARAMETERS="${OMNI_OPSD_DDP_FIND_UNUSED_PARAMETERS:-false}"
export OMNI_OPSD_ATTN_IMPL="${OMNI_OPSD_ATTN_IMPL:-sdpa}"
export OMNI_OPSD_GKD_SAFE_MODE="${OMNI_OPSD_GKD_SAFE_MODE:-0}"
export OMNI_OPSD_GKD_MAX_GRAD_NORM="${OMNI_OPSD_GKD_MAX_GRAD_NORM:-0}"
export OMNI_OPSD_MAX_GRAD_NORM="${OMNI_OPSD_MAX_GRAD_NORM:-0}"
export OMNI_OPSD_OPSD_EMA_ALPHA=0.0
export OMNI_OPSD_CLUE_EMA_ALPHA="${OMNI_OPSD_CLUE_EMA_ALPHA:-0.05}"
export OMNI_OPSD_FULL_EMA_TEACHER="${OMNI_OPSD_FULL_EMA_TEACHER:-false}"
export OMNI_OPSD_FULL_EMA_ALPHA="${OMNI_OPSD_FULL_EMA_ALPHA:-0.05}"
export OMNI_OPSD_FULL_EMA_OFFLOAD="${OMNI_OPSD_FULL_EMA_OFFLOAD:-false}"
export OMNI_OPSD_GOLD_CE_ALPHA="${OMNI_OPSD_GOLD_CE_ALPHA:-0.25}"
export OMNI_OPSD_SPLIT_DATASET_RATIO="${OMNI_OPSD_SPLIT_DATASET_RATIO:-0}"
export OMNI_OPSD_EVAL_STRATEGY="${OMNI_OPSD_EVAL_STRATEGY:-no}"
export OMNI_OPSD_EVAL_STEPS="${OMNI_OPSD_EVAL_STEPS:-${OMNI_OPSD_SAVE_STEPS}}"
export OMNI_OPSD_LOG_COMPLETIONS="${OMNI_OPSD_LOG_COMPLETIONS:-false}"
export OMNI_OPSD_DIAG_ENABLED="${OMNI_OPSD_DIAG_ENABLED:-true}"
export OMNI_OPSD_DIAG_TOP_K="${OMNI_OPSD_DIAG_TOP_K:-20}"
export OMNI_OPSD_DIAG_TEMPERATURE="${OMNI_OPSD_DIAG_TEMPERATURE:-1.0}"
export OMNI_OPSD_DIAG_FREQUENCY="${OMNI_OPSD_DIAG_FREQUENCY:-1}"
export OMNI_OPSD_DIAG_CHUNK_SIZE="${OMNI_OPSD_DIAG_CHUNK_SIZE:-256}"
# vLLM 0.11.x has an engine-side overlapping audio/video placeholder issue.
# Keep audio enabled for the actual Transformers training/teacher forward; the
# switch is explicit so a smoke failure can be retried as a visual-only rollout
# without silently changing the recorded experiment contract.
export OMNI_OPSD_VLLM_DROP_AUDIO="${OMNI_OPSD_VLLM_DROP_AUDIO:-0}"
export USE_AUDIO_IN_VIDEO="${USE_AUDIO_IN_VIDEO:-1}"
export MAX_NUM_WORKERS_FETCH_VIDEO="${MAX_NUM_WORKERS_FETCH_VIDEO:-1}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
# Keep local MP4 paths on disk instead of converting them to BytesIO.  The
# previous run hit rank-specific EOF failures while seeking long BytesIO
# videos; a larger retry budget and one decord thread make the restart robust.
export OMNI_OPSD_DIRECT_LOCAL_VIDEO="${OMNI_OPSD_DIRECT_LOCAL_VIDEO:-1}"
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"
export DECORD_NUM_THREADS="${DECORD_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${allocator_conf}"
export PATH="${python_env}/bin:${PATH}"
export PYTHONPATH="${project_root}/src:${project_root}:${swift_root}${PYTHONPATH:+:${PYTHONPATH}}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29941}"
export OMNI_OPSD_EXPERIMENT_LABEL="${OMNI_OPSD_EXPERIMENT_LABEL:-${label}}"

global_batch_size=$((nproc * OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE * OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS))
if [[ "${global_batch_size}" -ne 32 && "${OMNI_OPSD_ALLOW_NON_32_GLOBAL_BATCH:-0}" != "1" ]]; then
  echo "effective batch must be 32: ${nproc}*${OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE}*${OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS}=${global_batch_size}" >&2
  exit 2
fi
{
  printf 'mode=%s\nroot=%s\ndataset=%s\ndevices=%s\nnproc=%s\nmax_steps=%s\nnum_train_epochs=%s\nper_device_train_batch_size=%s\ngradient_accumulation_steps=%s\nglobal_batch_size=%s\n' \
    "${mode}" "${root}" "${dataset}" "${devices}" "${nproc}" "${max_steps}" \
    "${OMNI_OPSD_NUM_TRAIN_EPOCHS}" "${OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE}" \
    "${OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS}" "${global_batch_size}"
  printf 'arm=clue_opsd\nmodel=%s\nvideo_reader=%s\nuse_audio_in_video=%s\ngkd_logits_topk=%s\ndiag_enabled=%s\ndiag_top_k=%s\ndiag_temperature=%s\ndiag_frequency=%s\ndiag_chunk_size=%s\nrollout_top_p=%s\nrollout_top_k=%s\nlora_rank=%s\nlora_alpha=%s\nlearning_rate=%s\nclue_ema_alpha=%s\nfull_ema_teacher=%s\nfull_ema_alpha=%s\nfull_ema_offload=%s\ngold_ce_alpha=%s\nsplit_dataset_ratio=%s\neval_strategy=%s\neval_steps=%s\nlog_completions=%s\nuse_vllm=%s\nvllm_mode=%s\nvllm_tensor_parallel_size=%s\nvllm_gpu_memory_utilization=%s\nvllm_sleep_level=%s\nsteps_per_generation=%s\nddp_find_unused_parameters=%s\nvllm_drop_audio=%s\nallocator_conf=%s\nffmpeg_bin=%s\nstarted_at=%s\n' \
    "${model}" "${FORCE_QWENVL_VIDEO_READER}" "${USE_AUDIO_IN_VIDEO}" \
    "${OMNI_OPSD_GKD_LOGITS_TOPK}" "${OMNI_OPSD_DIAG_ENABLED}" "${OMNI_OPSD_DIAG_TOP_K}" \
    "${OMNI_OPSD_DIAG_TEMPERATURE}" "${OMNI_OPSD_DIAG_FREQUENCY}" "${OMNI_OPSD_DIAG_CHUNK_SIZE}" \
    "${OMNI_OPSD_ROLLOUT_TOP_P}" "${OMNI_OPSD_ROLLOUT_TOP_K}" \
    "${OMNI_OPSD_LORA_RANK}" "${OMNI_OPSD_LORA_ALPHA}" "${OMNI_OPSD_LEARNING_RATE}" \
    "${OMNI_OPSD_CLUE_EMA_ALPHA}" "${OMNI_OPSD_FULL_EMA_TEACHER}" "${OMNI_OPSD_FULL_EMA_ALPHA}" \
    "${OMNI_OPSD_FULL_EMA_OFFLOAD}" "${OMNI_OPSD_GOLD_CE_ALPHA}" "${OMNI_OPSD_SPLIT_DATASET_RATIO}" \
    "${OMNI_OPSD_EVAL_STRATEGY}" "${OMNI_OPSD_EVAL_STEPS}" "${OMNI_OPSD_LOG_COMPLETIONS}" \
    "${OMNI_OPSD_USE_VLLM}" "${OMNI_OPSD_VLLM_MODE}" \
    "${OMNI_OPSD_VLLM_TENSOR_PARALLEL_SIZE}" "${OMNI_OPSD_VLLM_GPU_MEMORY_UTILIZATION}" \
    "${OMNI_OPSD_VLLM_SLEEP_LEVEL}" "${OMNI_OPSD_STEPS_PER_GENERATION}" \
    "${OMNI_OPSD_DDP_FIND_UNUSED_PARAMETERS}" "${OMNI_OPSD_VLLM_DROP_AUDIO}" \
    "${allocator_conf}" "${ffmpeg_bin:-PATH lookup}" "$(date --iso-8601=seconds)"
} > "${root}/launch_config.txt"

exec bash "${project_root}/scripts/run_training_arm_a100.sh" clue_opsd \
  > "${root}/clue_opsd.log" 2>&1
