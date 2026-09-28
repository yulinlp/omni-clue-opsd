#!/usr/bin/env bash
set -euo pipefail

# Qwen2.5-Omni-3B standard OPSD run with full-parameter student training.
# This separate entry point leaves the 7B LoRA experiment defaults unchanged.

project_root="${OMNI_OPSD_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_env="${OMNI_OPSD_ENV:-/share/home/ylhu/.conda/envs/vllm}"
swift_root="${OMNI_OPSD_MS_SWIFT_ROOT:-${project_root}/ms-swift}"
model="${OMNI_OPSD_MODEL:-/share/home/ylhu/models/Qwen2.5-Omni-3B}"
devices="${OMNI_OPSD_CUDA_DEVICES:-0,1,2,3}"
dataset="${OMNI_OPSD_DATASET:-${project_root}/data/gap5000/dynamic_budget_v1/opsd/formal/data/reasoning.jsonl}"
tag="${OMNI_OPSD_OPSD_TAG:-$(date +%Y%m%d_%H%M%S)}"
root="${OMNI_OPSD_OPSD_ROOT:-${project_root}/output/gap5000_opsd_3b_full_cuda_${tag}}"
ffmpeg_bin="${OMNI_OPSD_FFMPEG_BIN:-}"
if [[ -z "${ffmpeg_bin}" && -x "/share/home/ylhu/.conda/envs/omniagent_gyh/bin/ffmpeg" ]]; then
  ffmpeg_bin="/share/home/ylhu/.conda/envs/omniagent_gyh/bin/ffmpeg"
fi
if [[ -n "${ffmpeg_bin}" ]]; then
  [[ -x "${ffmpeg_bin}" ]] || { echo "ffmpeg is not executable: ${ffmpeg_bin}" >&2; exit 2; }
  export PATH="$(dirname "${ffmpeg_bin}"):${PATH}"
fi

IFS=',' read -r -a cuda_devices <<< "${devices}"
if [[ "${#cuda_devices[@]}" -ne 4 ]]; then
  echo "Qwen2.5-Omni-3B full OPSD requires four CUDA devices; got ${devices}" >&2
  exit 2
fi

mkdir -p "${root}"
export CUDA_VISIBLE_DEVICES="${devices}"
export ASCEND_RT_VISIBLE_DEVICES="${devices}"
export OMNI_OPSD_ARM=opsd
export OMNI_OPSD_NPROC_PER_NODE=4
export OMNI_OPSD_PROJECT_ROOT="${project_root}"
export OMNI_OPSD_MS_SWIFT_ROOT="${swift_root}"
export OMNI_OPSD_PYTHON_DEPS="${project_root}"
export OMNI_OPSD_PYTHON_BIN="${OMNI_OPSD_PYTHON_BIN:-${python_env}/bin/python}"
export OMNI_OPSD_SWIFT_BIN="${OMNI_OPSD_SWIFT_BIN:-${python_env}/bin/swift}"
export OMNI_OPSD_MODEL="${model}"
export OMNI_OPSD_DATASET="${dataset}"
export OMNI_OPSD_OUTPUT_DIR="${root}/opsd"
export OMNI_OPSD_MAX_STEPS="${OMNI_OPSD_MAX_STEPS:-157}"
export OMNI_OPSD_NUM_TRAIN_EPOCHS="${OMNI_OPSD_NUM_TRAIN_EPOCHS:-1}"
export OMNI_OPSD_SAVE_STEPS="${OMNI_OPSD_SAVE_STEPS:-25}"
export OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE="${OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS="${OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS:-8}"
export OMNI_OPSD_TUNER_TYPE=full
export OMNI_OPSD_FREEZE_LLM=false
export OMNI_OPSD_FREEZE_VIT=false
export OMNI_OPSD_FREEZE_ALIGNER=false
# The local EMA implementation is LoRA-shadow based; full-parameter training
# uses the current-policy teacher rather than a misleading LoRA shadow.
export OMNI_OPSD_OPSD_EMA_ALPHA=0.0
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
export OMNI_OPSD_VLLM_GPU_MEMORY_UTILIZATION="${OMNI_OPSD_VLLM_GPU_MEMORY_UTILIZATION:-0.10}"
export OMNI_OPSD_VLLM_SLEEP_LEVEL="${OMNI_OPSD_VLLM_SLEEP_LEVEL:-2}"
# Keep the optimizer's effective batch at 32 while reducing the peak memory
# of each rollout/encoding cycle. This does not alter the train batch.
export OMNI_OPSD_STEPS_PER_GENERATION="${OMNI_OPSD_STEPS_PER_GENERATION:-1}"
export OMNI_OPSD_ALLOC_CONF="${OMNI_OPSD_ALLOC_CONF:-expandable_segments:True}"
export OMNI_OPSD_DDP_FIND_UNUSED_PARAMETERS="${OMNI_OPSD_DDP_FIND_UNUSED_PARAMETERS:-true}"
export OMNI_OPSD_USE_LOGITS_TO_KEEP="${OMNI_OPSD_USE_LOGITS_TO_KEEP:-1}"
# vLLM 0.11.x cannot match Qwen2.5-Omni's overlapping audio/video prompt
# updates.  Rollouts therefore use the visual stream; the Transformers
# training forward still keeps USE_AUDIO_IN_VIDEO=1.
export OMNI_OPSD_VLLM_DROP_AUDIO=1
export OMNI_OPSD_ATTN_IMPL="${OMNI_OPSD_ATTN_IMPL:-sdpa}"
export OMNI_OPSD_GKD_SAFE_MODE="${OMNI_OPSD_GKD_SAFE_MODE:-0}"
export OMNI_OPSD_GKD_MAX_GRAD_NORM="${OMNI_OPSD_GKD_MAX_GRAD_NORM:-0}"
export OMNI_OPSD_MAX_GRAD_NORM="${OMNI_OPSD_MAX_GRAD_NORM:-0}"
export USE_AUDIO_IN_VIDEO="${USE_AUDIO_IN_VIDEO:-1}"
export MAX_NUM_WORKERS_FETCH_VIDEO="${MAX_NUM_WORKERS_FETCH_VIDEO:-1}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
# Keep local MP4 paths as filesystem paths for decord.  Converting long,
# high-FPS videos to BytesIO makes random seeks near EOF fail under concurrent
# DDP decoding.  The larger retry budget is an additional guard for containers
# whose final packets require more EOF retries.
export OMNI_OPSD_DIRECT_LOCAL_VIDEO="${OMNI_OPSD_DIRECT_LOCAL_VIDEO:-1}"
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"
export DECORD_NUM_THREADS="${DECORD_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PATH="${python_env}/bin:${PATH}"
export PYTHONPATH="${project_root}/src:${project_root}:${swift_root}${PYTHONPATH:+:${PYTHONPATH}}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29941}"
export OMNI_OPSD_EXPERIMENT_LABEL="${OMNI_OPSD_EXPERIMENT_LABEL:-gap5000_opsd_3b_full_parameter}"

cat > "${root}/launch_config.txt" <<EOF
model=${model}
dataset=${dataset}
devices=${devices}
nproc=4
max_steps=${OMNI_OPSD_MAX_STEPS}
num_train_epochs=${OMNI_OPSD_NUM_TRAIN_EPOCHS}
per_device_train_batch_size=${OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE}
gradient_accumulation_steps=${OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS}
global_batch_size=$((4 * OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE * OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS))
tuner_type=${OMNI_OPSD_TUNER_TYPE}
freeze_llm=${OMNI_OPSD_FREEZE_LLM}
freeze_vit=${OMNI_OPSD_FREEZE_VIT}
freeze_aligner=${OMNI_OPSD_FREEZE_ALIGNER}
learning_rate=${OMNI_OPSD_LEARNING_RATE}
opsd_ema_alpha=${OMNI_OPSD_OPSD_EMA_ALPHA}
gkd_logits_topk=${OMNI_OPSD_GKD_LOGITS_TOPK}
rollout_top_p=${OMNI_OPSD_ROLLOUT_TOP_P}
rollout_top_k=${OMNI_OPSD_ROLLOUT_TOP_K}
use_vllm=${OMNI_OPSD_USE_VLLM}
vllm_mode=${OMNI_OPSD_VLLM_MODE}
vllm_tensor_parallel_size=${OMNI_OPSD_VLLM_TENSOR_PARALLEL_SIZE}
vllm_drop_audio=${OMNI_OPSD_VLLM_DROP_AUDIO}
vllm_gpu_memory_utilization=${OMNI_OPSD_VLLM_GPU_MEMORY_UTILIZATION}
vllm_sleep_level=${OMNI_OPSD_VLLM_SLEEP_LEVEL}
steps_per_generation=${OMNI_OPSD_STEPS_PER_GENERATION}
allocator_conf=${OMNI_OPSD_ALLOC_CONF}
ddp_find_unused_parameters=${OMNI_OPSD_DDP_FIND_UNUSED_PARAMETERS}
use_logits_to_keep=${OMNI_OPSD_USE_LOGITS_TO_KEEP}
use_audio_in_video=${USE_AUDIO_IN_VIDEO}
direct_local_video=${OMNI_OPSD_DIRECT_LOCAL_VIDEO}
decord_eof_retry_max=${DECORD_EOF_RETRY_MAX}
decord_num_threads=${DECORD_NUM_THREADS}
ffmpeg_bin=${ffmpeg_bin:-PATH lookup}
started_at=$(date --iso-8601=seconds)
EOF

exec bash "${project_root}/scripts/run_training_arm_a100.sh" opsd \
  > "${root}/opsd.log" 2>&1
