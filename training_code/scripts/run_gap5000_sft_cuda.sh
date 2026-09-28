#!/usr/bin/env bash
set -euo pipefail

# Standalone dynamic-budget SFT runner.  The historical 3B script is kept
# unchanged; this entry point is the current 7B/5,000-row experiment described
# in sections 6 and 7 of the experiment summary.

project_root="${OMNI_OPSD_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_env="${OMNI_OPSD_ENV:-/share/home/ylhu/.conda/envs/vllm}"
swift_root="${OMNI_OPSD_MS_SWIFT_ROOT:-${project_root}/ms-swift}"
model="${OMNI_OPSD_MODEL:-/share/home/ylhu/models/Qwen2.5-Omni-7B}"
devices="${OMNI_OPSD_CUDA_DEVICES:-0,1,2,3}"
mode="${1:-formal}"
tag="${OMNI_OPSD_SFT_TAG:-$(date +%Y%m%d_%H%M%S)}"
root="${OMNI_OPSD_SFT_ROOT:-${project_root}/output/gap5000_sft_cuda_${tag}}"
ffmpeg_bin="${OMNI_OPSD_FFMPEG_BIN:-}"
if [[ -z "${ffmpeg_bin}" && -x "/share/home/ylhu/.conda/envs/omniagent_gyh/bin/ffmpeg" ]]; then
  ffmpeg_bin="/share/home/ylhu/.conda/envs/omniagent_gyh/bin/ffmpeg"
fi
if [[ -n "${ffmpeg_bin}" ]]; then
  [[ -x "${ffmpeg_bin}" ]] || { echo "ffmpeg is not executable: ${ffmpeg_bin}" >&2; exit 2; }
  export PATH="$(dirname "${ffmpeg_bin}"):${PATH}"
fi

case "${mode}" in
  gate)
    dataset_default="${project_root}/data/gap5000/dynamic_budget_v1/sft/gate/data/sft.jsonl"
    max_steps_default=1
    save_steps_default=1
    ;;
  formal)
    dataset_default="${project_root}/data/gap5000/dynamic_budget_v1/sft/formal/data/sft.jsonl"
    max_steps_default=157
    save_steps_default=25
    ;;
  *)
    echo "usage: $0 gate|formal" >&2
    exit 2
    ;;
esac

IFS=',' read -r -a cuda_devices <<< "${devices}"
if [[ "${#cuda_devices[@]}" -ne 4 ]]; then
  echo "The dynamic SFT arm requires four CUDA devices; got ${devices}" >&2
  exit 2
fi

mkdir -p "${root}"
export CUDA_VISIBLE_DEVICES="${devices}"
export ASCEND_RT_VISIBLE_DEVICES="${devices}"
export OMNI_OPSD_ARM=sft
export OMNI_OPSD_NPROC_PER_NODE="${OMNI_OPSD_NPROC_PER_NODE:-4}"
if [[ "${OMNI_OPSD_NPROC_PER_NODE}" -ne 4 ]]; then
  echo "OMNI_OPSD_NPROC_PER_NODE must be 4 for dynamic SFT" >&2
  exit 2
fi
export OMNI_OPSD_PROJECT_ROOT="${project_root}"
export OMNI_OPSD_MS_SWIFT_ROOT="${swift_root}"
export OMNI_OPSD_PYTHON_DEPS="${project_root}"
export OMNI_OPSD_PYTHON_BIN="${OMNI_OPSD_PYTHON_BIN:-${python_env}/bin/python}"
export OMNI_OPSD_SWIFT_BIN="${OMNI_OPSD_SWIFT_BIN:-${python_env}/bin/swift}"
export OMNI_OPSD_MODEL="${model}"
export OMNI_OPSD_DATASET="${OMNI_OPSD_DATASET:-${dataset_default}}"
export OMNI_OPSD_OUTPUT_DIR="${root}/sft"
export OMNI_OPSD_MAX_STEPS="${OMNI_OPSD_MAX_STEPS:-${max_steps_default}}"
export OMNI_OPSD_NUM_TRAIN_EPOCHS="${OMNI_OPSD_NUM_TRAIN_EPOCHS:-1}"
export OMNI_OPSD_SAVE_STEPS="${OMNI_OPSD_SAVE_STEPS:-${save_steps_default}}"
export OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE="${OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS="${OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS:-4}"
export OMNI_OPSD_LORA_RANK="${OMNI_OPSD_LORA_RANK:-64}"
export OMNI_OPSD_LORA_ALPHA="${OMNI_OPSD_LORA_ALPHA:-128}"
export OMNI_OPSD_LEARNING_RATE="${OMNI_OPSD_LEARNING_RATE:-1e-5}"
export OMNI_OPSD_MAX_LENGTH="${OMNI_OPSD_MAX_LENGTH:-32768}"
export OMNI_OPSD_MIN_PIXELS="${OMNI_OPSD_MIN_PIXELS:-3136}"
export OMNI_OPSD_MAX_GRAD_NORM="${OMNI_OPSD_MAX_GRAD_NORM:-0}"
export OMNI_OPSD_ATTN_IMPL="${OMNI_OPSD_ATTN_IMPL:-sdpa}"
export OMNI_OPSD_OPSD_EMA_ALPHA=0.0
export USE_AUDIO_IN_VIDEO="${USE_AUDIO_IN_VIDEO:-1}"
export MAX_NUM_WORKERS_FETCH_VIDEO="${MAX_NUM_WORKERS_FETCH_VIDEO:-1}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PATH="${python_env}/bin:${PATH}"
export PYTHONPATH="${project_root}/src:${project_root}:${swift_root}${PYTHONPATH:+:${PYTHONPATH}}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29911}"
export OMNI_OPSD_EXPERIMENT_LABEL="${OMNI_OPSD_EXPERIMENT_LABEL:-gap5000_dynamic_sft_${mode}}"

mkdir -p "${root}/sft"
cat > "${root}/launch_config.txt" <<EOF
mode=${mode}
model=${model}
dataset=${OMNI_OPSD_DATASET}
devices=${devices}
nproc=${OMNI_OPSD_NPROC_PER_NODE}
max_steps=${OMNI_OPSD_MAX_STEPS}
num_train_epochs=${OMNI_OPSD_NUM_TRAIN_EPOCHS}
per_device_train_batch_size=${OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE}
gradient_accumulation_steps=${OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS}
global_batch_size=$((OMNI_OPSD_NPROC_PER_NODE * OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE * OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS))
lora_rank=${OMNI_OPSD_LORA_RANK}
lora_alpha=${OMNI_OPSD_LORA_ALPHA}
learning_rate=${OMNI_OPSD_LEARNING_RATE}
max_grad_norm=${OMNI_OPSD_MAX_GRAD_NORM}
attn_impl=${OMNI_OPSD_ATTN_IMPL}
use_audio_in_video=${USE_AUDIO_IN_VIDEO}
started_at=$(date --iso-8601=seconds)
EOF

exec bash "${project_root}/scripts/run_training_arm_a100.sh" sft \
  > "${root}/sft.log" 2>&1
