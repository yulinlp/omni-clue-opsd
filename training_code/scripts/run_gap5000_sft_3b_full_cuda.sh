#!/usr/bin/env bash
set -euo pipefail

# Qwen2.5-Omni-3B reasoning SFT with full-parameter student tuning.
#
# This wrapper fixes the historical 3B SFT mismatch by selecting the current
# dynamic-budget reasoning JSONL, exporting USE_AUDIO_IN_VIDEO=1 explicitly,
# and disabling all multimodal freezes.  A100-80G full tuning uses batch=1 and
# accumulation=8 to preserve the required global batch of 32 on four cards.

project_root="${OMNI_OPSD_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_env="${OMNI_OPSD_ENV:-/share/home/ylhu/.conda/envs/vllm}"
model="${OMNI_OPSD_MODEL:-/share/home/ylhu/models/Qwen2.5-Omni-3B}"
devices="${OMNI_OPSD_CUDA_DEVICES:-0,1,2,3}"
mode="${1:-formal}"

export OMNI_OPSD_PROJECT_ROOT="${project_root}"
export OMNI_OPSD_ENV="${python_env}"
export OMNI_OPSD_MODEL="${model}"
export OMNI_OPSD_TUNER_TYPE=full
export OMNI_OPSD_FREEZE_LLM=false
export OMNI_OPSD_FREEZE_VIT=false
export OMNI_OPSD_FREEZE_ALIGNER=false
export OMNI_OPSD_CUDA_DEVICES="${devices}"
export OMNI_OPSD_NPROC_PER_NODE=4
export OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE="${OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS="${OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS:-8}"
export OMNI_OPSD_MAX_STEPS="${OMNI_OPSD_MAX_STEPS:-157}"
export OMNI_OPSD_NUM_TRAIN_EPOCHS=1
export OMNI_OPSD_SFT_LEARNING_RATE="${OMNI_OPSD_SFT_LEARNING_RATE:-1e-5}"
export OMNI_OPSD_LEARNING_RATE="${OMNI_OPSD_LEARNING_RATE:-${OMNI_OPSD_SFT_LEARNING_RATE}}"
export OMNI_OPSD_MAX_LENGTH="${OMNI_OPSD_MAX_LENGTH:-32768}"
export OMNI_OPSD_MIN_PIXELS="${OMNI_OPSD_MIN_PIXELS:-3136}"
export OMNI_OPSD_MAX_GRAD_NORM="${OMNI_OPSD_MAX_GRAD_NORM:-0}"
export OMNI_OPSD_ATTN_IMPL="${OMNI_OPSD_ATTN_IMPL:-sdpa}"
export OMNI_OPSD_ALLOC_CONF="${OMNI_OPSD_ALLOC_CONF:-expandable_segments:True}"
export OMNI_OPSD_USE_LOGITS_TO_KEEP=1
export USE_AUDIO_IN_VIDEO=1
export MAX_NUM_WORKERS_FETCH_VIDEO="${MAX_NUM_WORKERS_FETCH_VIDEO:-1}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export OMNI_OPSD_EXPERIMENT_LABEL="${OMNI_OPSD_EXPERIMENT_LABEL:-gap5000_sft_3b_full_cuda}"

if [[ -z "${OMNI_OPSD_DATASET:-}" ]]; then
  if [[ "${mode}" == formal ]]; then
    export OMNI_OPSD_DATASET="${project_root}/data/gap5000/dynamic_budget_v1/sft/formal/data/sft.jsonl"
  elif [[ "${mode}" == gate ]]; then
    export OMNI_OPSD_DATASET="${project_root}/data/gap5000/dynamic_budget_v1/sft/gate/data/sft.jsonl"
  else
    echo "usage: $0 formal|gate" >&2
    exit 2
  fi
fi

exec bash "${project_root}/scripts/run_gap5000_sft_cuda.sh" "${mode}"
