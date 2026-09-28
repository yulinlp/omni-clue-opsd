#!/usr/bin/env bash
set -euo pipefail

# Full-parameter Qwen2.5-Omni-3B Clue-OPSD on the direct-answer OE 5k set.
# This launcher is intended for wxzhao/gpu07 with GPUs 0 and 2 excluded
# because existing reference jobs occupy them; six idle GPUs are passed explicitly.

project_root="${OMNI_OPSD_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
devices="${OMNI_OPSD_CUDA_DEVICES:-1,3,4,5,6,7}"
IFS=',' read -r -a device_list <<< "${devices}"
if [[ "${#device_list[@]}" -ne 6 ]]; then
  echo "This launcher requires six idle GPUs (GPU0 and GPU2 are reserved): ${devices}" >&2
  exit 2
fi

tag="${OMNI_OPSD_CLUE_OPSD_TAG:-$(date +%Y%m%d_%H%M%S)}"
export OMNI_OPSD_PROJECT_ROOT="${project_root}"
export OMNI_OPSD_ENV="${OMNI_OPSD_ENV:-/share/home/ylhu/.conda/envs/vllm}"
export OMNI_OPSD_MS_SWIFT_ROOT="${OMNI_OPSD_MS_SWIFT_ROOT:-${project_root}/ms-swift}"
export OMNI_OPSD_MODEL="${OMNI_OPSD_MODEL:-/share/home/ylhu/models/Qwen2.5-Omni-3B}"
export OMNI_OPSD_CUDA_DEVICES="${devices}"
export OMNI_OPSD_NPROC_PER_NODE=6
export OMNI_OPSD_DATASET="${OMNI_OPSD_DATASET:-${project_root}/data/omnivideo_oe_5k_clue_only/omnivideo_oe_5k.clue_opsd.jsonl}"
export OMNI_OPSD_CLUE_OPSD_TAG="${tag}"
export OMNI_OPSD_CLUE_OPSD_ROOT="${OMNI_OPSD_CLUE_OPSD_ROOT:-${project_root}/output/omnivideo_oe_5k_clue_opsd_3b_full_ema_gpu07_${tag}}"
export OMNI_OPSD_EXPERIMENT_LABEL="${OMNI_OPSD_EXPERIMENT_LABEL:-omnivideo_oe_5k_clue_opsd_3b_full_ema_gpu07_6gpu}"

# Full-parameter student and detached full-parameter EMA teacher.
export OMNI_OPSD_TUNER_TYPE=full
export OMNI_OPSD_FREEZE_LLM=false
export OMNI_OPSD_FREEZE_VIT=false
export OMNI_OPSD_FREEZE_ALIGNER=false
export OMNI_OPSD_FULL_EMA_TEACHER=true
export OMNI_OPSD_FULL_EMA_ALPHA="${OMNI_OPSD_FULL_EMA_ALPHA:-0.05}"
export OMNI_OPSD_FULL_EMA_OFFLOAD=true
# Full EMA is the only teacher state for this run; do not also request the
# legacy LoRA-shadow clue_ema_alpha CLI option.
export OMNI_OPSD_ALLOW_NON_EMA_CLUE=1

# Six ranks cannot form an exact global batch of 32 with integer accumulation.
# Use 6*1*5=30 and record this explicit protocol exception.
export OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE=1
export OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS=5
export OMNI_OPSD_ALLOW_NON_32_GLOBAL_BATCH=1
export OMNI_OPSD_NUM_TRAIN_EPOCHS=1
export OMNI_OPSD_MAX_STEPS="${OMNI_OPSD_MAX_STEPS:-141}"
export OMNI_OPSD_LEARNING_RATE="${OMNI_OPSD_LEARNING_RATE:-2e-6}"
export OMNI_OPSD_MAX_LENGTH=32768
export OMNI_OPSD_MAX_COMPLETION_LENGTH="${OMNI_OPSD_MAX_COMPLETION_LENGTH:-512}"
export OMNI_OPSD_SAVE_STEPS="${OMNI_OPSD_SAVE_STEPS:-25}"
export OMNI_OPSD_SPLIT_DATASET_RATIO=0.1
export OMNI_OPSD_EVAL_STRATEGY=steps
export OMNI_OPSD_EVAL_STEPS="${OMNI_OPSD_EVAL_STEPS:-25}"

# The OE gold answer is free text and is deliberately withheld from the teacher
# prompt; the previous A-D answer-token CE is therefore not applicable.
export OMNI_OPSD_GOLD_CE_ALPHA=0
export OMNI_OPSD_SFT_ALPHA=0
export OMNI_OPSD_LOG_COMPLETIONS=true
export OMNI_OPSD_DIAG_ENABLED=true
export OMNI_OPSD_DIAG_TOP_K=20
export OMNI_OPSD_DIAG_TEMPERATURE=1.0
export OMNI_OPSD_DIAG_FREQUENCY=1
export OMNI_OPSD_DIAG_CHUNK_SIZE=256

# Transformers rollout keeps audio/video settings identical to the teacher
# forward and avoids vLLM's multimodal audio path on this full-parameter run.
export OMNI_OPSD_USE_VLLM=false
export OMNI_OPSD_VLLM_DROP_AUDIO=0
export USE_AUDIO_IN_VIDEO=1
export MAX_NUM_WORKERS_FETCH_VIDEO="${MAX_NUM_WORKERS_FETCH_VIDEO:-1}"
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export OMNI_OPSD_OFFLOAD_MODEL=false
export OMNI_OPSD_OFFLOAD_OPTIMIZER=true
export OMNI_OPSD_DDP_FIND_UNUSED_PARAMETERS=true
export OMNI_OPSD_ATTN_IMPL=sdpa
export OMNI_OPSD_MAX_GRAD_NORM=0
export OMNI_OPSD_GKD_SAFE_MODE=0
export OMNI_OPSD_ALLOC_CONF="${OMNI_OPSD_ALLOC_CONF:-expandable_segments:True}"

exec bash "${project_root}/scripts/run_gap5000_clue_opsd_cuda.sh" formal
