#!/usr/bin/env bash
set -euo pipefail

# Qwen2.5-Omni-3B full-parameter CLUE-OPSD launcher.
# The underlying CUDA launcher uses the section-6/7 dynamic 16k visual budget
# and the four-card effective batch contract.  This 3B rerun uses the new
# full-parameter EMA teacher and an answer-token auxiliary CE term.

project_root="${OMNI_OPSD_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export OMNI_OPSD_PROJECT_ROOT="${project_root}"
export OMNI_OPSD_MODEL="${OMNI_OPSD_MODEL:-/share/home/ylhu/models/Qwen2.5-Omni-3B}"
export OMNI_OPSD_TUNER_TYPE="full"
export OMNI_OPSD_FREEZE_LLM="false"
export OMNI_OPSD_FREEZE_VIT="false"
export OMNI_OPSD_FREEZE_ALIGNER="false"
export OMNI_OPSD_ALLOW_NON_EMA_CLUE="1"
export OMNI_OPSD_FULL_EMA_TEACHER="true"
export OMNI_OPSD_FULL_EMA_ALPHA="${OMNI_OPSD_FULL_EMA_ALPHA:-0.05}"
# A detached 5.5B-parameter BF16 EMA copy cannot remain resident alongside
# the full student and Adam states on an 80-GiB card.  Keep it CPU-offloaded
# by default; the launcher can be overridden for larger-memory hardware.
export OMNI_OPSD_FULL_EMA_OFFLOAD="${OMNI_OPSD_FULL_EMA_OFFLOAD:-true}"
export OMNI_OPSD_GOLD_CE_ALPHA="${OMNI_OPSD_GOLD_CE_ALPHA:-0.25}"
# Full-parameter Adam states are offloaded only around Transformers rollout;
# keeping the student on CUDA preserves normal multimodal training speed while
# avoiding the post-first-step generation OOM on 80-GiB A100 cards.
export OMNI_OPSD_OFFLOAD_MODEL="${OMNI_OPSD_OFFLOAD_MODEL:-false}"
export OMNI_OPSD_OFFLOAD_OPTIMIZER="${OMNI_OPSD_OFFLOAD_OPTIMIZER:-true}"
export OMNI_OPSD_MAX_COMPLETION_LENGTH="${OMNI_OPSD_MAX_COMPLETION_LENGTH:-256}"
export OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE="${OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS="${OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS:-8}"
export OMNI_OPSD_USE_VLLM="${OMNI_OPSD_USE_VLLM:-false}"
export OMNI_OPSD_VLLM_GPU_MEMORY_UTILIZATION="${OMNI_OPSD_VLLM_GPU_MEMORY_UTILIZATION:-0.10}"
export OMNI_OPSD_VLLM_SLEEP_LEVEL="${OMNI_OPSD_VLLM_SLEEP_LEVEL:-2}"
export OMNI_OPSD_DDP_FIND_UNUSED_PARAMETERS="${OMNI_OPSD_DDP_FIND_UNUSED_PARAMETERS:-true}"
export OMNI_OPSD_VLLM_DROP_AUDIO="0"
export USE_AUDIO_IN_VIDEO="1"
export OMNI_OPSD_SPLIT_DATASET_RATIO="${OMNI_OPSD_SPLIT_DATASET_RATIO:-0.1}"
export OMNI_OPSD_EVAL_STRATEGY="${OMNI_OPSD_EVAL_STRATEGY:-steps}"
export OMNI_OPSD_EVAL_STEPS="${OMNI_OPSD_EVAL_STEPS:-25}"
export OMNI_OPSD_LOG_COMPLETIONS="true"
export OMNI_OPSD_SAVE_STEPS="${OMNI_OPSD_SAVE_STEPS:-25}"
export OMNI_OPSD_MAX_STEPS="${OMNI_OPSD_MAX_STEPS:-141}"
export OMNI_OPSD_DATASET="${OMNI_OPSD_DATASET:-${project_root}/data/gap5000/dynamic_budget_v1_16k_20260913/clue_opsd/formal/data/reasoning.jsonl}"
export OMNI_OPSD_EXPERIMENT_LABEL="${OMNI_OPSD_EXPERIMENT_LABEL:-gap5000_clue_opsd_3b_full_ema_cuda}"

exec bash "${project_root}/scripts/run_gap5000_clue_opsd_cuda.sh" "${1:-formal}"
