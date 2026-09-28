#!/usr/bin/env bash
set -euo pipefail

# 1-step memory/decode gate for the 15k-visual-budget CLUE-OPSD full-EMA run.
# Environment matches run_omnivideo_oe5k_clue_opsd_gpu05_v15k.sh exactly;
# only max_steps/save_steps/root/mode differ so this is a true short gate for
# the formal 133-step run (batch=1, grad_accum=8, full-EMA offload, audio on).

project_root="/share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907"
tag="gpu05_v15k_gate_20260918"

export OMNI_OPSD_PROJECT_ROOT="${project_root}"
export OMNI_OPSD_ENV="/share/home/ylhu/.conda/envs/vllm"
export OMNI_OPSD_MS_SWIFT_ROOT="${project_root}/ms-swift"
export OMNI_OPSD_MODEL="/share/home/ylhu/models/Qwen2.5-Omni-3B"
export OMNI_OPSD_CUDA_DEVICES="0,1,2,3"
export OMNI_OPSD_NPROC_PER_NODE=4
export OMNI_OPSD_DATASET="${project_root}/data/omnivideo_oe_5k_v15k_clue_only_repaired/omnivideo_oe_5k.clue_opsd.jsonl"
export OMNI_OPSD_CLUE_OPSD_TAG="${tag}"
export OMNI_OPSD_CLUE_OPSD_ROOT="${project_root}/output/omnivideo_oe5k_clue_opsd_3b_full_ema_gpu05_v15k_gate_20260918"
export OMNI_OPSD_EXPERIMENT_LABEL="omnivideo_oe5k_clue_opsd_3b_full_ema_gpu05_v15k_gate"

export OMNI_OPSD_TUNER_TYPE=full
export OMNI_OPSD_FREEZE_LLM=false
export OMNI_OPSD_FREEZE_VIT=false
export OMNI_OPSD_FREEZE_ALIGNER=false
export OMNI_OPSD_FULL_EMA_TEACHER=true
export OMNI_OPSD_FULL_EMA_ALPHA=0.05
export OMNI_OPSD_FULL_EMA_OFFLOAD=true
export OMNI_OPSD_ALLOW_NON_EMA_CLUE=1

export OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE=1
export OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS=8
export OMNI_OPSD_ALLOW_NON_32_GLOBAL_BATCH=0
export OMNI_OPSD_NUM_TRAIN_EPOCHS=1
export OMNI_OPSD_MAX_STEPS=1
export OMNI_OPSD_LEARNING_RATE=2e-6
export OMNI_OPSD_MAX_LENGTH=32768
export OMNI_OPSD_MAX_COMPLETION_LENGTH=512
export OMNI_OPSD_SAVE_STEPS=1
export OMNI_OPSD_SPLIT_DATASET_RATIO=0.1
export OMNI_OPSD_EVAL_STRATEGY=no
export OMNI_OPSD_EVAL_STEPS=1

export OMNI_OPSD_GOLD_CE_ALPHA=0
export OMNI_OPSD_SFT_ALPHA=0
export OMNI_OPSD_LOG_COMPLETIONS=true
export OMNI_OPSD_DIAG_ENABLED=true
export OMNI_OPSD_DIAG_TOP_K=20
export OMNI_OPSD_DIAG_TEMPERATURE=1.0
export OMNI_OPSD_DIAG_FREQUENCY=1
export OMNI_OPSD_DIAG_CHUNK_SIZE=256

export OMNI_OPSD_USE_VLLM=false
export OMNI_OPSD_VLLM_DROP_AUDIO=0
export USE_AUDIO_IN_VIDEO=1
export MAX_NUM_WORKERS_FETCH_VIDEO=1
export FORCE_QWENVL_VIDEO_READER=decord
export OMNI_OPSD_DIRECT_LOCAL_VIDEO=1
export DECORD_EOF_RETRY_MAX=20480
export DECORD_NUM_THREADS=1
export OMNI_OPSD_OFFLOAD_MODEL=false
export OMNI_OPSD_OFFLOAD_OPTIMIZER=true
export OMNI_OPSD_DDP_FIND_UNUSED_PARAMETERS=true
export OMNI_OPSD_ATTN_IMPL=sdpa
export OMNI_OPSD_MAX_GRAD_NORM=0
export OMNI_OPSD_GKD_SAFE_MODE=0
export OMNI_OPSD_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=30011

exec bash "${project_root}/scripts/run_gap5000_clue_opsd_cuda.sh" smoke
