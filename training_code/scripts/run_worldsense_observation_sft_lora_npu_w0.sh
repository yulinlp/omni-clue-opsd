#!/usr/bin/env bash
# Single-node, eight-NPU SFT. Same effective batch 32 as the previous SFT.
set -euo pipefail
repo=/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd
runtime=/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_sft_lora_npu24_20260928
export OMNI_OPSD_RUN_ROOT="$repo/training_runs/worldsense_observation_sft_lora_20260930"
export OMNI_OPSD_DATASET="${OMNI_OPSD_DATASET:-$OMNI_OPSD_RUN_ROOT/data/sft_observation_openqa.jsonl}"
export OMNI_OPSD_OUTPUT_DIR="${OMNI_OPSD_OUTPUT_DIR:-$OMNI_OPSD_RUN_ROOT/outputs/formal}"
export OMNI_OPSD_NNODES=1 OMNI_OPSD_NPROC_PER_NODE=8
export OMNI_OPSD_MASTER_ADDR=127.0.0.1 OMNI_OPSD_MASTER_PORT=29551
export OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS=4
export OMNI_OPSD_NUM_TRAIN_EPOCHS=3 OMNI_OPSD_SAVE_STRATEGY=epoch OMNI_OPSD_SAVE_TOTAL_LIMIT=3
export OMNI_OPSD_LORA_RANK=64 OMNI_OPSD_LORA_ALPHA=128 OMNI_OPSD_LEARNING_RATE=1e-5
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLDSENSE_CKPT_LM_HEAD=1
# Dynamic HCCL rendezvous for a single node; do not use the platform's 24-card table.
unset RANK_TABLE_FILE RANK_TABLE_FILE_V_1_0 || true
exec bash "$runtime/scripts/run_sft_lora_npu24.sh" 0
