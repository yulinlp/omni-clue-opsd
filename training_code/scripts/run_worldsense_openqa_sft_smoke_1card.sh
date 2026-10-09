#!/usr/bin/env bash
# One-card, one-step validation of the open-QA SFT data and training path.
set -euo pipefail
repo="/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd"
runtime="/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_sft_lora_npu24_20260928"
export OMNI_OPSD_RUN_ROOT="${repo}/training_runs/worldsense_openqa_20260929"
export OMNI_OPSD_DATASET="${OMNI_OPSD_RUN_ROOT}/data/sft_smoke_heavy1.jsonl"
export OMNI_OPSD_OUTPUT_DIR="${OMNI_OPSD_RUN_ROOT}/outputs/sft_smoke"
export OMNI_OPSD_NNODES=1
export OMNI_OPSD_NPROC_PER_NODE=1
export OMNI_OPSD_MASTER_ADDR=127.0.0.1
export OMNI_OPSD_MASTER_PORT=29524
export OMNI_OPSD_MAX_STEPS=1
export OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS=1
export ASCEND_RT_VISIBLE_DEVICES=0
unset RANK_TABLE_FILE RANK_TABLE_FILE_V_1_0 || true
exec bash "${runtime}/scripts/run_sft_lora_npu24.sh" 0
