#!/usr/bin/env bash
# Run in npu96 worker-6 (rank 0) or worker-7 (rank 1).
set -uo pipefail
phase="${1:?smoke|formal}"
rank="${2:?0|1}"
[[ "$rank" == 0 || "$rank" == 1 ]] || exit 2
repo=/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd
root="$repo/training_runs/worldsense_observation_sft_full_npu96_w6w7_20260930"
worker=$((rank+6))
echo "$$" > "$root/logs/${phase}_worker-${worker}.pid"
export OMNI_OPSD_RUN_ROOT="$root"
export OMNI_OPSD_DATASET="$root/data/sft_observation_openqa.jsonl"
export OMNI_OPSD_OUTPUT_DIR="$root/outputs/$phase"
export OMNI_OPSD_NNODES=2 OMNI_OPSD_NPROC_PER_NODE=8
export OMNI_OPSD_MASTER_ADDR=172.16.15.143 OMNI_OPSD_MASTER_PORT=29571
export OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS=2
export OMNI_OPSD_NUM_TRAIN_EPOCHS=3 OMNI_OPSD_SAVE_STRATEGY=epoch OMNI_OPSD_SAVE_TOTAL_LIMIT=3
export OMNI_OPSD_LEARNING_RATE=1e-5 WORLDSENSE_CKPT_LM_HEAD=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RANK_TABLE_FILE="$root/data/ranktable_w6w7.json"
export RANK_TABLE_FILE_V_1_0="$RANK_TABLE_FILE"
export HCCL_CONNECT_TIMEOUT=600 PYTHONUNBUFFERED=1
if [[ "$phase" == smoke ]]; then
  export OMNI_OPSD_DATASET="$root/data/smoke32.jsonl"
  export OMNI_OPSD_MAX_STEPS=1
fi
bash "$repo/training_code/scripts/run_sft_full_observation_common.sh" "$rank"
code=$?
echo "$code" > "$root/logs/${phase}_worker-${worker}.exit"
exit "$code"
