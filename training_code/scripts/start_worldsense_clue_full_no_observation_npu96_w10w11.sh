#!/usr/bin/env bash
# CLUE ablation: teacher receives gold clue and answer, no observation.
set -uo pipefail
phase="${1:?smoke|formal}"
rank="${2:?0|1}"
[[ "$rank" == 0 || "$rank" == 1 ]] || exit 2
repo=/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd
root="$repo/training_runs/worldsense_clue_full_no_observation_npu96_w10w11_20260930"
worker=$((rank+10))
echo "$$" > "$root/logs/${phase}_worker-${worker}.pid"
export OMNI_OPSD_RUN_ROOT="$root"
export OMNI_OPSD_MASTER_ADDR=172.16.13.102 OMNI_OPSD_MASTER_PORT=29581
export OMNI_OPSD_RANK_TABLE_FILE="$root/data/ranktable_w10w11.json"
export OMNI_OPSD_OUTPUT_DIR="$root/outputs/$phase"
if [[ "$phase" == smoke ]]; then
  export OMNI_OPSD_DATASET="$root/data/smoke32.jsonl"
  export OMNI_OPSD_MAX_STEPS=1 OMNI_OPSD_SAVE_STEPS=1
fi
bash "$repo/training_code/scripts/run_worldsense_openqa_thinking_full_npu_w1w2.sh" "$rank"
code=$?
echo "$code" > "$root/logs/${phase}_worker-${worker}.exit"
exit "$code"
