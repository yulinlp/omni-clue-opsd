#!/usr/bin/env bash
set -uo pipefail
phase="${1:-formal}"
repo=/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd
root="$repo/training_runs/worldsense_observation_sft_lora_20260930"
echo "$$" > "$root/logs/${phase}_worker-0.pid"
if [[ "$phase" == smoke ]]; then
  export OMNI_OPSD_DATASET="$root/data/smoke32.jsonl"
  export OMNI_OPSD_OUTPUT_DIR="$root/outputs/smoke"
  export OMNI_OPSD_MAX_STEPS=1
fi
bash "$repo/training_code/scripts/run_worldsense_observation_sft_lora_npu_w0.sh"
code=$?
echo "$code" > "$root/logs/${phase}_worker-0.exit"
exit "$code"
