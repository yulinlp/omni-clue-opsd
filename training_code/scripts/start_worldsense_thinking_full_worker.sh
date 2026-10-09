#!/usr/bin/env bash
# Called via SSH with nohup; keep a PID and exit code for each worker.
set -uo pipefail
phase="${1:?smokeN|formal}"
rank="${2:?0|1}"
repo=/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd
root="${OMNI_OPSD_RUN_ROOT:-$repo/training_runs/worldsense_openqa_thinking_full_20260930}"
worker=$((rank+1))
echo "$$" > "$root/logs/${phase}_worker-${worker}.pid"
if [[ "$phase" == smoke* ]]; then
  export OMNI_OPSD_DATASET="$root/data/smoke32.jsonl"
  export OMNI_OPSD_OUTPUT_DIR="$root/outputs/$phase"
  export OMNI_OPSD_MAX_STEPS=1 OMNI_OPSD_SAVE_STEPS=1
fi
bash "$repo/training_code/scripts/run_worldsense_openqa_thinking_full_npu_w1w2.sh" "$rank"
code=$?
echo "$code" > "$root/logs/${phase}_worker-${worker}.exit"
exit "$code"
