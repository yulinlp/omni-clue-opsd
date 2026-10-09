#!/usr/bin/env bash
# Run on worker-0. Stop monitoring when the formal trainer exits.
set -uo pipefail
root=/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_sft_lora_20260930
snapshot=/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_sft_lora_npu24_20260928/scripts/npu-snapshot.sh
while [[ ! -f "$root/logs/formal_worker-0.exit" ]]; do
  date --iso-8601=seconds
  bash "$snapshot"
  tail -1 "$root"/outputs/formal/v*/logging.jsonl 2>/dev/null || true
  sleep 60
done
printf 'Trainer exited: '
cat "$root/logs/formal_worker-0.exit"
