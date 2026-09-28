#!/usr/bin/env bash
# Full 3079-question gap screening: full on gpu07, gold on gpu02.
set -uo pipefail
cd /share/home/ylhu/Omni-OPSD
export PYTHONPATH=/share/home/ylhu/Omni-OPSD
OUT=outputs/worldsense_gap/full
mkdir -p $OUT
setsid nohup python3 scripts/worldsense_gap/score.py \
  --canonical outputs/worldsense_gap/candidates.jsonl \
  --audit outputs/worldsense_gap/audit.json \
  --model-dir /share/home/ylhu/models/Qwen2.5-Omni-7B \
  --endpoint http://gpu07:8091/v1/chat/completions \
  --output-dir $OUT/full_run --views full --workers 12 \
  > $OUT/full_run.log 2>&1 < /dev/null &
echo "full scorer pid=$!"
setsid nohup python3 scripts/worldsense_gap/score.py \
  --canonical outputs/worldsense_gap/candidates.jsonl \
  --audit outputs/worldsense_gap/audit.json \
  --model-dir /share/home/ylhu/models/Qwen2.5-Omni-7B \
  --endpoint http://gpu02:8092/v1/chat/completions \
  --output-dir $OUT/gold_run --views gold --workers 12 \
  > $OUT/gold_run.log 2>&1 < /dev/null &
echo "gold scorer pid=$!"
setsid nohup python3 scripts/worldsense_gap/monitor.py \
  --run-dir $OUT --expected 3079 --interval 1800 \
  > $OUT/monitor_daemon.log 2>&1 < /dev/null &
echo "monitor pid=$!"
