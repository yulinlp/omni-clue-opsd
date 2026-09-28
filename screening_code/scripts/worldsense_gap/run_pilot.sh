#!/usr/bin/env bash
# 200-question pilot: full on gpu07, gold on gpu02, both scored from ln02.
set -uo pipefail
cd /share/home/ylhu/Omni-OPSD
export PYTHONPATH=/share/home/ylhu/Omni-OPSD
OUT=outputs/worldsense_gap/pilot
mkdir -p $OUT
setsid nohup python3 scripts/worldsense_gap/score.py \
  --canonical outputs/worldsense_gap/pilot200.jsonl \
  --audit outputs/worldsense_gap/audit.json \
  --model-dir /share/home/ylhu/models/Qwen2.5-Omni-7B \
  --endpoint http://gpu07:8091/v1/chat/completions \
  --output-dir $OUT/full_run --views full --workers 6 \
  > $OUT/full_run.log 2>&1 < /dev/null &
echo "full scorer pid=$!"
setsid nohup python3 scripts/worldsense_gap/score.py \
  --canonical outputs/worldsense_gap/pilot200.jsonl \
  --audit outputs/worldsense_gap/audit.json \
  --model-dir /share/home/ylhu/models/Qwen2.5-Omni-7B \
  --endpoint http://gpu02:8092/v1/chat/completions \
  --output-dir $OUT/gold_run --views gold --workers 6 \
  > $OUT/gold_run.log 2>&1 < /dev/null &
echo "gold scorer pid=$!"
