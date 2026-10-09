#!/usr/bin/env bash
set -euo pipefail
repo=/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd
export OMNI_OPSD_RUN_ROOT="$repo/training_runs/worldsense_openqa_thinking_full_observation_20260930"
export OMNI_OPSD_MASTER_PORT=29561
exec bash "$repo/training_code/scripts/start_worldsense_thinking_full_worker.sh" formal "${1:?0|1}"
