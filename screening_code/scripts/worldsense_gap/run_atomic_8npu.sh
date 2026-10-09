#!/usr/bin/env bash
# Launch eight disjoint WorldSense atomic-view score shards on one 8-NPU worker.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
workspace_root="$(dirname "$project_root")"
source_root="$workspace_root/Omni-OPSD"
canonical="$project_root/data/atomic_eval_20260929/worldsense_candidates_3079.atomic.jsonl"
model="$workspace_root/omni-opsd-assets/models/Qwen2.5-Omni-7B"
python_bin="/home/ma-user/anaconda3/envs/PyTorch-2.9.0/bin/python"
python_deps="$workspace_root/omni-opsd-runtime/python-deps"
scorer="$project_root/screening_code/scripts/worldsense_gap/score_atomic_views.py"
run_root="$project_root/data/atomic_eval_20260929/${ATOMIC_RUN_NAME:-full_8npu}"
cpu_ranges=(1-7 24-31 32-39 40-47 48-55 56-63 64-71 72-79)
IFS=',' read -r -a devices <<< "${ATOMIC_DEVICES:-0,1,2,3,4,5,6,7}"
variants="${ATOMIC_VARIANTS:-E2_G_exact,E4_H_halo3,E7_T_8fps,E8_S_highres,E12_gold_v,E13_A_audio_exact}"
if [[ "${#devices[@]}" -eq 0 ]]; then
  echo "no devices selected" >&2
  exit 2
fi

for required in "$canonical" "$model/config.json" "$python_bin" "$scorer"; do
  [[ -e "$required" ]] || { echo "missing $required" >&2; exit 2; }
done
mkdir -p "$run_root"

for device in "${devices[@]}"; do
  if [[ ! "$device" =~ ^[0-7]$ ]]; then
    echo "invalid device: $device" >&2
    exit 2
  fi
  shard_dir="$run_root/$(printf 'shard%02d' "$device")"
  mkdir -p "$shard_dir"
  if [[ -f "$shard_dir/PID" ]]; then
    prior_pid="$(cat "$shard_dir/PID")"
    if kill -0 "$prior_pid" 2>/dev/null; then
      echo "shard $device already running as $prior_pid" >&2
      exit 2
    fi
  fi
done

for device in "${devices[@]}"; do
  shard_dir="$run_root/$(printf 'shard%02d' "$device")"
  nohup env \
    ASCEND_RT_VISIBLE_DEVICES="$device" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMNI_OPSD_TORCH_THREADS=1 \
    PYTHONPATH="$source_root/src:$python_deps" \
    taskset -c "${cpu_ranges[$device]}" "$python_bin" "$scorer" \
      --source-root "$source_root" \
      --annotation "$canonical" \
      --model "$model" \
      --output "$shard_dir/results.jsonl" \
      --unscored-output "$shard_dir/unscored.jsonl" \
      --num-shards 8 --shard-index "$device" --device npu:0 --resume \
      --variants "$variants" \
      > "$shard_dir/score.log" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$shard_dir/PID"
  echo "launched shard=$device npu=$device pid=$pid"
done
