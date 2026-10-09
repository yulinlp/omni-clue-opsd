#!/usr/bin/env bash
# Start on npu96 worker-10 (sft) or npu24 worker-1 (clue).
# Each arm waits only for its own training, evaluates OmniVideoBench 200,
# waits at a common barrier, then evaluates video-disjoint WorldSense 200.
set -euo pipefail

arm="${1:?usage: $0 sft|clue}"
case "${arm}" in sft|clue) ;; *) echo "arm must be sft or clue" >&2; exit 2 ;; esac
repo="/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd"
train_root="${repo}/training_runs/worldsense_openqa_20260929"
omni="${train_root}/eval_omnivideobench200"
world="${train_root}/eval_worldsense_holdout200"
runner="${repo}/training_code/scripts/run_worldsense_omnivideobench200_npu.sh"
aggregate="${repo}/training_code/scripts/aggregate_video_odyssey_training_eval.py"
python_bin="/home/ma-user/anaconda3/envs/PyTorch-2.9.0/bin/python"
if [[ "${arm}" == sft ]]; then
  run="${train_root}/outputs/sft_formal/v0-20260929-174408"
  train_log="${train_root}/sft_worker-10.log"
  train_data="${train_root}/data/sft_openqa.jsonl"
  eval_arm=sft
else
  run="${train_root}/outputs/clue_formal/v0-20260929-174734"
  train_log="${train_root}/clue_worker-1.log"
  train_data="${train_root}/data/clue_openqa.jsonl"
  eval_arm=clue_opsd
fi
mkdir -p "${omni}" "${world}"
exec 9> "${omni}/${arm}.lock"
flock -n 9 || { echo "${arm} queue already running" >&2; exit 1; }

status() {
  printf 'status=%s\ntime=%s\n' "$1" "$(date --iso-8601=seconds)" > "${omni}/${arm}.STATUS.txt"
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$1"
}
on_exit() {
  local rc=$?
  if ((rc != 0)); then status "FAILED exit_code=${rc}"; fi
}
trap on_exit EXIT

training_complete() {
  "${python_bin}" - "${run}" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
checkpoint = run / "checkpoint-135"
for name in ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json"):
    path = checkpoint / name
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(1)
try:
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    records = [json.loads(line) for line in (run / "logging.jsonl").read_text().splitlines() if line.strip()]
except (OSError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
if state.get("global_step") != 135 or not any(
    "train_runtime" in item and item.get("global_step/max_steps") == "135/135" for item in records
):
    raise SystemExit(1)
PY
}

wait_training() {
  status "WAITING_${arm^^}_TRAINING"
  until training_complete; do
    if [[ -f "${train_log}" ]]; then
      age=$(( $(date +%s) - $(stat -c %Y "${train_log}") ))
      if ((age > 7200)); then
        echo "training log is stale for over two hours: ${train_log}" >&2
        exit 1
      fi
    fi
    sleep "${OMNI_OPSD_EVAL_POLL_SECONDS:-60}"
  done
  # The final trainer summary may precede torchrun's process exit.
  while pgrep -f -- "${train_data}" >/dev/null; do
    status "WAITING_${arm^^}_TRAIN_PROCESSES_EXIT"
    sleep "${OMNI_OPSD_EVAL_POLL_SECONDS:-60}"
  done
  status "${arm^^}_TRAINING_COMPLETE"
}

run_eval() {
  local benchmark="$1" root="$2" label="$3" adapter="$4"
  local data labels output="${root}/results/${label}"
  if [[ "${benchmark}" == OmniVideoBench ]]; then
    data="${root}/data/omnivideobench.answer_free.jsonl"
    labels="${root}/data/omnivideobench.labels.jsonl"
  else
    data="${root}/data/worldsense.answer_free.jsonl"
    labels="${root}/data/worldsense.labels.jsonl"
  fi
  if [[ -s "${output}/summary.json" && -s "${output}/results.jsonl" ]]; then
    status "SKIPPED_${benchmark}_${label}_COMPLETE"
    return
  fi
  status "RUNNING_${benchmark}_${label}"
  OMNI_OPSD_EVAL_BENCHMARK="${benchmark}" \
  OMNI_OPSD_EVAL_DATASET="${data}" OMNI_OPSD_EVAL_LABELS="${labels}" \
  OMNI_OPSD_EVAL_MASTER_PORT="${OMNI_OPSD_EVAL_MASTER_PORT:-29780}" \
    bash "${runner}" "${eval_arm}" "${adapter}" "${output}"
  status "COMPLETED_${benchmark}_${label}"
}

smoke() {
  local benchmark="$1" root="$2" adapter="$3" output="${2}/smoke_${arm}"
  local data labels
  [[ ! -s "${output}/summary.json" ]] || return 0
  if [[ "${benchmark}" == OmniVideoBench ]]; then
    data="${root}/smoke_data/omnivideobench.answer_free.jsonl"
    labels="${root}/smoke_data/omnivideobench.labels.jsonl"
  else
    data="${root}/smoke_data/worldsense.answer_free.jsonl"
    labels="${root}/smoke_data/worldsense.labels.jsonl"
  fi
  status "SMOKE_${benchmark}_${arm}"
  OMNI_OPSD_EVAL_BENCHMARK="${benchmark}" \
  OMNI_OPSD_EVAL_DATASET="${data}" OMNI_OPSD_EVAL_LABELS="${labels}" \
  OMNI_OPSD_EVAL_EXPECTED_ROWS=1 OMNI_OPSD_EVAL_NPROC=1 OMNI_OPSD_EVAL_DEVICES=0 \
  OMNI_OPSD_EVAL_MASTER_PORT=29779 \
    bash "${runner}" "${eval_arm}" "${adapter}" "${output}"
}

barrier_complete() {
  local root="$1" name
  for name in base sft_epoch3 sft_epoch2 sft_epoch1 clue_epoch3 clue_epoch2 clue_epoch1; do
    [[ -s "${root}/results/${name}/summary.json" ]] || return 1
  done
}

wait_barrier() {
  local root="$1" benchmark="$2" other
  if [[ "${arm}" == sft ]]; then other=clue; else other=sft; fi
  status "WAITING_${benchmark}_BARRIER"
  until barrier_complete "${root}"; do
    if [[ -s "${omni}/${other}.STATUS.txt" ]] && \
       grep -q '^status=FAILED' "${omni}/${other}.STATUS.txt"; then
      echo "${other} queue failed; see ${omni}/${other}.queue.log" >&2
      exit 1
    fi
    sleep "${OMNI_OPSD_EVAL_POLL_SECONDS:-60}"
  done
  status "${benchmark}_BARRIER_COMPLETE"
}

compare_all() {
  local root="$1" benchmark="$2" data labels
  if [[ "${benchmark}" == OmniVideoBench ]]; then
    data="${root}/data/omnivideobench.answer_free.jsonl"
    labels="${root}/data/omnivideobench.labels.jsonl"
  else
    data="${root}/data/worldsense.answer_free.jsonl"
    labels="${root}/data/worldsense.labels.jsonl"
  fi
  status "AGGREGATING_${benchmark}"
  PYTHONPATH="${repo}/training_code/src" "${python_bin}" "${aggregate}" \
    --benchmark "${benchmark}" --labels "${labels}" --dataset "${data}" \
    --run "base=${root}/results/base/results.jsonl" \
    --run "sft_epoch3=${root}/results/sft_epoch3/results.jsonl" \
    --run "clue_epoch3=${root}/results/clue_epoch3/results.jsonl" \
    --run "sft_epoch2=${root}/results/sft_epoch2/results.jsonl" \
    --run "clue_epoch2=${root}/results/clue_epoch2/results.jsonl" \
    --run "sft_epoch1=${root}/results/sft_epoch1/results.jsonl" \
    --run "clue_epoch1=${root}/results/clue_epoch1/results.jsonl" \
    --reference base --output-dir "${root}/comparison_all"
}

wait_training
smoke OmniVideoBench "${omni}" "${run}/checkpoint-135"
run_eval OmniVideoBench "${omni}" "${arm}_epoch3" "${run}/checkpoint-135"
if [[ "${arm}" == sft ]]; then
  eval_arm=base
  run_eval OmniVideoBench "${omni}" base -
  eval_arm=sft
fi
run_eval OmniVideoBench "${omni}" "${arm}_epoch2" "${run}/checkpoint-90"
run_eval OmniVideoBench "${omni}" "${arm}_epoch1" "${run}/checkpoint-45"

# This barrier enforces "finish the OmniVideoBench 200-question evaluation,
# then start WorldSense" across both arms, while allowing SFT to start early.
wait_barrier "${omni}" OmniVideoBench
if [[ "${arm}" == sft ]]; then compare_all "${omni}" OmniVideoBench; fi

smoke WorldSense "${world}" "${run}/checkpoint-135"
run_eval WorldSense "${world}" "${arm}_epoch3" "${run}/checkpoint-135"
if [[ "${arm}" == sft ]]; then
  eval_arm=base
  run_eval WorldSense "${world}" base -
  eval_arm=sft
fi
run_eval WorldSense "${world}" "${arm}_epoch2" "${run}/checkpoint-90"
run_eval WorldSense "${world}" "${arm}_epoch1" "${run}/checkpoint-45"
if [[ "${arm}" == sft ]]; then
  wait_barrier "${world}" WorldSense
  compare_all "${world}" WorldSense
fi
status "COMPLETED_${arm^^}_ALL_EVALUATIONS"
