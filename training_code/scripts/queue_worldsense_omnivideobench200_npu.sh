#!/usr/bin/env bash
# Run on npu24 worker-1. Wait for both three-epoch trainings, then evaluate
# the same 200 OmniVideoBench questions in epoch order 3 -> 2 -> 1.
set -euo pipefail

repo="/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd"
train_root="${repo}/training_runs/worldsense_openqa_20260929"
root="${train_root}/eval_omnivideobench200"
sft="${train_root}/outputs/sft_formal/v0-20260929-174408"
clue="${train_root}/outputs/clue_formal/v0-20260929-174734"
runner="${repo}/training_code/scripts/run_worldsense_omnivideobench200_npu.sh"
aggregate="${repo}/training_code/scripts/aggregate_video_odyssey_training_eval.py"
python_bin="/home/ma-user/anaconda3/envs/PyTorch-2.9.0/bin/python"
data="${root}/data/omnivideobench.answer_free.jsonl"
labels="${root}/data/omnivideobench.labels.jsonl"
mkdir -p "${root}"
exec 9> "${root}/sequence.lock"
flock -n 9 || { echo "evaluation queue is already running" >&2; exit 1; }

status() {
  printf 'status=%s\ntime=%s\n' "$1" "$(date --iso-8601=seconds)" > "${root}/STATUS.txt"
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$1"
}
on_exit() {
  local rc=$?
  if ((rc != 0)); then status "FAILED exit_code=${rc}"; fi
}
trap on_exit EXIT

training_complete() {
  "${python_bin}" - "${sft}" "${clue}" <<'PY'
import json
import sys
from pathlib import Path

for raw in sys.argv[1:]:
    run = Path(raw)
    checkpoint = run / "checkpoint-135"
    for name in ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json"):
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(1)
    try:
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if state.get("global_step") != 135:
            raise SystemExit(1)
        records = [json.loads(line) for line in (run / "logging.jsonl").read_text().splitlines() if line.strip()]
        if not any("train_runtime" in row and row.get("global_step/max_steps") == "135/135" for row in records):
            raise SystemExit(1)
    except (OSError, ValueError, json.JSONDecodeError):
        raise SystemExit(1)
PY
}

run_one() {
  local label="$1" arm="$2" adapter="$3" output="${root}/results/${1}"
  if [[ -s "${output}/summary.json" && -s "${output}/results.jsonl" ]]; then
    status "SKIPPED ${label} (complete)"
    return
  fi
  status "RUNNING ${label}"
  bash "${runner}" "${arm}" "${adapter}" "${output}"
  status "COMPLETED ${label}"
}

compare_epoch() {
  local epoch="$1"
  status "AGGREGATING epoch${epoch}"
  PYTHONPATH="${repo}/training_code/src" "${python_bin}" "${aggregate}" \
    --benchmark OmniVideoBench --labels "${labels}" --dataset "${data}" \
    --run "base=${root}/results/base/results.jsonl" \
    --run "sft_epoch${epoch}=${root}/results/sft_epoch${epoch}/results.jsonl" \
    --run "clue_epoch${epoch}=${root}/results/clue_epoch${epoch}/results.jsonl" \
    --reference base --output-dir "${root}/comparison_epoch${epoch}"
}

[[ -s "${data}" && -s "${labels}" ]] || { echo "sampled data is missing" >&2; exit 2; }
status WAITING_FOR_EPOCH3_TRAINING
while ! training_complete; do
  for item in sft clue; do
    if [[ "${item}" == sft ]]; then
      run="${sft}"
      log="${train_root}/sft_worker-10.log"
    else
      run="${clue}"
      log="${train_root}/clue_worker-1.log"
    fi
    # The SFT run finishes hours before CLUE. Its log then stops growing by
    # design, so only treat a *pending* run's old log as a failure signal.
    if [[ -s "${run}/logging.jsonl" ]] && \
       grep -q '"train_runtime".*"global_step/max_steps": "135/135"' "${run}/logging.jsonl"; then
      continue
    fi
    if [[ -f "${log}" ]]; then
      age=$(( $(date +%s) - $(stat -c %Y "${log}") ))
      if ((age > 7200)); then
        echo "training log is stale for over two hours: ${log}" >&2
        exit 1
      fi
    fi
  done
  sleep "${OMNI_OPSD_EVAL_POLL_SECONDS:-60}"
done
status TRAINING_COMPLETE
while pgrep -f -- "${train_root}/data/clue_openqa.jsonl" >/dev/null; do
  status WAITING_FOR_CLUE_PROCESSES_TO_EXIT
  sleep "${OMNI_OPSD_EVAL_POLL_SECONDS:-60}"
done

# A one-question base smoke validates NPU inference and the scoring path.
if [[ ! -s "${root}/smoke_base/summary.json" ]]; then
  status RUNNING_SMOKE
  OMNI_OPSD_EVAL_DATASET="${root}/smoke_data/omnivideobench.answer_free.jsonl" \
  OMNI_OPSD_EVAL_LABELS="${root}/smoke_data/omnivideobench.labels.jsonl" \
  OMNI_OPSD_EVAL_EXPECTED_ROWS=1 OMNI_OPSD_EVAL_NPROC=1 OMNI_OPSD_EVAL_DEVICES=0 \
  OMNI_OPSD_EVAL_MASTER_PORT=29779 \
    bash "${runner}" base - "${root}/smoke_base"
fi

# Do the three-epoch adapters first, as requested. The base run then supplies
# their reference score; subsequent epochs reuse this exact base result.
run_one sft_epoch3 sft "${sft}/checkpoint-135"
run_one clue_epoch3 clue_opsd "${clue}/checkpoint-135"
run_one base base -
compare_epoch 3
run_one sft_epoch2 sft "${sft}/checkpoint-90"
run_one clue_epoch2 clue_opsd "${clue}/checkpoint-90"
compare_epoch 2
run_one sft_epoch1 sft "${sft}/checkpoint-45"
run_one clue_epoch1 clue_opsd "${clue}/checkpoint-45"
compare_epoch 1

status AGGREGATING_ALL
PYTHONPATH="${repo}/training_code/src" "${python_bin}" "${aggregate}" \
  --benchmark OmniVideoBench --labels "${labels}" --dataset "${data}" \
  --run "base=${root}/results/base/results.jsonl" \
  --run "sft_epoch3=${root}/results/sft_epoch3/results.jsonl" \
  --run "clue_epoch3=${root}/results/clue_epoch3/results.jsonl" \
  --run "sft_epoch2=${root}/results/sft_epoch2/results.jsonl" \
  --run "clue_epoch2=${root}/results/clue_epoch2/results.jsonl" \
  --run "sft_epoch1=${root}/results/sft_epoch1/results.jsonl" \
  --run "clue_epoch1=${root}/results/clue_epoch1/results.jsonl" \
  --reference base --output-dir "${root}/comparison_all"
status COMPLETED_ALL
