#!/usr/bin/env bash
# Evaluate the fixed, video-disjoint Tier A/B/D WorldSense set on npu24 workers 0-2.
# Run on worker-0. Each model uses eight independent single-card inference shards.
set -euo pipefail

repo=/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd
train_root="${repo}/training_runs/worldsense_openqa_20260929"
root="${train_root}/eval_worldsense_tier_abd518"
data="${root}/data/worldsense.answer_free.jsonl"
labels="${root}/data/worldsense.labels.jsonl"
sft="${train_root}/outputs/sft_formal/v0-20260929-174408"
clue="${train_root}/outputs/clue_formal/v0-20260929-174734"
runner="${repo}/training_code/scripts/run_worldsense_omnivideobench200_npu.sh"
aggregate="${repo}/training_code/scripts/aggregate_video_odyssey_training_eval.py"
python_bin=/home/ma-user/anaconda3/envs/PyTorch-2.9.0/bin/python
host_w1=172.16.12.237
host_w2=172.16.12.70

mkdir -p "${root}/logs" "${root}/results"
exec 9> "${root}/sequence.lock"
flock -n 9 || { echo 'Tier A/B/D evaluation queue already running' >&2; exit 1; }

status() {
  local temporary="${root}/STATUS.${BASHPID}.tmp"
  printf 'status=%s\ntime=%s\n' "$1" "$(date --iso-8601=seconds)" > "${temporary}"
  mv -f "${temporary}" "${root}/STATUS.txt"
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$1"
}
on_exit() {
  local rc=$?
  if ((rc != 0)); then status "FAILED exit_code=${rc}"; fi
}
trap on_exit EXIT

run_one() {
  local label="$1" arm="$2" adapter="$3" host="$4"
  local output="${root}/results/${label}"
  if [[ -s "${output}/summary.json" && -s "${output}/results.jsonl" ]]; then
    status "SKIPPED_${label}_COMPLETE"
    return 0
  fi
  status "RUNNING_${label}_on_${host}"
  if [[ "${host}" == local ]]; then
    OMNI_OPSD_EVAL_BENCHMARK=WorldSense \
    OMNI_OPSD_EVAL_DATASET="${data}" \
    OMNI_OPSD_EVAL_LABELS="${labels}" \
    OMNI_OPSD_EVAL_EXPECTED_ROWS=518 \
      bash "${runner}" "${arm}" "${adapter}" "${output}" \
      > "${root}/logs/${label}.log" 2>&1
  else
    ssh -F /dev/null -p 2222 -o BatchMode=yes -o ConnectTimeout=8 "${host}" \
      "env OMNI_OPSD_EVAL_BENCHMARK=WorldSense OMNI_OPSD_EVAL_DATASET='${data}' OMNI_OPSD_EVAL_LABELS='${labels}' OMNI_OPSD_EVAL_EXPECTED_ROWS=518 bash '${runner}' '${arm}' '${adapter}' '${output}'" \
      > "${root}/logs/${label}.log" 2>&1
  fi
  # Shared storage may expose the remote worker's final rename a few seconds
  # after ssh exits. Wait for both files before declaring a failed run.
  for ((attempt=0; attempt<30; attempt++)); do
    if [[ -s "${output}/summary.json" && -s "${output}/results.jsonl" ]]; then
      status "COMPLETED_${label}"
      return 0
    fi
    sleep 1
  done
  echo "missing completed result after remote worker exit: ${label}" >&2
  return 1
}

phase() {
  local name="$1"
  shift
  local -a pids=()
  local failed=0
  status "STARTING_${name}"
  while (($#)); do
    run_one "$1" "$2" "$3" "$4" &
    pids+=("$!")
    shift 4
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then failed=1; fi
  done
  if ((failed)); then
    echo "${name} failed; inspect ${root}/logs and results/*/shards/card_*/infer.log" >&2
    return 1
  fi
  status "COMPLETED_${name}"
}

[[ -s "${data}" && -s "${labels}" ]] || { echo 'missing frozen evaluation input' >&2; exit 2; }
phase epoch3_and_base \
  base base - local \
  sft_epoch3 sft "${sft}/checkpoint-135" "${host_w1}" \
  clue_epoch3 clue_opsd "${clue}/checkpoint-135" "${host_w2}"
phase epoch2 \
  sft_epoch2 sft "${sft}/checkpoint-90" local \
  clue_epoch2 clue_opsd "${clue}/checkpoint-90" "${host_w1}"
phase epoch1 \
  sft_epoch1 sft "${sft}/checkpoint-45" local \
  clue_epoch1 clue_opsd "${clue}/checkpoint-45" "${host_w1}"

status AGGREGATING_ALL
PYTHONPATH="${repo}/training_code/src" "${python_bin}" "${aggregate}" \
  --benchmark WorldSense --labels "${labels}" --dataset "${data}" \
  --tier-metrics "${repo}/data/screening/per_question.jsonl" \
  --run "base=${root}/results/base/results.jsonl" \
  --run "sft_epoch3=${root}/results/sft_epoch3/results.jsonl" \
  --run "clue_epoch3=${root}/results/clue_epoch3/results.jsonl" \
  --run "sft_epoch2=${root}/results/sft_epoch2/results.jsonl" \
  --run "clue_epoch2=${root}/results/clue_epoch2/results.jsonl" \
  --run "sft_epoch1=${root}/results/sft_epoch1/results.jsonl" \
  --run "clue_epoch1=${root}/results/clue_epoch1/results.jsonl" \
  --reference base --output-dir "${root}/comparison_all" \
  > "${root}/logs/aggregate.log" 2>&1
status COMPLETED_ALL
