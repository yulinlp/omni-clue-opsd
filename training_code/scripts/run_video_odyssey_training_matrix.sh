#!/usr/bin/env bash
set -euo pipefail

# Launch four two-NPU arms concurrently.  A one-step smoke gate must pass for
# every arm before fresh 300-step runs are admitted.

project_root="${OMNI_OPSD_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
matrix_root="${OMNI_OPSD_MATRIX_ROOT:?set OMNI_OPSD_MATRIX_ROOT}"
dataset_root="${OMNI_OPSD_MATRIX_DATASET_ROOT:?set OMNI_OPSD_MATRIX_DATASET_ROOT}"
arm_launcher="${project_root}/scripts/run_video_odyssey_training_arm.sh"
mkdir -p "${matrix_root}/logs" "${matrix_root}/pids"

arms=(sft grpo opsd clue_opsd)
devices=(0,1 2,3 4,5 6,7)
ports=(29801 29802 29803 29804)

dataset_for() {
  printf '%s/video_odyssey_train.%s.jsonl' "${dataset_root}" "$1"
}

launch_phase() {
  local phase="$1"
  local steps="$2"
  local save_steps="$3"
  local -a phase_pids=()
  local index arm output log pid
  for index in "${!arms[@]}"; do
    arm="${arms[$index]}"
    output="${matrix_root}/${phase}/${arm}"
    log="${matrix_root}/logs/${phase}.${arm}.log"
    mkdir -p "${output}"
    nohup env \
      MASTER_PORT="${ports[$index]}" \
      ASCEND_RT_VISIBLE_DEVICES="${devices[$index]}" \
      OMNI_OPSD_ARM="${arm}" \
      OMNI_OPSD_DATASET="$(dataset_for "${arm}")" \
      OMNI_OPSD_OUTPUT_DIR="${output}" \
      OMNI_OPSD_NPROC_PER_NODE=2 \
      OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS=16 \
      OMNI_OPSD_MAX_STEPS="${steps}" \
      OMNI_OPSD_SAVE_STEPS="${save_steps}" \
      OMNI_OPSD_EXPERIMENT_LABEL="video_odyssey_${phase}_${arm}" \
      bash "${arm_launcher}" >"${log}" 2>&1 &
    pid=$!
    phase_pids+=("${pid}")
    printf '%s\n' "${pid}" > "${matrix_root}/pids/${phase}.${arm}.pid"
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "${phase}" "${arm}" "${pid}" "${devices[$index]}" \
      >> "${matrix_root}/launch_manifest.tsv"
  done

  local failed=0 status
  for index in "${!arms[@]}"; do
    if wait "${phase_pids[$index]}"; then
      status=0
    else
      status=$?
      failed=1
    fi
    printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "${phase}" "${arms[$index]}" "${status}" \
      >> "${matrix_root}/completion_manifest.tsv"
  done
  return "${failed}"
}

printf 'timestamp\tphase\tarm\tpid\tdevices\n' > "${matrix_root}/launch_manifest.tsv"
printf 'timestamp\tphase\tarm\texit_code\n' > "${matrix_root}/completion_manifest.tsv"

if ! launch_phase smoke 1 1; then
  echo "At least one matrix smoke arm failed; full runs were not launched" >&2
  exit 1
fi

launch_phase full "${OMNI_OPSD_FULL_MAX_STEPS:-300}" "${OMNI_OPSD_FULL_SAVE_STEPS:-25}"
