#!/usr/bin/env bash
set -euo pipefail

# Run one or more full-node (8 NPU) arms on a single ModelArts worker.  Every
# arm must pass a fresh one-step smoke before any full run in this queue starts.

project_root="${OMNI_OPSD_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
queue_root="${OMNI_OPSD_QUEUE_ROOT:?set OMNI_OPSD_QUEUE_ROOT}"
dataset_root="${OMNI_OPSD_MATRIX_DATASET_ROOT:?set OMNI_OPSD_MATRIX_DATASET_ROOT}"
queue_arms_csv="${OMNI_OPSD_QUEUE_ARMS:?set OMNI_OPSD_QUEUE_ARMS, for example sft,opsd}"
arm_launcher="${project_root}/scripts/run_video_odyssey_training_arm.sh"
IFS=',' read -r -a arms <<< "${queue_arms_csv}"
if [[ "${#arms[@]}" -lt 1 ]]; then
  echo "The arm queue is empty" >&2
  exit 2
fi

for arm in "${arms[@]}"; do
  case "${arm}" in
    sft|grpo|opsd|clue_opsd) ;;
    *) echo "Unsupported queued arm: ${arm}" >&2; exit 2 ;;
  esac
  if [[ ! -s "${dataset_root}/video_odyssey_train.${arm}.jsonl" ]]; then
    echo "Missing dataset for ${arm}" >&2
    exit 2
  fi
done

mkdir -p "${queue_root}/logs" "${queue_root}/pids"
printf 'timestamp\tphase\tarm\texit_code\n' > "${queue_root}/completion_manifest.tsv"

run_arm() {
  local phase="$1"
  local arm="$2"
  local steps save_steps port output log
  if [[ "${phase}" == "smoke" ]]; then
    steps=1
    save_steps=1
  else
    steps="${OMNI_OPSD_FULL_MAX_STEPS:-300}"
    save_steps="${OMNI_OPSD_FULL_SAVE_STEPS:-25}"
  fi
  case "${arm}" in
    sft) port=29901 ;;
    grpo) port=29902 ;;
    opsd) port=29903 ;;
    clue_opsd) port=29904 ;;
  esac
  output="${queue_root}/${phase}/${arm}"
  log="${queue_root}/logs/${phase}.${arm}.log"
  mkdir -p "${output}"
  printf '%s\n' "$$" > "${queue_root}/pids/${phase}.${arm}.queue.pid"
  set +e
  env \
    MASTER_PORT="${port}" \
    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    OMNI_OPSD_ARM="${arm}" \
    OMNI_OPSD_DATASET="${dataset_root}/video_odyssey_train.${arm}.jsonl" \
    OMNI_OPSD_OUTPUT_DIR="${output}" \
    OMNI_OPSD_NPROC_PER_NODE=8 \
    OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS=4 \
    OMNI_OPSD_MAX_STEPS="${steps}" \
    OMNI_OPSD_SAVE_STEPS="${save_steps}" \
    OMNI_OPSD_EXPERIMENT_LABEL="video_odyssey_${phase}_${arm}_8npu" \
    bash "${arm_launcher}" > "${log}" 2>&1
  local status=$?
  set -e
  printf '%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "${phase}" "${arm}" "${status}" \
    >> "${queue_root}/completion_manifest.tsv"
  if [[ "${status}" -ne 0 ]]; then
    echo "${phase} ${arm} failed with exit code ${status}; see ${log}" >&2
    return "${status}"
  fi
}

for arm in "${arms[@]}"; do
  run_arm smoke "${arm}"
done
for arm in "${arms[@]}"; do
  run_arm full "${arm}"
done
