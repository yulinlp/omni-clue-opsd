#!/usr/bin/env bash
set -euo pipefail

project_root="${OMNI_OPSD_PROJECT_ROOT:-/share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907}"
python_env="${OMNI_OPSD_ENV:-/share/home/ylhu/.conda/envs/omniopsd_train}"
root="${OMNI_OPSD_OPSD_EVAL_ROOT:?set OMNI_OPSD_OPSD_EVAL_ROOT}"
dataset="${OMNI_OPSD_EVAL_DATASET:-${project_root}/data/OmniVideoBench/omnivideobench.answer_free.jsonl}"
labels="${OMNI_OPSD_EVAL_LABELS:-${project_root}/data/OmniVideoBench/omnivideobench.labels.jsonl}"
ms_swift_root="${OMNI_OPSD_MS_SWIFT_ROOT:-${project_root}/ms-swift}"
python_bin="${OMNI_OPSD_PYTHON_BIN:-${python_env}/bin/python}"

base3b="/share/home/ylhu/models/Qwen2.5-Omni-3B"
base7b="/share/home/ylhu/models/Qwen2.5-Omni-7B"
adapter3b="${project_root}/output/xfu_gpu05_opsd_formal_1epoch_20260909_023127/qwen25_omni_3b/opsd/v0-20260909-141329/checkpoint-78"
adapter7b="${project_root}/output/xfu_gpu05_opsd_formal_1epoch_20260909_023127/qwen25_omni_7b/opsd/v0-20260909-023200/checkpoint-78"
base_result3b="${project_root}/output/omnivideobench_qwen25_omni3b_base_20260909/results.jsonl"
base_result7b="${project_root}/output/omnivideobench_qwen25_omni7b_base_20260909/results.jsonl"

mkdir -p "${root}"
exec > >(tee -a "${root}/sequence.log") 2>&1

printf '%s\n' \
  "benchmark=OmniVideoBench" \
  "dataset=${dataset}" \
  "labels=${labels}" \
  "base_results_reused=true" \
  "base_3b_results=${base_result3b}" \
  "base_7b_results=${base_result7b}" \
  "opsd_3b_adapter=${adapter3b}" \
  "opsd_7b_adapter=${adapter7b}" \
  "cuda_devices=${OMNI_OPSD_CUDA_DEVICES:-0,1,2,3}" \
  "nproc_per_node=${OMNI_OPSD_NPROC_PER_NODE:-4}" \
  "use_audio_in_video=1" \
  "started_at=$(date --iso-8601=seconds)" \
  > "${root}/SEQUENCE_CONFIG.txt"

[[ -s "${dataset}" ]] || { echo "missing dataset: ${dataset}" >&2; exit 2; }
[[ -s "${labels}" ]] || { echo "missing labels: ${labels}" >&2; exit 2; }
[[ -s "${base_result3b}" && -s "${base_result7b}" ]] || { echo "missing existing base results" >&2; exit 2; }
for path in "${adapter3b}" "${adapter7b}"; do
  for file in adapter_config.json adapter_model.safetensors trainer_state.json; do
    [[ -s "${path}/${file}" ]] || { echo "missing adapter file: ${path}/${file}" >&2; exit 2; }
  done
done

run_one() {
  local label="$1" model="$2" adapter="$3" out="$4" port="$5"
  local status_file="${out}/STATUS.txt"
  local arm=opsd
  mkdir -p "${out}"
  if [[ -s "${out}/summary.json" && -s "${out}/results.jsonl" ]]; then
    echo "already complete: ${label}"
    return 0
  fi
  printf '%s\n' \
    "label=${label}" "model=${model}" "adapter=${adapter}" \
    "started_at=$(date --iso-8601=seconds)" "status=RUNNING" > "${status_file}"
  echo "starting ${label}"
  set +e
  env \
    OMNI_OPSD_PROJECT_ROOT="${project_root}" \
    OMNI_OPSD_ENV="${python_env}" \
    OMNI_OPSD_MS_SWIFT_ROOT="${ms_swift_root}" \
    OMNI_OPSD_PYTHON_BIN="${python_bin}" \
    OMNI_OPSD_MODEL="${model}" \
    OMNI_OPSD_EVAL_ADAPTER="${adapter}" \
    OMNI_OPSD_EVAL_ARM=opsd \
    OMNI_OPSD_EVAL_DATASET="${dataset}" \
    OMNI_OPSD_EVAL_LABELS="${labels}" \
    OMNI_OPSD_EVAL_OUTPUT_DIR="${out}" \
    OMNI_OPSD_CUDA_DEVICES="${OMNI_OPSD_CUDA_DEVICES:-0,1,2,3}" \
    OMNI_OPSD_NPROC_PER_NODE="${OMNI_OPSD_NPROC_PER_NODE:-4}" \
    OMNI_OPSD_EVAL_EXPECTED_ROWS=1000 \
    OMNI_OPSD_EVAL_USE_AUDIO_IN_VIDEO=1 \
    OMNI_OPSD_EVAL_ATTN_IMPL="${OMNI_OPSD_EVAL_ATTN_IMPL:-sdpa}" \
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" \
    MASTER_PORT="${port}" \
    DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    OMNI_OPSD_DIRECT_LOCAL_VIDEO=1 \
    bash "${project_root}/scripts/run_omnivideobench_eval.sh"
  local rc=$?
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    printf '%s\n' "ended_at=$(date --iso-8601=seconds)" "exit_code=${rc}" "status=FAILED" >> "${status_file}"
    echo "${label} failed with exit_code=${rc}" >&2
    return "${rc}"
  fi
  printf '%s\n' "ended_at=$(date --iso-8601=seconds)" "exit_code=0" "status=COMPLETED" >> "${status_file}"
  echo "completed ${label}"
}

aggregate_one() {
  local label="$1" base_result="$2" opsd_result="$3" out="$4"
  mkdir -p "${out}"
  "${python_bin}" "${project_root}/scripts/aggregate_video_odyssey_training_eval.py" \
    --labels "${labels}" --dataset "${dataset}" \
    --run "base=${base_result}" --run "opsd=${opsd_result}" \
    --reference base --output-dir "${out}"
  echo "comparison completed ${label}: ${out}"
}

run_one qwen25_omni3b_opsd "${base3b}" "${adapter3b}" "${root}/qwen25_omni3b_opsd" 29992
aggregate_one qwen25_omni3b "${base_result3b}" "${root}/qwen25_omni3b_opsd/results.jsonl" "${root}/comparison_3b"
run_one qwen25_omni7b_opsd "${base7b}" "${adapter7b}" "${root}/qwen25_omni7b_opsd" 29993
aggregate_one qwen25_omni7b "${base_result7b}" "${root}/qwen25_omni7b_opsd/results.jsonl" "${root}/comparison_7b"
printf '%s\n' \
  "sequence_completed_at=$(date --iso-8601=seconds)" \
  "qwen25_omni3b_opsd=COMPLETED" \
  "qwen25_omni7b_opsd=COMPLETED" \
  > "${root}/SEQUENCE_COMPLETED.txt"
echo "OmniVideoBench OPSD sequence completed"
