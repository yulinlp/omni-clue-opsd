#!/usr/bin/env bash
set -euo pipefail

project_root="/share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907"
job_id="279089"
dataset="${project_root}/data/OmniVideoBench/omnivideobench.answer_free.jsonl"
labels="${project_root}/data/OmniVideoBench/omnivideobench.labels.jsonl"
base3b="/share/home/ylhu/models/Qwen2.5-Omni-3B"
base7b="/share/home/ylhu/models/Qwen2.5-Omni-7B"
adapter3b="${project_root}/output/qwen25_omni3b_sft_1epoch_gpu02_20260908_193353/v0-20260908-200025/checkpoint-157"
adapter7b="${project_root}/output/gap5000_formal_20260908_010630/sft/v0-20260908-010643/checkpoint-175"
common_env=(
  "OMNI_OPSD_EVAL_DATASET=${dataset}"
  "OMNI_OPSD_EVAL_LABELS=${labels}"
  "OMNI_OPSD_EVAL_EXPECTED_ROWS=1000"
  "OMNI_OPSD_EVAL_USE_AUDIO_IN_VIDEO=1"
  "OMNI_OPSD_EVAL_ATTN_IMPL=sdpa"
  "DECORD_EOF_RETRY_MAX=20480"
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
)

wait_for_summary() {
  local summary="$1"
  local infer_log="$2"
  while [[ ! -s "${summary}" ]]; do
    if [[ -s "${infer_log}" ]] && rg -q 'ChildFailedError|torch\.OutOfMemoryError|ValueError: torchvision returned no frames' "${infer_log}"; then
      echo "evaluation failed before summary: ${infer_log}" >&2
      return 1
    fi
    sleep 45
  done
}

run_arm() {
  local arm="$1"
  local model="$2"
  local adapter="$3"
  local devices="$4"
  local nproc="$5"
  local output_dir="$6"
  local summary="${output_dir}/summary.json"
  local infer_log="${output_dir}/infer.log"

  if [[ -s "${summary}" ]]; then
    echo "already complete: ${output_dir}"
    return 0
  fi
  mkdir -p "${output_dir}"
  echo "starting arm=${arm} model=${model} devices=${devices} nproc=${nproc}"
  local adapter_env=""
  if [[ -n "${adapter}" ]]; then
    adapter_env="OMNI_OPSD_EVAL_ADAPTER=${adapter}"
  fi
  env \
    "${common_env[@]}" \
    "OMNI_OPSD_MODEL=${model}" \
    "OMNI_OPSD_EVAL_ARM=${arm}" \
    "OMNI_OPSD_EVAL_OUTPUT_DIR=${output_dir}" \
    "OMNI_OPSD_CUDA_DEVICES=${devices}" \
    "OMNI_OPSD_NPROC_PER_NODE=${nproc}" \
    ${adapter_env} \
    srun --overlap --jobid="${job_id}" --nodelist=gpu02 --gres=gpu:4 --export=ALL \
      bash -lc "cd '${project_root}' && exec bash scripts/run_omnivideobench_eval.sh"
  wait_for_summary "${summary}" "${infer_log}"
  echo "completed arm=${arm} output=${output_dir}"
}

echo "waiting for the already-running 3B SFT evaluation"
wait_for_summary \
  "${project_root}/output/omnivideobench_qwen25_omni3b_sft_checkpoint157_20260909/summary.json" \
  "${project_root}/output/omnivideobench_qwen25_omni3b_sft_checkpoint157_20260909/infer.log"

run_arm sft "${base3b}" "${adapter3b}" "0,1,2,3" 4 \
  "${project_root}/output/omnivideobench_qwen25_omni3b_sft_checkpoint157_20260909"
run_arm base "${base3b}" "" "0,1,2,3" 4 \
  "${project_root}/output/omnivideobench_qwen25_omni3b_base_20260909"
run_arm base "${base7b}" "" "0,1,3" 3 \
  "${project_root}/output/omnivideobench_qwen25_omni7b_base_20260909"
run_arm sft "${base7b}" "${adapter7b}" "0,1,3" 3 \
  "${project_root}/output/omnivideobench_qwen25_omni7b_sft_checkpoint175_20260909"
echo "OmniVideoBench matrix completed"
