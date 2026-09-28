#!/usr/bin/env bash
set -euo pipefail

# CUDA evaluation entry point for the local OmniVideoBench release.  It keeps
# the strict ID/media checks from run_video_odyssey_training_eval.sh while
# using the 1,000-row OmniVideoBench contract and audio-enabled Qwen-Omni
# inputs.

project_root="${OMNI_OPSD_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_env="${OMNI_OPSD_ENV:-/share/home/ylhu/.conda/envs/omniopsd_train}"
swift_root="${OMNI_OPSD_MS_SWIFT_ROOT:-${project_root}/ms-swift}"
python_bin="${OMNI_OPSD_PYTHON_BIN:-${python_env}/bin/python}"
if [[ -n "${OMNI_OPSD_SWIFT_BIN:-}" ]]; then
  swift_cmd=("${OMNI_OPSD_SWIFT_BIN}")
elif [[ -x "${python_env}/bin/swift" ]]; then
  swift_cmd=("${python_env}/bin/swift")
else
  swift_cmd=("${python_bin}" -m swift.cli.main)
fi
model="${OMNI_OPSD_MODEL:?set OMNI_OPSD_MODEL}"
dataset="${OMNI_OPSD_EVAL_DATASET:-${project_root}/data/OmniVideoBench/omnivideobench.answer_free.jsonl}"
labels="${OMNI_OPSD_EVAL_LABELS:-${project_root}/data/OmniVideoBench/omnivideobench.labels.jsonl}"
output_dir="${OMNI_OPSD_EVAL_OUTPUT_DIR:?set OMNI_OPSD_EVAL_OUTPUT_DIR}"
adapter="${OMNI_OPSD_EVAL_ADAPTER:-}"
arm="${OMNI_OPSD_EVAL_ARM:-sft}"
devices="${OMNI_OPSD_CUDA_DEVICES:-0,1,2,3}"
nproc="${OMNI_OPSD_NPROC_PER_NODE:-4}"
expected_rows="${OMNI_OPSD_EVAL_EXPECTED_ROWS:-1000}"
audio_flag="${OMNI_OPSD_EVAL_USE_AUDIO_IN_VIDEO:-1}"
attn_impl="${OMNI_OPSD_EVAL_ATTN_IMPL:-sdpa}"
master_addr="${MASTER_ADDR:-127.0.0.1}"
master_port="${MASTER_PORT:-29921}"

[[ "${arm}" == "base" || "${arm}" == "full" || "${arm}" == "sft" || "${arm}" == "opsd" ]] || {
  echo "OMNI_OPSD_EVAL_ARM must be base, full, sft, or opsd" >&2
  exit 2
}
[[ "${audio_flag}" == "1" ]] || {
  echo "OmniVideoBench evaluation requires OMNI_OPSD_EVAL_USE_AUDIO_IN_VIDEO=1" >&2
  exit 2
}
IFS=',' read -r -a device_list <<< "${devices}"
[[ "${#device_list[@]}" -eq "${nproc}" ]] || {
  echo "nproc=${nproc} does not match CUDA devices=${devices}" >&2
  exit 2
}
for path in "${model}" "${dataset}" "${labels}" "${swift_root}/swift/cli/infer.py"; do
  [[ -e "${path}" ]] || { echo "Required path is missing: ${path}" >&2; exit 2; }
done
if [[ "${arm}" == "base" || "${arm}" == "full" ]]; then
  [[ -z "${adapter}" ]] || { echo "base evaluation cannot use an adapter" >&2; exit 2; }
else
  for file in adapter_config.json adapter_model.safetensors trainer_state.json; do
    [[ -s "${adapter}/${file}" ]] || { echo "Incomplete adapter: ${adapter}/${file}" >&2; exit 2; }
  done
fi

PYTHONPATH="${project_root}/src:${project_root}:${swift_root}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${python_bin}" - "${dataset}" "${labels}" "${expected_rows}" <<'PY'
import json
import os
import sys

dataset, labels, expected_text = sys.argv[1:]
expected = int(expected_text)
rows = [json.loads(line) for line in open(dataset, encoding='utf-8') if line.strip()]
gold = [json.loads(line) for line in open(labels, encoding='utf-8') if line.strip()]
if len(rows) != expected or len(gold) != expected:
    raise SystemExit(f'cardinality mismatch: dataset={len(rows)} labels={len(gold)} expected={expected}')
row_ids = []
for row in rows:
    sample_id = str(row.get('case_id') or row.get('prompt_id') or '')
    row_ids.append(sample_id)
    if not sample_id or row.get('benchmark') != 'OmniVideoBench':
        raise SystemExit(f'invalid OmniVideoBench row id/benchmark: {sample_id}')
    forbidden = {'answer', 'solution', 'correct_option', 'reasoning_steps', 'teacher_prompt', 'teacher_videos'}
    leaked = sorted(forbidden.intersection(row))
    if leaked:
        raise SystemExit(f'answer/teacher leakage in {sample_id}: {leaked}')
    messages = row.get('messages') or []
    if len(messages) != 1 or messages[0].get('role') != 'user':
        raise SystemExit(f'{sample_id} must contain exactly one user message')
    videos = row.get('videos') or []
    if len(videos) != 1 or not isinstance(videos[0], dict):
        raise SystemExit(f'{sample_id} must contain one structured video')
    media_path = videos[0].get('video')
    if not media_path or not os.path.isfile(media_path) or os.path.getsize(media_path) == 0:
        raise SystemExit(f'{sample_id} references missing video: {media_path}')
    contract = row.get('sampling_contract') or {}
    if not contract.get('held_out_evaluation') or contract.get('use_audio_in_video') is not True:
        raise SystemExit(f'{sample_id} has an invalid audio/held-out contract')
if len(set(row_ids)) != expected:
    raise SystemExit('evaluation case IDs are not unique')
label_ids = [str(row.get('sample_id') or '') for row in gold]
if set(row_ids) != set(label_ids) or len(set(label_ids)) != expected:
    raise SystemExit('evaluation and label IDs differ')
for row in gold:
    if str(row.get('answer', '')).strip().upper() not in {'A', 'B', 'C', 'D'}:
        raise SystemExit(f"invalid label for {row.get('sample_id')}")
print(f'validated OmniVideoBench rows={expected} no_answer_leakage=true use_audio_in_video=true')
PY

mkdir -p "${output_dir}"
result_path="${output_dir}/results.jsonl"
[[ ! -e "${result_path}" ]] || { echo "Refusing to overwrite existing result: ${result_path}" >&2; exit 2; }
adapter_args=()
[[ -z "${adapter}" ]] || adapter_args=(--adapters "${adapter}")
dataset_sha256="$(sha256sum "${dataset}" | awk '{print $1}')"
labels_sha256="$(sha256sum "${labels}" | awk '{print $1}')"
printf '%s\n' \
  "benchmark=OmniVideoBench" \
  "arm=${arm}" \
  "model=${model}" \
  "adapter=${adapter}" \
  "dataset=${dataset}" \
  "dataset_sha256=${dataset_sha256}" \
  "labels=${labels}" \
  "labels_sha256=${labels_sha256}" \
  "expected_rows=${expected_rows}" \
  "devices=${devices}" \
  "nproc=${nproc}" \
  "use_audio_in_video=true" \
  "attn_impl=${attn_impl}" \
  "generation=greedy_max_new_tokens_8" \
  > "${output_dir}/RUN_CLASSIFICATION.txt"

export CUDA_VISIBLE_DEVICES="${devices}"
export ASCEND_RT_VISIBLE_DEVICES="${devices}"
export NPROC_PER_NODE="${nproc}"
export MASTER_ADDR="${master_addr}"
export MASTER_PORT="${master_port}"
export USE_AUDIO_IN_VIDEO=1
export ENABLE_AUDIO_OUTPUT=0
export PYTHONPATH="${project_root}/src:${project_root}:${swift_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${python_env}/bin:${PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"
export DECORD_NUM_THREADS="${DECORD_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMNI_OPSD_DIRECT_LOCAL_VIDEO="${OMNI_OPSD_DIRECT_LOCAL_VIDEO:-1}"

# The Qwen-Omni audio path uses audioread, which looks for an ffmpeg or
# avconv executable on PATH.  Some GPU-node shells do not inherit a system
# multimedia binary even though the shared conda installation contains one.
# Add a deterministic user-space fallback without changing the audio contract.
if ! command -v ffmpeg >/dev/null 2>&1 && ! command -v avconv >/dev/null 2>&1; then
  for ffmpeg_candidate in \
    "${OMNI_OPSD_FFMPEG_BIN:-}" \
    "/share/home/ylhu/.conda/envs/omniagent_gyh/bin/ffmpeg" \
    "/share/home/ylhu/.conda/envs/omniopsd_train/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"; do
    if [[ -n "${ffmpeg_candidate}" && -x "${ffmpeg_candidate}" ]]; then
      export PATH="$(dirname "${ffmpeg_candidate}"):${PATH}"
      break
    fi
  done
fi

"${swift_cmd[@]}" infer \
  --model "${model}" \
  "${adapter_args[@]}" \
  --val_dataset "${dataset}" \
  --result_path "${result_path}" \
  --infer_backend transformers \
  --max_batch_size 1 \
  --max_new_tokens 8 \
  --temperature 0 \
  --stream false \
  --torch_dtype bfloat16 \
  --attn_impl "${attn_impl}" \
  --max_length 32768 \
  --dataset_num_proc 1 \
  --val_dataset_shuffle false \
  --seed 20260904 \
  > "${output_dir}/infer.log" 2>&1

PYTHONPATH="${project_root}/src:${project_root}:${swift_root}" \
  "${python_bin}" "${project_root}/scripts/summarize_video_odyssey_training_eval.py" \
  --results "${result_path}" \
  --labels "${labels}" \
  --dataset "${dataset}" \
  --output "${output_dir}/summary.json" \
  --arm "${arm}" \
  --adapter "${adapter}" \
  > "${output_dir}/summarize.log" 2>&1
