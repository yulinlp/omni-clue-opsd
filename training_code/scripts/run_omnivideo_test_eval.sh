#!/usr/bin/env bash
set -euo pipefail

project_root="${OMNI_OPSD_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_env="${OMNI_OPSD_ENV:-/share/home/ylhu/.conda/envs/omniopsd_train}"
swift_root="${OMNI_OPSD_MS_SWIFT_ROOT:-${project_root}/ms-swift}"
python_bin="${OMNI_OPSD_PYTHON_BIN:-${python_env}/bin/python}"
if [[ -x "${python_env}/bin/swift" ]]; then
  swift_cmd=("${python_env}/bin/swift")
else
  swift_cmd=("${python_bin}" -m swift.cli.main)
fi

model="${OMNI_OPSD_MODEL:?set OMNI_OPSD_MODEL}"
dataset="${OMNI_OPSD_EVAL_DATASET:-${project_root}/output/omnivideo_test_505_eval/omnivideo_test_505.answer_free.jsonl}"
labels="${OMNI_OPSD_EVAL_LABELS:-${project_root}/output/omnivideo_test_505_eval/omnivideo_test_505.labels.jsonl}"
output_dir="${OMNI_OPSD_EVAL_OUTPUT_DIR:?set OMNI_OPSD_EVAL_OUTPUT_DIR}"
arm="${OMNI_OPSD_EVAL_ARM:?set OMNI_OPSD_EVAL_ARM}"
devices="${OMNI_OPSD_CUDA_DEVICES:-0,1,2,3}"
nproc="${OMNI_OPSD_NPROC_PER_NODE:-4}"
attn_impl="${OMNI_OPSD_EVAL_ATTN_IMPL:-sdpa}"

[[ -s "${model}/config.json" ]] || { echo "missing model: ${model}" >&2; exit 2; }
[[ -s "${dataset}" && -s "${labels}" ]] || { echo "missing evaluation files" >&2; exit 2; }
IFS=',' read -r -a device_list <<< "${devices}"
[[ "${#device_list[@]}" -eq "${nproc}" ]] || { echo "nproc/devices mismatch" >&2; exit 2; }

PYTHONPATH="${project_root}/src:${project_root}:${swift_root}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${python_bin}" - "${dataset}" "${labels}" <<'PY'
import json, os, sys
dataset, labels = sys.argv[1:]
rows = [json.loads(x) for x in open(dataset, encoding='utf-8') if x.strip()]
gold = [json.loads(x) for x in open(labels, encoding='utf-8') if x.strip()]
if len(rows) != 505 or len(gold) != 505:
    raise SystemExit(f'expected 505 rows, got dataset={len(rows)} labels={len(gold)}')
ids = []
for row in rows:
    sid = str(row.get('case_id') or row.get('prompt_id') or '')
    ids.append(sid)
    if row.get('benchmark') != 'OmniVideo-Test':
        raise SystemExit(f'wrong benchmark for {sid}')
    if any(k in row for k in ('answer','solution','correct_option','teacher_prompt','teacher_videos')):
        raise SystemExit(f'answer leakage in {sid}')
    if len(row.get('messages') or []) != 1 or len(row.get('videos') or []) != 1:
        raise SystemExit(f'invalid media/message structure for {sid}')
    media = row['videos'][0].get('video')
    if not media or not os.path.isfile(media) or os.path.getsize(media) == 0:
        raise SystemExit(f'missing video for {sid}: {media}')
    if (row.get('sampling_contract') or {}).get('use_audio_in_video') is not True:
        raise SystemExit(f'audio contract disabled for {sid}')
if len(set(ids)) != 505 or {x['sample_id'] for x in gold} != set(ids):
    raise SystemExit('ID mismatch')
print('validated OmniVideo-Test rows=505 audio=true answer_free=true')
PY

mkdir -p "${output_dir}"
result_path="${output_dir}/results.jsonl"
[[ ! -e "${result_path}" ]] || { echo "refusing to overwrite ${result_path}" >&2; exit 2; }
printf '%s\n' \
  "benchmark=OmniVideo-Test" \
  "arm=${arm}" \
  "model=${model}" \
  "dataset=${dataset}" \
  "labels=${labels}" \
  "expected_rows=505" \
  "devices=${devices}" \
  "nproc=${nproc}" \
  "use_audio_in_video=true" \
  "attn_impl=${attn_impl}" \
  "generation=greedy_max_new_tokens_8" \
  > "${output_dir}/RUN_CLASSIFICATION.txt"

export CUDA_VISIBLE_DEVICES="${devices}"
export ASCEND_RT_VISIBLE_DEVICES="${devices}"
export NPROC_PER_NODE="${nproc}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29981}"
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

if ! command -v ffmpeg >/dev/null 2>&1 && ! command -v avconv >/dev/null 2>&1; then
  ffmpeg_candidate="/share/home/ylhu/.conda/envs/omniagent_gyh/bin/ffmpeg"
  [[ -x "${ffmpeg_candidate}" ]] && export PATH="$(dirname "${ffmpeg_candidate}"):${PATH}"
fi

"${swift_cmd[@]}" infer \
  --model "${model}" \
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
  > "${output_dir}/summarize.log" 2>&1

echo "completed ${arm}: ${output_dir}/summary.json"
