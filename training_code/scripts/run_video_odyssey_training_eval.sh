#!/usr/bin/env bash
set -euo pipefail

arm="${OMNI_OPSD_EVAL_ARM:?set OMNI_OPSD_EVAL_ARM}"
case "${arm}" in
  base|sft|grpo|opsd|clue_opsd) ;;
  *) echo "Unsupported evaluation arm: ${arm}" >&2; exit 2 ;;
esac

model="${OMNI_OPSD_MODEL:?set OMNI_OPSD_MODEL}"
dataset="${OMNI_OPSD_EVAL_DATASET:?set OMNI_OPSD_EVAL_DATASET}"
labels="${OMNI_OPSD_EVAL_LABELS:?set OMNI_OPSD_EVAL_LABELS}"
output_dir="${OMNI_OPSD_EVAL_OUTPUT_DIR:?set OMNI_OPSD_EVAL_OUTPUT_DIR}"
ms_swift_root="${OMNI_OPSD_MS_SWIFT_ROOT:?set OMNI_OPSD_MS_SWIFT_ROOT}"
python_deps="${OMNI_OPSD_PYTHON_DEPS:?set OMNI_OPSD_PYTHON_DEPS}"
project_root="${OMNI_OPSD_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
swift_bin="${OMNI_OPSD_SWIFT_BIN:-swift}"
adapter="${OMNI_OPSD_EVAL_ADAPTER:-}"
nproc="${OMNI_OPSD_NPROC_PER_NODE:-8}"
expected_rows="${OMNI_OPSD_EVAL_EXPECTED_ROWS:-108}"
use_audio_in_video="${OMNI_OPSD_EVAL_USE_AUDIO_IN_VIDEO:-0}"

case "${use_audio_in_video}" in
  0|1) ;;
  *) echo "OMNI_OPSD_EVAL_USE_AUDIO_IN_VIDEO must be 0 or 1" >&2; exit 2 ;;
esac

: "${ASCEND_RT_VISIBLE_DEVICES:?set ASCEND_RT_VISIBLE_DEVICES explicitly}"
IFS=',' read -r -a visible_devices <<< "${ASCEND_RT_VISIBLE_DEVICES}"
if [[ "${#visible_devices[@]}" -ne "${nproc}" ]]; then
  echo "Expected ${nproc} visible NPUs, got ${#visible_devices[@]}" >&2
  exit 2
fi
for path in "${model}" "${dataset}" "${labels}" "${ms_swift_root}/swift"; do
  [[ -e "${path}" ]] || { echo "Required path is missing: ${path}" >&2; exit 2; }
done
if [[ "${arm}" == "base" ]]; then
  [[ -z "${adapter}" ]] || { echo "Base evaluation must not load an adapter" >&2; exit 2; }
else
  for file in adapter_config.json adapter_model.safetensors trainer_state.json; do
    [[ -s "${adapter}/${file}" ]] || { echo "Incomplete adapter: ${adapter}/${file}" >&2; exit 2; }
  done
fi

python_bin="$(command -v python)"
PYTHONPATH="${project_root}/src:${python_deps}:${ms_swift_root}" \
  "${python_bin}" - "${dataset}" "${labels}" "${expected_rows}" \
  "${use_audio_in_video}" <<'PY'
import json
import os
import sys

dataset, labels, expected_text, audio_flag = sys.argv[1:]
expected = int(expected_text)
expected_audio = audio_flag == "1"
rows = [json.loads(line) for line in open(dataset, encoding="utf-8") if line.strip()]
gold = [json.loads(line) for line in open(labels, encoding="utf-8") if line.strip()]
if len(rows) != expected or len(gold) != expected:
    raise SystemExit(f"cardinality mismatch: dataset={len(rows)} labels={len(gold)} expected={expected}")
ids = []
for row in rows:
    case_id = str(row.get("case_id") or row.get("prompt_id") or "")
    ids.append(case_id)
    if "answer" in row or "solution" in row:
        raise SystemExit(f"answer leakage in evaluation row {case_id}")
    if "teacher_prompt" in row or "teacher_videos" in row or "teacher_audios" in row:
        raise SystemExit(f"teacher-only field leaked into evaluation row {case_id}")
    messages = row.get("messages") or []
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise SystemExit(f"evaluation row {case_id} must contain only a user message")
    videos = row.get("videos") or []
    if len(videos) != 1 or not isinstance(videos[0], (str, dict)):
        raise SystemExit(f"evaluation row {case_id} lacks one materialized video")
    media_path = videos[0] if isinstance(videos[0], str) else videos[0].get("video")
    if not media_path or not os.path.isfile(media_path):
        raise SystemExit(f"evaluation row {case_id} references missing media")
    sampling = row.get("sampling_contract", {})
    if not sampling.get("held_out_evaluation"):
        raise SystemExit(f"evaluation row {case_id} lacks held-out contract")
    if bool(sampling.get("use_audio_in_video")) != expected_audio:
        raise SystemExit(
            f"evaluation audio contract mismatch for {case_id}: "
            f"dataset={sampling.get('use_audio_in_video')} runtime={expected_audio}"
        )
if len(set(ids)) != expected:
    raise SystemExit("evaluation case IDs are not unique")
if set(ids) != {str(row["sample_id"]) for row in gold}:
    raise SystemExit("evaluation and label IDs differ")
print(f"validated held-out evaluation rows={expected} no_answer_leakage=true")
PY

mkdir -p "${output_dir}"
result_path="${output_dir}/results.jsonl"
[[ ! -e "${result_path}" ]] || { echo "Refusing to append to existing result: ${result_path}" >&2; exit 2; }
adapter_args=()
[[ -z "${adapter}" ]] || adapter_args=(--adapters "${adapter}")

dataset_sha256="$(sha256sum "${dataset}" | awk '{print $1}')"
labels_sha256="$(sha256sum "${labels}" | awk '{print $1}')"
project_commit="$(git -C "${project_root}" rev-parse HEAD 2>/dev/null || printf unknown)"
printf '%s\n' \
  "arm=${arm}" \
  "model=${model}" \
  "adapter=${adapter}" \
  "dataset=${dataset}" \
  "dataset_sha256=${dataset_sha256}" \
  "labels=${labels}" \
  "labels_sha256=${labels_sha256}" \
  "expected_rows=${expected_rows}" \
  "nproc=${nproc}" \
  "visible_devices=${ASCEND_RT_VISIBLE_DEVICES}" \
  "project_commit=${project_commit}" \
  "protocol_classification=$([[ "${use_audio_in_video}" == 1 ]] && printf omni-audio-video || printf visual-control)" \
  "use_audio_in_video=$([[ "${use_audio_in_video}" == 1 ]] && printf true || printf false)" \
  "generation=greedy_max_new_tokens_4" \
  > "${output_dir}/RUN_CLASSIFICATION.txt"

export PYTHONPATH="${project_root}/src:${python_deps}:${ms_swift_root}${PYTHONPATH:+:${PYTHONPATH}}"
export NPROC_PER_NODE="${nproc}"
export USE_AUDIO_IN_VIDEO="${use_audio_in_video}"
export ENABLE_AUDIO_OUTPUT=0
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

"${swift_bin}" infer \
  --model "${model}" \
  "${adapter_args[@]}" \
  --val_dataset "${dataset}" \
  --result_path "${result_path}" \
  --infer_backend transformers \
  --max_batch_size 1 \
  --max_new_tokens 4 \
  --temperature 0 \
  --stream false \
  --torch_dtype bfloat16 \
  --attn_impl eager \
  --dataset_num_proc 1 \
  --val_dataset_shuffle false \
  --seed 20260903 \
  > "${output_dir}/infer.log" 2>&1

PYTHONPATH="${project_root}/src" "${python_bin}" \
  "${project_root}/scripts/summarize_video_odyssey_training_eval.py" \
  --results "${result_path}" --labels "${labels}" \
  --dataset "${dataset}" \
  --output "${output_dir}/summary.json" --arm "${arm}" --adapter "${adapter}" \
  > "${output_dir}/summarize.log" 2>&1
