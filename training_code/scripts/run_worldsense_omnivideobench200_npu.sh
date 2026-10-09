#!/usr/bin/env bash
# Run one answer-free OmniVideoBench split through Qwen2.5-Omni-7B on one NPU node.
# Usage: bash run_worldsense_omnivideobench200_npu.sh ARM ADAPTER_OR_DASH OUTPUT_DIR
set -euo pipefail

arm="${1:?ARM is required}"
adapter="${2:?ADAPTER_OR_DASH is required}"
output_dir="${3:?OUTPUT_DIR is required}"
case "${arm}" in base|sft|clue_opsd) ;; *) echo "invalid arm: ${arm}" >&2; exit 2 ;; esac
[[ "${adapter}" != - ]] || adapter=""
if [[ "${arm}" == base ]]; then
  [[ -z "${adapter}" ]] || { echo "base cannot have an adapter" >&2; exit 2; }
else
  for name in adapter_config.json adapter_model.safetensors trainer_state.json; do
    [[ -s "${adapter}/${name}" ]] || { echo "incomplete adapter: ${adapter}/${name}" >&2; exit 2; }
  done
fi

repo="/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd"
run_root="${repo}/training_runs/worldsense_openqa_20260929/eval_omnivideobench200"
data="${OMNI_OPSD_EVAL_DATASET:-${run_root}/data/omnivideobench.answer_free.jsonl}"
labels="${OMNI_OPSD_EVAL_LABELS:-${run_root}/data/omnivideobench.labels.jsonl}"
expected="${OMNI_OPSD_EVAL_EXPECTED_ROWS:-200}"
benchmark="${OMNI_OPSD_EVAL_BENCHMARK:-OmniVideoBench}"
nproc="${OMNI_OPSD_EVAL_NPROC:-8}"
devices="${OMNI_OPSD_EVAL_DEVICES:-0,1,2,3,4,5,6,7}"
model="/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-assets/models/Qwen2.5-Omni-7B"
swift_root="/opt/huawei/dataset/hyl_ulan/ylhu/third_party/ms-swift"
deps="/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_npu24_deps"
python_env="/home/ma-user/anaconda3/envs/PyTorch-2.9.0"
IFS=',' read -r -a cards <<< "${devices}"
[[ "${#cards[@]}" -eq "${nproc}" ]] || { echo "device count does not match nproc" >&2; exit 2; }
for path in "${data}" "${labels}" "${model}/config.json" "${swift_root}/swift/cli/infer.py"; do
  [[ -s "${path}" ]] || { echo "missing: ${path}" >&2; exit 2; }
done

# Keep labels separate from the model input and verify the same complete ID set.
"${python_env}/bin/python" - "${data}" "${labels}" "${expected}" "${benchmark}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

data_path, label_path, expected, benchmark = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
manifest_path = Path(data_path).parent / "manifest.json"
if manifest_path.is_file():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path, key in ((data_path, "answer_free_sha256"), (label_path, "labels_sha256")):
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if digest != manifest.get(key):
            raise SystemExit(f"input hash differs from frozen manifest: {path}")
data = [json.loads(line) for line in open(data_path, encoding="utf-8") if line.strip()]
labels = [json.loads(line) for line in open(label_path, encoding="utf-8") if line.strip()]
if len(data) != expected or len(labels) != expected:
    raise SystemExit(f"expected {expected} rows, got data={len(data)} labels={len(labels)}")
ids = []
for row in data:
    sample_id = row.get("case_id")
    ids.append(sample_id)
    if not sample_id or row.get("benchmark") != benchmark:
        raise SystemExit(f"invalid sample: {sample_id}")
    if {"answer", "solution", "correct_option", "teacher_prompt", "teacher_videos"} & row.keys():
        raise SystemExit(f"answer or teacher data leaked into {sample_id}")
    if len(row.get("messages", [])) != 1 or row["messages"][0].get("role") != "user":
        raise SystemExit(f"invalid messages in {sample_id}")
    videos = row.get("videos", [])
    if len(videos) != 1 or not isinstance(videos[0], dict):
        raise SystemExit(f"invalid media in {sample_id}")
    video = Path(videos[0].get("video", ""))
    if not video.is_file() or video.stat().st_size == 0:
        raise SystemExit(f"missing video in {sample_id}: {video}")
    contract = row.get("sampling_contract", {})
    if contract.get("held_out_evaluation") is not True or contract.get("use_audio_in_video") is not True:
        raise SystemExit(f"invalid audio/held-out contract in {sample_id}")
label_ids = [row.get("sample_id") for row in labels]
if len(set(ids)) != expected or len(set(label_ids)) != expected or set(ids) != set(label_ids):
    raise SystemExit("input and label IDs do not match exactly")
if any(str(row.get("answer", "")) not in {"A", "B", "C", "D"} for row in labels):
    raise SystemExit("invalid option letter in labels")
print(f"validated {expected} answer-free {benchmark} questions and videos")
PY

if [[ "${OMNI_OPSD_EVAL_VALIDATE_ONLY:-0}" == 1 ]]; then exit 0; fi
mkdir -p "${output_dir}"
result="${output_dir}/results.jsonl"
[[ ! -e "${result}" ]] || { echo "refusing to append to existing result: ${result}" >&2; exit 2; }
dataset_hash="$(sha256sum "${data}" | awk '{print $1}')"
label_hash="$(sha256sum "${labels}" | awk '{print $1}')"
printf '%s\n' \
  "benchmark=${benchmark}" "arm=${arm}" "model=${model}" "adapter=${adapter}" \
  "dataset=${data}" "dataset_sha256=${dataset_hash}" \
  "labels=${labels}" "labels_sha256=${label_hash}" \
  "expected_rows=${expected}" "devices=${devices}" "nproc=${nproc}" \
  "inference_layout=independent_single_card_shards" \
  "use_audio_in_video=true" "generation=greedy_max_new_tokens_8" \
  "started_at=$(date --iso-8601=seconds)" > "${output_dir}/RUN_CLASSIFICATION.txt"

export PYTHONPATH="${deps}:${repo}/training_code/src:${swift_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${python_env}/bin:${PATH}"
unset RANK_TABLE_FILE RANK_TABLE_FILE_V_1_0
export USE_AUDIO_IN_VIDEO=1
export ENABLE_AUDIO_OUTPUT=0
export FORCE_QWENVL_VIDEO_READER=pyav_seek
export MAX_NUM_WORKERS_FETCH_VIDEO=1
export WORLDSENSE_DROP_TALKER=1
export HCCL_CONNECT_TIMEOUT=600
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
adapter_args=()
[[ -z "${adapter}" ]] || adapter_args=(--adapters "${adapter}")

# Run each card as an independent one-card process. Distributed Swift inference
# gathers results across ranks; variable video decoding times can make one rank
# wait long enough for HCCL to time out, even though all cards are healthy.
shard_root="${output_dir}/shards"
mkdir -p "${shard_root}"
"${python_env}/bin/python" - "${data}" "${shard_root}" "${nproc}" <<'PY'
import json
import sys
from pathlib import Path

source, root, count = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
rows = [line for line in source.read_text().splitlines() if line.strip()]
for index in range(count):
    folder = root / f"card_{index}"
    folder.mkdir(parents=True, exist_ok=True)
    shard = rows[index::count]
    (folder / "input.jsonl").write_text("\n".join(shard) + "\n")
    for line in shard:
        json.loads(line)
PY

pids=()
for ((i=0; i<nproc; i++)); do
  shard="${shard_root}/card_${i}"
  shard_result="${shard}/results.jsonl"
  if [[ -s "${shard_result}" ]]; then
    # A prior interrupted attempt may have left a partial file. Preserve it.
    if "${python_env}/bin/python" - "${shard}/input.jsonl" "${shard_result}" <<'PY'
import json
import sys
from pathlib import Path

source, result = (Path(p) for p in sys.argv[1:])
expected = sum(bool(line.strip()) for line in source.open())
try:
    actual = sum(bool(line.strip()) and bool(json.loads(line)) for line in result.open())
except (OSError, ValueError):
    actual = -1
if actual == expected:
    raise SystemExit(0)
raise SystemExit(1)
PY
    then
      continue
    fi
    mv "${shard_result}" "${shard_result}.partial.$(date +%s)"
  fi
  (
    export ASCEND_RT_VISIBLE_DEVICES="${cards[$i]}"
    export NPROC_PER_NODE=1 NNODES=1 MASTER_ADDR=127.0.0.1
    export MASTER_PORT="$(( ${OMNI_OPSD_EVAL_MASTER_PORT:-29780} + i ))"
    "${python_env}/bin/swift" infer \
      --model "${model}" "${adapter_args[@]}" \
      --val_dataset "${shard}/input.jsonl" --result_path "${shard_result}" \
      --infer_backend transformers --max_batch_size 1 --write_batch_size 64 \
      --max_new_tokens 8 --temperature 0 --stream false \
      --torch_dtype bfloat16 --attn_impl sdpa --max_length 32768 \
      --dataset_num_proc 1 --val_dataset_shuffle false --seed 20260930
  ) > "${shard}/infer.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if ((failed)); then
  echo "one or more independent NPU inference shards failed; see ${shard_root}/card_*/infer.log" >&2
  exit 1
fi
"${python_env}/bin/python" - "${shard_root}" "${nproc}" "${result}" "${expected}" <<'PY'
import json
import sys
from pathlib import Path

root, count, output, expected = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
rows = []
for index in range(count):
    path = root / f"card_{index}/results.jsonl"
    shard = [json.loads(line) for line in path.open() if line.strip()]
    rows.extend(shard)
if len(rows) != expected:
    raise SystemExit(f"shard result count mismatch: {len(rows)} != {expected}")
output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
print(f"merged {len(rows)} independent-card predictions into {output}")
PY

PYTHONPATH="${repo}/training_code/src" "${python_env}/bin/python" \
  "${repo}/training_code/scripts/summarize_video_odyssey_training_eval.py" \
  --results "${result}" --labels "${labels}" --dataset "${data}" \
  --output "${output_dir}/summary.json" --arm "${arm}" --adapter "${adapter}" \
  > "${output_dir}/summarize.log" 2>&1
printf '%s\n' "completed_at=$(date --iso-8601=seconds)" >> "${output_dir}/RUN_CLASSIFICATION.txt"
