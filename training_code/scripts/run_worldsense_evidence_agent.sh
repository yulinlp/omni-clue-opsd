#!/usr/bin/env bash
set -euo pipefail

# Launch the WorldSense adaptive evidence-localization agent on a GPU node.
# Localization-only stage: explores the video with the inspect tool and writes
# candidate evidence intervals with per-interval observations.

project_root="/share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907"
python_env="${OMNI_OPSD_AGENT_ENV:-/share/home/ylhu/.conda/envs/omniopsd_train}"
model="${OMNI_OPSD_AGENT_MODEL:-/share/home/ylhu/models/Qwen3-Omni-30B-A3B-Instruct}"
qa="${OMNI_OPSD_WORLDSENSE_QA:-/share/home/ylhu/Light-Omni/data/omni_benchmarks/WorldSense/worldsense_qa.json}"
video_root="${OMNI_OPSD_WORLDSENSE_VIDEOS:-/share/home/ylhu/Light-Omni/data/omni_benchmarks/WorldSense/videos}"
output="${OMNI_OPSD_AGENT_OUTPUT:-${project_root}/output/worldsense_evidence/evidence_localization.jsonl}"
trace="${OMNI_OPSD_AGENT_TRACE:-${project_root}/output/worldsense_evidence/evidence_localization.trace.jsonl}"
media_cache="${OMNI_OPSD_AGENT_MEDIA_CACHE:-${project_root}/output/worldsense_evidence/media_index.json}"
device="${OMNI_OPSD_AGENT_DEVICE:-cuda:0}"
dtype="${OMNI_OPSD_AGENT_DTYPE:-bfloat16}"
attn="${OMNI_OPSD_AGENT_ATTN:-sdpa}"
max_turns="${OMNI_OPSD_AGENT_MAX_TURNS:-6}"
max_inspect="${OMNI_OPSD_AGENT_MAX_INSPECT:-3}"
temperature="${OMNI_OPSD_AGENT_TEMPERATURE:-0.2}"
limit_args=()
if [[ -n "${OMNI_OPSD_AGENT_LIMIT:-}" ]]; then
  limit_args=(--limit "${OMNI_OPSD_AGENT_LIMIT}")
fi
filter_args=()
if [[ -n "${OMNI_OPSD_AGENT_QUESTION_IDS:-}" ]]; then
  filter_args+=(--question-ids "${OMNI_OPSD_AGENT_QUESTION_IDS}")
fi
if [[ -n "${OMNI_OPSD_AGENT_TASK_TYPES:-}" ]]; then
  filter_args+=(--task-types "${OMNI_OPSD_AGENT_TASK_TYPES}")
fi

ffmpeg_bin="/share/home/ylhu/.conda/envs/omniagent_gyh/bin"
if [[ -x "${ffmpeg_bin}/ffmpeg" ]]; then
  export PATH="${ffmpeg_bin}:${PATH}"
fi

export PYTHONPATH="${project_root}/src:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
export FORCE_QWENVL_VIDEO_READER=decord
export USE_AUDIO_IN_VIDEO=1
export DECORD_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "${output}")"
echo "agent model : ${model}"
echo "device      : ${device}"
echo "questions   : ${qa}"
echo "videos      : ${video_root}"
echo "output      : ${output}"
echo "trace       : ${trace}"
echo "budget      : max_turns=${max_turns} max_inspect=${max_inspect} temperature=${temperature}"

exec "${python_env}/bin/python" "${project_root}/scripts/annotate_worldsense_evidence.py" \
  --qa "${qa}" \
  --video-root "${video_root}" \
  --output "${output}" \
  --trace "${trace}" \
  --media-index-cache "${media_cache}" \
  --model "${model}" \
  --device "${device}" \
  --dtype "${dtype}" \
  --attn "${attn}" \
  --max-turns "${max_turns}" \
  --max-inspect "${max_inspect}" \
  --temperature "${temperature}" \
  "${limit_args[@]}" \
  "${filter_args[@]}"
