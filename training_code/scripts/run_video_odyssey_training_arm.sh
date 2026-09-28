#!/usr/bin/env bash
set -euo pipefail

# Run one comparable VideoOdyssey post-training arm on an explicit accelerator slice.
# The matrix builder guarantees identical full-video student inputs; this
# launcher changes only the supervision mechanism selected by OMNI_OPSD_ARM.

arm="${OMNI_OPSD_ARM:?set OMNI_OPSD_ARM to sft, grpo, opsd, or clue_opsd}"
case "${arm}" in
  sft|grpo|opsd|clue_opsd) ;;
  *) echo "Unsupported OMNI_OPSD_ARM=${arm}" >&2; exit 2 ;;
esac

model="${OMNI_OPSD_MODEL:?set OMNI_OPSD_MODEL}"
dataset="${OMNI_OPSD_DATASET:?set OMNI_OPSD_DATASET}"
output_dir="${OMNI_OPSD_OUTPUT_DIR:?set OMNI_OPSD_OUTPUT_DIR}"
resume_checkpoint="${OMNI_OPSD_RESUME_FROM_CHECKPOINT:-}"
ms_swift_root="${OMNI_OPSD_MS_SWIFT_ROOT:?set OMNI_OPSD_MS_SWIFT_ROOT}"
python_deps="${OMNI_OPSD_PYTHON_DEPS:?set OMNI_OPSD_PYTHON_DEPS}"
project_root="${OMNI_OPSD_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
swift_bin="${OMNI_OPSD_SWIFT_BIN:-swift}"
nproc="${OMNI_OPSD_NPROC_PER_NODE:-2}"
max_steps="${OMNI_OPSD_MAX_STEPS:-300}"
num_train_epochs="${OMNI_OPSD_NUM_TRAIN_EPOCHS:-}"
per_device_batch="${OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
gradient_accumulation="${OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS:-16}"
tuner_type="${OMNI_OPSD_TUNER_TYPE:-lora}"
lora_rank="${OMNI_OPSD_LORA_RANK:-16}"
lora_alpha="${OMNI_OPSD_LORA_ALPHA:-32}"
freeze_llm="${OMNI_OPSD_FREEZE_LLM:-false}"
freeze_vit="${OMNI_OPSD_FREEZE_VIT:-true}"
freeze_aligner="${OMNI_OPSD_FREEZE_ALIGNER:-true}"
learning_rate="${OMNI_OPSD_LEARNING_RATE:-2e-6}"
max_length="${OMNI_OPSD_MAX_LENGTH:-32768}"
allocator_conf="${OMNI_OPSD_ALLOC_CONF:-}"
min_pixels="${OMNI_OPSD_MIN_PIXELS:-3136}"
max_completion_length="${OMNI_OPSD_MAX_COMPLETION_LENGTH:-8}"
gkd_logits_topk="${OMNI_OPSD_GKD_LOGITS_TOPK:-100}"
rollout_top_p="${OMNI_OPSD_ROLLOUT_TOP_P:-1.0}"
rollout_top_k="${OMNI_OPSD_ROLLOUT_TOP_K:-20}"
vllm_gpu_memory_utilization="${OMNI_OPSD_VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
vllm_sleep_level="${OMNI_OPSD_VLLM_SLEEP_LEVEL:-1}"
steps_per_generation="${OMNI_OPSD_STEPS_PER_GENERATION:-}"
ddp_find_unused_parameters="${OMNI_OPSD_DDP_FIND_UNUSED_PARAMETERS:-}"
if [[ "${arm}" == "opsd" || "${arm}" == "clue_opsd" ]]; then
  # GKD/OPSD always uses the vLLM rollout path by default.  This avoids the
  # TransformersEngine branch; an explicit environment override remains
  # available for a platform-specific diagnostic run.
  use_vllm="${OMNI_OPSD_USE_VLLM:-true}"
else
  # SFT has no rollout engine; keep vLLM checks and arguments out of this arm.
  use_vllm=false
fi
vllm_mode="${OMNI_OPSD_VLLM_MODE:-colocate}"
vllm_tensor_parallel_size="${OMNI_OPSD_VLLM_TENSOR_PARALLEL_SIZE:-1}"
save_steps="${OMNI_OPSD_SAVE_STEPS:-25}"
attn_impl="${OMNI_OPSD_ATTN_IMPL:-eager}"
ema_alpha="${OMNI_OPSD_CLUE_EMA_ALPHA:-0.05}"
opsd_ema_alpha="${OMNI_OPSD_OPSD_EMA_ALPHA:-0.0}"
full_ema_teacher="${OMNI_OPSD_FULL_EMA_TEACHER:-false}"
full_ema_alpha="${OMNI_OPSD_FULL_EMA_ALPHA:-0.05}"
full_ema_offload="${OMNI_OPSD_FULL_EMA_OFFLOAD:-false}"
gold_ce_alpha="${OMNI_OPSD_GOLD_CE_ALPHA:-0.25}"
split_dataset_ratio="${OMNI_OPSD_SPLIT_DATASET_RATIO:-0}"
eval_strategy="${OMNI_OPSD_EVAL_STRATEGY:-no}"
eval_steps="${OMNI_OPSD_EVAL_STEPS:-${save_steps}}"
log_completions="${OMNI_OPSD_LOG_COMPLETIONS:-false}"
offload_model="${OMNI_OPSD_OFFLOAD_MODEL:-false}"
offload_optimizer="${OMNI_OPSD_OFFLOAD_OPTIMIZER:-false}"
allow_non_ema_clue="${OMNI_OPSD_ALLOW_NON_EMA_CLUE:-0}"
allow_non_32_global_batch="${OMNI_OPSD_ALLOW_NON_32_GLOBAL_BATCH:-0}"
max_grad_norm="${OMNI_OPSD_MAX_GRAD_NORM:-1.0}"
use_logits_to_keep="${OMNI_OPSD_USE_LOGITS_TO_KEEP:-0}"
diag_enabled="${OMNI_OPSD_DIAG_ENABLED:-false}"
diag_top_k="${OMNI_OPSD_DIAG_TOP_K:-20}"
diag_temperature="${OMNI_OPSD_DIAG_TEMPERATURE:-1.0}"
diag_frequency="${OMNI_OPSD_DIAG_FREQUENCY:-1}"
diag_chunk_size="${OMNI_OPSD_DIAG_CHUNK_SIZE:-256}"
diag_answer_labels_path="${OMNI_OPSD_DIAG_ANSWER_LABELS_PATH:-}"
# The full multimodal GKD gradient can exhaust 910B1 HBM in the Ascend
# LpNormV2 kernel.  Keep this guard on by default for OPSD/CLUE-OPSD, while
# allowing an explicit platform override for machines where clipping fits.
gkd_safe_mode="${OMNI_OPSD_GKD_SAFE_MODE:-1}"
gkd_max_grad_norm="${OMNI_OPSD_GKD_MAX_GRAD_NORM:-0}"
seed="${OMNI_OPSD_SEED:-20260904}"
experiment_label="${OMNI_OPSD_EXPERIMENT_LABEL:-video_odyssey_${arm}}"
use_audio_in_video="${USE_AUDIO_IN_VIDEO:-0}"
allow_sparse_pilot="${OMNI_OPSD_ALLOW_SPARSE_ENGINEERING_PILOT:-0}"

case "${use_audio_in_video}" in
  0|1) ;;
  *) echo "USE_AUDIO_IN_VIDEO must be 0 or 1" >&2; exit 2 ;;
esac
case "${tuner_type}" in
  lora|full) ;;
  *) echo "OMNI_OPSD_TUNER_TYPE must be lora or full" >&2; exit 2 ;;
esac
for flag_name in freeze_llm freeze_vit freeze_aligner; do
  flag_value="${!flag_name}"
  case "${flag_value}" in
    true|false) ;;
    *) echo "OMNI_OPSD_${flag_name^^} must be true or false" >&2; exit 2 ;;
  esac
done
case "${allow_sparse_pilot}" in
  0|1) ;;
  *) echo "OMNI_OPSD_ALLOW_SPARSE_ENGINEERING_PILOT must be 0 or 1" >&2; exit 2 ;;
esac
case "${allow_non_ema_clue}" in
  0|1) ;;
  *) echo "OMNI_OPSD_ALLOW_NON_EMA_CLUE must be 0 or 1" >&2; exit 2 ;;
esac
case "${allow_non_32_global_batch}" in
  0|1) ;;
  *) echo "OMNI_OPSD_ALLOW_NON_32_GLOBAL_BATCH must be 0 or 1" >&2; exit 2 ;;
esac
case "${full_ema_teacher}" in
  true|false) ;;
  *) echo "OMNI_OPSD_FULL_EMA_TEACHER must be true or false" >&2; exit 2 ;;
esac
case "${full_ema_offload}" in
  true|false) ;;
  *) echo "OMNI_OPSD_FULL_EMA_OFFLOAD must be true or false" >&2; exit 2 ;;
esac
case "${offload_model}" in
  true|false) ;;
  *) echo "OMNI_OPSD_OFFLOAD_MODEL must be true or false" >&2; exit 2 ;;
esac
case "${offload_optimizer}" in
  true|false) ;;
  *) echo "OMNI_OPSD_OFFLOAD_OPTIMIZER must be true or false" >&2; exit 2 ;;
esac
case "${gkd_safe_mode}" in
  0|1) ;;
  *) echo "OMNI_OPSD_GKD_SAFE_MODE must be 0 or 1" >&2; exit 2 ;;
esac
case "${use_logits_to_keep}" in
  0|1) ;;
  *) echo "OMNI_OPSD_USE_LOGITS_TO_KEEP must be 0 or 1" >&2; exit 2 ;;
esac
case "${diag_enabled}" in
  true|false) ;;
  *) echo "OMNI_OPSD_DIAG_ENABLED must be true or false" >&2; exit 2 ;;
esac
if [[ ! "${diag_top_k}" =~ ^[1-9][0-9]*$ || ! "${diag_frequency}" =~ ^[1-9][0-9]*$ ||
      ! "${diag_chunk_size}" =~ ^[1-9][0-9]*$ ]]; then
  echo "OMNI_OPSD_DIAG_TOP_K, DIAG_FREQUENCY and DIAG_CHUNK_SIZE must be positive integers" >&2
  exit 2
fi
if [[ ! "${diag_temperature}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]] ||
   (( $(awk -v value="${diag_temperature}" 'BEGIN {print (value <= 0)}') )); then
  echo "OMNI_OPSD_DIAG_TEMPERATURE must be positive" >&2
  exit 2
fi
case "${attn_impl}" in
  eager|sdpa|flash_attention_2) ;;
  *) echo "Unsupported OMNI_OPSD_ATTN_IMPL=${attn_impl}" >&2; exit 2 ;;
esac
if [[ ! "${min_pixels}" =~ ^[1-9][0-9]*$ ]]; then
  echo "OMNI_OPSD_MIN_PIXELS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${gkd_logits_topk}" =~ ^[1-9][0-9]*$ ]]; then
  echo "OMNI_OPSD_GKD_LOGITS_TOPK must be a positive integer" >&2
  exit 2
fi
if [[ ! "${lora_rank}" =~ ^[1-9][0-9]*$ || ! "${lora_alpha}" =~ ^[1-9][0-9]*([.][0-9]+)?$ ]]; then
  echo "OMNI_OPSD_LORA_RANK and OMNI_OPSD_LORA_ALPHA must be positive numbers" >&2
  exit 2
fi
if [[ ! "${learning_rate}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]]; then
  echo "OMNI_OPSD_LEARNING_RATE must be a positive numeric value" >&2
  exit 2
fi
if [[ ! "${vllm_tensor_parallel_size}" =~ ^[1-9][0-9]*$ ]]; then
  echo "OMNI_OPSD_VLLM_TENSOR_PARALLEL_SIZE must be a positive integer" >&2
  exit 2
fi
if [[ -n "${num_train_epochs}" && ! "${num_train_epochs}" =~ ^[1-9][0-9]*$ ]]; then
  echo "OMNI_OPSD_NUM_TRAIN_EPOCHS must be a positive integer when set" >&2
  exit 2
fi
if [[ ! "${split_dataset_ratio}" =~ ^(0([.]([0-9]+))?|1([.]0+)?)$ ]]; then
  echo "OMNI_OPSD_SPLIT_DATASET_RATIO must be in [0, 1]" >&2
  exit 2
fi
if [[ "${eval_strategy}" != no && "${eval_strategy}" != steps && "${eval_strategy}" != epoch ]]; then
  echo "OMNI_OPSD_EVAL_STRATEGY must be no, steps, or epoch" >&2
  exit 2
fi
if [[ ! "${opsd_ema_alpha}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] ||
   (( $(awk -v value="${opsd_ema_alpha}" 'BEGIN {print (value < 0 || value >= 1)}') )); then
  echo "OMNI_OPSD_OPSD_EMA_ALPHA must be in [0, 1)" >&2
  exit 2
fi
if [[ ! "${rollout_top_p}" =~ ^(0\.[0-9]*[1-9][0-9]*|1([.]0+)?)$ ]]; then
  echo "OMNI_OPSD_ROLLOUT_TOP_P must be in (0, 1]" >&2
  exit 2
fi
if [[ ! "${rollout_top_k}" =~ ^-?[1-9][0-9]*$|^0$ ]]; then
  echo "OMNI_OPSD_ROLLOUT_TOP_K must be an integer" >&2
  exit 2
fi
if [[ -n "${steps_per_generation}" && ! "${steps_per_generation}" =~ ^[1-9][0-9]*$ ]]; then
  echo "OMNI_OPSD_STEPS_PER_GENERATION must be a positive integer when set" >&2
  exit 2
fi
if [[ -n "${ddp_find_unused_parameters}" && "${ddp_find_unused_parameters}" != true &&
      "${ddp_find_unused_parameters}" != false ]]; then
  echo "OMNI_OPSD_DDP_FIND_UNUSED_PARAMETERS must be true or false when set" >&2
  exit 2
fi
if [[ "${arm}" == "opsd" || "${arm}" == "clue_opsd" ]]; then
  if [[ ! "${vllm_gpu_memory_utilization}" =~ ^(0\.[0-9]+|1([.]0+)?)$ ]]; then
    echo "OMNI_OPSD_VLLM_GPU_MEMORY_UTILIZATION must be in (0, 1]" >&2
    exit 2
  fi
  if [[ ! "${vllm_sleep_level}" =~ ^[012]$ ]]; then
    echo "OMNI_OPSD_VLLM_SLEEP_LEVEL must be 0, 1, or 2" >&2
    exit 2
  fi
fi
case "${use_vllm}" in
  true|false) ;;
  *) echo "OMNI_OPSD_USE_VLLM must be true or false" >&2; exit 2 ;;
esac
if [[ "${use_vllm}" == "true" ]]; then
  case "${vllm_mode}" in
    server|colocate) ;;
    *) echo "OMNI_OPSD_VLLM_MODE must be server or colocate when vLLM is enabled" >&2; exit 2 ;;
  esac
fi
if [[ ("${arm}" == "opsd" || "${arm}" == "clue_opsd") && "${gkd_safe_mode}" == "1" ]]; then
  max_grad_norm="${gkd_max_grad_norm}"
fi
if [[ ! "${max_grad_norm}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
  echo "OMNI_OPSD_MAX_GRAD_NORM must be a non-negative number" >&2
  exit 2
fi
if [[ -n "${resume_checkpoint}" && ! -d "${resume_checkpoint}" ]]; then
  echo "Resume checkpoint does not exist: ${resume_checkpoint}" >&2
  exit 2
fi

: "${ASCEND_RT_VISIBLE_DEVICES:?set ASCEND_RT_VISIBLE_DEVICES explicitly}"
IFS=',' read -r -a visible_devices <<< "${ASCEND_RT_VISIBLE_DEVICES}"
if [[ "${#visible_devices[@]}" -ne "${nproc}" ]]; then
  echo "Expected ${nproc} visible NPUs, got ${#visible_devices[@]}" >&2
  exit 2
fi
for path in "${model}" "${dataset}" "${ms_swift_root}/swift/rl_core/data.py"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path is missing: ${path}" >&2
    exit 2
  fi
done
if [[ "${arm}" == "grpo" && ! -f "${project_root}/scripts/video_odyssey_grpo_reward.py" ]]; then
  echo "GRPO reward plugin is missing" >&2
  exit 2
fi
if [[ "${arm}" == "opsd" || "${arm}" == "clue_opsd" ]]; then
  if ! grep -q 'teacher_videos' "${ms_swift_root}/swift/rl_core/data.py"; then
    echo "ms-swift checkout lacks teacher video support" >&2
    exit 2
  fi
fi
if [[ "${arm}" == "clue_opsd" && "${allow_non_ema_clue}" == "0" ]]; then
  if ! grep -q 'clue_ema_alpha' "${ms_swift_root}/swift/rlhf_trainers/gkd_trainer.py"; then
    echo "ms-swift checkout lacks the LoRA-shadow EMA teacher patch" >&2
    exit 2
  fi
fi

if [[ "${arm}" == "clue_opsd" && "${allow_non_ema_clue}" == "0" ]]; then
  # GKDConfig has the CLUE field, but the top-level swift rlhf parser uses
  # RLHFArguments.  Patch/check that parser before launching any distributed
  # workers; otherwise all eight ranks fail immediately on an unknown option.
  bash "${project_root}/scripts/ensure_ms_swift_clue_cli.sh" "${ms_swift_root}"
  if ! PYTHONPATH="${project_root}/src:${python_deps}:${ms_swift_root}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${swift_bin}" rlhf --help 2>&1 | grep -- '--clue_ema_alpha' >/dev/null; then
    echo "ms-swift swift rlhf parser lacks --clue_ema_alpha" >&2
    exit 2
  fi
fi

python_bin="${OMNI_OPSD_PYTHON_BIN:-$(command -v python)}"
if [[ "${use_vllm}" == "true" ]]; then
  if ! PYTHONPATH="${project_root}/src:${python_deps}:${ms_swift_root}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${python_bin}" -c 'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("vllm") else 1)'; then
    echo "vLLM is enabled but is not installed in ${python_bin}; install a CUDA/PyTorch-compatible vLLM before training" >&2
    exit 2
  fi
fi
PYTHONPATH="${project_root}/src:${python_deps}:${ms_swift_root}" \
  "${python_bin}" - <<'PY'
import importlib.metadata

import msgspec
import qwen_omni_utils

version = importlib.metadata.version("qwen_omni_utils")
if version != "0.0.9":
    raise SystemExit(f"expected qwen_omni_utils 0.0.9, found {version}")
print(f"validated training dependencies: msgspec={msgspec.__version__} qwen_omni_utils={version}")
PY

PYTHONPATH="${project_root}/src:${python_deps}:${ms_swift_root}" \
  "${python_bin}" - "${dataset}" "${arm}" "${use_audio_in_video}" \
  "${allow_sparse_pilot}" "${max_steps}" "${min_pixels}" <<'PY'
import json
import os
import sys

path, arm, audio_flag, allow_sparse_flag, max_steps_text, min_pixels_text = sys.argv[1:]
expected_audio = audio_flag == "1"
allow_sparse = allow_sparse_flag == "1"
max_steps = int(max_steps_text)
min_pixels = int(min_pixels_text)
rows = 0
case_ids = set()
student_frame_budgets = []
direct_answer_rows = 0
with open(path, encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        rows += 1
        case_ids.add(str(row.get("case_id")))
        messages = row.get("messages") or []
        if str(row.get("response_format", "")) == "direct_answer":
            direct_answer_rows += 1
        if not messages or messages[0].get("role") != "user":
            raise SystemExit("dataset row lacks a user prompt")
        if not row.get("videos"):
            raise SystemExit("dataset row lacks student video")
        for media in row["videos"] + (row.get("teacher_videos") or []):
            media_path = media if isinstance(media, str) else media.get("video")
            if not media_path or not os.path.isfile(media_path):
                raise SystemExit(f"missing media: {media_path}")
        contract = row.get("supervision_contract") or {}
        sampling = row.get("sampling_contract") or {}
        if bool(sampling.get("use_audio_in_video")) != expected_audio:
            raise SystemExit(
                f"audio contract mismatch for {row.get('case_id')}: "
                f"dataset={sampling.get('use_audio_in_video')} runtime={expected_audio}"
            )
        frame_budget = sampling.get(
            "frames_per_video_input", sampling.get("max_frames_per_view")
        )
        if not isinstance(frame_budget, int) or frame_budget < 2:
            raise SystemExit(f"missing student frame budget for {row.get('case_id')}")
        for media in row.get("videos") or []:
            if isinstance(media, dict) and int(media.get("min_pixels", min_pixels)) != min_pixels:
                raise SystemExit(f"minimum pixel contract mismatch for {row.get('case_id')}")
        student_frame_budgets.append(frame_budget)
        if row.get("experiment_arm") != arm:
            raise SystemExit(f"arm mismatch: {row.get('experiment_arm')} != {arm}")
        if arm == "sft":
            if len(messages) != 2 or messages[-1].get("role") != "assistant":
                raise SystemExit("SFT row lacks gold assistant target")
        elif arm == "grpo":
            if not row.get("solution") or len(messages) != 1:
                raise SystemExit("GRPO row lacks reward solution or leaks an assistant target")
        elif arm == "opsd":
            if not contract.get("gold_answer_in_teacher_prompt"):
                raise SystemExit("OPSD row lacks answer-privileged teacher contract")
            if row.get("teacher_videos") != row.get("videos"):
                raise SystemExit("standard OPSD teacher must use the same full video")
        else:
            response_format = str(row.get("response_format", "reasoning"))
            if response_format == "direct_answer":
                # OE Clue-OPSD deliberately gives the teacher only the
                # annotated clue intervals.  The open-ended gold answer must
                # not be present in the row or teacher prompt.
                if contract.get("gold_answer_in_teacher_prompt"):
                    raise SystemExit("direct-answer Clue-OPSD teacher must not receive gold_answer")
                if "gold_answer" in row:
                    raise SystemExit("direct-answer Clue-OPSD row leaks gold_answer")
            else:
                if not contract.get("gold_answer_in_teacher_prompt"):
                    raise SystemExit("Clue-OPSD row lacks gold-answer teacher contract")
                gold_answer = str(row.get("gold_answer", "")).strip()
                if not gold_answer:
                    raise SystemExit("Clue-OPSD row lacks a gold_answer")
                option = gold_answer.upper()
                if option not in {"A", "B", "C", "D"}:
                    raise SystemExit("Clue-OPSD row lacks a valid multiple-choice gold_answer")
                if f"correct option is {option}" not in str(row.get("teacher_prompt", "")):
                    raise SystemExit("Clue-OPSD teacher prompt does not contain gold_answer")
            if "solution" in row or "answer" in row:
                raise SystemExit("Clue-OPSD row contains raw answer/solution supervision")
            if not row.get("teacher_videos"):
                raise SystemExit("Clue-OPSD row lacks clue videos")
if rows == 0 or len(case_ids) != rows:
    raise SystemExit(f"invalid dataset cardinality: rows={rows}, unique_case_ids={len(case_ids)}")
min_frames = min(student_frame_budgets)
max_frames = max(student_frame_budgets)
if max_steps > 1 and min_frames < 64 and not allow_sparse and direct_answer_rows != rows:
    raise SystemExit(
        f"refusing multi-step VideoOdyssey training with only {min_frames} student frames; "
        "set OMNI_OPSD_ALLOW_SPARSE_ENGINEERING_PILOT=1 only for a labeled engineering pilot; "
        "direct-answer datasets use their materialized dynamic frame budget"
    )
print(
    f"validated arm={arm} rows={rows} unique_case_ids={len(case_ids)} "
    f"student_frames={min_frames}..{max_frames} use_audio_in_video={expected_audio}"
)
PY

mkdir -p "${output_dir}"
dataset_sha256="$(sha256sum "${dataset}" | awk '{print $1}')"
dataset_rows="$(awk 'NF {count++} END {print count+0}' "${dataset}")"
project_commit="$(git -C "${project_root}" rev-parse HEAD 2>/dev/null || printf unknown)"
swift_commit="$(git -C "${ms_swift_root}" rev-parse HEAD 2>/dev/null || printf unknown)"
if [[ ! "${per_device_batch}" =~ ^[1-9][0-9]*$ ]]; then
  echo "OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
effective_batch="$((nproc * per_device_batch * gradient_accumulation))"
if [[ "${effective_batch}" -ne 32 && "${allow_non_32_global_batch}" != "1" ]]; then
  echo "effective batch must be 32: ${nproc}*${per_device_batch}*${gradient_accumulation}=${effective_batch}" >&2
  exit 2
fi
supervision=""
teacher_mode="none"
clue_ema_classification="not-applicable"
case "${arm}" in
  sft) supervision="gold-answer-teacher-forcing" ;;
  grpo) supervision="gold-answer-exact-match-sequence-reward" ;;
  opsd)
    supervision="answer-privileged-full-video-gkd"
    if [[ "${opsd_ema_alpha}" != "0" && "${opsd_ema_alpha}" != "0.0" ]]; then
      teacher_mode="lora-shadow-ema"
    else
      teacher_mode="dynamic-current-policy"
    fi
    ;;
  clue_opsd)
    supervision="clue-privileged-video-gkd"
    if [[ "${full_ema_teacher}" == "true" ]]; then
      teacher_mode="full-parameter-ema"
      clue_ema_classification="full:${full_ema_alpha}"
    elif [[ "${allow_non_ema_clue}" == "1" ]]; then
      teacher_mode="dynamic-current-policy-not-ema"
      clue_ema_classification="disabled"
    else
      teacher_mode="lora-shadow-ema"
      clue_ema_classification="${ema_alpha}"
    fi
    ;;
esac
if [[ "${allow_sparse_pilot}" == "1" ]]; then
  protocol_classification="engineering-pilot"
elif [[ "${use_audio_in_video}" == "1" ]]; then
  protocol_classification="omni-audio-video"
else
  protocol_classification="visual-control"
fi
printf '%s\n' \
  "experiment_label=${experiment_label}" \
  "arm=${arm}" \
  "framework=ms-swift" \
  "supervision=${supervision}" \
  "teacher_mode=${teacher_mode}" \
  "backbone=${model}" \
  "tuner=${tuner_type}" \
  "freeze_llm=${freeze_llm}" \
  "freeze_vit=${freeze_vit}" \
  "freeze_aligner=${freeze_aligner}" \
  "full_parameter_paper_setting=$([[ "${tuner_type}" == full ]] && printf true || printf false)" \
  "paper_exact=false" \
  "protocol_classification=${protocol_classification}" \
  "divergence_support=$([[ "${arm}" == opsd || "${arm}" == clue_opsd ]] && printf topk_with_tail_mass || printf not-applicable)" \
  "gkd_logits_topk=$([[ "${arm}" == opsd || "${arm}" == clue_opsd ]] && printf '%s' "${gkd_logits_topk}" || printf not-applicable)" \
  "rollout_top_p=$([[ "${arm}" == opsd || "${arm}" == clue_opsd ]] && printf '%s' "${rollout_top_p}" || printf not-applicable)" \
  "rollout_top_k=$([[ "${arm}" == opsd || "${arm}" == clue_opsd ]] && printf '%s' "${rollout_top_k}" || printf not-applicable)" \
  "lora_rank=${lora_rank}" \
  "lora_alpha=${lora_alpha}" \
  "learning_rate=${learning_rate}" \
  "opsd_ema_alpha=$([[ "${arm}" == opsd ]] && printf '%s' "${opsd_ema_alpha}" || printf not-applicable)" \
  "use_vllm=$([[ "${arm}" == opsd || "${arm}" == clue_opsd ]] && printf '%s' "${use_vllm}" || printf not-applicable)" \
  "vllm_mode=$([[ "${arm}" == opsd || "${arm}" == clue_opsd ]] && printf '%s' "${vllm_mode}" || printf not-applicable)" \
  "vllm_tensor_parallel_size=$([[ "${arm}" == opsd || "${arm}" == clue_opsd ]] && printf '%s' "${vllm_tensor_parallel_size}" || printf not-applicable)" \
  "vllm_gpu_memory_utilization=$([[ "${arm}" == opsd || "${arm}" == clue_opsd ]] && printf '%s' "${vllm_gpu_memory_utilization}" || printf not-applicable)" \
  "vllm_sleep_level=$([[ "${arm}" == opsd || "${arm}" == clue_opsd ]] && printf '%s' "${vllm_sleep_level}" || printf not-applicable)" \
  "steps_per_generation=$([[ "${arm}" == opsd || "${arm}" == clue_opsd ]] && printf '%s' "${steps_per_generation:-default}" || printf not-applicable)" \
  "ddp_find_unused_parameters=${ddp_find_unused_parameters:-default}" \
  "clue_ema_alpha=${clue_ema_classification}" \
  "full_ema_teacher=${full_ema_teacher}" \
  "full_ema_alpha=${full_ema_alpha}" \
  "full_ema_offload=${full_ema_offload}" \
  "gold_ce_alpha=${gold_ce_alpha}" \
  "allow_non_ema_clue=${allow_non_ema_clue}" \
  "allow_non_32_global_batch=${allow_non_32_global_batch}" \
  "use_audio_in_video=$([[ "${use_audio_in_video}" == 1 ]] && printf true || printf false)" \
  "model=${model}" \
  "dataset=${dataset}" \
  "dataset_sha256=${dataset_sha256}" \
  "dataset_rows=${dataset_rows}" \
  "resume_from_checkpoint=${resume_checkpoint}" \
  "project_commit=${project_commit}" \
  "ms_swift_commit=${swift_commit}" \
  "nproc=${nproc}" \
  "visible_devices=${ASCEND_RT_VISIBLE_DEVICES}" \
  "max_steps=${max_steps}" \
  "num_train_epochs=${num_train_epochs:-unset}" \
  "allocator_conf=${allocator_conf:-default}" \
  "min_pixels=${min_pixels}" \
  "max_grad_norm=${max_grad_norm}" \
  "split_dataset_ratio=${split_dataset_ratio}" \
  "eval_strategy=${eval_strategy}" \
  "eval_steps=${eval_steps}" \
  "log_completions=${log_completions}" \
  "offload_model=${offload_model}" \
  "offload_optimizer=${offload_optimizer}" \
  "use_logits_to_keep=$([[ "${use_logits_to_keep}" == 1 ]] && printf true || printf false)" \
  "diag_enabled=${diag_enabled}" \
  "diag_top_k=${diag_top_k}" \
  "diag_temperature=${diag_temperature}" \
  "diag_frequency=${diag_frequency}" \
  "diag_chunk_size=${diag_chunk_size}" \
  "diag_answer_labels_path=${diag_answer_labels_path:-unset}" \
  "gkd_safe_mode=${gkd_safe_mode}" \
  "gkd_max_grad_norm=${gkd_max_grad_norm}" \
  "per_device_train_batch_size=${per_device_batch}" \
  "gradient_accumulation_steps=${gradient_accumulation}" \
  "effective_batch_size=${effective_batch}" \
  "attn_impl=${attn_impl}" \
  "seed=${seed}" \
  > "${output_dir}/RUN_CLASSIFICATION.txt"

export PYTHONPATH="${project_root}/src:${python_deps}:${ms_swift_root}${PYTHONPATH:+:${PYTHONPATH}}"
export NPROC_PER_NODE="${nproc}"
export USE_AUDIO_IN_VIDEO="${use_audio_in_video}"
export OMNI_OPSD_GKD_SAFE_MODE="${gkd_safe_mode}"
export OMNI_OPSD_GKD_MAX_GRAD_NORM="${gkd_max_grad_norm}"
export OMNI_OPSD_USE_LOGITS_TO_KEEP="${use_logits_to_keep}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MAX_NUM_WORKERS_FETCH_VIDEO="${MAX_NUM_WORKERS_FETCH_VIDEO:-1}"
if [[ -n "${allocator_conf}" ]]; then
  export PYTORCH_NPU_ALLOC_CONF="${allocator_conf}"
  export PYTORCH_CUDA_ALLOC_CONF="${allocator_conf}"
fi
export PYTHONUNBUFFERED=1

common_args=(
  --model "${model}"
  --dataset "${dataset}"
  --split_dataset_ratio "${split_dataset_ratio}"
  --output_dir "${output_dir}"
  --tuner_type "${tuner_type}"
  --torch_dtype bfloat16
  --max_steps "${max_steps}"
  --per_device_train_batch_size "${per_device_batch}"
  --gradient_accumulation_steps "${gradient_accumulation}"
  --learning_rate "${learning_rate}"
  --lr_scheduler_type cosine
  --warmup_ratio 0.03
  --save_steps "${save_steps}"
  --save_total_limit "${OMNI_OPSD_SAVE_TOTAL_LIMIT:-2}"
  --logging_steps 1
  --max_length "${max_length}"
  --max_grad_norm "${max_grad_norm}"
  --gradient_checkpointing true
  --attn_impl "${attn_impl}"
  --dataset_num_proc 1
  --dataloader_num_workers 0
  --dataloader_persistent_workers false
  --no_dataset_shuffle
  --seed "${seed}"
  --report_to none
)

if [[ "${arm}" == "opsd" || "${arm}" == "clue_opsd" ]]; then
  # Full-parameter Adam states are temporarily moved to CPU during
  # Transformers rollouts so the next multimodal generation has room for
  # its KV cache and visual/audio activations.
  common_args+=(
    --offload_model "${offload_model}"
    --offload_optimizer "${offload_optimizer}"
  )
fi

if [[ "${use_logits_to_keep}" == "1" &&
      ( "${arm}" == "sft" || "${arm}" == "opsd" || "${arm}" == "clue_opsd" ) ]]; then
  common_args+=(--use_logits_to_keep true)
fi

if [[ -n "${OMNI_OPSD_DEEPSPEED:-}" ]]; then
  common_args+=(--deepspeed "${OMNI_OPSD_DEEPSPEED}")
fi
if [[ -n "${OMNI_OPSD_FSDP:-}" ]]; then
  # torch/accelerate built-in sharding; used because deepspeed is not installed
  # in the vllm environment.  full_shard shards params+grads+optimizer.
  common_args+=(--fsdp "${OMNI_OPSD_FSDP}")
fi

if [[ -n "${ddp_find_unused_parameters}" ]]; then
  common_args+=(--ddp_find_unused_parameters "${ddp_find_unused_parameters}")
fi

if [[ "${tuner_type}" == "lora" ]]; then
  common_args+=(
    --lora_rank "${lora_rank}"
    --lora_alpha "${lora_alpha}"
    --target_modules all-linear
  )
else
  # ms-swift's multimodal full tuner freezes the vision tower and aligner by
  # default.  Full student fine-tuning must explicitly unfreeze them.
  common_args+=(
    --freeze_llm "${freeze_llm}"
    --freeze_vit "${freeze_vit}"
    --freeze_aligner "${freeze_aligner}"
  )
fi

if [[ -n "${num_train_epochs}" ]]; then
  common_args+=(--num_train_epochs "${num_train_epochs}")
fi

if [[ -n "${resume_checkpoint}" ]]; then
  common_args+=(--resume_from_checkpoint "${resume_checkpoint}")
fi

if [[ "${eval_strategy}" != "no" ]]; then
  common_args+=(
    --eval_strategy "${eval_strategy}"
    --eval_steps "${eval_steps}"
    --metric_for_best_model eval_loss
    --greater_is_better false
    --load_best_model_at_end true
  )
fi

vllm_args=()
if [[ "${use_vllm}" == "true" ]]; then
  # The ms-swift parser requires an explicit mode whenever use_vllm=true.
  # There is no external rollout server in this project, so colocate the
  # engine with each training rank by default.
  vllm_args+=(
    --vllm_mode "${vllm_mode}"
    --vllm_gpu_memory_utilization "${vllm_gpu_memory_utilization}"
    --vllm_tensor_parallel_size "${vllm_tensor_parallel_size}"
    --sleep_level "${vllm_sleep_level}"
  )
  if [[ -n "${steps_per_generation}" ]]; then
    vllm_args+=(--steps_per_generation "${steps_per_generation}")
  fi
fi

case "${arm}" in
  sft)
    exec "${swift_bin}" sft "${common_args[@]}"
    ;;
  grpo)
    exec "${swift_bin}" rlhf \
      --rlhf_type grpo \
      "${common_args[@]}" \
      --external_plugins "${project_root}/scripts/video_odyssey_grpo_reward.py" \
      --reward_funcs video_odyssey_mcq_accuracy \
      --reward_weights 1.0 \
      --loss_type grpo \
      --beta 0.0 \
      --num_generations 8 \
      --max_completion_length "${max_completion_length}" \
      --temperature 1.0 \
      --top_p 0.95 \
      --top_k 20 \
      --use_vllm false \
      --log_completions true
    ;;
  opsd|clue_opsd)
    diag_answer_args=()
    if [[ -n "${diag_answer_labels_path}" ]]; then
      diag_answer_args+=(--diag_answer_labels_path "${diag_answer_labels_path}")
    fi
    ema_args=()
    if [[ "${arm}" == "opsd" && "${opsd_ema_alpha}" != "0" && "${opsd_ema_alpha}" != "0.0" ]]; then
      ema_args+=(--opsd_ema_alpha "${opsd_ema_alpha}")
    fi
    if [[ "${arm}" == "clue_opsd" && "${allow_non_ema_clue}" == "0" ]]; then
      ema_args+=(--clue_ema_alpha "${ema_alpha}")
    fi
    if [[ "${arm}" == "clue_opsd" && "${full_ema_teacher}" == "true" ]]; then
      ema_args+=(
        --full_ema_teacher true
        --full_ema_alpha "${full_ema_alpha}"
        --full_ema_offload "${full_ema_offload}"
        --gold_ce_alpha "${gold_ce_alpha}"
      )
    fi
    exec "${swift_bin}" rlhf \
      --rlhf_type gkd \
      "${common_args[@]}" \
      --lmbda 1.0 \
      --beta 0.5 \
      --temperature 1.0 \
      "${ema_args[@]}" \
      --sft_alpha 0 \
      --max_completion_length "${max_completion_length}" \
      --gkd_logits_topk "${gkd_logits_topk}" \
      --top_p "${rollout_top_p}" \
      --top_k "${rollout_top_k}" \
      --use_vllm "${use_vllm}" \
      --log_completions "${log_completions}" \
      --diag_enabled "${diag_enabled}" \
      --diag_top_k "${diag_top_k}" \
      --diag_temperature "${diag_temperature}" \
      --diag_frequency "${diag_frequency}" \
      --diag_chunk_size "${diag_chunk_size}" \
      "${diag_answer_args[@]}" \
      "${vllm_args[@]}"
    ;;
esac
