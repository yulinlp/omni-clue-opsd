#!/usr/bin/env bash
set -euo pipefail

# This launcher exercises ms-swift's on-policy GKD path with separate full-video
# Student and clue-video Teacher media. The public checkout does not implement
# the paper's EMA Teacher; the EMA mode below is accepted only when a functional
# fork containing the trainer-side shadow update is installed. The remaining
# modes are explicitly classified as non-EMA diagnostics.

run_kind="${OMNI_OPSD_RUN_KIND:-dynamic_teacher_non_ema_smoke}"
case "${run_kind}" in
  dynamic_teacher_non_ema_smoke|dynamic_teacher_non_ema_stability|clue_opsd_lora_ema_core) ;;
  *)
    echo "Unsupported run kind: ${run_kind}; EMA Teacher is not implemented yet" >&2
    exit 2
    ;;
esac

model="${OMNI_OPSD_MODEL:?set OMNI_OPSD_MODEL}"
dataset="${OMNI_OPSD_DATASET:?set OMNI_OPSD_DATASET}"
output_dir="${OMNI_OPSD_OUTPUT_DIR:?set OMNI_OPSD_OUTPUT_DIR}"
ms_swift_root="${OMNI_OPSD_MS_SWIFT_ROOT:?set OMNI_OPSD_MS_SWIFT_ROOT}"
python_deps="${OMNI_OPSD_PYTHON_DEPS:?set OMNI_OPSD_PYTHON_DEPS}"
swift_bin="${OMNI_OPSD_SWIFT_BIN:-swift}"
nproc="${OMNI_OPSD_NPROC_PER_NODE:-8}"
max_steps="${OMNI_OPSD_MAX_STEPS:-1}"
gradient_accumulation="${OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS:-4}"
max_length="${OMNI_OPSD_MAX_LENGTH:-32768}"
max_completion_length="${OMNI_OPSD_MAX_COMPLETION_LENGTH:-8}"
save_steps="${OMNI_OPSD_SAVE_STEPS:-1}"
gkd_logits_topk="${OMNI_OPSD_GKD_LOGITS_TOPK:-100}"
rollout_top_p="${OMNI_OPSD_ROLLOUT_TOP_P:-1.0}"
rollout_top_k="${OMNI_OPSD_ROLLOUT_TOP_K:-20}"
use_vllm="${OMNI_OPSD_USE_VLLM:-true}"
vllm_mode="${OMNI_OPSD_VLLM_MODE:-colocate}"
vllm_gpu_memory_utilization="${OMNI_OPSD_VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
vllm_sleep_level="${OMNI_OPSD_VLLM_SLEEP_LEVEL:-1}"
ema_alpha="${OMNI_OPSD_CLUE_EMA_ALPHA:-}"
resume_checkpoint="${OMNI_OPSD_RESUME_FROM_CHECKPOINT:-}"
experiment_label="${OMNI_OPSD_EXPERIMENT_LABEL:-unspecified}"

teacher_mode="dynamic-current-policy-not-ema"
core_algorithm_faithful="false"
ema_args=()
resume_args=()
vllm_args=()
if [[ "${run_kind}" == "clue_opsd_lora_ema_core" ]]; then
  if [[ -z "${ema_alpha}" ]]; then
    echo "OMNI_OPSD_CLUE_EMA_ALPHA is required for clue_opsd_lora_ema_core" >&2
    exit 2
  fi
  # Do not label a run as EMA merely because the CLI flag is present.  The
  # public checkout lacks the trainer-side shadow parameter update; require
  # the actual implementation before accepting this mode.
  if ! grep -q 'clue_ema_alpha' "${ms_swift_root}/swift/rlhf_trainers/gkd_trainer.py" || \
     ! grep -q 'clue_ema_alpha' "${ms_swift_root}/swift/arguments/rlhf_args.py"; then
    echo "ms-swift checkout lacks the LoRA-shadow EMA implementation; use a functional fork or a non-EMA diagnostic" >&2
    exit 2
  fi
  teacher_mode="lora-shadow-ema"
  core_algorithm_faithful="true"
  ema_args=(--clue_ema_alpha "${ema_alpha}")
fi
if [[ -n "${resume_checkpoint}" ]]; then
  if [[ ! -d "${resume_checkpoint}" ]]; then
    echo "Resume checkpoint does not exist: ${resume_checkpoint}" >&2
    exit 2
  fi
  resume_args=(--resume_from_checkpoint "${resume_checkpoint}")
fi

if [[ ! "${gkd_logits_topk}" =~ ^[1-9][0-9]*$ ]]; then
  echo "OMNI_OPSD_GKD_LOGITS_TOPK must be a positive integer" >&2
  exit 2
fi
if [[ "${rollout_top_p}" != "1.0" ]]; then
  echo "OMNI_OPSD_ROLLOUT_TOP_P must be 1.0 for the paper-aligned CLUE-OPSD run" >&2
  exit 2
fi
if [[ "${rollout_top_k}" != "20" ]]; then
  echo "OMNI_OPSD_ROLLOUT_TOP_K must be 20 for the paper-aligned CLUE-OPSD run" >&2
  exit 2
fi
if [[ ! "${vllm_gpu_memory_utilization}" =~ ^(0\.[0-9]+|1([.]0+)?)$ ]]; then
  echo "OMNI_OPSD_VLLM_GPU_MEMORY_UTILIZATION must be in (0, 1]" >&2
  exit 2
fi
if [[ ! "${vllm_sleep_level}" =~ ^[012]$ ]]; then
  echo "OMNI_OPSD_VLLM_SLEEP_LEVEL must be 0, 1, or 2" >&2
  exit 2
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
  vllm_args+=(
    --vllm_mode "${vllm_mode}"
    --vllm_gpu_memory_utilization "${vllm_gpu_memory_utilization}"
    --sleep_level "${vllm_sleep_level}"
  )
fi

: "${ASCEND_RT_VISIBLE_DEVICES:?set ASCEND_RT_VISIBLE_DEVICES explicitly}"
IFS=',' read -r -a visible_devices <<< "${ASCEND_RT_VISIBLE_DEVICES}"
if [[ "${#visible_devices[@]}" -ne "${nproc}" ]]; then
  echo "Expected ${nproc} visible NPUs, got ${#visible_devices[@]}" >&2
  exit 2
fi
if [[ "${USE_AUDIO_IN_VIDEO:-0}" != "0" ]]; then
  echo "VideoOdyssey full audio is 1-4 hours; this visual CLUE control requires USE_AUDIO_IN_VIDEO=0" >&2
  exit 2
fi
for path in "${model}" "${dataset}" "${ms_swift_root}/swift/rl_core/data.py"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path is missing: ${path}" >&2
    exit 2
  fi
done
if ! grep -q 'teacher_videos' "${ms_swift_root}/swift/rl_core/data.py" || \
   ! grep -q 'teacher_audios' "${ms_swift_root}/swift/rl_core/data.py"; then
  echo "ms-swift checkout lacks the required teacher video/audio patch" >&2
  exit 2
fi

python_bin="$(command -v python)"
if [[ "${use_vllm}" == "true" ]]; then
  if ! PYTHONPATH="${python_deps}:${ms_swift_root}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${python_bin}" -c 'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("vllm") else 1)'; then
    echo "vLLM is enabled but is not installed in ${python_bin}; install a CUDA/PyTorch-compatible vLLM before training" >&2
    exit 2
  fi
fi
PYTHONPATH="${python_deps}:${ms_swift_root}" "${python_bin}" - "${dataset}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    row = json.loads(next(line for line in handle if line.strip()))
if not row.get("teacher_videos"):
    raise SystemExit("dataset row has no teacher_videos")
if row.get("sampling_contract", {}).get("use_audio_in_video") is not False:
    raise SystemExit("dataset contract must disable unbounded VideoOdyssey audio")
if "answer" in row:
    raise SystemExit("answer label leaked into the model-facing training row")
PY

mkdir -p "${output_dir}"
dataset_sha256="$(sha256sum "${dataset}" | awk '{print $1}')"
dataset_rows="$(awk 'NF {count++} END {print count+0}' "${dataset}")"
printf '%s\n' \
  "experiment_label=${experiment_label}" \
  "run_kind=${run_kind}" \
  "teacher_mode=${teacher_mode}" \
  "core_algorithm_faithful=${core_algorithm_faithful}" \
  "paper_faithful=false" \
  "paper_exact=false" \
  "clue_ema_alpha=${ema_alpha}" \
  "resume_from_checkpoint=${resume_checkpoint}" \
  "use_audio_in_video=false" \
  "gkd_logits_topk=${gkd_logits_topk}" \
  "rollout_top_p=${rollout_top_p}" \
  "rollout_top_k=${rollout_top_k}" \
  "use_vllm=${use_vllm}" \
  "vllm_mode=${vllm_mode}" \
  "vllm_gpu_memory_utilization=${vllm_gpu_memory_utilization}" \
  "vllm_sleep_level=${vllm_sleep_level}" \
  "model=${model}" \
  "dataset=${dataset}" \
  "dataset_sha256=${dataset_sha256}" \
  "dataset_rows=${dataset_rows}" \
  "ms_swift_root=${ms_swift_root}" \
  "max_steps=${max_steps}" \
  > "${output_dir}/RUN_CLASSIFICATION.txt"

export PYTHONPATH="${python_deps}:${ms_swift_root}${PYTHONPATH:+:${PYTHONPATH}}"
export NPROC_PER_NODE="${nproc}"
export USE_AUDIO_IN_VIDEO=0
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

exec "${swift_bin}" rlhf \
  --rlhf_type gkd \
  --model "${model}" \
  --dataset "${dataset}" \
  --output_dir "${output_dir}" \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --lmbda 1.0 \
  --beta 0.5 \
  --temperature 1.0 \
  "${ema_args[@]}" \
  "${resume_args[@]}" \
  --sft_alpha 0 \
  --torch_dtype bfloat16 \
  --max_steps "${max_steps}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${gradient_accumulation}" \
  --learning_rate 2e-6 \
  --warmup_ratio 0.03 \
  --save_steps "${save_steps}" \
  --save_total_limit 2 \
  --logging_steps 1 \
  --max_length "${max_length}" \
  --max_completion_length "${max_completion_length}" \
  --gkd_logits_topk "${gkd_logits_topk}" \
  --top_p "${rollout_top_p}" \
  --top_k "${rollout_top_k}" \
  --gradient_checkpointing true \
  --attn_impl eager \
  --use_vllm "${use_vllm}" \
  "${vllm_args[@]}" \
  --dataset_num_proc 1 \
  --dataloader_num_workers 0 \
  --no_dataset_shuffle \
  --report_to none
