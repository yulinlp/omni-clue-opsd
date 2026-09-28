#!/usr/bin/env bash
set -euo pipefail

# CUDA/A100 entry point for one of the four training arms.  The shared arm
# launcher historically used ASCEND_RT_VISIBLE_DEVICES for its device-count
# guard.  On CUDA this wrapper mirrors CUDA_VISIBLE_DEVICES into that variable
# only for the guard; PyTorch/ms-swift still select GPUs from CUDA_VISIBLE_DEVICES.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
arm="${1:?usage: run_training_arm_a100.sh sft|grpo|opsd|clue_opsd}"
case "${arm}" in
  sft|grpo|opsd|clue_opsd) ;;
  *) echo "Unsupported arm: ${arm}" >&2; exit 2 ;;
esac

if [[ -n "${OMNI_OPSD_ARM:-}" && "${OMNI_OPSD_ARM}" != "${arm}" ]]; then
  echo "OMNI_OPSD_ARM=${OMNI_OPSD_ARM} disagrees with positional arm=${arm}" >&2
  exit 2
fi
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES, for example 0,1,2,3}"

IFS=',' read -r -a cuda_devices <<< "${CUDA_VISIBLE_DEVICES}"
for device in "${cuda_devices[@]}"; do
  [[ "${device}" =~ ^[0-9]+$ ]] || {
    echo "CUDA_VISIBLE_DEVICES must be a comma-separated list of integer IDs" >&2
    exit 2
  }
done
if [[ "${#cuda_devices[@]}" -lt 1 ]]; then
  echo "CUDA_VISIBLE_DEVICES is empty" >&2
  exit 2
fi

nproc="${OMNI_OPSD_NPROC_PER_NODE:-${#cuda_devices[@]}}"
if [[ "${nproc}" != "${#cuda_devices[@]}" ]]; then
  echo "OMNI_OPSD_NPROC_PER_NODE=${nproc} must match CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

export OMNI_OPSD_ARM="${arm}"
export OMNI_OPSD_NPROC_PER_NODE="${nproc}"
export ASCEND_RT_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export OMNI_OPSD_GKD_SAFE_MODE="${OMNI_OPSD_GKD_SAFE_MODE:-0}"
export OMNI_OPSD_GKD_MAX_GRAD_NORM="${OMNI_OPSD_GKD_MAX_GRAD_NORM:-1.0}"
export OMNI_OPSD_ATTN_IMPL="${OMNI_OPSD_ATTN_IMPL:-sdpa}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29801}"

exec bash "${script_dir}/run_video_odyssey_training_arm.sh"
