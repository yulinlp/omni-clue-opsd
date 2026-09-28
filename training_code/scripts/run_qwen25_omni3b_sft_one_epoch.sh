#!/usr/bin/env bash
set -euo pipefail

# One-epoch CUDA SFT launcher for Qwen2.5-Omni-3B.  The script is intended to
# run inside the user's existing gpu02 Slurm allocation via srun --overlap.
# It keeps the data/sampling contract from the SFT handoff document and makes
# the effective batch-size invariant explicit.

project_root="${OMNI_OPSD_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_env="${OMNI_OPSD_ENV:-/share/home/ylhu/.conda/envs/omniopsd_train}"
swift_root="${OMNI_OPSD_MS_SWIFT_ROOT:-${project_root}/ms-swift}"
model="${OMNI_OPSD_MODEL:-/share/home/ylhu/models/Qwen2.5-Omni-3B}"
# Default to the ID-frozen 5k_oe direct-answer SFT set.  Callers can still
# override this with OMNI_OPSD_DATASET for an older reproduction run.
dataset="${OMNI_OPSD_DATASET:-${project_root}/data/omnivideo_oe_5k/omnivideo_oe_5k.sft.jsonl}"
devices="${OMNI_OPSD_CUDA_DEVICES:-0,1,2,3}"
nproc="${OMNI_OPSD_NPROC_PER_NODE:-4}"
per_device_batch="${OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
gradient_accumulation="${OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS:-8}"
output_dir="${OMNI_OPSD_OUTPUT_DIR:?set OMNI_OPSD_OUTPUT_DIR to a unique output directory}"
master_addr="${MASTER_ADDR:-127.0.0.1}"
master_port="${MASTER_PORT:-29911}"

IFS=',' read -r -a device_list <<< "${devices}"
if [[ "${#device_list[@]}" -ne 4 || "${nproc}" -ne 4 ]]; then
  echo "This launcher requires four CUDA devices and nproc=4: devices=${devices} nproc=${nproc}" >&2
  exit 2
fi
if [[ "$((nproc * per_device_batch * gradient_accumulation))" -ne 32 ]]; then
  echo "effective batch must be 32: ${nproc}*${per_device_batch}*${gradient_accumulation}" >&2
  exit 2
fi
for path in "${model}" "${dataset}" "${swift_root}/swift/cli/sft.py"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path is missing: ${path}" >&2
    exit 2
  fi
done

rows="$(awk 'NF {count++} END {print count+0}' "${dataset}")"
if [[ "${rows}" -ne 5000 ]]; then
  echo "Expected 5000 training rows, found ${rows}" >&2
  exit 2
fi

mkdir -p "${output_dir}"
dataset_sha256="$(sha256sum "${dataset}" | awk '{print $1}')"
project_commit="$(git -C "${project_root}" rev-parse HEAD 2>/dev/null || printf unknown)"
swift_commit="$(git -C "${swift_root}" rev-parse HEAD 2>/dev/null || printf unknown)"
cat > "${output_dir}/launch_config.txt" <<EOF
model=${model}
dataset=${dataset}
dataset_rows=${rows}
dataset_sha256=${dataset_sha256}
devices=${devices}
nproc=${nproc}
per_device_train_batch_size=${per_device_batch}
gradient_accumulation_steps=${gradient_accumulation}
effective_batch_size=$((nproc * per_device_batch * gradient_accumulation))
num_train_epochs=1
max_steps=-1
max_length=32768
min_pixels=3136
max_pixels=28672
max_frames=768
use_audio_in_video=1
project_commit=${project_commit}
ms_swift_commit=${swift_commit}
started_at=$(date --iso-8601=seconds)
EOF

export CUDA_VISIBLE_DEVICES="${devices}"
export ASCEND_RT_VISIBLE_DEVICES="${devices}"
export NPROC_PER_NODE="${nproc}"
export MASTER_ADDR="${master_addr}"
export MASTER_PORT="${master_port}"
export PYTHONPATH="${project_root}/src:${project_root}:${swift_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${python_env}/bin:${PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MAX_NUM_WORKERS_FETCH_VIDEO="${MAX_NUM_WORKERS_FETCH_VIDEO:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# The original 3B run recorded audio in its launch contract but did not export
# this runtime switch, so ms-swift silently used USE_AUDIO_IN_VIDEO=False.
# Keep the historical entry point aligned with the Omni audio/video contract.
export USE_AUDIO_IN_VIDEO="${USE_AUDIO_IN_VIDEO:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "${python_env}/bin/swift" sft \
  --model "${model}" \
  --dataset "${dataset}" \
  --split_dataset_ratio 0 \
  --output_dir "${output_dir}" \
  --tuner_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --max_steps -1 \
  --per_device_train_batch_size "${per_device_batch}" \
  --gradient_accumulation_steps "${gradient_accumulation}" \
  --learning_rate 2e-6 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --save_steps 25 \
  --save_total_limit 2 \
  --logging_steps 1 \
  --max_length 32768 \
  --max_grad_norm 1.0 \
  --gradient_checkpointing true \
  --attn_impl sdpa \
  --dataset_num_proc 1 \
  --dataloader_num_workers 0 \
  --dataloader_persistent_workers false \
  --no_dataset_shuffle \
  --seed 20260904 \
  --report_to none
