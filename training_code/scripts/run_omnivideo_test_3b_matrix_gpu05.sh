#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset="${project_root}/output/omnivideo_test_505_eval/omnivideo_test_505.answer_free.jsonl"
labels="${project_root}/output/omnivideo_test_505_eval/omnivideo_test_505.labels.jsonl"
base="/share/home/ylhu/models/Qwen2.5-Omni-3B"
eval_root="${project_root}/output/omnivideo_test_505_matrix_gpu05_20260917"

declare -a arms=(clue_opsd sft opsd)
declare -a models=(
  "${project_root}/output/omnivideo_oe5k_clue_opsd_3b_full_ema_gpu07_6gpu_20260916_resume100_repaired_v2/clue_opsd/v0-20260916-222515/checkpoint-141"
  "${project_root}/output/omnivideo_oe5k_sft_3b_full_gpu04_20260915_oe5k_sft_3b_full_gpu04_retry2/sft/v0-20260915-151958/checkpoint-157"
  "${project_root}/output/omnivideo_oe5k_opsd_3b_full_gpu05_20260916_oe5k_opsd_3b_full_gpu05_repaired_v1/opsd/v0-20260916-120504/checkpoint-157"
)

mkdir -p "${eval_root}"
for i in "${!arms[@]}"; do
  arm="${arms[$i]}"
  model="${models[$i]}"
  out="${eval_root}/${arm}"
  if [[ -s "${out}/summary.json" ]]; then
    echo "already complete: ${arm}"
    continue
  fi
  echo "[$(date '+%F %T')] starting ${arm}"
  env \
    OMNI_OPSD_PROJECT_ROOT="${project_root}" \
    OMNI_OPSD_MODEL="${model}" \
    OMNI_OPSD_EVAL_DATASET="${dataset}" \
    OMNI_OPSD_EVAL_LABELS="${labels}" \
    OMNI_OPSD_EVAL_OUTPUT_DIR="${out}" \
    OMNI_OPSD_EVAL_ARM="${arm}" \
    OMNI_OPSD_CUDA_DEVICES=0,1,2,3 \
    OMNI_OPSD_NPROC_PER_NODE=4 \
    OMNI_OPSD_EVAL_ATTN_IMPL=sdpa \
    MASTER_PORT=29981 \
    bash "${project_root}/scripts/run_omnivideo_test_eval.sh"
  echo "[$(date '+%F %T')] completed ${arm}"
done
echo "[$(date '+%F %T')] all requested OmniVideo-Test evaluations complete"
