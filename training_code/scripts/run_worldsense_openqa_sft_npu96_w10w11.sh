#!/usr/bin/env bash
# Open-QA SFT on npu96 worker-10/11. The existing MCQ SFT launcher is unchanged.
set -euo pipefail

node_rank="${1:?usage: $0 NODE_RANK(0|1)}"
[[ "${node_rank}" == 0 || "${node_rank}" == 1 ]] || exit 2

repo="/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd"
runtime="/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_sft_lora_npu24_20260928"
run_root="${repo}/training_runs/worldsense_openqa_20260929"
export OMNI_OPSD_DATASET="${OMNI_OPSD_DATASET:-${run_root}/data/sft_openqa.jsonl}"
export OMNI_OPSD_OUTPUT_DIR="${OMNI_OPSD_OUTPUT_DIR:-${run_root}/outputs/sft_formal}"
export OMNI_OPSD_MASTER_PORT="${OMNI_OPSD_MASTER_PORT:-29523}"

[[ -s "${OMNI_OPSD_DATASET}" ]] || { echo "missing data: ${OMNI_OPSD_DATASET}" >&2; exit 2; }
exec bash "${runtime}/scripts/run_sft_lora_npu96_w10w11.sh" "${node_rank}"
