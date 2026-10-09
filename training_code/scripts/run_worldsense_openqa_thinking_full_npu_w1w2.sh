#!/usr/bin/env bash
set -euo pipefail
node_rank="${1:?usage: $0 NODE_RANK(0|1)}"
[[ "$node_rank" == 0 || "$node_rank" == 1 ]] || exit 2
repo=/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd
run_root="${OMNI_OPSD_RUN_ROOT:-$repo/training_runs/worldsense_openqa_thinking_full_20260930}"
deps=/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_npu24_deps
swift_root=/opt/huawei/dataset/hyl_ulan/ylhu/third_party/ms-swift
python_bin=/home/ma-user/anaconda3/envs/PyTorch-2.9.0/bin/python
export PYTHONPATH="$deps:$swift_root${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RANK_TABLE_FILE="${OMNI_OPSD_RANK_TABLE_FILE:-$run_root/data/ranktable_w1w2.json}"
export RANK_TABLE_FILE_V_1_0="$RANK_TABLE_FILE"
export USE_AUDIO_IN_VIDEO=1 FORCE_QWENVL_VIDEO_READER=pyav_seek MAX_NUM_WORKERS_FETCH_VIDEO=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export WORLDSENSE_LOGITS_TO_KEEP=1 WORLDSENSE_DROP_TALKER=1 WORLDSENSE_CKPT_LM_HEAD=1
export HCCL_CONNECT_TIMEOUT=600 PYTHONUNBUFFERED=1
export OMNI_FULL_CPU_THREADS=4
dataset="${OMNI_OPSD_DATASET:-$run_root/data/clue_openqa_thinking.jsonl}"
output="${OMNI_OPSD_OUTPUT_DIR:-$run_root/outputs/formal}"
args=(
  --rlhf_type gkd
  --model /opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-assets/models/Qwen2.5-Omni-7B
  --dataset "$dataset" --output_dir "$output" --split_dataset_ratio 0
  --tuner_type full --freeze_llm false --freeze_vit false --freeze_aligner false
  --lmbda 1.0 --sft_alpha 0 --beta 0.5 --temperature 1.0 --clue_ema_alpha 0.05
  --deepspeed "$repo/training_code/configs/zero2_npu_cpu_offload.json"
  --optim adamw_torch --torch_dtype bfloat16 --num_train_epochs 3
  --per_device_train_batch_size 1 --gradient_accumulation_steps 2
  --learning_rate 2e-6 --lr_scheduler_type cosine --warmup_ratio 0.03
  --save_strategy epoch --save_total_limit 3
  --logging_steps 1 --max_length 32768 --max_completion_length 512
  --use_logits_to_keep true --gradient_checkpointing true
  --gradient_checkpointing_kwargs '{"use_reentrant":false}'
  --attn_impl sdpa --use_vllm false
  --dataset_num_proc 1 --dataloader_num_workers 0 --no_dataset_shuffle
  --seed 20260904 --report_to none --log_completions true
)
if [[ -n "${OMNI_OPSD_MAX_STEPS:-}" ]]; then args+=(--max_steps "$OMNI_OPSD_MAX_STEPS"); fi
if [[ -n "${OMNI_OPSD_SAVE_STEPS:-}" ]]; then args+=(--save_strategy steps --save_steps "$OMNI_OPSD_SAVE_STEPS"); fi
exec "$python_bin" -m torch.distributed.run \
  --nproc_per_node 8 --nnodes 2 --node_rank "$node_rank" \
  --master_addr "${OMNI_OPSD_MASTER_ADDR:-172.16.12.237}" --master_port "${OMNI_OPSD_MASTER_PORT:-29541}" \
  "$repo/training_code/scripts/worldsense_full_gkd_entry.py" "${args[@]}"
