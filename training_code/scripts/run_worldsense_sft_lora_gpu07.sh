#!/usr/bin/env bash
# WorldSense SFT (LoRA r64/a128) on gpu07, 4 GPUs, 3 epochs, save every epoch.
set -uo pipefail
project_root="/share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907"
cd "$project_root"
export OMNI_OPSD_DATASET="${project_root}/data/worldsense_gap/sft/formal/data/sft.jsonl"
export OMNI_OPSD_MODEL=/share/home/ylhu/models/Qwen2.5-Omni-7B
export OMNI_OPSD_CUDA_DEVICES=0,1,2,3
export OMNI_OPSD_NPROC_PER_NODE=4
export OMNI_OPSD_MAX_STEPS=138          # 3 epochs x (1453/32 ~ 46)
export OMNI_OPSD_SAVE_STEPS=46          # one checkpoint per epoch
export OMNI_OPSD_NUM_TRAIN_EPOCHS=3
export OMNI_OPSD_TUNER_TYPE=lora
export OMNI_OPSD_LORA_RANK=64
export OMNI_OPSD_LORA_ALPHA=128
export OMNI_OPSD_LEARNING_RATE=1e-5
export OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE=1
export OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS=8
export OMNI_OPSD_USE_LOGITS_TO_KEEP=1   # only last-token logits: saves ~20GB (batch x 32k x 152k vocab)
export OMNI_OPSD_SFT_ROOT="${project_root}/output/worldsense_sft_lora_gpu07"
export OMNI_OPSD_EXPERIMENT_LABEL=worldsense_sft_lora_gpu07
exec bash "${project_root}/scripts/run_gap5000_sft_cuda.sh" formal
