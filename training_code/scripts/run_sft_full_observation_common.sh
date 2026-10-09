#!/usr/bin/env bash
# Full SFT argument template, derived from the observation LoRA launcher.
# Same data, prompt, optimization settings and media preprocessing;
# all thinker parameters train, with ZeRO-2 CPU optimizer offload.
set -euo pipefail   # 安全模式：出错即停 / 未定义变量报错 / 管道失败即失败

# ---- 读取第一个参数并校验 ------------------------------------------------------
node_rank="${1:?usage: $0 NODE_RANK}"
case "${node_rank}" in
  0|1|2) ;;
  *) echo "NODE_RANK must be 0, 1 or 2" >&2; exit 2 ;;
esac

# ---- 路径类变量（全部支持环境变量覆盖） ----------------------------------------
# 语法说明："${VAR:-默认}" = 有外部值用外部值，否则用默认值。
# 这样上一层只需导出要改的变量，其余保持默认。
run_root="${OMNI_OPSD_RUN_ROOT:-/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_sft_lora_npu24_20260928}"
model="${OMNI_OPSD_MODEL:-/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-assets/models/Qwen2.5-Omni-7B}"          # 基座模型
dataset="${OMNI_OPSD_DATASET:-${run_root}/data/sft_worldsense_npu24.jsonl}"                                      # 训练数据
output_dir="${OMNI_OPSD_OUTPUT_DIR:-${run_root}/outputs/sft}"                                                    # 输出目录
deps_dir="${OMNI_OPSD_DEPS_DIR:-/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_npu24_deps}"      # 本机依赖补丁目录
ms_swift_root="${OMNI_OPSD_MS_SWIFT_ROOT:-/opt/huawei/dataset/hyl_ulan/ylhu/third_party/ms-swift}"               # ms-swift 源码目录
python_env="${OMNI_OPSD_PYTHON_ENV:-/home/ma-user/anaconda3/envs/PyTorch-2.9.0}"                                 # Python 环境
swift_bin="${OMNI_OPSD_SWIFT_BIN:-${python_env}/bin/swift}"                                                      # swift 可执行文件

# ---- 分布式配置（多机训练的核心 4+1 个变量） ------------------------------------
nnodes="${OMNI_OPSD_NNODES:-3}"                    # 机器总数（默认 3 节点）
nproc="${OMNI_OPSD_NPROC_PER_NODE:-8}"             # 每机进程数 = 每机 NPU 数
master_addr="${OMNI_OPSD_MASTER_ADDR:-172.16.2.181}"  # 主节点 IP（默认 worker-0）
master_port="${OMNI_OPSD_MASTER_PORT:-29501}"      # 主节点端口

# ---- 训练超参数（默认值即本次实验口径，可被上一层覆盖） ------------------------
num_train_epochs="${OMNI_OPSD_NUM_TRAIN_EPOCHS:-3}"                 # 训练轮数
save_strategy="${OMNI_OPSD_SAVE_STRATEGY:-epoch}"                   # 每轮结束保存
save_total_limit="${OMNI_OPSD_SAVE_TOTAL_LIMIT:-3}"                 # 最多保留 3 个 checkpoint
per_device_batch="${OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"      # 每卡 batch=1（视频样本很大）
gradient_accumulation="${OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS:-1}" # 梯度累积步数
learning_rate="${OMNI_OPSD_LEARNING_RATE:-1e-5}"                    # 学习率
max_length="${OMNI_OPSD_MAX_LENGTH:-32768}"                         # 单样本最大 token 数（模型上限）
attn_impl="${OMNI_OPSD_ATTN_IMPL:-sdpa}"                            # 注意力实现：sdpa 省显存
max_steps="${OMNI_OPSD_MAX_STEPS:-}"                                # 可选：限制总步数（冒烟时用）

# ---- 前置检查：关键路径必须存在，否则立刻报错退出（避免跑到一半才失败） ---------
for path in "${model}" "${dataset}" "${ms_swift_root}/swift/cli/sft.py" "${deps_dir}/qwen_omni_utils"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path is missing: ${path}" >&2
    exit 2
  fi
done

# ---- 导出 Python 搜索路径与分布式环境变量 --------------------------------------
# PYTHONPATH 的顺序很重要：放在前面的目录优先被 import。
#   1) deps_dir    ：本机补丁（pyav_seek 视频读取器 + sitecustomize 显存适配）
#   2) ms_swift_root：打过补丁的 ms-swift 4.6.0.dev0 源码
#   3) 原有的 PYTHONPATH（如有）
export PYTHONPATH="${deps_dir}:${ms_swift_root}${PYTHONPATH:+:${PYTHONPATH}}"
export NPROC_PER_NODE="${nproc}"     # ms-swift 看到这些变量后会自动调用 torchrun
export NNODES="${nnodes}"
export NODE_RANK="${node_rank}"
export MASTER_ADDR="${master_addr}"
export MASTER_PORT="${master_port}"
# 本进程可见的 NPU 编号（8 卡全用）。
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# ---- 多媒体相关环境变量（保持与筛选阶段一致的音视频口径） ----------------------
# 说明：这些设置决定“模型看到什么样的视频/音频”，属于实验协议的一部分。
#   USE_AUDIO_IN_VIDEO=1        ：读取视频时同时读取音轨；
#   FORCE_QWENVL_VIDEO_READER   ：指定视频解码后端为 pyav_seek（本机 aarch64 没有
#                                 decord；torchvision 整片解码会触发 swscale 报错）；
#   MAX_NUM_WORKERS_FETCH_VIDEO ：解码视频的线程数（1 个，避免内存峰值）；
#   OMP/MKL_NUM_THREADS=1       ：限制 CPU 线程，防止多进程互相抢核；
#   TOKENIZERS_PARALLELISM=false：避免 tokenizer 多进程告警。
export USE_AUDIO_IN_VIDEO=1
export FORCE_QWENVL_VIDEO_READER=pyav_seek
export MAX_NUM_WORKERS_FETCH_VIDEO=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
# NPU 显存分配器使用“可扩展段”，减少碎片导致的 OOM。
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# ---- 本机显存适配开关（由 sitecustomize.py 读取，详见文档 §5.4） ----------------
#   DROP_TALKER=1     ：SFT 只用 thinker（理解），卸载用不到的语音生成模块（省 ~4.5GB）；
#   CKPT_LM_HEAD=0    ：默认不用“lm_head 重算”，保留备用；
#   LOGITS_TO_KEEP=1  ：只对答案 token 计算 logits/grad（等效 A100 的 use_logits_to_keep）。
export WORLDSENSE_DROP_TALKER="${WORLDSENSE_DROP_TALKER:-1}"
export WORLDSENSE_CKPT_LM_HEAD="${WORLDSENSE_CKPT_LM_HEAD:-0}"
export WORLDSENSE_LOGITS_TO_KEEP="${WORLDSENSE_LOGITS_TO_KEEP:-1}"
# HCCL 建链超时 600 秒：多机首次建链较慢时给足时间。
export HCCL_CONNECT_TIMEOUT=600
# 让 Python 输出不缓冲，日志能实时看到（否则可能攒一大块才落盘）。
export PYTHONUNBUFFERED=1

# ---- 写一份“运行配置单”，方便事后审计（记录本次实际使用的参数） ----------------
mkdir -p "${output_dir}" "${run_root}/logs"
cat > "${output_dir}/RUN_CLASSIFICATION_npu24_node${node_rank}.txt" <<EOF
experiment_label=worldsense_observation_sft_full_npu96_w6w7
arm=sft
framework=ms-swift
supervision=observation-plus-answer-teacher-forcing
backbone=${model}
tuner=full
freeze_llm=false
freeze_vit=false
freeze_aligner=false
learning_rate=${learning_rate}
num_train_epochs=${num_train_epochs}
save_strategy=${save_strategy}
save_total_limit=${save_total_limit}
per_device_train_batch_size=${per_device_batch}
gradient_accumulation_steps=${gradient_accumulation}
effective_batch_size=$((nnodes * nproc * per_device_batch * gradient_accumulation))
max_length=${max_length}
attn_impl=${attn_impl}
gradient_checkpointing=true
use_logits_to_keep=true (sitecustomize routes the mask into lm_head slicing)
max_grad_norm=0
dataset=${dataset}
dataset_rows=$(awk 'NF {n++} END {print n+0}' "${dataset}")
node_rank=${node_rank}/${nnodes}
nproc_per_node=${nproc}
master=${master_addr}:${master_port}
use_audio_in_video=true
video_reader=pyav_seek
min_pixels=3136 (per-row, unchanged)
max_pixels=per-row, unchanged
started_at=$(date --iso-8601=seconds)
EOF

# ---- 组装 swift sft 的参数数组 --------------------------------------------------
# Bash 数组：args=( 元素1 元素2 ... )，之后用 "${args[@]}" 原样展开。
# 每个 --参数 的含义见行内注释；这些是 ms-swift/transformers 的训练参数。
args=(
  --model "${model}"                        # 基座模型路径
  --dataset "${dataset}"                    # 训练数据（JSONL）
  --split_dataset_ratio 0                   # 不切验证集（1453 条全部训练）
  --output_dir "${output_dir}"              # checkpoint/日志输出目录
  --tuner_type full                         # 全参数微调
  --freeze_llm false --freeze_vit false --freeze_aligner false
  --deepspeed /opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_code/configs/zero2_npu_cpu_offload.json
  --optim adamw_torch
  --torch_dtype bfloat16                    # 计算/权重精度 bf16
  --num_train_epochs "${num_train_epochs}"  # 3 轮
  --save_strategy "${save_strategy}"        # 保存策略：每 epoch
  --save_total_limit "${save_total_limit}"  # 保留最近 3 个
  --per_device_train_batch_size "${per_device_batch}"     # 每卡 batch
  --gradient_accumulation_steps "${gradient_accumulation}" # 梯度累积
  --learning_rate "${learning_rate}"        # 学习率
  --lr_scheduler_type cosine                # 余弦学习率调度
  --warmup_ratio 0.03                       # 前 3% 步数做 warmup
  --logging_steps 1                         # 每步都打印日志
  --max_length "${max_length}"              # 单样本最大 token 数
  --max_grad_norm 0                         # 关闭梯度裁剪（与原实验一致）
  --gradient_checkpointing true             # 用重算换显存（视频序列很长，必须开）
  --attn_impl "${attn_impl}"                # 注意力实现 sdpa
  --use_logits_to_keep true                 # 只算答案位置的 logits（配合 sitecustomize 生效）
  --dataset_num_proc 1                      # 数据预处理进程数
  --dataloader_num_workers 0                # 主进程内解码视频（避免多进程读视频的坑）
  --dataloader_persistent_workers false
  --no_dataset_shuffle                      # 不打乱顺序（复现性）
  --seed 20260904                           # 随机种子
  --report_to none                          # 不上报 wandb 等平台
)
# 冒烟测试时通过环境变量限制步数；正式训练不设置该变量。
if [[ -n "${max_steps}" ]]; then
  args+=(--max_steps "${max_steps}")
fi

if [[ -n "${OMNI_OPSD_RESUME_FROM_CHECKPOINT:-}" ]]; then
  args+=(--resume_from_checkpoint "$OMNI_OPSD_RESUME_FROM_CHECKPOINT")
fi

# ---- 启动！---------------------------------------------------------------------
# exec：用 swift 进程替换当前 shell；swift 内部读取 NPROC_PER_NODE/NNODES/... 后
# 调用 torch.distributed.run，在每台机器上生成 8 个训练进程。
exec "${swift_bin}" sft "${args[@]}"
