#!/usr/bin/env bash
# =============================================================================
# run_worldsense_clue_opsd_npu_w1w2.sh —— CLUE-OPSD 正式训练入口（worker-1/2，16 卡）
# =============================================================================
# 作用：
#   在**当前 npu24 作业**的 worker-1 / worker-2 容器内运行，各执行一次：
#     - 设置双机分布式参数（2 节点 × 8 卡 = 16 卡）；
#     - 设置 GKD（广义知识蒸馏）/ EMA 教师 / LoRA 等超参数；
#     - 通过 torch.distributed.run 启动 ms-swift 的 GKD 训练入口。
#
# 与 SFT 的区别（一句话）：
#   SFT 是“背标准答案”；CLUE-OPSD 是学生看全片、教师看证据段，
#   学生先自己生成 completion，再让自己的分布去对齐教师的分布。
#
# 用法（在目标容器内，两个节点各执行一次）：
#   bash run_worldsense_clue_opsd_npu_w1w2.sh 0   # worker-1（主节点）
#   bash run_worldsense_clue_opsd_npu_w1w2.sh 1   # worker-2（从节点）
# =============================================================================

set -euo pipefail   # 安全模式：出错即停 / 未定义变量报错 / 管道失败即失败

# ---- 参数校验：只接受 0 或 1 ---------------------------------------------------
node_rank="${1:?usage: $0 NODE_RANK(0|1)}"
[[ "${node_rank}" == 0 || "${node_rank}" == 1 ]] || exit 2

# ---- 路径类变量 ----------------------------------------------------------------
repo="/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd"                                  # 代码仓库
run_root="${repo}/training_runs/worldsense_clue_opsd_thinking_gold_r64_w1w2"              # 本次运行的资料目录
model="/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-assets/models/Qwen2.5-Omni-7B"        # 基座模型
deps="/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_npu24_deps"          # 本机补丁目录
swift_root="/opt/huawei/dataset/hyl_ulan/ylhu/third_party/ms-swift"                       # ms-swift 源码
python_bin="/home/ma-user/anaconda3/envs/PyTorch-2.9.0/bin/python"                        # Python 解释器
# 数据与输出支持环境变量覆盖：方便冒烟测试时换成小数据集/临时输出目录。
dataset="${OMNI_OPSD_DATASET:-${run_root}/data/clue_opsd_thinking_gold.jsonl}"
output="${OMNI_OPSD_OUTPUT_DIR:-${run_root}/outputs/formal}"

# ---- Python 搜索路径：本机补丁在最前，其次 ms-swift 源码 -----------------------
export PYTHONPATH="${deps}:${swift_root}${PYTHONPATH:+:${PYTHONPATH}}"

# ---- 分布式参数 ----------------------------------------------------------------
# 本脚本直接调用 torch.distributed.run（不是 swift CLI 包装），所以这些变量会作为
# torchrun 的显式命令行参数使用（见文件末尾）。
export NPROC_PER_NODE="${OMNI_OPSD_NPROC_PER_NODE:-8}"     # 每机 8 个进程
export NNODES="${OMNI_OPSD_NNODES:-2}"                     # 2 台机器
export NODE_RANK="${node_rank}"                            # 本机序号（0 或 1）
export MASTER_ADDR="${OMNI_OPSD_MASTER_ADDR:-172.16.12.237}"   # 主节点 = worker-1
export MASTER_PORT="${OMNI_OPSD_MASTER_PORT:-29531}"       # 主节点端口
# 使用本机全部 8 张 NPU。
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# ---- HCCL 通信表：单卡调试时不用；正式多机必须用子集表 -------------------------
# 单机单卡冒烟（NNODES=1 且 NPROC=1）时清掉平台总表，避免编号不匹配；
# 正式 16 卡时用提前生成的 worker-1/2 子集表（rank 0–7 / 8–15）。
if [[ "${NNODES}" == 1 && "${NPROC_PER_NODE}" == 1 ]]; then
  unset RANK_TABLE_FILE RANK_TABLE_FILE_V_1_0 || true
else
  export RANK_TABLE_FILE="${run_root}/data/ranktable_w1w2.json"
  export RANK_TABLE_FILE_V_1_0="${RANK_TABLE_FILE}"
fi

# ---- 多媒体环境变量（与 SFT 相同的口径：全片 + 音频 + pyav_seek 解码） ----------
export USE_AUDIO_IN_VIDEO=1
export FORCE_QWENVL_VIDEO_READER=pyav_seek
export MAX_NUM_WORKERS_FETCH_VIDEO=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# ---- 本机显存适配开关（sitecustomize.py 读取） --------------------------------
#   LOGITS_TO_KEEP=1：只对学生 completion 的位置算 logits（省显存）；
#   DROP_TALKER=1   ：卸载理解任务用不到的语音生成模块（省 ~4.5GB）；
export WORLDSENSE_LOGITS_TO_KEEP=1
export WORLDSENSE_DROP_TALKER=1
export HCCL_CONNECT_TIMEOUT=600
export PYTHONUNBUFFERED=1

# ---- 前置检查：数据、模型、依赖、（多机时的）rank table 必须存在 ---------------
# 语法说明：${RANK_TABLE_FILE:+"${RANK_TABLE_FILE}"} 表示“变量已设置才展开”，
# 单机调试时 RANK_TABLE_FILE 未设置，这一项就不会出现在检查列表里。
for path in "${dataset}" "${model}/config.json" "${deps}/qwen_omni_utils" ${RANK_TABLE_FILE:+"${RANK_TABLE_FILE}"}; do
  [[ -e "${path}" ]] || { echo "missing: ${path}" >&2; exit 2; }
done
mkdir -p "${output}"

# ---- 组装训练参数数组 ----------------------------------------------------------
# 参数分三类：
#   1) GKD 蒸馏：rlhf_type=gkd、lmbda/beta/temperature、gkd_logits_topk、sft_alpha；
#   2) CLUE/EMA：clue_ema_alpha（LoRA-shadow EMA 教师，本机 ms-swift 补丁实现）；
#   3) 常规训练：LoRA、batch、LR、保存、精度、注意力等。
args=(
  --rlhf_type gkd                          # 训练类型：GKD（广义知识蒸馏）
  --model "${model}"                       # 基座模型
  --dataset "${dataset}"                   # CLUE-OPSD 数据（含 teacher_videos/teacher_prompt）
  --output_dir "${output}"                 # 输出目录
  --split_dataset_ratio 0                  # 不切验证集
  --tuner_type lora                        # LoRA 微调
  --lora_rank 64                           # LoRA 秩（本次 CLUE 用 r64）
  --lora_alpha 128                         # LoRA alpha
  --target_modules all-linear              # 对全部线性层加 LoRA
  --lmbda "${OMNI_OPSD_LMBDA:-0.5}"       # 一半学生 rollout，一半带标准分析/答案的数据 completion
  --beta 0.5                               # JSD 插值系数（0.5 = 对称 JSD）
  --temperature 1.0                        # 蒸馏温度
  --clue_ema_alpha 0.05                    # 教师权重按 EMA 缓慢跟随学生（补丁实现）
  --sft_alpha 0.25                         # 数据 completion 上增加完整分析和答案的交叉熵
  --gkd_logits_topk 100                    # 只比较前 100 个候选 token（含尾部概率）
  --torch_dtype bfloat16                   # bf16 精度
  --num_train_epochs 3                     # 3 轮
  --per_device_train_batch_size 1          # 每卡 batch=1（长视频样本大）
  --gradient_accumulation_steps "${OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS:-2}"   # 梯度累积 2 → 全局 batch 32
  --learning_rate 2e-6                     # 蒸馏学习率（比 SFT 小一个量级）
  --lr_scheduler_type cosine               # 余弦调度
  --warmup_ratio 0.03                      # warmup 3%
  --save_strategy epoch                    # 每轮保存
  --save_total_limit 3                     # 保留最近 3 个
  --logging_steps 1                        # 每步打印
  --max_length 32768                       # 单样本最大 token
  --max_completion_length 320              # 留足最多 120 个英文词的分析及 <answer> 标签
  --use_logits_to_keep true                # 只算需要监督的位置（配合本机补丁生效）
  --gradient_checkpointing true            # 重算换显存
  --attn_impl sdpa                         # 省显存的注意力实现
  --use_vllm false                         # 用 Transformers 做 rollout（本机未验证 vLLM 路径）
  --dataset_num_proc 1                     # 数据预处理进程数
  --dataloader_num_workers 0               # 主进程内解码视频
  --no_dataset_shuffle                     # 不打乱（复现性）
  --seed 20260904                          # 随机种子
  --report_to none                         # 不上报外部平台
)
# 冒烟测试用：限制总步数（正式训练不设置）。
if [[ -n "${OMNI_OPSD_MAX_STEPS:-}" ]]; then args+=(--max_steps "${OMNI_OPSD_MAX_STEPS}"); fi
# 可选：改成按固定步数保存（例如每 100 步存一次）。
if [[ -n "${OMNI_OPSD_SAVE_STEPS:-}" ]]; then args+=(--save_strategy steps --save_steps "${OMNI_OPSD_SAVE_STEPS}"); fi

# ---- 启动 torchrun -------------------------------------------------------------
# 直接调用 torch.distributed.run：
#   --nproc_per_node/--nnodes/--node_rank/--master_addr/--master_port 是标准的
#   多机分布式启动参数；随后是要运行的 Python 入口和训练参数。
# 使用 ms-swift 的 GKD Trainer；数据分支的 SFT CE 在该训练器内计算。
exec "${python_bin}" -m torch.distributed.run \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  "${swift_root}/swift/cli/rlhf.py" "${args[@]}"
