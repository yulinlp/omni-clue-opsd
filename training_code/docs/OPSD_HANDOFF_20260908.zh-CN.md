# OPSD 训练交接文档（2026-09-08）

本文档交给下一个 Codex session 使用，目标是在 GPU02 上继续完成 OmniVideo gap5000 的 OPSD 训练。当前推荐的训练是解决显存问题后的工程安全版本；它可以稳定运行，但由于改变了视频采样上限，不能标记为论文原始 768 帧/28672 像素协议。

## 2026-09-08 参数变更（后续新运行采用）

根据论文的 top-K 蒸馏定义，后续 OPSD/CLUE-OPSD 的 GKD 支持集固定为 `K=100`：先取学生分布的 top-100 token，再取教师在这些 token 上的对应概率，并为学生和教师各保留一个“其余词表”的 tail mass。实现位于 `ms-swift/swift/rlhf_trainers/gkd_loss.py`，会在响应 token 掩码之后再做 top-K，适用于普通 OPSD 中学生与 teacher prompt 长度不同的情况。

后续 GKD rollout 使用 vLLM colocate 引擎，参数为 `use_vllm=true`、`vllm_mode=colocate`、`top_p=1.0`、`top_k=20`。这会完全绕过 `TransformersEngine`。为避免 colocate 引擎和训练模型争抢 HBM，启动器同时默认使用 `vllm_gpu_memory_utilization=0.30`、`sleep_level=1`；这两个是显存工程参数，不改变论文中的采样分布，可用 `OMNI_OPSD_VLLM_GPU_MEMORY_UTILIZATION` 和 `OMNI_OPSD_VLLM_SLEEP_LEVEL` 覆盖。启动前必须在训练环境安装与 CUDA、PyTorch 和当前 ms-swift 兼容的 vLLM；当前环境此前检查到尚未安装 `vllm`，未安装前启动会在初始化阶段明确失败。SFT 分支不使用 rollout，也不受这些 GKD 参数影响。旧日志中出现的 `full-vocabulary` 或 `vllm=false` 只代表历史运行，不能用于描述这次新配置。

## 先看当前结论

- 原始完整分辨率 OPSD 正式任务在 xysui 的 gpu04 上发生 CUDA OOM，不能继续使用原始 768 帧、28672 像素配置直接训练。
- 新生成的 CUDA 安全数据版本在 GPU02 四卡上完成了真实单步冒烟：无 OOM、完成反向传播并保存 `checkpoint-1`。
- 安全版本的 300 步正式任务后来按“只训练一个 epoch”的要求停止在 step 2/300。它没有到 `save_steps=25`，所以没有正式 adapter checkpoint；不要从这个目录恢复或评测。
- 下一 session 应从基座模型重新启动一个新的安全版本 OPSD one-epoch 任务。按当前 4 卡、每卡 batch=1、梯度累积=16 的 dataloader 语义，建议 `max_steps=78`，并把 `save_steps` 设为 13，使最终 `checkpoint-78` 一定落盘。
- 启动命令应在仍然有效的 Slurm GPU 作业内使用 `nohup setsid`。这样 SSH 或 Codex session 断开不会主动杀掉进程，但 Slurm 作业结束或被取消时训练仍会停止。

## 当前目录、数据和状态

| 项目 | 当前值 |
| --- | --- |
| 项目根目录 | `/share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907` |
| 基座模型 | `/share/home/ylhu/models/Qwen2.5-Omni-7B` |
| Conda 环境 | `/share/home/ylhu/.conda/envs/omniopsd_train` |
| ms-swift 源码 | `/share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907/ms-swift` |
| ms-swift commit | `06c7d80d8fd104f773149c662011cb3e2885ac6a` |
| 项目 commit | `1cbedee34325ca76e8f0e4c5c171824baf054528` |
| 原始 OPSD 矩阵 | `data/gap5000/training_matrix/omnivideo_100k_train.opsd.jsonl` |
| 原始 OPSD SHA-256 | `909c8484928a223805e9ba96580305b33743ca71f3b8c813172dd3d66116fc3d` |
| 安全 OPSD 矩阵 | `data/gap5000/training_matrix/omnivideo_100k_train.opsd.cuda_safe_f256_p7840.jsonl` |
| 安全矩阵 SHA-256 | `a66b25cfd8c97ba93fc81e9601c12bb24374952d285e5d3934b388f16ba13ec7` |
| 安全矩阵行数/唯一 case_id | `5000 / 5000` |
| 推荐节点 | GPU02，实际分配到的 4 张卡 |
| 已停止安全正式 runner PID | `1861756`（仅供日志追溯） |

安全矩阵由原始 OPSD 矩阵逐行复制生成，保留行顺序、case_id、问题、选项、正确答案、完整视频时间区间和音频开关；仅把学生和 teacher 视图中的采样预算限制为 `max_frames=256`、`max_pixels=7840`，并在 `sampling_contract` 中记录原始预算和安全版本信息。原始文件没有被覆盖。

安全矩阵生成脚本：

```text
scripts/make_opsd_cuda_safe_dataset.py
```

如需重新生成，使用：

```bash
cd /share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907
/share/home/ylhu/.conda/envs/omniopsd_train/bin/python scripts/make_opsd_cuda_safe_dataset.py \
  --input data/gap5000/training_matrix/omnivideo_100k_train.opsd.jsonl \
  --output data/gap5000/training_matrix/omnivideo_100k_train.opsd.cuda_safe_f256_p7840.jsonl \
  --max-frames 256 --max-pixels 7840
```

## 为什么原始 OPSD 会 OOM

OPSD 的一次优化更新包含学生 forward 和 answer-privileged teacher forward。二者都处理完整音视频；teacher 的 prompt 额外提供正确答案。原始矩阵允许最多 768 帧、每帧最多 28672 像素，在 80 GiB A100 上同时进行两路多模态 forward 时峰值显存过高。

gpu04 的原始正式任务目录为：

```text
output/xysui_opsd_formal_decord2_20260908_0222
```

该任务是在 xysui 的 Slurm 作业 `279142`、gpu04 四卡上启动的；作业和进程信息只用于追溯，下一次任务应重新确认当前有效的 GPU 分配。

它在第一个优化步附近失败。日志中的典型信息是：

- GPU2 试图额外申请约 11.52 GiB，但只有约 1.97 GiB 空闲；进程已占约 77.18 GiB，allocated 约 61.48 GiB，reserved-but-unallocated 约 14.06 GiB。
- GPU3 试图额外申请约 11.05 GiB，但只有约 3.19 GiB 空闲，显存占用约 75.95 GiB。

降低梯度累积只能减少更新之间的累计，不会降低一个学生/teacher forward 的峰值。因此本次采取了两项措施：

1. 安全数据集把每个视图限制到 256 帧、7840 像素，仍在原视频完整时间区间上均匀采样；
2. `scripts/run_gap5000_opsd_cuda.sh` 默认导出 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，减轻 CUDA allocator 的碎片影响。

安全配置是显存工程方案，不等价于原始论文数据协议。所有安全版本输出都必须标记：

```text
paper_exact=false
protocol_classification=engineering-pilot
```

不能把安全版本的结果与原始 768/28672 协议的结果混写成同一实验。

## 已完成的 GPU 验证

### 安全版本单步冒烟

目录：

```text
output/gpu02_opsd_cuda_safe_f256_p7840_smoke_20260908
```

配置是 GPU02 四卡、4 个 rank、`max_steps=1`、每卡 batch=1、梯度累积=1、decord、音频开启、expandable allocator。验证行：

```text
validated arm=opsd rows=5000 unique_case_ids=5000 student_frames=256..256 use_audio_in_video=True
```

结果：约 53.8 秒完成一步，`loss=0.2005`、`grad_norm=0.8523`、显存约 33.14 GiB，并保存：

```text
output/gpu02_opsd_cuda_safe_f256_p7840_smoke_20260908/opsd/v0-20260908-150444/checkpoint-1
```

这是“安全版本可以运行”的证据，不是正式训练模型。

### 安全版本正式任务（已停止）

目录：

```text
output/gpu02_opsd_cuda_safe_f256_p7840_formal_20260908_151157
```

实际启动配置见：

```text
output/gpu02_opsd_cuda_safe_f256_p7840_formal_20260908_151157/launch_config.txt
output/gpu02_opsd_cuda_safe_f256_p7840_formal_20260908_151157/opsd/RUN_CLASSIFICATION.txt
```

它使用 4 卡、`max_steps=300`、梯度累积=16，step 1 用时约 8 分 22 秒，step 2 用时约 8 分 16 秒；日志显示显存约 50.77–53.09 GiB，外部采样约 53–59 GiB，没有 OOM。之后按用户要求停止在 step 2，日志末尾的 `SignalException: Process ... got signal: 15` 是主动 SIGTERM，不是 CUDA 错误。

由于 `save_steps=25`，该目录没有 `adapter_model.safetensors`。只有 `args.json`、`logging.jsonl` 和运行日志，不能把它当作可恢复 checkpoint。下一次训练应从基座重新开始，而不是把该目录作为 `resume_from_checkpoint`。

## 下一次 one-epoch OPSD 的参数

推荐保持当前已经验证过的安全配置，只把训练步数改为一个 epoch：

| 参数 | 推荐值 | 说明 |
| --- | --- | --- |
| 数据 | `omnivideo_100k_train.opsd.cuda_safe_f256_p7840.jsonl` | 5000 行，安全采样版本 |
| GPU | GPU02 实际分配的 `0,1,2,3` | 4 个分布式进程 |
| 每卡 batch | 1 | 单个多模态样本 |
| 梯度累积 | 16 | 有效 batch=`4×1×16=64` |
| `max_steps` | 78 | 当前 trainer 语义下约等于一个 epoch |
| `save_steps` | 13 | 13、26、39、52、65、78 都会保存，最终 checkpoint 可用 |
| LoRA | rank16、alpha32、`all-linear` | 只训练 adapter |
| dtype | BF16 | 与冒烟、正式部分任务一致 |
| learning rate | `2e-6` | cosine scheduler，warmup ratio 0.03 |
| max length | 32768 | 保持已验证设置 |
| completion length | 8 | OPSD GKD 配置 |
| `lambda`/`lmbda` | 1.0 | GKD 项权重 |
| `beta` | 0.5 | 学生监督与蒸馏混合参数 |
| `gkd_logits_topk` | 100 | 学生 top-100 支持集 + 学生/教师 tail mass |
| temperature | 1.0 | top-K + tail mass divergence 的温度 |
| `sft_alpha` | 0 | 不额外混入 SFT loss |
| vLLM | true (`colocate`) | GKD rollout 使用 vLLM，替代 `TransformersEngine` |
| `vllm_gpu_memory_utilization` | 0.30 | 为训练模型预留 HBM；启动器可通过环境变量覆盖 |
| `sleep_level` | 1 | rollout 后暂时释放 vLLM 显存，降低与训练 forward 的峰值冲突 |
| rollout `top_p` | 1.0 | 不做 nucleus 截断 |
| rollout `top_k` | 20 | 每一步只在最高概率的 20 个 token 中采样 |
| teacher | dynamic current-policy | `no_grad` 的当前策略 teacher |
| EMA | 不使用 | EMA 设计属于 CLUE-OPSD 路径，当前标准 OPSD 不依赖 EMA |
| 视频读取 | `decord` | 避免四 rank 同时 PyAV/torchvision 解码资源错误 |
| 音频 | `USE_AUDIO_IN_VIDEO=1` | 学生和 teacher 都保持音频 |
| attention | SDPA | `OMNI_OPSD_ATTN_IMPL=sdpa` |
| allocator | `expandable_segments:True` | 缓解碎片，脚本会导出该变量 |
| GKD safe mode | 0 | 保持完整 GKD；`gkd_max_grad_norm=1.0` |
| shuffle | 关闭 | `--no_dataset_shuffle` |
| seed | 20260904 | 与 SFT/冒烟一致 |

### 78 步为什么算一个 epoch

日志中每个优化步的 epoch 增量约为 `0.01282`，即约 `1/78`。因此在当前 4 rank、有效 batch 64 的 dataloader 语义下，`max_steps=78` 到达约 `epoch=1.0`。用 `floor(5000/64)=78` 可以保留现有 batch 和梯度累积设置，但尾部约 8 条样本不会形成一次完整的同步优化更新。

如果实验要求“5000 条每一条都必须参与一次 optimizer update”，可把梯度累积改为 10：有效 batch=`4×1×10=40`，训练步数改为 125。这样会改变有效 batch、学习率调度和训练动力学；除非明确选择该方案，不要在下一 session 中临时改动。推荐先按 78 步跑通，并在实验记录中写明尾部处理方式。

## GPU02 持久化启动命令

以下命令必须在已经分配到 GPU02 的有效 Slurm 作业内、并且当前用户确实能使用四张卡时执行。它不会替用户申请或绕过调度器资源：

```bash
cd /share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907
# 脚本通过 OMNI_OPSD_ENV 和绝对路径使用已验证环境，不依赖当前 shell 的 python。

TAG=$(date +%Y%m%d_%H%M%S)
ROOT="$PWD/output/gpu02_opsd_cuda_safe_f256_p7840_oneepoch_${TAG}"
mkdir -p "$ROOT"

export OMNI_OPSD_PROJECT_ROOT="$PWD"
export OMNI_OPSD_ENV=/share/home/ylhu/.conda/envs/omniopsd_train
export OMNI_OPSD_MS_SWIFT_ROOT="$PWD/ms-swift"
export OMNI_OPSD_MODEL=/share/home/ylhu/models/Qwen2.5-Omni-7B
export OMNI_OPSD_CUDA_DEVICES=0,1,2,3
export OMNI_OPSD_NPROC_PER_NODE=4
export OMNI_OPSD_DATASET="$PWD/data/gap5000/training_matrix/omnivideo_100k_train.opsd.cuda_safe_f256_p7840.jsonl"
export OMNI_OPSD_OPSD_ROOT="$ROOT"
export OMNI_OPSD_EXPERIMENT_LABEL=opsd_cuda_safe_f256_p7840_oneepoch
export OMNI_OPSD_MAX_STEPS=78
export OMNI_OPSD_SAVE_STEPS=13
export OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE=1
export OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS=16
export OMNI_OPSD_MAX_LENGTH=32768
export OMNI_OPSD_MIN_PIXELS=3136
export OMNI_OPSD_MAX_COMPLETION_LENGTH=8
export OMNI_OPSD_GKD_LOGITS_TOPK=100
export OMNI_OPSD_ROLLOUT_TOP_P=1.0
export OMNI_OPSD_ROLLOUT_TOP_K=20
export OMNI_OPSD_USE_VLLM=true
export OMNI_OPSD_VLLM_MODE=colocate
export OMNI_OPSD_VLLM_GPU_MEMORY_UTILIZATION=0.30
export OMNI_OPSD_VLLM_SLEEP_LEVEL=1
export OMNI_OPSD_ATTN_IMPL=sdpa
export OMNI_OPSD_GKD_SAFE_MODE=0
export OMNI_OPSD_GKD_MAX_GRAD_NORM=1.0
export OMNI_OPSD_ALLOC_CONF=expandable_segments:True
export FORCE_QWENVL_VIDEO_READER=decord
export USE_AUDIO_IN_VIDEO=1
export MAX_NUM_WORKERS_FETCH_VIDEO=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMNI_OPSD_ALLOW_SPARSE_ENGINEERING_PILOT=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29951

export CUDA_VISIBLE_DEVICES="$OMNI_OPSD_CUDA_DEVICES"
export ASCEND_RT_VISIBLE_DEVICES="$OMNI_OPSD_CUDA_DEVICES"
export PYTHONPATH="$PWD/src:$PWD/ms-swift${PYTHONPATH:+:$PYTHONPATH}"

nohup setsid bash scripts/run_gap5000_opsd_cuda.sh formal \
  > "$ROOT/launcher.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$ROOT/runner.pid"
printf 'ROOT=%s\nPID=%s\n' "$ROOT" "$PID"
```

`run_gap5000_opsd_cuda.sh` 会再次把训练主日志写入 `$ROOT/opsd.log`，并将实际环境写入 `$ROOT/launch_config.txt`。外层 `nohup setsid` 使终端断开时进程继续运行；不要仅根据 shell 返回的 `[1] Done` 判断任务结束，应查看 worker、日志和 GPU 显存。若 Slurm 作业的时间上限或作业状态结束，后台进程也会被调度器清理。

启动后立即确认：

```bash
ROOT=<刚才打印的绝对路径>
cat "$ROOT/runner.pid"
cat "$ROOT/launch_config.txt"
tail -50 "$ROOT/opsd.log"
ps -ef | rg 'torch.distributed.run|swift/cli/rlhf.py|run_gap5000_opsd_cuda' | rg -v 'rg '
nvidia-smi
```

## 训练过程监控

建议每 10–20 分钟检查一次，不要频繁 attach 到训练进程：

```bash
ROOT=<训练根目录>
tail -f "$ROOT/opsd.log"
```

另一个终端可用：

```bash
rg -n 'Train:|global_step|max_steps|loss|epoch|memory\(GiB\)|Saving model checkpoint|CUDA out of memory|Traceback|SIGTERM' \
  "$ROOT/opsd.log" | tail -80

nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
find "$ROOT" -name adapter_model.safetensors -o -name trainer_state.json | sort
```

正常 one-epoch 任务应看到 1/78、2/78，直到 78/78；每个安全版本更新通常约 8–9 分钟，粗略总时长约 10–12 小时，实际以前 3–5 个优化步的 `train_speed(s/it)` 为准。若再次出现 OOM，先保留完整日志、`nvidia-smi` 输出和失败 rank，再考虑进一步降低采样预算；不要只降低梯度累积后把实验仍标为同一配置。

完成后应至少存在：

```text
$ROOT/opsd/RUN_CLASSIFICATION.txt
$ROOT/opsd/v0-*/checkpoint-78/adapter_model.safetensors
$ROOT/opsd/v0-*/checkpoint-78/adapter_config.json
$ROOT/opsd/v0-*/checkpoint-78/trainer_state.json
```

检查 `RUN_CLASSIFICATION.txt` 必须包含 `dataset_rows=5000`、安全数据 SHA、`paper_exact=false`、`protocol_classification=engineering-pilot`、`nproc=4`、`max_steps=78` 和 `allocator_conf=expandable_segments:True`。记录最终 checkpoint 的绝对路径和 SHA-256，供后续评测加载。

## OPSD 核心代码和数据流

主要入口关系如下：

```text
scripts/run_gap5000_opsd_cuda.sh
  -> scripts/run_training_arm_a100.sh
    -> scripts/run_video_odyssey_training_arm.sh
      -> ms-swift/swift/cli/rlhf.py --rlhf_type gkd
        -> ms-swift/swift/rlhf_trainers/gkd_trainer.py
```

OPSD 的一条样本包含学生输入和 teacher 输入：

1. 学生读取完整视频和音频，回答问题；
2. teacher 读取相同的完整视频和音频，但 teacher prompt 额外含 gold answer；
3. 学生 forward 保留梯度，当前策略 teacher forward 在 `no_grad` 下提供蒸馏目标；
4. 使用学生 top-100、教师对应值和双方 tail mass 组成的压缩 divergence，并按 `lambda=1.0`、`beta=0.5`、`sft_alpha=0` 组合。

当前标准 OPSD 不使用 EMA teacher。EMA 只会影响 CLUE-OPSD 的原设计路径；不能因为 `clue_ema_alpha` 字段存在或缺失，就把标准 OPSD 描述成 EMA 实验。

安全数据集之所以必须落盘，是因为 Qwen-Omni 的 `smart_nframes` 会尊重每一行媒体映射中的 `max_frames`；只设置 `FPS_MAX_FRAMES` 环境变量无法覆盖原始行里的 768 帧上限。`VIDEO_MAX_PIXELS` 也只能限制空间预算，不能单独解决帧数问题。

## 文件修改和复核建议

本次为显存修复新增或修改的文件：

```text
scripts/make_opsd_cuda_safe_dataset.py
scripts/run_gap5000_opsd_cuda.sh
data/gap5000/training_matrix/omnivideo_100k_train.opsd.cuda_safe_f256_p7840.jsonl
```

它们已经被安全冒烟和安全正式前两步实际使用。由于项目工作区还包含此前会话产生的其他未提交文件，下一个 session 不要用 `git reset --hard` 或大范围清理工作区；如需提交补丁，先单独查看：

```bash
git -C /share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907 status --short
git -C /share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907 diff -- scripts/make_opsd_cuda_safe_dataset.py scripts/run_gap5000_opsd_cuda.sh
```

## 推荐的接手顺序

1. 在 GPU02 的有效 Slurm 作业内确认四张卡空闲，并确认没有残留 `torch.distributed` worker。
2. 检查安全数据行数、SHA-256 和 `student_frames=256`；保留原始矩阵，不覆盖它。
3. 用上面的持久化命令从基座模型启动新的 `max_steps=78` 任务，保存间隔 13。
4. 前三步重点观察显存和单步耗时；确认没有 OOM 后再让任务后台运行。
5. 监控 78/78 和 `checkpoint-78`，不要把旧的 step2 部分目录当成正式模型。
6. 训练完成后记录 adapter 路径、训练日志、`RUN_CLASSIFICATION.txt` 和数据 SHA；后续评测时使用基座模型 + OPSD adapter。

背景运行记录和通用参数说明还可查看：

```text
docs/GAP5000_RUN_20260908.zh-CN.md
docs/GAP5000_TRAINING_GUIDE.zh-CN.md
docs/PROJECT_WALKTHROUGH.zh-CN.md
scripts/run_training_arm_a100.sh
scripts/run_video_odyssey_training_arm.sh
```
