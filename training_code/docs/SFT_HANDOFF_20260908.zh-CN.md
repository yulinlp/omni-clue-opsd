# SFT 训练交接文档（2026-09-08）

本文档交给下一个 Codex session 使用，目标是接着完成 OmniVideo gap5000 的 SFT 评测。文档中的路径均以当前机器的共享文件系统为准：

```text
/share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907
```

## 先看当前结论

- SFT 正式任务已经停止，没有正在运行的 SFT worker。
- 按“至少完成一个数据集遍历后停止”的要求，当前选定的评测模型是保存下来的 LoRA adapter：

  ```text
  /share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907/output/gap5000_formal_20260908_010630/sft/v0-20260908-010643/checkpoint-175
  ```

- `checkpoint-175` 的记录为 `global_step=175`、`epoch=1.1152`。原始日志后来还走到了未保存的 step 198，但没有 `checkpoint-198`，因此评测不要使用日志中的 step 198 数值，也不要把它当成可加载模型。
- 目前没有独立的 OmniVideo 测试集和标签文件。下一 session 必须先找到或构造 answer-free、与训练集隔离的评测集，再运行评测；不能直接把这 5000 条训练数据当作泛化测试集。
- 这是 LoRA adapter 训练，没有生成合并后的完整模型。评测时直接加载基座模型和 adapter 即可，不需要先合并。

## 训练、数据和源码位置

| 项目 | 当前值 |
| --- | --- |
| 项目根目录 | `/share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907` |
| 基座模型 | `/share/home/ylhu/models/Qwen2.5-Omni-7B` |
| Conda 环境 | `/share/home/ylhu/.conda/envs/omniopsd_train` |
| Python | `/share/home/ylhu/.conda/envs/omniopsd_train/bin/python` |
| Swift CLI | `/share/home/ylhu/.conda/envs/omniopsd_train/bin/swift` |
| ms-swift 源码 | `/share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907/ms-swift` |
| ms-swift 实际 commit | `06c7d80d8fd104f773149c662011cb3e2885ac6a` |
| 项目实际 commit | `1cbedee34325ca76e8f0e4c5c171824baf054528` |
| SFT 矩阵 | `data/gap5000/training_matrix/omnivideo_100k_train.sft.jsonl` |
| SFT 矩阵绝对路径 | `/share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907/data/gap5000/training_matrix/omnivideo_100k_train.sft.jsonl` |
| 数据行数/唯一 case_id | `5000 / 5000` |
| SFT 数据 SHA-256 | `61d003fc53fb549bafe926adc345b14363fe180191d55204118f8dcb753a8047` |
| 正式输出根目录 | `output/gap5000_formal_20260908_010630` |
| 训练设备 | `CUDA_VISIBLE_DEVICES=2,3`，2 个分布式进程 |
| 历史外层 runner PID | `1684832`（已停止，仅供日志追溯） |

训练数据由 OmniVideo-100K 的 MCQ-30K 按项目根目录下的
`omnivideo_gap5000.aggregate.sample_ids.txt` 精确筛选得到。样本顺序和 ID 不应重新打乱或重新抽样。视频缓存位于：

```text
/share/home/ylhu/Omni-OPSD/data/cache/omnivideo_5k/videos
```

SFT 的每条学生输入包含完整视频、音频、问题和选项；assistant 目标是正确答案字母。SFT 本身没有独立的 GKD teacher 或 EMA teacher。四组实验保持学生输入一致，只改变监督字段。

训练采样契约是：`fps=2.0`、`min_pixels=3136`、`max_pixels=28672`、`max_frames=768`、`use_audio_in_video=true`。正式日志确认学生视图为 768 帧并启用音频。正式 SFT 使用了 torchvision 视频读取路径；OPSD 因四卡完整视频读取的资源问题另行固定使用 decord，二者不要混淆。

## 环境与 ms-swift 说明

当前环境已经验证过关键包，主要版本如下：

```text
Python 3.11
torch 2.6.0+cu124
torchvision 0.21.0+cu124
transformers 5.16.1
trl 0.29.1
peft 0.20.0
accelerate 1.14.0
qwen-omni-utils 0.0.9
av 18.1.0
modelscope 1.39.1
ms-swift 4.6.0.dev0（源码 commit 见上表）
```

历史文档中出现过的目标 commit
`960c5bf2cb070d1e3483ed93965f2e338d3ae93a` 在公开仓库中无法取得；本次实际运行的是上表的 `06c7d80...`。项目中能从补丁确定恢复的结构化视频/音频输入、teacher 媒体透传和 Qwen-Omni Tensor 兼容处理已经用于训练。完整的 LoRA-shadow EMA teacher 仍未恢复，这不影响当前 SFT，也不应在 SFT 评测中声称使用了 EMA。

下一个 session 开始评测前，先确认环境和源码没有被覆盖：

```bash
cd /share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907
# 当前 shell 若未激活环境，可用集群实际的 conda 初始化脚本激活；以下命令全部使用环境绝对路径。
export OMNI_OPSD_MS_SWIFT_ROOT="$PWD/ms-swift"
export OMNI_OPSD_PYTHON_DEPS="$PWD"
export PYTHONPATH="$PWD/src:$PWD/ms-swift${PYTHONPATH:+:$PYTHONPATH}"
/share/home/ylhu/.conda/envs/omniopsd_train/bin/python -m pip check
git -C "$PWD/ms-swift" rev-parse HEAD
```

## SFT 正式任务的实际配置和状态

正式任务的启动记录在：

```text
output/gap5000_formal_20260908_010630/environment/launch_config.txt
output/gap5000_formal_20260908_010630/sft.status
output/gap5000_formal_20260908_010630/sft/STOPPED_AFTER_ONE_EPOCH.txt
```

实际核心命令为：

```bash
/share/home/ylhu/.conda/envs/omniopsd_train/bin/python -m torch.distributed.run \
  --nproc_per_node 2 --master_port 29901 --master_addr 127.0.0.1 \
  /share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907/ms-swift/swift/cli/sft.py \
  --model /share/home/ylhu/models/Qwen2.5-Omni-7B \
  --dataset /share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907/data/gap5000/training_matrix/omnivideo_100k_train.sft.jsonl \
  --split_dataset_ratio 0 \
  --output_dir /share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907/output/gap5000_formal_20260908_010630/sft \
  --tuner_type lora --lora_rank 16 --lora_alpha 32 --target_modules all-linear \
  --torch_dtype bfloat16 --max_steps 300 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --learning_rate 2e-6 --lr_scheduler_type cosine --warmup_ratio 0.03 \
  --save_steps 25 --save_total_limit 2 --logging_steps 1 \
  --max_length 32768 --max_grad_norm 1.0 \
  --gradient_checkpointing true --attn_impl sdpa \
  --dataset_num_proc 1 --dataloader_num_workers 0 \
  --dataloader_persistent_workers false --no_dataset_shuffle \
  --seed 20260904 --report_to none
```

参数含义和实际影响：

- 2 个进程、每卡 batch=1、梯度累积 16，名义有效 batch 是 `2×1×16=32`。300 个优化步对应约 9600 个样本实例，日志中的 epoch 约为 1.9；`max_steps=300` 是原启动框架的上限，不是“一个 epoch”的自动计算结果。
- LoRA rank=16、alpha=32，目标模块使用 `all-linear`；BF16；冻结视觉编码器和 aligner；启用 gradient checkpointing 和 SDPA。
- 最大上下文长度 32768；不打乱数据；数据处理进程 1，DataLoader worker 0 且关闭 persistent workers。这些设置曾分别修复过数据截断、persistent worker 和音视频加载问题。
- 日志每一步记录 loss、梯度范数、token accuracy 和显存。正式 SFT 没有出现 CUDA OOM，显存约 77.25 GiB/卡，接近 80 GiB A100 上限。

SFT 的停止记录是：

```text
STOPPED 2026-09-08T13:50:59+08:00 reason=completed_more_than_one_epoch latest_checkpoint=checkpoint-175 global_step=175 epoch=1.1152
```

step 175 时的记录包括 `loss=0.3828`、`token_acc=0.8646`、`grad_norm=1.504`、`memory=77.25 GiB`。之后 step 198 只出现在日志，记录为 `loss=0.5179`、`epoch=1.262`，随后收到 SIGTERM。停止是按用户要求执行的分布式进程组终止，不是训练异常。

## SFT 产物和 adapter 加载

当前保留下来的 SFT checkpoint 是：

```text
/share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907/output/gap5000_formal_20260908_010630/sft/v0-20260908-010643/checkpoint-150
/share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907/output/gap5000_formal_20260908_010630/sft/v0-20260908-010643/checkpoint-175
```

评测应使用 `checkpoint-175`。其中的主要文件如下：

```text
adapter_model.safetensors   161,536,328 bytes
adapter_config.json         1,153 bytes
additional_config.json         67 bytes
args.json                  16,825 bytes
optimizer.pt             323,290,858 bytes
trainer_state.json        37,024 bytes
scheduler.pt               1,000 bytes
training_args.bin          6,456 bytes
```

`adapter_config.json` 表明：基座是 Qwen2.5-Omni-7B，PEFT 类型是 LoRA，`r=16`、`lora_alpha=32`、`bias=none`、任务类型为 `CAUSAL_LM`。实际 target regex 只匹配 thinker 模块中的线性投影层（`q/k/v/o/up/down/gate_proj`）。

评测时应让 ms-swift 以“基座模型 + adapter”的形式加载，例如使用项目已有评测入口的 `OMNI_OPSD_EVAL_ADAPTER`。不要把 `adapter_model.safetensors` 当作完整模型目录，也不要单独把它传给 `--model`。

如果未来必须生成合并后的完整模型，先在当前环境核实 CLI：

```bash
/share/home/ylhu/.conda/envs/omniopsd_train/bin/swift export --help
```

然后再按照该版本帮助中的 adapter merge 参数执行，并对合并目录做一次加载检查。Qwen-Omni 是多模态模型，不能未经验证地套用普通文本模型的合并命令；当前评测不需要合并，因此不要为了评测额外生成一份大模型副本。

## 下一 session 的评测步骤

### 1. 先准备独立评测数据

评测集必须满足：

1. 与 5000 条训练样本按视频或官方划分隔离；
2. 每条输入只有用户问题、选项和所需媒体，不包含 `answer`、`solution`、`teacher_*` 或其他答案泄漏字段；
3. 标签文件中的 ID 与评测输入一一对应，不能重复、缺失或多出；
4. 视频/音频文件在当前节点可读，并记录实际的 fps、帧数上限、像素上限和音频开关；
5. 如果使用的是项目历史的 VideoOdyssey 108 条评测入口，应明确把结果标注为 VideoOdyssey 评测，不能称为 OmniVideo-100K gap5000 泛化结果。

项目现有入口：

```text
scripts/run_video_odyssey_training_eval.sh
scripts/summarize_video_odyssey_training_eval.py
scripts/aggregate_video_odyssey_training_eval.py
src/omni_opsd/evaluation.py
```

该入口默认 `expected_rows=108`、`use_audio_in_video=0`，并会严格检查 answer-free 输入、标签 ID 和媒体存在性。OmniVideo 评测使用前必须覆盖这些环境变量，尤其是实际行数和音频设置。

### 2. 使用 checkpoint-175 运行 SFT 评测

在已分配的 GPU 节点上，从项目根目录执行。下面是命令模板，尖括号路径必须替换成真实的独立评测集和标签，不要照抄占位符：

```bash
cd /share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907
# 当前 shell 若未激活环境，可用集群实际的 conda 初始化脚本激活；以下变量固定到已验证环境。

export CUDA_VISIBLE_DEVICES=0,1
export ASCEND_RT_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export OMNI_OPSD_NPROC_PER_NODE=2
export OMNI_OPSD_EVAL_ARM=sft
export OMNI_OPSD_MODEL=/share/home/ylhu/models/Qwen2.5-Omni-7B
export OMNI_OPSD_EVAL_ADAPTER=/share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907/output/gap5000_formal_20260908_010630/sft/v0-20260908-010643/checkpoint-175
export OMNI_OPSD_EVAL_DATASET=<绝对路径/answer_free_heldout.jsonl>
export OMNI_OPSD_EVAL_LABELS=<绝对路径/heldout.labels.jsonl>
export OMNI_OPSD_EVAL_EXPECTED_ROWS=<评测集实际行数>
export OMNI_OPSD_EVAL_USE_AUDIO_IN_VIDEO=<按评测集契约填写0或1>
export OMNI_OPSD_EVAL_OUTPUT_DIR=$PWD/output/eval_sft_checkpoint175_$(date +%Y%m%d_%H%M%S)
export OMNI_OPSD_MS_SWIFT_ROOT=$PWD/ms-swift
export OMNI_OPSD_PYTHON_DEPS=$PWD
export OMNI_OPSD_PYTHON_BIN=/share/home/ylhu/.conda/envs/omniopsd_train/bin/python
export OMNI_OPSD_SWIFT_BIN=/share/home/ylhu/.conda/envs/omniopsd_train/bin/swift
export PYTHONPATH="$PWD/src:$PWD/ms-swift${PYTHONPATH:+:$PYTHONPATH}"

bash scripts/run_video_odyssey_training_eval.sh
```

入口完成后，重点查看：

```text
$OMNI_OPSD_EVAL_OUTPUT_DIR/results.jsonl
$OMNI_OPSD_EVAL_OUTPUT_DIR/infer.log
$OMNI_OPSD_EVAL_OUTPUT_DIR/summary.json
```

`summary.json` 中应同时报告 accuracy、parse rate、缺失/重复/多余 ID 和无法解析的回答数。无法解析的回答按错误计入，不能静默删除。为了比较 SFT 增益，使用完全相同的评测输入和标签再运行一次 `OMNI_OPSD_EVAL_ARM=base`，并将两次输出交给 `aggregate_video_odyssey_training_eval.py` 做配对比较。

### 3. 评测前后检查

```bash
ADAPTER=/share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907/output/gap5000_formal_20260908_010630/sft/v0-20260908-010643/checkpoint-175
test -s "$ADAPTER/adapter_model.safetensors"
test -s "$ADAPTER/adapter_config.json"
python - <<'PY'
import json
from pathlib import Path
p = Path('/share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907/output/gap5000_formal_20260908_010630/sft/v0-20260908-010643/checkpoint-175/adapter_config.json')
cfg = json.loads(p.read_text())
print('base_model_name_or_path =', cfg.get('base_model_name_or_path'))
print('r =', cfg.get('r'), 'lora_alpha =', cfg.get('lora_alpha'))
PY
```

不要修改 checkpoint 内文件，也不要在没有备份的情况下运行清理命令。若评测脚本报路径、ID 或媒体校验错误，应先修正评测数据契约；不要关闭严格校验来得到一个看似完整的分数。

## 已验证的 SFT 冒烟记录

最终成功的单步冒烟目录是：

```text
output/gap5000_smoke_20260908_004156/sft_final/v0-20260908-004208/checkpoint-1
```

它使用 2 卡、`max_steps=1`、每卡 batch=1、梯度累积=1、真实视频和音频，成功反向传播并保存 checkpoint。结果为 `loss=0.2637`、`grad_norm=6.604`、`token_acc=0.8333`、显存约 55.91 GiB。早期 smoke 目录中有多次失败日志（torch.load 安全限制、数据截断、persistent worker、float.cpu），不要把这些失败目录当作成功验证；应以 `gap5000_smoke_20260908_004156/sft_final` 为准。

## 推荐的接手顺序

1. 确认当前没有残留训练进程，确认 `checkpoint-175` 和基座模型可读。
2. 找到或构造独立、answer-free 的评测集和标签，核对行数、ID、媒体和音频契约。
3. 先跑少量评测输入验证基座 + adapter 能加载，再跑完整评测。
4. 保存 `results.jsonl`、`infer.log`、`summary.json` 及输入/标签 SHA-256。
5. 在相同评测集上运行 base 对照，并使用项目聚合脚本比较。
6. 只有需要部署时才考虑合并 LoRA；合并前先验证当前 ms-swift 的 export 参数和 Qwen-Omni 加载方式。

相关的背景和脚本说明还可查看：

```text
docs/GAP5000_RUN_20260908.zh-CN.md
docs/GAP5000_TRAINING_GUIDE.zh-CN.md
docs/PROJECT_WALKTHROUGH.zh-CN.md
scripts/run_training_arm_a100.sh
scripts/run_video_odyssey_training_arm.sh
scripts/run_video_odyssey_training_eval.sh
```
