# WorldSense 开放式问答 SFT 与 CLUE-OPSD（2026-09-29）

初次接触双机训练时，可先读 [多机训练通信入门笔记](MULTI_NODE_TRAINING_COMMUNICATION_BEGINNER.zh-CN.md)。

## 输入和答案

这是一项新设置；旧选择题 JSONL、启动脚本和 checkpoint 保留。1453 道题的输入不再列 A/B/C/D 选项，模型直接回答正确选项的**文字内容**。第 1 题的三个文本如下：

```text
student：
<video>
Question: What reasons does she have for liking the anime version of Gundam?
Answer the question directly in natural language. Give only the answer, without analysis or an option letter.

SFT 目标：
It is more rugged.

CLUE teacher：
<video>
Question: What reasons does she have for liking the anime version of Gundam?
Verified correct answer: It is more rugged.
Answer the question directly in natural language. Give only the answer, without analysis or an option letter.
```

student 看完整视频和音轨；CLUE teacher 看标注的 clue 时间段和音轨。teacher 的 `Verified correct answer` 不进入 student prompt。原题 `xmHjHCiU::task0` 的正确选项是 “None of the above”；去掉选项后这个短语失去意义，所以根据原 SFT 分析改成 “Her old filming location.”。其余 1452 题使用正确选项原文。

生成脚本：[prepare_worldsense_openqa.py](../scripts/prepare_worldsense_openqa.py)。数据位于 `training_runs/worldsense_openqa_20260929/data/`：

- `sft_openqa.jsonl`：全片问答，assistant 目标为正确选项文字。
- `clue_openqa.jsonl`：同一 student 问答，另有带正确答案的 `teacher_prompt` 和证据段 `teacher_videos`。
- `ranktable_w1w2.json`：npu24 worker-1/2 的 HCCL 子集表。

原始选择题、SFT 分析文本和视频采样都没有被覆盖。新文件的 1453 行已经核对：student/teacher prompt 无选项清单、无分析指令，student 不含标准答案；SFT 和 CLUE 目标逐题一致。

## 训练方法和参数

| 设置 | SFT | CLUE-OPSD |
| --- | --- | --- |
| 节点 | npu96 worker-10/11，16 NPU | npu24 worker-1/2，16 NPU |
| LoRA | r64，alpha128 | r64，alpha128 |
| 学习率 | 1e-5 | 2e-6 |
| 训练批次 | 每卡 1、梯度累积 2，3 epoch | 每卡 1、梯度累积 2，3 epoch |
| 监督 | 对正确答案文字做 teacher-forced CE | 50% student rollout 的 GKD；50% 标准答案文字的 GKD + 0.25 倍 SFT CE |
| 教师 | 无 | clue 媒体 + 正确答案文字，EMA alpha 0.05 |
| 生成上限 | SFT 训练不生成 completion | 192 tokens；最长标准答案 145 tokens，99% 不超过 24 tokens |

CLUE 的 `lmbda=0.5` 使一半训练步使用 student 自己生成的回答，另一半使用数据中的正确答案文字；`sft_alpha=0.25` 只在后一分支增加逐 token 的交叉熵。这让标准答案文字真正参与损失，而不只作为 teacher 的提示。GKD/JSD 在两个分支都保留。SFT 以 teacher forcing 训练，因此没有 `max_completion_length` 参数；它的答案长度由 JSONL 目标决定。

## 数据和启动脚本

一次性重建数据：

```bash
cd /opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd
python training_code/scripts/prepare_worldsense_openqa.py \
  --sft-source /opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_sft_lora_npu24_20260928/data/sft_worldsense_npu24.jsonl \
  --clue-source training_runs/worldsense_clue_opsd_w1w2/data/clue_opsd.jsonl \
  --sft-output training_runs/worldsense_openqa_20260929/data/sft_openqa.jsonl \
  --clue-output training_runs/worldsense_openqa_20260929/data/clue_openqa.jsonl
```

正式训练在每台目标 worker 各执行一次，先启动 rank 1，再启动 rank 0：

```bash
# 在 npu96 worker-11 运行 SFT rank 1；在 worker-10 运行 SFT rank 0。
bash /opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_code/scripts/run_worldsense_openqa_sft_npu96_w10w11.sh NODE_RANK

# 在 npu24 worker-2 运行 CLUE rank 1；在 worker-1 运行 CLUE rank 0。
bash /opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_code/scripts/run_worldsense_openqa_clue_opsd_npu_w1w2.sh NODE_RANK
```

这里 `NODE_RANK` 必须替换成实际的 `0` 或 `1`。SFT 新脚本只覆盖数据、输出目录和端口，复用原 npu96 双机 SFT 参数层；CLUE 新脚本独立设置 GKD、teacher、LoRA 和生成参数。两种训练都从 Qwen2.5-Omni-7B 基座重新开始，不续接选择题 checkpoint。

新实验日志与模型输出统一放在 `training_runs/worldsense_openqa_20260929/`：`sft_worker-10.log`、`sft_worker-11.log`、`clue_worker-1.log`、`clue_worker-2.log`，以及 `outputs/sft_formal/`、`outputs/clue_formal/`。从仓库根目录执行 `bash training_code/scripts/npu24-watch.sh 0 1` 查看 CLUE 两节点，执行 `bash training_code/scripts/npu96-openqa-watch.sh 0 1` 查看 SFT 的 worker-10/11；去掉 `0 1` 则持续刷新。
