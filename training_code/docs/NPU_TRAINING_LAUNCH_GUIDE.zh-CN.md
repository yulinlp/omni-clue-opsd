# WorldSense 训练启动入门：SFT 与 CLUE-OPSD

> **2026-09-29 更新：**本文后面的命令记录原选择题实验，供理解和复现历史设置。当前重新训练的“去选项、直接回答选项文字”实验请看 [WORLDSENSE_OPENQA_TRAINING.zh-CN.md](WORLDSENSE_OPENQA_TRAINING.zh-CN.md)。旧 SFT/CLUE 输出目录保留，不能把本页的旧路径当成新实验的启动命令。

多机通信的概念、主节点和 rank 的区别，另见 [多机训练通信入门笔记](MULTI_NODE_TRAINING_COMMUNICATION_BEGINNER.zh-CN.md)。

本文说明这台 ModelArts 机器上**实际使用的** Qwen2.5-Omni-7B 训练脚本、手动启动命令，以及每层脚本做了什么。文中的命令针对当前路径和节点；换作业后，要先修改 IP、rank table、模型与数据路径。

> **先确认没有同一训练正在运行。** 对同一组 NPU、同一输出目录重复执行 `start` 或训练命令，会启动第二套进程，造成显存冲突或写入同一 checkpoint 目录。当前两项训练是否还在运行，应先看监控和日志。

## 1. 先理解一条训练命令

一项双机训练需要在**两台机器上各运行一次启动脚本**。每台机器启动 8 个进程，每个进程使用一张 NPU。两个脚本使用相同的模型、数据、输出目录、`MASTER_ADDR` 和 `MASTER_PORT`，但 `NODE_RANK` 不同：主节点为 0，另一节点为 1。

```text
准备好的 JSONL 数据 + Qwen2.5-Omni-7B 模型
                    │
                    ▼
每台 worker 上执行一次 Bash 启动脚本
                    │ 设置节点地址、卡数、环境变量和训练参数
                    ▼
ms-swift 的 swift sft 或 swift rlhf
                    │ 自动调用 torch.distributed.run，生成每节点 8 个进程
                    ▼
训练日志 + checkpoint
```

三个常见概念：

| 名词 | 在本项目中的含义 |
| --- | --- |
| `NODE_RANK` | **机器序号**，不是 NPU 卡号。两节点分别传 0、1。 |
| `NPROC_PER_NODE=8` | 每台机器启动 8 个训练进程，对应 8 张 NPU。 |
| `RANK_TABLE_FILE` | Ascend HCCL 通信使用的设备清单；双机子集必须重新编号为全局 rank 0–15。 |

全局 batch size 的算法是 `机器数 × 每机 NPU 数 × 每卡 batch × 梯度累积次数`。这里两项正式训练都是 `2 × 8 × 1 × 2 = 32`。梯度累积 2 指每个进程先处理两个小 batch，再同步并更新一次参数。因此日志中的一个 `step` 不是一条样本。

## 2. 两项正式训练分别在哪里运行

| 项目 | SFT-LoRA | CLUE-OPSD |
| --- | --- | --- |
| 算法入口 | `swift sft` | `swift rlhf --rlhf_type gkd` |
| 目标节点 | **另一项 npu96 作业**的 worker-10、worker-11 | **当前 npu24 作业**的 worker-1、worker-2 |
| 卡数 | 16 | 16 |
| 数据 | 1453 条 SFT JSONL，有 user 和 assistant 标答 | 同 1453 题转换成 CLUE JSONL，学生全片、教师证据区间 |
| 学习率 | `1e-5` | `2e-6` |
| LoRA | rank 64、alpha 128 | rank 16、alpha 32，EMA alpha 0.05 |
| 输出 | `omni-opsd-runtime/.../outputs/sft_npu96_w10w11/` | 本仓库 `training_runs/worldsense_clue_opsd_w1w2/outputs/formal/` |

这里的 `npu24` 是**当前作业有 3 节点、总共 24 卡**的名称。脚本名 `run_sft_lora_npu24.sh` 只是继承了最初的 3 节点版本；SFT 的上层脚本把它改成 **2 节点 16 卡**。不要仅凭文件名判断实际使用的节点数。

以下用两个路径变量简化命令：

```bash
REPO=/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd
SFT_RUN=/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_sft_lora_npu24_20260928
```

变量只在**设置它们的当前 shell** 中有效。SSH 到另一台机器后，要重新设置，或在远程命令中直接写完整路径。下面的启动示例因此使用完整路径。

## 3. SFT：真正用到哪些脚本

### 3.1 调用顺序

```text
scripts/launch_npu96_w10w11.sh start      可选：从入口机通过 SSH 编排
  └─ scripts/run_sft_lora_npu96_w10w11.sh 0 或 1
       └─ scripts/run_sft_lora_npu24.sh 0 或 1
            └─ /home/ma-user/anaconda3/envs/PyTorch-2.9.0/bin/swift sft
                 └─ torch.distributed.run（每节点 8 个进程）
```

前三个脚本都在 `$SFT_RUN/scripts/`。真正给模型执行训练的是最底层的 `swift sft`；上面的 Bash 脚本负责把路径、分布式参数和超参数正确传进去。

| 核心文件 | 功能与实现 |
| --- | --- |
| `scripts/launch_npu96_w10w11.sh` | **可选的远程编排器**。通过 SSH 连接 npu96 作业，提供 `check/start/status/stop`。`start` 先启动 rank 1，再启动 rank 0，并把输出重定向到两个日志。它不设模型超参数。 |
| `scripts/run_sft_lora_npu96_w10w11.sh` | **本次 SFT 的双节点配置层**。设 `NNODES=2`、每节点 8 卡、主节点 `172.16.13.102:29521`、梯度累积 2、3 个 epoch、每 epoch 保存一次，以及 worker-10/11 的 HCCL 子集表；最后调用下一层。 |
| `scripts/run_sft_lora_npu24.sh` | **通用 SFT 参数层**。检查模型、数据和依赖是否存在；设置 `PYTHONPATH`、音视频读取和显存环境变量；将 LoRA、学习率、batch、bf16、SDPA、checkpoint 等参数拼成 `swift sft` 命令。 |
| `scripts/build_dataset.py` | **一次性数据准备**。把仓库 `data/sft/sft.jsonl` 中旧服务器的视频绝对路径改成当前 NFS 视频路径，逐条检查视频文件和数据字段，生成 `data/sft_worldsense_npu24.jsonl`。正式启动时读取的是生成后的文件。 |
| `scripts/build_ranktable_subset.py` | **一次性通信配置准备**。从平台总 rank table 取 worker-10/11，并把 16 张卡重新编号为 0–15，生成 `data/ranktable_npu96_w10w11.json`。 |

`run_sft_lora_npu24.sh` 的原始默认值是 3 节点、主节点 worker-0。如果直接在 worker-10/11 上运行它，**不会得到本次 SFT 的配置**。必须运行外层 `run_sft_lora_npu96_w10w11.sh`，或自己完整设置所有覆盖变量。

### 3.2 如何手动启动 SFT

若已经分别进入 npu96 作业的两台目标容器，且数据及 rank table 已存在，可在各自 shell 执行：

```bash
# npu96 worker-11：先启动从节点（NODE_RANK=1）
bash /opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_sft_lora_npu24_20260928/scripts/run_sft_lora_npu96_w10w11.sh 1 \
  > /opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_sft_lora_npu24_20260928/logs/npu96_node-1.log 2>&1
```

```bash
# npu96 worker-10：接着启动主节点（NODE_RANK=0）
bash /opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_sft_lora_npu24_20260928/scripts/run_sft_lora_npu96_w10w11.sh 0 \
  > /opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_sft_lora_npu24_20260928/logs/npu96_node-0.log 2>&1
```

这两个命令会占据各自终端，便于初学者看到退出码；需要长期后台运行时，可用现成编排器从具备 `ulan_31445_npu96` SSH 别名的入口机执行：

```bash
bash /opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_sft_lora_npu24_20260928/scripts/launch_npu96_w10w11.sh check
bash /opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_sft_lora_npu24_20260928/scripts/launch_npu96_w10w11.sh start
bash /opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_sft_lora_npu24_20260928/scripts/launch_npu96_w10w11.sh status
```

逐项解释：`bash` 执行脚本；末尾的 `0/1` 是节点 rank；`>` 把正常输出写入日志；`2>&1` 把错误输出也写进同一日志。两个节点需要在较短时间内启动，否则先启动的一方会等待 rendezvous，最后超时。编排器中的 `setsid nohup ... &` 用来让进程脱离 SSH 会话并在后台运行。

### 3.3 SFT 的主要参数

`swift sft` 对有标准答案的样本做监督微调：模型看视频和问题，学习复现 JSONL `messages` 中 assistant 的回答。脚本设 LoRA rank 64、alpha 128、`all-linear`，以较少可训练参数适配 7B 基座；bf16 和 SDPA 降低显存；`gradient_checkpointing` 用额外计算换激活显存；`use_logits_to_keep` 只计算需要监督的位置；`WORLDSENSE_DROP_TALKER=1` 卸载理解任务不用的语音生成模块。`USE_AUDIO_IN_VIDEO=1` 和 `pyav_seek` 保持音视频输入和本机解码方式。保存策略为每个 epoch 一次，保留最近 3 个 checkpoint。

## 4. CLUE-OPSD：真正用到哪些脚本

### 4.1 调用顺序

```text
training_code/scripts/prepare_worldsense_clue_opsd_npu.py   一次性转换训练数据
                    │
                    ▼
training_runs/worldsense_clue_opsd_w1w2/data/clue_opsd.jsonl
                    │
                    ▼
worker-1/2 各运行一次 training_code/scripts/run_worldsense_clue_opsd_npu_w1w2.sh
                    │
                    ▼
swift rlhf --rlhf_type gkd → torch.distributed.run（每节点 8 个进程）
```

| 核心文件 | 功能与实现 |
| --- | --- |
| `training_code/scripts/prepare_worldsense_clue_opsd_npu.py` | 读已适配当前视频路径的 SFT JSONL 和 `data/annotation/merged.evidence.jsonl`；用 `case_id` 找到 1–4 个 `clue_intervals`；调用动态预算函数，按区间时长分配偶数帧；生成教师的 `teacher_videos`。学生的 `videos` 仍是全片；学生和教师 prompt 都不包含标准答案。输出 1453 行 `clue_opsd.jsonl`。 |
| `training_code/src/omni_opsd/data/dynamic_budget.py` | 数据转换脚本调用其中的 `dynamic_clue_budget_for`。把多个证据区间视为**同一个上下文**，共享视觉 token 预算，并计入音频、文本及 512 token 余量；将总帧数和缩放网格写入数据。它不负责启动训练。 |
| `training_code/scripts/run_worldsense_clue_opsd_npu_w1w2.sh` | **正式训练入口**。参数 0 对应 worker-1，参数 1 对应 worker-2；设置主节点 `172.16.12.237:29531`、16 卡子集 rank table、模型/数据/输出路径以及 GKD、EMA、LoRA 参数，最后执行 `swift rlhf`。 |
| `training_runs/worldsense_clue_opsd_w1w2/data/ranktable_w1w2.json` | 一次性生成的 HCCL 子集表，指定 worker-1 为全局 rank 0–7、worker-2 为 8–15。 |
| `training_code/scripts/npu24-watch.sh` | **只读监控**。SSH 读取两台机器的 8 卡 NPU 状态，并从 worker-1 日志取最新 loss/step/耗时；它不会启动训练。 |

数据转换的关键差异可以用一题理解：原 SFT 行包含“全片视频 + user 问题 + assistant 标答”；CLUE 行保留学生看到的全片视频，把标注的几段时间窗放入 `teacher_videos`，并删除 assistant 标答。GKD 让学生先生成短 completion，再比较学生和 EMA 教师对这些 token 的分布。教师看到的是证据段音视频，学生看到的是全片音视频。

### 4.2 如何手动启动 CLUE-OPSD

**只在数据已经准备好且 worker-1/2 的目标 NPU 空闲时执行。** 先在当前作业的 worker-0 检查输入和当前训练状态：

```bash
cd /opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd
wc -l training_runs/worldsense_clue_opsd_w1w2/data/clue_opsd.jsonl
bash training_code/scripts/npu24-watch.sh 0 1
```

`wc -l` 应显示 1453；监控表可看到是否已有训练进程占用显存。若分别登录到 worker-2 和 worker-1，可按下面顺序在两个终端运行前台命令：

```bash
# 当前作业 worker-2：先启动从节点，参数 1
cd /opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd
bash training_code/scripts/run_worldsense_clue_opsd_npu_w1w2.sh 1 \
  > training_runs/worldsense_clue_opsd_w1w2/worker-2.log 2>&1
```

```bash
# 当前作业 worker-1：随后启动主节点，参数 0
cd /opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd
bash training_code/scripts/run_worldsense_clue_opsd_npu_w1w2.sh 0 \
  > training_runs/worldsense_clue_opsd_w1w2/worker-1.log 2>&1
```

若从 worker-0 通过 SSH 启动后台任务，本次正式训练使用的是以下形式。`-p 2222` 是容器 SSH 端口；`-F /dev/null` 忽略本机有权限问题的系统 SSH 配置；`setsid -f` 让训练进程脱离登录会话；`< /dev/null` 使后台进程不再等待终端输入：

```bash
# 先在 worker-2 启动 NODE_RANK=1
ssh -F /dev/null -p 2222 -o BatchMode=yes 172.16.12.70 \
  'cd /opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd && setsid -f bash training_code/scripts/run_worldsense_clue_opsd_npu_w1w2.sh 1 > training_runs/worldsense_clue_opsd_w1w2/worker-2.log 2>&1 < /dev/null'

# 再在 worker-1 启动 NODE_RANK=0
ssh -F /dev/null -p 2222 -o BatchMode=yes 172.16.12.237 \
  'cd /opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd && setsid -f bash training_code/scripts/run_worldsense_clue_opsd_npu_w1w2.sh 0 > training_runs/worldsense_clue_opsd_w1w2/worker-1.log 2>&1 < /dev/null'
```

每个 `ssh` 命令中的单引号包着**在远端执行的整条命令**。远端先 `cd` 到仓库，再启动脚本并将日志写在共享目录。SSH 返回只表示远端命令已提交，**不能证明训练成功**；还要查看两个日志中的 `world_size: 16`、开始训练的 `Train: 0/135` 及首个 `{'loss': ... 'global_step/max_steps': '1/135'}`。

### 4.3 CLUE-OPSD 的主要参数

`--rlhf_type gkd` 指定广义知识蒸馏训练；`--lmbda 1.0` 使用学生自己生成的 completion；`--beta 0.5` 和 `--temperature 1.0` 控制分布比较；`--gkd_logits_topk 100` 把损失计算限制在前 100 个候选 token 等实现所需的尾部概率；`--sft_alpha 0` 关闭额外的 SFT 交叉熵项。`--clue_ema_alpha 0.05` 启用本机 ms-swift 补丁中的 LoRA-shadow EMA 教师。`--max_completion_length 8` 指每次学生最多生成 8 个 token，**不是**输入视频长度。`--use_vllm false` 指这次用 Transformers 执行 rollout；本机没有验证 vLLM colocate 正式路径。

输入视频可达约 300 秒。为在 64GB NPU 上跑通重样本，启动脚本同时设置 `--use_logits_to_keep true`、`WORLDSENSE_LOGITS_TO_KEEP=1` 和 `WORLDSENSE_DROP_TALKER=1`；三者共同让本机 `sitecustomize.py` 对 Omni 的 `lm_head` 只投影需要的 completion 位置，并卸载不用的语音输出模块。缺少这些设置时，重样本单步曾在把整段 logits 转为 fp32 时 OOM。`PYTHONPATH` 前面的依赖目录加载该适配补丁及 PyAV 视频读取器。

### 4.4 训练算法和本机补丁实际在哪里执行

启动器末行的 `exec "${swift_bin}" rlhf "${args[@]}"` 会**用 swift 进程替换当前 Bash 进程**。以后执行的算法不在启动器里，而在安装于 `/opt/huawei/dataset/hyl_ulan/ylhu/third_party/ms-swift/` 的代码中：

| 运行时文件 | 学习时应看什么 |
| --- | --- |
| `swift/cli/rlhf.py` | `swift rlhf` 的 Python 入口：解析命令行，创建模型、数据和训练器。SFT 对应的入口是 `swift/cli/sft.py`。 |
| `swift/rlhf_trainers/gkd_helpers.py` | 将数据行中的学生 `videos` 与教师 `teacher_videos` 分别送去编码，使双方能看同一题的不同媒体范围。 |
| `swift/rlhf_trainers/gkd_trainer.py` | CLUE 的核心训练器：生成 completion，分别执行学生和 EMA 教师前向，计算 GKD/JSD 损失、更新参数和 EMA 阴影权重。`clue_ema_alpha` 的实际实现也在这里；只有命令行参数并不代表功能已实现。 |
| `omni-opsd-runtime/worldsense_npu24_deps/sitecustomize.py` | 本机 Python 启动时自动加载的显存适配代码：卸载 talker/token2wav，并让 `logits_to_keep` 真正作用于 Qwen-Omni 的 `lm_head`。该目录被放在 `PYTHONPATH` 首位。 |

所以，修改训练方法时要看 ms-swift 训练器和数据协议；修改“在哪些节点、用多少卡、从哪里读数据”时才改 Bash 启动器。对于初次复现，先沿用已经验证的脚本，不必逐行手写一条很长的 `swift` 命令。

## 5. 一次性准备命令：什么时候才需要运行

这两类准备命令只在**新机器、新作业或重新构建数据**时执行。当前正式数据和 rank table 已存在；训练中不要覆盖正在读取的 JSONL。

### SFT 视频路径适配

```bash
python /opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_sft_lora_npu24_20260928/scripts/build_dataset.py \
  /opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_sft_lora_npu24_20260928/data/sft_worldsense_npu24.jsonl
```

这一步只改视频路径，不重新筛选题目或改帧预算。若原仓库的 `data/sft/sft.jsonl` 已是当前机器的视频路径，就不需要转换。

### CLUE 数据转换

```bash
cd /opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd
PYTHONPATH=training_code/src python training_code/scripts/prepare_worldsense_clue_opsd_npu.py \
  --sft /opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_sft_lora_npu24_20260928/data/sft_worldsense_npu24.jsonl \
  --annotation data/annotation/merged.evidence.jsonl \
  --output training_runs/worldsense_clue_opsd_w1w2/data/clue_opsd.jsonl
```

`PYTHONPATH=training_code/src` 让 Python 找到仓库中的 `omni_opsd.data.dynamic_budget`；`--sft` 是已适配本机视频路径的输入；`--annotation` 提供证据区间；`--output` 是训练脚本读取的文件。重建后应检查行数、视频路径和数据 SHA256，并用单步重样本测试确认模型能实际读入。

### HCCL 双机 rank table

```bash
# 当前 npu24 作业：选择平台表的 worker-1、worker-2
python /opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_sft_lora_npu24_20260928/scripts/build_ranktable_subset.py \
  /user/config/jobstart_hccl.json \
  /opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_clue_opsd_w1w2/data/ranktable_w1w2.json \
  1 2
```

`1 2` 是**平台总表中的 worker 下标**，不是训练脚本的 `NODE_RANK`。同理，npu96 SFT 使用那项作业的总表，选择 `10 11`，输出到 `$SFT_RUN/data/ranktable_npu96_w10w11.json`。两个作业的总表和设备 IP 不可混用。

## 6. 启动后如何确认训练真的在跑

```bash
# 当前作业 worker-1/2 的双列 NPU 实时表；Ctrl-C 退出监控，不会结束训练
bash /opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_code/scripts/npu24-watch.sh

# CLUE 主节点日志。关注 step、loss、OOM、Traceback、checkpoint。
tail -f /opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_clue_opsd_w1w2/worker-1.log

# npu96 SFT 的主节点日志
tail -f /opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_sft_lora_npu24_20260928/logs/npu96_node-0.log
```

日志中 `rank: 0` 和 `rank: 8` 表示两节点已加入 16 进程作业；首次出现 `Train: 0/...` 表示进入训练循环；**出现第一条 loss 且 step 从 0 变为 1**，才说明至少完成一次前向、反向与参数更新。checkpoint 通常到 epoch 结束才写入，不能用“暂时没有 checkpoint”判断训练未启动。

## 7. 哪些文件不是正式启动入口

`README.md` 和 `docs/Train_Settings.md` 是说明材料；`run_*smoke*.sh` 用于小样本单步冒烟；`npu24-watch.sh`、`npu96-watch.sh` 只负责监控；`prepare_*`、`build_*` 负责一次性准备数据或通信表。历史上的 `run_gap5000_*`、`run_omnivideo_*` 是其他数据集和旧硬件实验脚本。此次正式训练应按上面的调用链选择入口，不要把这些相近的脚本混用。
