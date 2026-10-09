# Omni CLUE-OPSD 项目完整交接文档

> 交接日期：2026-10-09，时区 Asia/Shanghai。文件名按交接要求使用 **HADOFF.md**。
> 目标仓库：https://github.com/yulinlp/omni-clue-opsd.git
> 原工作目录：`/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd`。
> 本文是当前交接入口。根目录 README.md 主要描述 9 月 25 日的第一版流程，不能代替本文中的最新实验和数据说明。

## 1. 接手后先读这几项

1. 阅读本文第 2、3、7、10 节，了解研究目标、最新数据、实验结果及当前任务状态。
2. 查看 `handoff/EXPORT_SUMMARY.json` 和导出清单。仓库包含代码、非视频数据、结果和分析；**模型权重、优化器状态、凭据与视频不随普通 Git 上传**。模型和优化器文件逻辑体积约 40 TB，可能包含硬链接重复计数，不能把这个数字当作实际磁盘占用。
3. 如 `handoff/bundles/` 中存在分卷，执行第 12 节的恢复命令，再查看 `training_runs/` 下的完整逐题结果、日志和训练轨迹。
4. 在新机器建立环境、准备原模型与媒体文件、修改路径和节点配置。**不要直接执行从旧机器复制来的控制器状态文件或 PID 文件。**
5. 先完成单卡读取音视频测试，再做多卡小样本训练、保存与恢复测试，最后启动正式任务。

本文区分三件事：已经完成的结果、源码中配置的下一批实验、尚未完成的迁移工作。状态快照只表示写入时间，不是实时监控。

## 2. 项目研究目标

基座是 **Qwen2.5-Omni-7B**，任务是输入视频及其音频，回答视频相关问题。

- **SFT**：给模型输入视频、问题和选项，直接监督标准答案；部分实验同时监督 observation 作为分析文本。
- **OPSD**：学生与教师都看完整视频。学生先生成回答，教师在学生已经生成的 token 前缀上提供概率分布，学生学习该分布。
- **CLUE-OPSD**：学生看完整视频，教师看人工或自动标注的 golden clue 区间。教师提示还可以包含标准答案、observation；不同实验必须以各自配置为准。
- **golden clue**：用于回答题目的视频时间区间。一题可以有多个区间。它不是标准答案，也不保证其中的证据一定完整。
- **observation**：标注流程产生的文字说明。历史文本可能含猜测、与答案矛盾或泄露“参考答案”等措辞，因此不能当作可靠事实直接使用。
- **answer**：选择题的标准选项字母。开放题实验曾将正确选项的文字作为监督目标。

最新一轮使用选择题，学生输出分析和选项。教师固定、不更新，使用 clue 和标准答案，不使用 observation。不要把历史 EMA 教师设置误当作当前设置。

## 3. 数据沿革与当前应使用的版本

### 3.1 早期版本，仅用于复查

`data/annotation/`、`data/screening/`、`data/selection/`、`data/sft/` 是最初标注、Full/Gold 筛选和 1453 条 SFT 数据。最初 WorldSense 为 3172 题，旧筛选集为 1500 题，准备训练后为 1453 题。

这些目录需要保留，以便重现旧结果，但**不是最新训练集**。

### 3.2 重新核验与标注

相关产物：

| 路径（相对 training_runs/） | 内容 |
|---|---|
| `worldsense_clue_only_agentic_20261004/` | 原 clue-only 答题、失败后的 agentic 重标、重新核验；包含 samples、results、items 和过程记录 |
| `worldsense_codex_reannotation_20261004/` | 后续人工式复查、区间建议、核验结果和审阅记录 |
| `worldsense_option_repair_20261004/` | 选项问题的检查与修复 |
| `worldsense_gap_reannotated_npu120_20261004/` | 重新标注版本的 Full/Gold 筛选 |
| `worldsense_gap_training_matched_npu120_20261004/` | 与训练媒体设置对齐的筛选版本 |
| `worldsense_observation_guess_audit_20261004/` | 猜测及不确定 observation 的筛查 |
| `worldsense_train671_observation_review_20261004/` | 第一批训练题的 observation 检查与重写 |
| `worldsense_train829_observation_rewrite_20261004_v2/` | 其余题的放宽要求后重写 |

曾有 666 题重标后仍未通过核验。它们不是一种统一错误：包含题目/答案争议、证据不足和模型答错等情况。复查记录不是所有题都已经证明正确。详细分类见对应目录报告，不能把“保留训练题”解释为“已证实标注正确”。

### 3.3 1500 题与 1000 题

- 1500 题成员及原 canonical 内容：
  `training_runs/worldsense_train1500_ab810_d690_20261004/train_1500.canonical.jsonl`
- 1000 题成员：
  `training_runs/worldsense_train1000_ab810_d190_20261004/train_1000.canonical.jsonl`
- 更新 observation 后、八组选择题实验的数据来源：
  `training_runs/worldsense_mcq8_latest1500_20261004/data/train_1500.updated_observations.jsonl`
- **最新九组修正实验的直接训练输入**：
  `training_runs/worldsense_clue_repair_ablation_20261007/<arm>/data/train.jsonl`

最新训练使用 1000 题：810 条 A/B 加 190 条 D。构建方式是从最新准备好的训练行中按 1000 题的 ID 取子集，保留更新后的文本；不是直接退回旧 canonical 文件中的 observation。

`sample_id` 常用于 canonical 数据，`case_id` 常用于准备后的训练数据。做关联前检查字段，不要按文件行号关联。

**训练顺序是固定种子控制的随机取样，不是按 Full/Gold 优势排序。** 只训练 20 步，不等于只取原文件前 960 行。

### 3.4 Full/Gold 筛选含义

对同一题，用原模型分别看 full video 和 golden clue，记录预测结果、正确选项概率及对数概率，再比较 Gold − Full 的差异。`screening_code/scripts/worldsense_gap/select_gap.py` 的分层为：A=Gold 对、Full 错；B=两者正确性相同，且正确选项概率 Gold − Full 至少为 0.10；C=Full 对、Gold 错；D=其余题。B 可以包含两者都错的题，D 也不等于两者都错。**具体版本必须核对该次 per_question 记录和选样脚本**，不能只看目录名称推导阈值。

排除超 300 秒媒体。历史任务中既有 `<300` 也有 `<=300` 的要求；最新两套外部评测实际采用 `0 < duration <= 300`，见 protocol 和 media_budget_audit。

## 4. 代码地图

### 4.1 通用代码

| 文件或目录 | 功能 |
|---|---|
| `training_code/src/omni_opsd/` | 数据处理、音视频预算、标注与辅助模块 |
| `training_code/src/omni_opsd/data/dynamic_budget.py` | 根据时长、音频 token 和上下文预算分配视频预算 |
| `training_code/scripts/worldsense_mcq_training_suite.py` | 历史多实验、多节点训练调度与 SSH 辅助函数 |
| `training_code/scripts/worldsense_mcq_sft_entry.py` | SFT 训练入口及答案/分析监督处理 |
| `training_code/scripts/worldsense_mcq_gkd_entry.py` | 选择题蒸馏入口 |
| `training_code/scripts/worldsense_full_gkd_entry.py` | 全参数 GKD 入口及教师处理 |
| `training_code/scripts/worldsense_mcq_checkpointing.py` | 检查点保存和完整性标记 |
| `training_code/scripts/worldsense_mcq8_eval_fleet.py` | 八组实验的检查点评测 |
| `training_code/scripts/worldsense_distill_answer_eval_fleet.py` | 蒸馏模型改用仅答案提示的评测 |
| `training_code/scripts/score_generalization_analysis_mcq_v4.py` | 兼容明确但不完全标准的答案格式；不能利用标准答案猜测模型输出 |
| `training_code/scripts/run_worldsense_observation_rewrite.py` | observation 重写流程 |
| `screening_code/`、`training_code/scripts/worldsense_gap/` | Full/Gold 筛选、评分及数据选择 |
| `training_code/patches/` | 框架补丁；还需核对外部 ms-swift 源码快照 |

### 4.2 最新实验必须使用其冻结代码

设 `R=training_runs/worldsense_clue_repair_ablation_20261007`。

| 文件 | 功能 |
|---|---|
| `$R/prepare.py` | 构建九组实验配置、1000 题输入和 rank table；包含旧机器假设，迁移后不要直接覆盖历史目录 |
| `$R/suite.json` | 每组数据路径、哈希、节点、训练参数、教师设置与保存规则 |
| `$R/code/worldsense_mcq_training_suite.py` | 本轮控制器，试跑、正式训练、断点重试和第二批门槛 |
| `$R/code/clue_ablation.py` | 本轮损失修改、教师处理、验证、诊断和提前停止 |
| `$R/code/mcq_distill_regions.py` | 分析、格式、答案区域的识别和加权 |
| `$R/code/mcq_complete_sampler.py` | 分布式样本覆盖与补齐；避免框架截断导致漏样本 |
| `$R/eval_code/eval_fleet.py` | 已保存检查点的两套 500 题评测、任务汇总和完成标记 |
| `$R/monitor_three.py` | 控制器存活及 worker 心跳监控；文件名沿用旧版，实际监控九组 |
| `$R/watch_logs.py` | 日志错误、非有限损失、长时间无进度检查 |
| `$R/test_objectives.py`、`$R/code/test_ablation_cpu.py` | 损失和数据处理的 CPU 检查 |

运行快照里的脚本仍会 import `training_code/scripts` 里的共享函数；因此迁移时不能只复制 `$R/code`。

## 5. 训练输入和提示词

### 5.1 学生提示

最新蒸馏主分支使用如下指令，前面还有题目、选项及完整视频输入：

```text
Briefly analyze the video and audio evidence in English using at most 120 words.
Put ALL analysis inside <analysis>...</analysis>.
Then select exactly ONE option.
Between <answer> and </answer>, write ONLY ONE uppercase option letter: A, B, C, or D.
Do NOT put analysis, explanations, option text, punctuation, or any other text inside <answer>...</answer>.
An analysis alone is incomplete: after </analysis>, you MUST write <answer>, your chosen letter, and </answer>.
Close the answer with </answer> and do not write any text after it.
```

### 5.2 教师提示和媒体

当前教师看 golden clue，提示中有标准答案，无 observation。学生不能接收这些字段。准确拼接内容查看该组 `data/train.jsonl` 中的教师字段及入口脚本。不要用本文的概述替换真实模板。真实样本的学生/教师文本、媒体路径和预算示例见 `handoff/PROMPT_EXAMPLE.json`。教师额外行包括 `Correct option: <字母>. <正确选项内容>`，并要求不要提及提供的标准选项。多个 `<video>` 对应多个 clue 片段。

历史带 observation 的实验，教师还包含 observation；OPSD 教师看 full video；这些不是当前实验设置。

### 5.3 答案监督为何单独改动

旧做法：学生先生成分析，随后在这段分析后监督标准答案。如果学生分析支持 B，却强制让后续答案为 A，会出现上下文与监督目标不一致。

`clean_ce` 做法：另起一次仅包含完整视频、题目、选项和仅答案指令的前向，监督 `<answer>标准字母</answer>`，不带学生刚才的分析。蒸馏主分支仍生成分析加答案，评测也仍要求分析加答案。

`lmbda=1.0` 表示蒸馏沿学生生成的回答计算；`sft_alpha=0` 关闭框架原生 SFT 混合项。**不代表本轮没有答案监督**：含 CE 的实验另加 `gold_ce=0.25` 的辅助损失。

没有完整答案时，不能把分析最后一个字母当作答案区域加权。查看 missing/ambiguous-answer 诊断和实际 token 区域。

## 6. 最新九组修正实验设置

统一设置：原模型起训，固定教师，无 observation；除冻结组外全参数训练；每组 3 worker、24 卡，每卡 batch 1、梯度累积 2，有效 batch 48；数据 1000 题；基础学习率 2e-6；学生生成上限 512 token。第一批最多 20 步，第二批最多 1 epoch（21 步）。

| 实验名 | 主要改动 | 批次 |
|---|---|---:|
| `clue_reference` | 旧学生分析上下文 CE + 旧分区域蒸馏加权 | 1 |
| `clue_clean_ce` | 只改为独立上下文的正确答案 CE | 1 |
| `clue_token_jsd` | 只改为按 token 权重总和归一化 JSD | 1 |
| `clue_p2_only` | 旧上下文 CE + 普通 JSD，无 JSD 答案加权 | 1 |
| `clue_p3_only` | 旧分区域 JSD，无辅助 CE | 1 |
| `clue_clean_token` | 独立 CE + 按 token 归一化 JSD | 2 |
| `clue_low_lr` | clean_token，学习率降为 5e-7 | 2 |
| `clue_freeze_media` | clean_token，冻结音频、视觉及连接模块 | 2 |
| `clue_plain` | 普通 JSD，无辅助 CE | 2 |

所有含 CE 组保持字母 CE 和格式 CE 分开平均，系数 0.25。本轮 P2-only 不是最早“所有答案 token 一起平均”的实现。

保存第 1–15 步及第 20 步，保存最终步和提前停止点。第二批最终为 21 步。固定 dev96 用于训练过程验证，从第 3 步起，连续三次验证比该组初始答对数少至少 10 题时提前停止。提前停止不是训练崩溃，也不是外部评测集挑出的停止点。

`suite.json` 继承了历史顶层字段，例如 `sampler.all_originals_per_epoch=1500`、旧 EMA 数值、旧 observation 统计等。**以当前每组 config、实际 train.jsonl 行数、启动参数和运行日志为准。** `EXPERIMENTS.zh-CN.md` 中也有保留的旧段落，顶部变更记录和当前配置优先。

## 7. 最新第一批完整结果

评测：OmniVideoBench 和 DailyOmni 各 500 题，视频不超过 300 秒，分析加选项，随机采样。共 55 检查点、110 项评测、55000 回答，全部完成。

### 7.1 最佳与最终结果

最佳按两套正确率平均选取，并列取较早步数；这是测试集事后选优，不是独立验证的泛化提升。

| 实验 | 最佳步 | 最佳 OVB / Daily (%) | 最终步 | 最终 OVB / Daily (%) |
|---|---:|---:|---:|---:|
| 历史原模型基线 | — | 36.0 / 57.6 | — | — |
| clue_reference | 1 | 36.2 / 55.8 | 8 | 26.6 / 35.0 |
| clue_clean_ce | 5 | 36.2 / 59.6 | 20 | 31.6 / 50.0 |
| clue_token_jsd | 4 | 36.0 / 58.0 | 7 | 28.8 / 35.0 |
| clue_p2_only | 2 | 34.8 / 57.8 | 8 | 30.8 / 36.0 |
| clue_p3_only | 1 | 36.2 / 55.8 | 20 | 31.0 / 50.6 |

reference、token_jsd、p2_only 按预设 dev96 规则提前停止，其余第一批组完成 20 步。

结果文件：

- `$R/evaluation/BATCH1_ALL_RESULTS.csv`：全部 110 项。
- `$R/evaluation/BATCH1_SUMMARY.json`：最佳与最终。
- `$R/evaluation/comparison.json`：控制器评分汇总。
- `$R/evaluation/BATCH1_COMPLETE.json`：第一批完成标记。
- `$R/evaluation/eval/` 及 worker 分片目录：逐题回答与评分；实际目录以导出清单为准。

准确率随训练步数的图：`handoff/BATCH1_ACCURACY.png`（PDF 同目录），可由 `handoff_tools/plot_batch1.py` 重新生成。

### 7.2 如何解释

第 6 步 DailyOmni：reference 41.2%、token_jsd 37.4%、p2_only 34.6%；clean_ce 56.0%、p3_only 56.2%。这支持“在可能错误的学生分析后监督标准答案”是早期骤降的重要候选原因。它还不是排除其他因素后的完整因果证明。

clean_ce 最佳平均 47.9%，历史原模型 46.8%，差 1.1 个百分点；最终仍全部低于原模型。修改 CE 缓解骤降，但没有解决后续退化。

原模型基线复用历史同题输出，本轮分片数与历史不同；随机采样消耗顺序会变。小幅差异需要用同分片、同种子复评，并报告多个种子或置信区间。不能把最高检查点的微小提升说成稳定有效。

## 8. 历史结果与分析索引

| 目录（相对 training_runs/） | 应查看的内容 |
|---|---|
| `worldsense_openqa_20260929/` | 开放题训练、早期 loss 和错误分析；`analysis_clue_20260930/` 中有曲线 |
| `worldsense_observation_eval_20261001/` | observation 相关四组实验评测 |
| `worldsense_training_matched_eval_20261003/` | 对齐采样和媒体预算后的评测 |
| `worldsense_generalization_mcq500_npu96_20261003/` | 外部选择题评测；`rescore_explicit_formats_v3/RESULTS.zh-CN.md` 为兼容格式重评分报告 |
| `generalization_openqa_npu120_20261003/` | 开放式评测；评分与选择题不可混合比较 |
| `worldsense_mcq8_latest1500_20261004/` | 八组选择题训练，注意废弃早期 formal 与后来完整采样版本的区别 |
| `worldsense_mcq8_eval500_20261005/` | 八组所有保存检查点评测；区分 answer 与 observation+answer 提示 |
| `worldsense_distill_answer_eval500_20261006/` | CLUE/OPSD 改用仅答案提示的评测 |
| `worldsense_atomic_latestclue_npu56_20261006/` | 原模型多原子视角评测及最新 clue |
| `worldsense_clue_p123_npu96_20261006/` | 前一轮三组 P1/P2/P3 组合实验，51 检查点、102 项评测；`analysis_20261007/REPORT.zh-CN.md` 有详细诊断 |
| `worldsense_clue_repair_ablation_20261007/` | 当前九组修正实验与第一批完整结果 |

主要阅读文档：

- `training_code/docs/NPU_TRAINING_LAUNCH_GUIDE.zh-CN.md`：新手启动说明；其中的早期超参数不是最新设置。
- `training_code/docs/MULTI_NODE_TRAINING_COMMUNICATION_BEGINNER.zh-CN.md`：多卡通信、batch、梯度累积。
- `training_code/docs/WORLDSENSE_TRAINING_FAILURE_BEGINNER_20261003.zh-CN.md`：输出重复、缺少最终答案、长度漂移等。
- `training_code/docs/WORLDSENSE_ANNOTATION_QUALITY_PIPELINE_TASK_B_20261003.zh-CN.md`：标注质量问题和流程建议。
- `training_code/docs/WORLDSENSE_AGENTIC_CASES_AND_DEMO_20261004.zh-CN.md`：时间戳 caption、反复看片与具体失败案例。
- `training_code/docs/WORLDSENSE_OBSERVATION_EXAMPLES_AND_REFERENCE_WORDING_20261004.zh-CN.md`：observation 中文示例及参考答案措辞。

## 9. 环境与外部依赖

### 9.1 原环境

原解释器：`/home/ma-user/anaconda3/envs/PyTorch-2.9.0/bin/python`。

交接时读到的安装元数据：Python 3.11.14、torch 2.9.0、torch-npu 2.9.0.post2、transformers 5.8.1、ms-swift 4.4.0、peft 0.14.0、accelerate 1.14.0、deepspeed 0.16.4、datasets 3.0.1、av 17.1.0、numpy 1.26.0、Pillow 11.3.0。

**这是交接时环境快照，不保证等于所有历史实验运行时版本。** 包版本号也不能证明本地补丁相同。实际脚本会通过 PYTHONPATH 加载本地源码。优先保留该源码及补丁，在新机器上核对 import 路径。

外部源码重要路径：`/opt/huawei/dataset/hyl_ulan/ylhu/third_party/ms-swift/`。另有 `/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_npu24_deps/`，包含 `sitecustomize.py` 和 `qwen_omni_utils` 0.0.9；启动脚本把它放进 PYTHONPATH，因此环境元数据中的 qwen-omni-utils 未安装不表示训练缺少此模块。其中 GKD trainer、helpers、模型和模板注册可能含本地修改。交接导出包含可用的外部源码快照时见 `handoff/external_sources/`；不能只 `pip install ms-swift==4.4.0` 就假设可复现。

`worldsense_npu24_deps` 中含编译扩展；如果新机器 CPU 架构或 Python ABI 不同，需要重新安装/编译对应扩展，不能直接复用二进制。

Ascend 环境还需要匹配 NPU 驱动、固件、CANN、torch-npu 和 HCCL。新机器如使用 GPU，需要改设备初始化、分布式后端和 NPU 专用环境变量，并重新做烟雾测试；不能直接复制 HCCL rank table。

### 9.2 必须单独准备的资产

| 资产 | 原位置或查找方法 |
|---|---|
| Qwen2.5-Omni-7B 基座 | `/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-assets/models/Qwen2.5-Omni-7B` |
| WorldSense 视频、golden clue 媒体 | 查看训练 JSONL 的 videos、teacher_videos 及媒体清单；clue 可按区间重新生成 |
| OmniVideoBench / DailyOmni 视频 | 查看评测 inputs.jsonl 和 media_budget_audit.jsonl；固定选题清单随结果保存 |
| 旧 SFT runtime | `/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_sft_lora_npu24_20260928/` |
| 模型检查点、优化器、教师权重 | 见导出外部文件清单，按需要 rsync；普通仓库未包含这些大文件 |
| API 密钥、SSH 密钥 | 新机器自行配置，不从公开仓库获取 |

视频虽不上传，非视频 JSONL 内仍保留原绝对路径用于溯源，迁移时必须映射。

### 9.3 音视频和推理一致性

最新主分支生成：temperature=1、top_p=1、top_k=50、最多 512 token；分析要求 120 words，不等于 120 tokens。上下文上限 32768，视频预算上限 24000，fps=2、最多 300 帧，音频 16k；实际视频预算会随音频长度动态减少，以实际 budget 记录为准。

评测必须保留音频输入，不能只看存在 video 路径就认为音频正常。检查 `USE_AUDIO_IN_VIDEO`、解码方式、processor 参数和音频 token 数。核对帧数、网格、max_pixels 与动态预算，而不只是核对 fps。

## 10. 当前运行状态、故障与未完成事项

截至 2026-10-09 13:23 的检查：第一批五组训练及 110 项评测完成；第二批四组已提交，节点报告 `waiting_for_free_cards`，尚无第二批结果。更新的只读快照见 `handoff/CURRENT_STATUS_SNAPSHOT.json`。迁移打包不代表已经停止旧机器上的任务。

当天发现并修复：`eval_fleet.py` 写入 `evaluation/arm_complete/<arm>.json` 时，父目录未创建，控制器退出。监控每 180 秒重启，仍重复同一错误，阻塞第一批完成标记和第二批启动。修复为保存 JSON 前先创建父目录，随后补齐最后五项评分，写入 BATCH1_COMPLETE。

**教训：进程能自动重启，不代表任务在前进。** 下一位维护者应同时检查 health 时间戳、成功任务数变化、重复异常和等待资源原因，不能只看 issues 是否为空。

待办：

1. 完成新机器资产迁移与环境验证。
2. 明确是否由新机器接续第二批，避免新旧控制器对同一输出目录双写。
3. 第二批按第一批之后的配置运行，完成所有保存检查点的两套 500 题评测。
4. 用同一推理分片和种子重评原模型，以确认小幅提升。
5. 将评测控制器重复报错和陈旧 health 视为告警；不要无限盲目重启。
6. 进一步核对独立 CE 的监督效果、teacher/student 选项概率、输出长度、重复分析与缺少答案的比例。

## 11. 新机器启动步骤和命令含义

### 11.1 克隆和恢复

```bash
git clone https://github.com/yulinlp/omni-clue-opsd.git
cd omni-clue-opsd
python handoff_tools/restore_bundles.py --repo . --verify-only
python handoff_tools/restore_bundles.py --repo .
```

第一条获取仓库；第二条只校验分卷完整性；第三条将结果分卷恢复到原相对路径。恢复工具拒绝越界路径和链接。若没有分卷，工具会明确报告，不能由此推断外部权重已经上传。

### 11.2 环境与路径迁移

1. 安装与硬件兼容的 Python/PyTorch/NPU 环境。
2. 恢复本地修改过的 ms-swift 等源码，并配置 PYTHONPATH。
3. 准备基座、视频及需继续使用的 checkpoint。
4. 在**新的运行目录**准备配置，不修改用于审计的旧实验快照。
5. 修改模型路径、训练/评测 JSONL 媒体路径、输出目录、Python 路径、REPO、SSH 节点和 rank table。
6. 数据或代码路径改变会影响 SHA256；重新生成新实验哈希并记录迁移关系，不删掉 verify 检查来绕过错误。

查找旧路径：

```bash
rg -n '/opt/huawei/|/home/ma-user/|172\.16\.|dev-modelarts' \
  training_code/scripts training_runs/worldsense_clue_repair_ablation_20261007/code \
  training_runs/worldsense_clue_repair_ablation_20261007/eval_code
```

不要对整个仓库无差别替换字符串。旧报告中的路径是历史记录；需要修改的是新任务实际加载的脚本、配置和数据。

### 11.3 控制器命令模板

下面命令只在**完成上述适配、生成新任务配置与 rank table 后**执行。`RUN_ROOT` 必须指向新的任务目录。不能把旧 COMPLETE、controller_state、PID 直接当成新任务状态。

```bash
PYTHON=/path/to/validated/env/bin/python
RUN_ROOT=/path/to/new_run

"$PYTHON" "$RUN_ROOT/code/worldsense_mcq_training_suite.py" \
  --root "$RUN_ROOT" --action launch-controller

"$PYTHON" "$RUN_ROOT/eval_code/eval_fleet.py" \
  --root "$RUN_ROOT/evaluation" --action prepare

"$PYTHON" "$RUN_ROOT/eval_code/eval_fleet.py" \
  --root "$RUN_ROOT/evaluation" --action launch-all
```

- `launch-controller`：验证代码/数据哈希并启动训练队列。控制器负责远程节点启动、检查点恢复与重试。
- `prepare`：建立固定评测输入、标签、媒体审计、任务表；源码有历史目录依赖，新机器需先改为已恢复的输入目录。
- `launch-all`：启动评测控制器和 worker；等待训练保存完整并退出后再评测。

监控脚本按自身所在目录定位任务：复制到新 RUN_ROOT 后，核对内部 Python/REPO 路径，再用独立日志启动。不要只改命令行 root 而忽略脚本内部固定路径。

### 11.4 最小验收

- 训练数据实际为 1000 个不同 ID，与成员清单一致。
- 学生输入没有 gold answer、observation 或 teacher 媒体泄漏。
- 学生完整视频和教师 clue 区间均可解码，音频存在且预算合法。
- 全部 rank 进入训练，loss/梯度有限，至少完成一次更新。
- 可保存完整模型、教师、优化器和每 rank RNG；从保存点恢复后能继续一步。
- 推理能输出 `<analysis>...</analysis><answer>A</answer>`；同时保留不合规输出统计。
- 评分按 sample_id 关联，500 条无缺失、无重复，媒体时长符合限制。

## 12. 上传内容、恢复与大文件迁移

GitHub 中保存可读代码/文档/关键结果，并用压缩分卷保存大量逐题记录、日志及非视频数据。详见 `handoff/EXPORT_SUMMARY.json`、`handoff/bundles/manifest.json` 和分卷内文件清单。

上传时排除：`.env`、私钥、认证信息、`.git`、Python 缓存、PID/锁文件、视频、模型/优化器等二进制训练状态。日志和文本遇到明确密钥内容会做替换，替换数量写入导出报告。原始工作目录不做脱敏覆盖。

权重迁移建议先选需要继续研究的检查点，不必搬全部历史模型。例如 clean_ce step5 和 step20、原模型及第二批需要的恢复点。完整断点续训要同时带学生权重、teacher 权重、优化器/ZeRO 分片、scheduler、trainer_state 和各 rank RNG；只有 safetensors 只能加载模型，不能保证精确续训。

```bash
# 在原机器执行；替换账号、地址和目标目录。
rsync -aH --info=progress2 \
  /path/to/source/checkpoint-5/ \
  user@new-host:/path/to/destination/checkpoint-5/
```

`-H` 保留传输范围内的硬链接；检查点之间若要保留共享链接，应放在同一次传输中。传输完成后验证大小、文件数和哈希。不能只复制 `MCQ_CHECKPOINT_COMPLETE.json` 后就认为检查点完整。

## 13. 标注 API 与安全配置

标注调用使用项目中称为 Qwen3.8-Omni-Flash 的 API 配置，实际 endpoint/model 以当时 manifest 和环境变量为准，不凭名称推断服务实现。`.env` 不上传。新机器按代码读取的变量自行配置，变量名模板见 `handoff/API_ENVIRONMENT.template`。

用户要求标注 API 不走网络代理。相关进程应清除大小写 HTTP_PROXY、HTTPS_PROXY、ALL_PROXY，并在 HTTP 客户端关闭继承代理；仅改 shell 环境不一定覆盖客户端或 Git 的显式代理。

本机旧 Git 显式代理 `proxy-notebook.modelarts.com:8083` 失效。临时直连示例：

```bash
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY \
    -u https_proxy -u http_proxy -u all_proxy \
    git -c http.proxy= -c https.proxy= fetch origin
```

不要把 API key、SSH 私钥或 GitHub token 写入仓库、日志、命令参数或聊天。仓库权限和网络连通是两件事；能读取公开仓库不代表能 push。

## 14. 交接完成的判断标准

- GitHub 可看到本文、核心代码、数据清单、结果表和分析。
- 所有分卷都有 SHA256 且能安全恢复。
- 未上传的媒体和模型状态有明确迁移清单。
- 新机器能读取同一批训练/评测题，运行一个小训练并加载其检查点完成评测。
- 旧机器是否继续运行、新机器接续哪些实验，记录明确。

最后一项硬件复现需要在新机器完成；上传仓库本身不能证明新机器环境已通过验证。

## 15. 本次交接包发布状态

文档编写和本地打包已执行；**远端上传需要以最终推送验证为准**。本次最初检查遇到两种认证错误：本机 Git 的 VSCode 凭据 socket 失效；GitHub 连接器实际写入返回 403，尽管读取仓库元数据时显示 push 权限。不要把“有本地提交”解释为“已经上传”。具体状态见 `handoff/UPLOAD_STATUS.json`。

恢复 Git 写权限后，在原机器对独立导出目录执行：

```bash
python handoff_tools/publish_snapshot.py \
  --repo /opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd-handoff-20261009 \
  --push
```

工具将脱敏文件分批提交、逐批普通推送，并比较远端 main 与本地 HEAD。它不会强制覆盖远端历史。若远端出现独立的新提交，会停止，要求先合并。恢复凭据时不要把 token 粘贴到聊天中。
