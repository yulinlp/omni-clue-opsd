# Clue-OPSD 训练诊断指标接入任务

请检查我们实际使用的 Clue-OPSD / ms-swift 训练代码，在不改变训练目标、采样策略、teacher EMA 和数据输入的前提下，增加以下只读诊断指标。先定位真实代码与版本，再做最小修改和测试；不要启动全量训练。

## 1. 需要接入的指标

在 teacher 与 student 的**同一条 student completion、相同预测位置**上计算，默认诊断 K=20，可配置且独立于训练损失 K：

- `diag/overlap_ratio`：双方各自 Top-K 的交集大小 / K，对有效 completion 位置平均。
- `diag/empty_overlap_rate`：交集为空的位置比例。
- `diag/student_overlap_mass`、`diag/teacher_overlap_mass`：双方完整词表分布在交集上的概率之和；空交集计 0。
- `diag/overlap_adv_paper`：双方在交集内分别重新归一化，逐位置计算 `-KL(p_student_intersection || p_teacher_intersection) / intersection_size`，再对非空交集位置平均。同时记录有效位置数；可额外记录不除交集大小的 `diag/overlap_kl`。
- `diag/student_entropy`、`diag/teacher_entropy`：完整词表分布的熵。
- `diag/entropy_gap_abs`：逐位置计算 `abs(H_teacher - H_student)` 再平均；不能先平均双方熵再取差。
- MCQ 答案诊断：在实际生成的答案槽、字母生成前的位置，统计四选项归一化分布的 `answer_jsd`、`answer_gt_probability_gap = q(GT)-p(GT)`，以及答案解析失败率。先核实字母 token 编码；多 token 情形不能随意用首 token 替代。无合法答案槽时跳过该项并报告覆盖率。GT 仅用于无梯度统计，不加入 student 输入或训练损失；若训练行无 GT，用 case_id 从独立标注映射读取。

以上指标尽量分别记录 `all / analysis / answer`。可选按有效回答长度划分前、中、后段；我们最多生成 512 token，不要直接照搬 1024-token 分段。

## 2. 可参考的开源实现

参考仓库：/share/home/ylhu/Rethinking-OPD-main。以下行号是当前快照的大致位置，修改前按函数/键名重新定位：

- `verl/verl/workers/fsdp_workers.py:1829`：`_compute_teacher_top_k_log_probs`，提取 teacher Top-K，与 student Top-K ID 比较，生成 `overlap_mask`，取得 teacher 在 student 候选上的完整分布 log-probability。
- 同文件 `:1806`：`_compute_entropy_safe`，分块计算全词表熵。
- `verl/verl/trainer/ppo/ray_trainer.py:1391`：Top-K 统计入口；`:1883` 为 overlap ratio，`:1898` 为双方交集概率质量；搜索 `val-topk/` 可找到更多指标和分段统计。虽然日志带 val 前缀，这段实际统计训练 batch。
- 同文件 `:1289`：`actor/entropy`、`teacher/entropy` 的归约；`:2337` 为日志输出。未找到直接的 entropy gap 计算，应在逐位置熵被丢弃前新增。
- `verl/verl/workers/actor/dp_actor.py:560` 和 `verl/verl/trainer/ppo/core_algos.py:855`：训练 reward/advantage 来源。**不要原样复制 `val-topk/adv_intersection` 当作论文公式（7）**：默认分支使用训练 advantage，归一化范围及跨位置平均方式与论文不同；请按上面的定义独立计算诊断量。
- `on_policy_distillation.sh:65` 附近：Top-K、strategy 配置；只作参考，不执行此脚本，也不照搬其训练参数。

我们之前记录的 ms-swift 入口为 `swift/rlhf_trainers/gkd_helpers.py`（双路编码）、`gkd_loss.py`（Top-K JSD）、`gkd_trainer.py`（训练与 EMA），以及 `reasoning_pilot_audit_plugin.py`（completion 对齐）。请定位实际运行版本，优先在 completion logits 对齐之后、Top-K 截断归一化之前接入统计。若真实代码不在当前环境，说明缺失路径，不修改无关副本。

## 3. 实现约束与验证

- Full/Clue prompt 和媒体 token 长度不同，按各自 completion 起点和 causal shift 对齐，断言 completion token IDs 一致。只统计有效回答，排除 prompt、媒体、padding。
- 必须取得双方各自 Top-K；仅有 student 在 teacher Top-K 上的 logits 不足以计算重合率。概率质量和熵使用完整词表归一化，不能用 Top-K softmax 冒充。统一诊断温度（默认 1），不使用 top-p/top-k 采样过滤后的分布。
- 全部统计 stop-gradient，分块计算，不额外保留完整 `[B,T,V]`；多卡按 sum/count 汇总，不平均各卡均值。支持开关和统计频率，日志附带 K、温度、有效计数。EMA 下另留固定题目/前缀诊断集，有助于区分输入变化与模型变化。
- `gap_recovery_rate` 未在参考仓库找到专用实现：在评测汇总阶段计算 `(当前 student 准确率 - 初始 student 准确率) / (固定 teacher 准确率 - 初始 student 准确率)`。三者须在同一有证据标注的评测集合、同一评分协议下评测，student 看 Full、teacher 看 Clue；不得混用训练集与外部 benchmark。分母非正或过小返回不可用并保留原始准确率；缺评测结果时先留独立汇总接口，不伪造数据。
- 用小张量测试相同分布、部分/空交集、padding、不同 prompt 长度；确认熵差是“先绝对值再平均”，交集质量在 [0,1]，论文 advantage 非正。固定 batch 对比开关前后 loss/梯度应一致；有运行条件时再做少量 smoke steps，不做全量训练。

完成后报告修改文件、日志键名、启用方式、测试结果与额外开销；明确尚未验证的部分。指标只用于分析，不能把重合率提高直接解释成准确率提高。
