# OmniVideo-100K：5k 数据筛选与实验小结

更新：2026-09-10。第6–8节记录最新版设置、实际prompt及训练后测试结果；第1–5节保留筛选与原子视角实验。当前选用 **准确率差、概率差优先（aggregate）** 的5k版本。

## 1. 实验设置与筛选方案

目标是构建具有明显证据条件增益的训练集，用于研究将 evidence-conditioned teacher 的优势转移给 full-video student。

| 项目     | 设置                                                         |
| -------- | ------------------------------------------------------------ |
| 候选数据 | OmniVideo-100K中的20,831道题，1,563个视频，视频时长60–180秒 |
| 评分模型 | 冻结的Qwen2.5-Omni-7B，同一模型分别处理Full与Gold            |
| Full输入 | 完整视频时间线及完整音频                                     |
| Gold输入 | 官方`designated_segments`指定的音视频片段                  |
| 采样     | 2 FPS；                                                      |
| 答案评分 | 首token的A/B/C/D选项概率归一化，取最大概率选项作为预测       |
| 配对条件 | 问题、选项、模型及评分规则一致，仅改变媒体输入范围           |

对每道题计算：

- **准确率贡献差**：`Δacc = 1[Gold答对] − 1[Full答对]`，取值为+1、0或−1。
- **正确答案概率差**：`Δp = P(GT | Gold) − P(GT | Full)`。

筛选步骤：

1. 检查源媒体时间戳、音视频时长、重复选项及重复/重叠证据区间，合计排除873条，剩余19,958条。
2. 先按`Δacc`降序，再按`Δp`降序排序；继续同分时依次比较对数概率差、Gold正确答案概率和样本ID。历史text-only评分仅作诊断，未参与排序。
3. 每个视频最多选5题，取满5,000个唯一样本。各任务不设置数量配额。

该方案优化集合层面的teacher–student gap，允许双方都答对或都答错的样本入选。Gold的特权来自证据位置标注。下文5k成绩均为筛选后训练集统计。

## 2. 筛选结果

| 数据集合                                 | 题数            | Full准确率       | Gold准确率       | 准确率差（pp）   | 平均Δp           |
| ---------------------------------------- | --------------- | ---------------- | ---------------- | ---------------- | ----------------- |
| 完整候选池                               | 20,831          | 80.51%           | 83.05%           | +2.54            | —                |
| **当前选定：准确率差、概率差优先** | **5,000** | **56.66%** | **80.88%** | **+24.22** | **+0.2284** |
| 对照方案：Gold正确率优先                 | 5,000           | 75.78%           | 100.00%          | +24.22           | +0.2061           |

当前版本覆盖1,471个视频，包含1,211条Full错→Gold对、3,789条双方对错状态相同的题；其中956条双方都答错。实际采用的QA改写数为0。选择当前版本，是为了保留更具挑战性的Full输入任务，同时获得更大的正确答案概率增益。

**任务分布及当前版本的Full/Gold结果：**

| 任务           | 数量（占比）            | Full             | Gold             | 准确率差（pp）   | 平均Δp           |
| -------------- | ----------------------- | ---------------- | ---------------- | ---------------- | ----------------- |
| 事件顺序排序   | 1,053（21.06%）         | 52.80%           | 77.87%           | +25.07           | +0.2205           |
| 比较           | 864（17.28%）           | 65.16%           | 83.68%           | +18.52           | +0.2125           |
| 假设推理       | 826（16.52%）           | 57.26%           | 79.30%           | +22.03           | +0.2011           |
| 情感分析       | 740（14.80%）           | 49.19%           | 79.19%           | +30.00           | +0.2472           |
| 因果推理       | 731（14.62%）           | 54.45%           | 80.44%           | +25.99           | +0.2636           |
| 未来预测       | 511（10.22%）           | 58.51%           | 82.39%           | +23.87           | +0.2313           |
| 总结           | 275（5.50%）            | 65.45%           | 91.27%           | +25.82           | +0.2404           |
| **整体** | **5,000（100%）** | **56.66%** | **80.88%** | **+24.22** | **+0.2284** |

七类任务均有正向增益。情感分析的准确率提升最大（+30.00 pp），因果推理的正确答案概率提升最大（+0.2636）。

## 3. 已完成的稳健性实验

每版5k固定抽取128个不同视频的题目，按任务轮转选取，种子为20260906。每题进行四种循环选项排列，答案随选项文本同步重映射；四种顺序等权统计。

| 候选方案             | 置换输入数 | Full   | Gold   | 准确率差（pp） | 平均Δp |
| -------------------- | ---------- | ------ | ------ | -------------- | ------- |
| **当前选定版** | 512        | 60.74% | 79.88% | +19.14         | +0.1935 |
| Gold正确率优先版     | 512        | 74.80% | 92.58% | +17.77         | +0.1700 |

另构造与官方证据匹配片段数和时长、且不与官方区间重叠的对照，使用原始选项顺序进行配对比较：

| 候选方案             | 可配对题数 | 官方Gold准确率 | 区间外对照准确率 | 官方−对照（pp） |
| -------------------- | ---------- | -------------- | ---------------- | ---------------- |
| **当前选定版** | 102        | 82.35%         | 42.16%           | +40.20           |
| Gold正确率优先版     | 103        | 100.00%        | 51.46%           | +48.54           |

选项换序后差距仍然保留；官方证据明显优于等时长区间外片段，支持证据位置对增益的作用。

## 4. 此前的相关实验

### OmniVideo原子视角实验

使用Qwen2.5-Omni-7B、2 FPS、768帧上限及相同像素设置。先跑448题均衡pilot（每任务64题），再跑2,084题、156视频的旧内部dev集合。采用贪心生成选项答案、最多4个新token。

| 视角 | 输入设置                       | 448题准确率 | 2,084题准确率 |
| ---- | ------------------------------ | ----------- | ------------- |
| A0   | Full视觉＋Full音频             | 79.91%      | 82.25%        |
| A1   | 证据视觉＋Full音频             | 80.36%      | 83.49%        |
| A2   | Full视觉＋证据音频             | 80.80%      | 82.92%        |
| A3   | 证据视觉＋证据音频             | 80.58%      | 83.88%        |
| A4   | 与证据匹配时长的均匀音视频片段 | 71.88%      | 73.94%        |
| A5   | Full音视频＋证据时间提示       | 80.13%      | 81.67%        |

2,084题上，A3−A0为+1.63 pp，A3−A4为+9.93 pp。前期实验显示官方证据优于匹配时长的均匀片段，但未经gap筛选时，相对Full的平均提升较小。

### 更早的VideoOdyssey探索

使用原生Qwen3-Omni-30B-A3B，在固定584题上比较视觉输入；音频/字幕实验使用其中具有字幕的485题。

| 实验                                  | 基线准确率 | 实验准确率 | 提升（pp） |
| ------------------------------------- | ---------- | ---------- | ---------- |
| Full视觉→精确证据视觉（584题）       | 28.60%     | 36.82%     | +8.22      |
| 证据视觉→证据视觉＋证据音频（485题） | 38.14%     | 49.48%     | +11.34     |
| 证据视觉→证据视觉＋证据字幕（485题） | 38.14%     | 55.05%     | +16.91     |

这些探索提示：证据定位和模态选择都可能带来收益，适合进一步按任务类型拆解。各表按其各自模型、数据和评分设置报告。

### QA改写小试

尝试使用Qwen3-Omni-30B-A3B辅助改写及证据审核，出现原题复述、输出结构不符及媒体描述不一致。当前5k全部来自原题筛选。

## 5. 5k原子视角实验结果

在固定的全部5,000题上运行A0–A5，并加入Full和Gold的分流输入对照C0/C3。评分沿用筛选时的首token概率口径；所有视角覆盖同一批题，没有另划测试集。

| 视角                    | 准确率 | 平均正确答案概率 |
| ----------------------- | ------ | ---------------- |
| A0 Full音视频           | 56.66% | 0.5084           |
| A1 证据视觉＋Full音频   | 62.50% | 0.5704           |
| A2 Full视觉＋证据音频   | 69.98% | 0.6286           |
| A3 证据音视频           | 80.88% | 0.7367           |
| A4 匹配时长均匀片段     | 55.52% | 0.5057           |
| A5 Full音视频＋时间提示 | 57.52% | 0.5162           |
| C0 Full分流输入         | 56.72% | 0.5137           |
| C3 Gold分流输入         | 77.30% | 0.7054           |

相对A0，A3提升 **+24.22 pp / +0.2284**（视频聚类bootstrap 95%区间分别为 `[+22.90,+25.54]` pp 和 `[+0.2216,+0.2356]`）；A1为 **+5.84 pp / +0.0620**，A2为 **+13.32 pp / +0.1202**。A3相对A4为 **+25.36 pp / +0.2311**，说明增益主要来自证据位置而非只减少输入时长；A5相对A0仅 **+0.86 pp / +0.0078**。A3和A0之间的分流对照为 C3−A3 **−3.58 pp / −0.0313**、C0−A0 **+0.06 pp / +0.0053**。

七类任务的A3−A0准确率增益均为正：事件顺序排序 +25.07 pp、比较 +18.52 pp、假设推理 +22.03 pp、情感分析 +30.00 pp、因果推理 +25.99 pp、未来预测 +23.87 pp、总结 +25.82 pp。按正确答案概率，因果推理增益最大（+0.2636）。

原子实验的最终文件位于远端 `atomic_gap5000_v1/report/`：`summary.json`、`accuracy_probability_by_task.csv` 和 `contrasts_by_task.csv`，并已生成 `ATOMIC_ANALYSIS_SUCCESS`；汇总修订和报告哈希记录在 `finalizer.manual.json`。其中1个视频的19条分流音频记录出现标注时长与解码端点偏差（155秒标注、149.376秒解码）；记录在汇总诊断中，未修改固定输入、评分或样本。

去掉该视频对应的5条题后，A3−A0 为 +24.20 pp / +0.2282，和全量 +24.22 pp / +0.2284 基本一致。

### E系列特权视角与 case-level teacher 选择

这部分在同一批5,000题上重新按最初定义的 E 系列运行，与上面的 A0–A5 拆分实验分开统计。模型、2 FPS、768帧上限和首 token 选项概率口径保持不变；除 E5 组合诊断外，各视角只改变一种特权输入条件。

#### 最终采用的原子视角定义

最终把原子视角分成“论文主实验因子”和“case-level teacher 候选”两层。论文主实验验证 G/H/T/S 四种证据变换，再验证显式证据音频 A；在逐 case teacher 选择中加入 V 作为关闭音频的模态对照，因此报告中的动态候选集合是 **G/H/T/S/V/A 六个视角**。每个视角都使用同一问题、选项、Qwen2.5-Omni-7B 和评分口径；只有下面列出的特权输入条件改变。

| 视角                             | 具体输入构造                                                                                                                                                                             | 改变的因素         | 在方案中的角色                        |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | ------------------------------------- |
| **G — E2_G_exact**        | 只输入数据集`designated_segments` 的 Gold 视觉区间；区间按原始时间戳裁剪、排序并合并。视频 2 FPS、最多768帧、`min_pixels=3,136`、`max_pixels=28,672`；同步音频跟随同一 Gold 区间。 | 证据定位           | 核心 Gold teacher anchor              |
| **H — E4_H_halo3**        | 将每个 Gold 区间前后各扩展3秒，裁剪到视频边界并合并重叠区间；视频和同步音频都覆盖扩展后的区间，其他采样设置同 G。                                                                        | 时间覆盖范围       | 核心时间邻域视角                      |
| **T — E7_T_8fps**         | 保持 Gold 区间和空间预算不变，把视觉采样从2 FPS提高到8 FPS，仍受最多768帧限制；音频仍跟随 Gold 区间。                                                                                    | 时间采样密度       | 核心高时间密度视角                    |
| **S — E8_S_highres**      | 保持 Gold 区间、2 FPS和音频不变，只把视觉`max_pixels` 提高到409,600。                                                                                                                  | 空间分辨率         | 核心高分辨率视角                      |
| **A — E13_A_audio_exact** | 视频仍是 Gold 视觉区间（2 FPS、基准空间预算），另外以独立音频 descriptor 输入 Gold 音频区间；不把整段视频音频重复附加。                                                                  | 音频模态及时间定位 | 核心显式音频视角                      |
| **V — E12_gold_v**        | 只输入 Gold 视觉区间，关闭视频内原生音频，也不提供独立音频。视觉采样设置同 G。                                                                                                           | 音频移除           | 模态对照，同时保留为动态 teacher 候选 |

这里的“Gold”只表示证据位置特权，不改变题目文本或正确选项。H/T/S/A/V 均从同一 Gold 区间出发，因此可以分别归因于时间邻域、时间密度、空间分辨率和音频输入。E0 Full、E1/E3 timestamp 和 E5 coarse+dense 是基线、时间提示控制或组合诊断，不放进最终六视角 teacher 候选；E5 同时改变全局粗览和 Gold 密集流，不能称为单一原子因子。

字幕和 query-bbox 保留为可选扩展：若样本有字幕，字幕视角在 Gold 视觉区间旁加入与证据时间相交的时间戳字幕；本 5k 清单字幕文件和字幕证据均为0，所以不人为补字幕。query-bbox 视角需要 answer-blind 的 YOLO 类检测器按 query 生成目标框，并把红框实际渲染到视频帧（再输入对应 Gold 时间段）；检测器和物化红框视频尚未提供，因此该视角暂不进入结果或动态选择。

| 视角                   | 输入变化                                         | 准确率           | 平均正确答案概率 | 相对 E2 准确率（pp） | 相对 E2 概率      |
| ---------------------- | ------------------------------------------------ | ---------------- | ---------------- | -------------------- | ----------------- |
| E0 Full AV             | 完整视频＋原生同步音频                           | 56.66%           | 0.5084           | −22.14              | −0.2130          |
| E1 Full + timestamp    | Full＋Gold区间文字时间提示                       | 57.26%           | 0.5133           | −21.54              | −0.2080          |
| **E2 G exact**   | **只保留 Gold 视觉区间＋原生音频**         | **78.80%** | **0.7213** | **—**         | **—**      |
| E3 G + timestamp       | G＋原视频绝对时间提示                            | 77.98%           | 0.7069           | −0.82               | −0.0144          |
| E4 H halo ±3 s        | Gold区间前后各扩3秒                              | 72.96%           | 0.6613           | −5.84               | −0.0600          |
| E5 coarse + dense      | Gold 8 FPS密集流＋全局0.25 FPS粗览               | 72.84%           | 0.6679           | −5.96               | −0.0534          |
| E7 T 8 FPS             | Gold视觉提高到8 FPS                              | 74.68%           | 0.6847           | −4.12               | −0.0366          |
| **E8 S highres** | **Gold视觉，空间预算提高到409,600 pixels** | **80.36%** | **0.7424** | **+1.56**      | **+0.0211** |
| E12 V visual-only      | Gold视觉，关闭原生视频音频                       | 57.80%           | 0.5282           | −21.00              | −0.1932          |
| E13 A audio-exact      | Gold视觉＋Gold区间显式音频                       | 75.50%           | 0.6959           | −3.30               | −0.0254          |

E8 是总体唯一超过精确 Gold 视角 E2 的单一视角，但它的收益取决于任务：事件顺序排序为 **+7.41 pp**，未来预测为 **−3.91 pp**。两类任务的差值相差 **11.33 pp**（视频聚类 bootstrap 95% CI `[6.94, 15.76]`），提示高分辨率的平均收益具有任务差异。字幕视角跳过是因为5k清单没有可用字幕证据；query-bbox 红框视角待外部检测器和物化视频提供后再运行。

### 为什么做多视角 OPSD：最强固定 teacher 仍有可补充的错误

本轮比较同一冻结模型在不同特权输入下的表现。Highres 是平均准确率最高的固定视角，但其他视角仍能在部分 case 上提供更好的监督。我们关心的是这些视角能否补充最强 teacher，而不是各视角的胜出比例是否均匀。以下统计重新读取六个视角的全部5,000题评分，按 sample ID 配对，覆盖1,471个视频。

**选择规则必须明确：这里的 oracle 使用真实答案，不是选择模型自身最高置信度。** 对题目 i、真实答案 yᵢ 和候选视角集合 V，选择：

`v*ᵢ = argmax(v∈V) P(yᵢ | questionᵢ, privileged_viewᵢᵛ)`。

然后取所选视角概率最高的 A/B/C/D 选项作为预测，计算其是否答对。六视角顺序为 G/H/T/S/V/A，同分取先出现的视角。模型最高置信度对照则选择 `argmax_v max_a P(a | questionᵢ, viewᵢᵛ)`，完全不使用真实答案。两者分别衡量标签指导的 teacher 选择与不看标签的置信度选择。

#### 1. 逐 case 选择比固定最强视角多解决多少题？

| Teacher 输入策略                       | 是否用真实答案选择视角 | 正确题数 / 5,000 | 准确率           | 相对固定 Highres    |
| -------------------------------------- | ---------------------- | ---------------- | ---------------- | ------------------- |
| 固定 G                                 | 否                     | 3,940            | 78.80%           | −1.56 pp           |
| 固定 Highres                           | 否                     | 4,018            | 80.36%           | —                  |
| G/S 两视角，按正确答案概率选择         | 是                     | 4,318            | 86.36%           | +6.00 pp            |
| G/H/T/S/V/A 六视角，按正确答案概率选择 | 是                     | 4,552            | **91.04%** | **+10.68 pp** |
| 六视角，按模型最高置信度选择           | 否                     | 3,820            | 76.40%           | −3.96 pp           |

G/S 二选一相对 Highres 增加300道正确答案；扩展到六视角后又净增加234道（**+4.68 pp，视频聚类 bootstrap 95% CI [4.09, 5.26]**）。因此互补性不只来自 G 和 Highres。六视角相对 Highres 的净增益为 **+10.68 pp，CI [9.80, 11.58]**，平均正确答案概率从0.7424提高到0.8388（+0.0964）。区间使用2,000次视频聚类重采样、种子20260908。

精确到错误转换，六视角选择救回了535道 Highres 错题，同时使1道原本正确的题变错，净增加534道。因为选的是最大正确答案概率，91.04% 不等于“只要任何视角答对就算对”；后者的正确覆盖率为91.10%。

多视角 teacher 的互补性：策略准确率、错题补救和概率增益

**读图：** A比较固定 teacher、标签指导的逐 case 选择和最高置信度选择；B统计其他视角能救回多少 Highres 错题，橙色条进一步限定为 G 和 Highres 都答错的题；C的横轴为相对 Highres 的正确答案概率增益门槛，纵轴是超过门槛的题数，每条曲线均统计全部5,000题。

#### 2. 平均分较低的视角有没有独立价值？

Highres 共答错982题，其中682题 G 也答错。下表直接统计其他视角对这些错误的补充，以及超过微小概率波动的收益。

| 视角           | 救回 Highres 错题 | G/S 都错时仍答对 | 六视角中仅本视角答对 | 正确答案概率比 S 至少高0.10的题数 |
| -------------- | ----------------- | ---------------- | -------------------- | --------------------------------- |
| G 精确证据     | 300               | 0                | 36                   | 826                               |
| H 前后扩3秒    | 238               | 66               | 29                   | 609                               |
| T 8 FPS        | 226               | 72               | 26                   | 634                               |
| V 仅视觉       | 250               | 126              | 80                   | 586                               |
| A 显式证据音频 | 279               | 68               | 31                   | 745                               |

各行补救样本有重叠，不能直接相加。H/T/V/A 都存在其余五个视角答错、只有本视角答对的题；这为保留这些候选提供了比“胜出占比”更直接的证据。V 的整体准确率只有57.80%，但有80题是六视角中独有的正确答案，说明平均排名不足以决定逐 case 的 teacher 价值。

#### 3. 从互补性到多视角 OPSD

训练数据提供真实答案，因此可以用正确答案概率离线选择每条训练样本的 teacher 视角，并由所选视角提供蒸馏监督。Student 接收完整音视频，学习利用这些更好的监督；推理时只运行 student。这个方案本身不要求额外学习一个不看答案的 selector。若未来希望训练时减少多视角评分开销，再单独验证低成本 selector。

论文动机可以表述为：**不同特权输入对同一模型产生样本相关的监督质量差异。尽管高分辨率证据视角具有最高整体准确率，其他视角仍能纠正其大量错误，包括 G 与高分辨率视角同时失败的样本。因此，我们在训练时利用标签对每条样本选择特权 teacher 视角，将互补监督通过 OPSD 转移给完整输入的 student。**

当前证据支持“teacher 监督存在互补空间”；91.04% 是训练集合上的标签指导选择成绩。将这一空间转化为 student 提升，需要下一步比较固定 G-OPSD、固定 S-OPSD、G/S 动态 OPSD 与六视角动态 OPSD，使用相同 student 初始化、训练样本、优化步数和蒸馏损失，在按视频隔离的独立评测集上比较。另加入六视角随机选择对照，并报告离线评分成本，以区分有依据的选择、输入多样性和计算量的作用。

逐任务分别选取本集合上平均准确率最高的固定视角，准确率为 **80.88%**，仍低于逐 case 六视角选择。这进一步说明任务标签无法完全概括样本内的视角差异。

移除单一视角后重新按正确答案概率选择，结果如下（完整六视角为91.04%）：

| 移除视角          | 剩余视角选择准确率 | 相对完整六视角下降 |
| ----------------- | ------------------ | ------------------ |
| E2_G_exact        | 90.30%             | 0.74 pp            |
| E4_H_halo3        | 90.48%             | 0.56 pp            |
| E7_T_8fps         | 90.52%             | 0.52 pp            |
| E8_S_highres      | 89.00%             | 2.04 pp            |
| E12_gold_v        | 89.42%             | 1.62 pp            |
| E13_A_audio_exact | 90.42%             | 0.62 pp            |

复现文件：[统计 JSON](../artifacts/omnivideo_view_complementarity.json)、[压缩逐题评分（无媒体）](../artifacts/omnivideo_e_series_paired_scores.json.gz)、[分析绘图脚本](../scripts/analyze_omnivideo_view_complementarity.py)、[矢量 PDF](../artifacts/omnivideo_view_complementarity.pdf)。

复现命令（本地CPU运行，无需模型或原视频）：

```bash
python scripts/analyze_omnivideo_view_complementarity.py \
  --scores artifacts/omnivideo_e_series_paired_scores.json.gz \
  --output artifacts/omnivideo_view_complementarity
```

## 6. 最新版实验设置：动态视频预算（2026-09-10）

本轮目标是先补齐相同高分辨率输入预算下的 SFT、标准 OPSD 和固定证据 Clue-OPSD，对照监督方式带来的收益，再推进逐 case 多特权视角选择。**动态视频预算按时长、音频长度和宽高比分配输入；它与使用正确答案选择特权 teacher 视角是两个不同环节。** 当前三组尚未接入六视角动态 teacher。

### 6.1 数据、模型和优化设置

| 项目         | 最新设置                                                                                              |
| ------------ | ----------------------------------------------------------------------------------------------------- |
| 训练集       | 冻结的 gap5000：5,000条、1,471个唯一视频，三组使用相同样本顺序                                        |
| 视频时长     | 60–180秒；按样本平均103.57秒，按唯一视频平均103.87秒，中位数100秒                                    |
| 模型         | Qwen2.5-Omni-7B，各组从原始基座重新初始化                                                             |
| 参数更新     | LoRA r64 / alpha128，target_modules=all-linear                                                        |
| 训练规模     | 每组4卡；每卡batch2、梯度累积4，全局batch32；157步，约1 epoch                                         |
| 学习率       | SFT：1e-5；标准OPSD、Clue-OPSD：2e-6                                                                  |
| 优化         | cosine schedule，warmup_ratio=0.03，BF16，gradient checkpointing，SDPA；max_grad_norm=0               |
| 输出         | 先分析音视频证据并比较选项，再输出`<answer>字母</answer>`                                           |
| OPSD rollout | student on-policy，vLLM colocate / TP2，同步调度；temperature=1、top_k=20、top_p=0.95，最多512 token  |
| 蒸馏         | teacher top-20词表蒸馏；JSD beta=0.5、temperature=1、lmbda=1、sft_alpha=0；LoRA-shadow EMA alpha=0.05 |
| 正式验证     | 按视频与训练集隔离的OmniVideo-Test 128题，Base和三个adapter使用相同动态预算，分别评choice和reasoning  |

SFT采用数据集已有`metadata.connections`证据解释，加最终正确选项作为assistant监督；解释来自数据集生成标注。标准OPSD teacher看Full音视频并获得标答；Clue-OPSD teacher看数据集发布的证据片段，不接收标答。三组student的Full音视频配置逐条一致。

### 6.2 动态多媒体预算

固定低分辨率版本给每帧`max_pixels=28,672`、2 FPS、最多768帧。按Qwen2.5-Omni的两帧时间合并口径，每个采样帧平均只分到约16–18个视觉token。新版将空间预算提高，同时控制完整输入长度。

1. 总上下文32,768 token；预留2,048给问题、回答及模板，音频按25 token/秒估算，并为每个视频区间额外预留64 token。
2. 可用视觉预算为`min(24,000, 32,768 − 文本预留 − 音频预算)`。
3. 目标2 FPS、总帧数最多300，每个采样帧平均100–128视觉token。若预算不足，减少均匀采样帧数，仍覆盖完整时间范围；音频不因减少视频帧而删减。
4. 显式保存`nframes`及按28对齐的`resized_height/resized_width`，结合实际宽高比选择网格。多个Clue证据区间共同分配一份预算，不对每个区间重复给予24k。

这里“每帧token”始终按**两帧合并后平均到每个采样帧**计算：

```text
视觉token = ceil(采样帧数 / 2) × (缩放后高度 / 28) × (缩放后宽度 / 28)
```

新版5,000条已分配的数据预算如下；这是根据媒体尺寸算出的网格预算，尚不是完整训练的显存实测结果。

| 指标                              | 数值           |
| --------------------------------- | -------------- |
| 每条student视频视觉token          | 15,120–24,000 |
| 平均视觉token                     | 21,454.94      |
| 实际最多采样帧数                  | 240            |
| 平均每个采样帧token               | 100–128       |
| SFT问题＋解释＋答案的最长原始文本 | 1,218 token    |

训练编码后检查实际长度，超过32,256即报错，避免静默截断并保留512 token余量。新配置先在最重32条样本上验证完整训练步、实际编码长度、峰值显存和非零参数更新，再从基座开始正式全量训练。100 token/帧是本次选定的实验下限，不是已证实的通用性能门槛。

### 6.3 当前执行状态

当前旧SFT、GRPO已按用户要求停止，GRPO不在新版三组队列中。现有低分辨率外部评测继续；动态预算的三个队列均等待这批20组评测完成，随后进行显存验证，再启动全量训练。**截至本次更新，新预算尚无训练后成绩。** 训练后自动评测同预算128题，另有同预算Base对照。

具体参数、运行目录及接续说明见[动态预算训练记录](DYNAMIC_VIDEO_BUDGET_TRAINING.md)。

## 7. Rollout、teacher与评测prompt

以下从实际训练JSONL提取，变量用花括号表示；底层使用Qwen2.5-Omni聊天模板。任务仍保留MCQ选项，变化是允许模型先生成分析，不是移除选项的open-ended训练。

### 7.1 Student reasoning rollout / reasoning评测

输入为完整音视频，user文本为：

```text
<video>
Question: {question}
Options:
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}
Briefly analyze the video and audio evidence relevant to the question and compare the answer options. Write the analysis first, then give exactly one option letter inside <answer>...</answer>. Keep the analysis concise (at most 120 words).
```

预期输出：

```text
{analysis of the video/audio evidence and comparison of options}
<answer>{option_letter}</answer>
```

不强制输出`<think>`标签，也未启用“teacher开thinking、student关thinking”的独立配置。训练rollout使用上节的随机采样参数；reasoning评测使用greedy、最多512 token。

### 7.2 标准OPSD teacher

Teacher使用与student完全相同的Full音视频。在上述reasoning prompt末尾增加：

```text

Privileged information: the correct option is {gold_letter}. Use the video and audio evidence to explain this answer briefly, then give the option inside <answer>...</answer>.
```

这段标答提示只进入teacher输入。Student先生成分析和答案，teacher在相同completion前缀上重算分布，提供top-20蒸馏监督；不是先让teacher独立生成一段解释，再把解释当作SFT标签。

### 7.3 Clue-OPSD teacher

Teacher文本的问题、选项和reasoning指令与student相同，开头的`<video>`数量与证据片段数量一致：

```text
<video>
... {one video placeholder per evidence interval}
<video>
Question: {question}
Options:
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}
Briefly analyze the video and audio evidence relevant to the question and compare the answer options. Write the analysis first, then give exactly one option letter inside <answer>...</answer>. Keep the analysis concise (at most 120 words).
```

对应媒体是数据集指定的证据音视频区间，保留区间顺序；teacher prompt没有gold letter，也没有额外输入标注解释。本轮是固定证据teacher，尚未按G/H/T/S/V/A逐case选teacher。

### 7.4 SFT与choice格式

新版reasoning SFT使用7.1的user prompt，assistant target为：

```text
{metadata.connections from the released training annotation}
<answer>{gold_letter}</answer>
```

保留原解释全文，不按prompt中的120词要求硬裁剪。SFT是teacher forcing，没有on-policy rollout。

旧choice-only训练及choice评测使用相同问题和选项，将最后的reasoning指令替换为：

```text
Answer with exactly one option letter.
```

旧SFT的assistant target仅为标答字母，例如`B`。Choice评测为greedy、最多8 token；解析失败计错。**让旧SFT在评测时输出reasoning，不代表它训练时监督过reasoning。**

### 7.5 实际JSONL数据组织与字段流向

每个训练文件一行对应一道题。同一视频可以对应多道题，`case_id`标识题目，`video_id`标识视频。OPSD两组的`messages`均只有user消息，没有预先写入assistant答案；生成的assistant消息在训练rollout时追加。

| 字段                                        | 标准OPSD                                | Clue-OPSD                    | 用途                                               |
| ------------------------------------------- | --------------------------------------- | ---------------------------- | -------------------------------------------------- |
| `messages`                                | 问题＋选项＋reasoning指令               | 相同                         | Student文本输入                                    |
| `videos`                                  | Full视频及动态采样参数                  | 逐题与标准OPSD一致           | Student媒体输入                                    |
| `teacher_prompt`                          | 同一问题与指令，末尾追加正确选项        | 同一问题与指令，无正确选项   | 构建teacher的user消息                              |
| `teacher_videos`                          | 与`videos`完全相同                    | 同一源视频的多个证据时间区间 | 替换teacher的媒体输入                              |
| `clue_intervals`                          | 保留的来源元数据，不用于裁剪本臂teacher | 对应证据区间                 | 记录数据来源；实际读取范围由`teacher_videos`指定 |
| `experiment_arm`                          | `opsd`                                | `clue_opsd`                | 指定监督方式                                       |
| `response_format`                         | `reasoning`                           | `reasoning`                | 输出格式标记                                       |
| `supervision_contract`                    | 标记标答只在teacher prompt中            | 标记没有标答监督             | 输入检查及记录                                     |
| `sampling_contract`、`dynamic_*_budget` | 记录预算                                | 分别记录Full与证据预算       | 配置和审计，不拼接到问题文本                       |

这两组文件没有单独的`answer`、`solution`或gold assistant target字段。**标准OPSD的标答已经写入**`teacher_prompt`**字符串；Clue-OPSD行内不提供标答。** `sampling_contract.answer_label_in_model_row=false`是沿用的student侧标记，不能据此判断标准OPSD teacher是否含标答；应查看`teacher_prompt`和`supervision_contract.gold_answer_in_teacher_prompt`。

#### 同一道真实训练题的结构示例

下面使用实际样本`tZ-QcbrBjaw_sentiment_analysis_0`，源视频129秒，正确选项B。为便于阅读，将媒体目录缩写为`{VIDEO_DIR}`，长prompt用7.1–7.3已逐字列出的模板表示，其余采样数值来自当前训练JSONL。这里是结构化展示，不是另一套简写prompt。

标准OPSD：

```json
{
  "case_id": "tZ-QcbrBjaw_sentiment_analysis_0",
  "prompt_id": "tZ-QcbrBjaw_sentiment_analysis_0",
  "video_id": "tZ-QcbrBjaw",
  "messages": [
    {"role": "user", "content": "{7.1的完整student prompt}"}
  ],
  "videos": [
    {
      "video": "{VIDEO_DIR}/tZ-QcbrBjaw.mp4",
      "video_start": 0.0,
      "video_end": 129.0,
      "nframes": 240,
      "resized_height": 280,
      "resized_width": 560,
      "min_pixels": 3136,
      "max_pixels": 156800
    }
  ],
  "teacher_prompt": "{同一student prompt}\n\nPrivileged information: the correct option is B. Use the video and audio evidence to explain this answer briefly, then give the option inside <answer>...</answer>.",
  "teacher_videos": [
    {
      "video": "{VIDEO_DIR}/tZ-QcbrBjaw.mp4",
      "video_start": 0.0,
      "video_end": 129.0,
      "nframes": 240,
      "resized_height": 280,
      "resized_width": 560,
      "min_pixels": 3136,
      "max_pixels": 156800
    }
  ],
  "experiment_arm": "opsd",
  "response_format": "reasoning",
  "supervision_contract": {
    "kind": "answer-privileged-on-policy-self-distillation",
    "gold_answer_in_student_target": false,
    "gold_answer_in_reward": false,
    "gold_answer_in_teacher_prompt": true,
    "teacher_view": "full-video-uniform"
  }
}
```

Clue-OPSD使用相同`case_id`、相同student `messages`和上述Full `videos`，teacher部分如下；预算记录等公共字段不重复展开：

```json
{
  "case_id": "tZ-QcbrBjaw_sentiment_analysis_0",
  "teacher_prompt": "<video>\n<video>\n<video>\nQuestion: {同一问题}\nOptions:\n{同一选项}\n{7.1末尾的完整reasoning指令}",
  "teacher_videos": [
    {
      "video": "{VIDEO_DIR}/tZ-QcbrBjaw.mp4",
      "video_start": 11.0,
      "video_end": 24.0,
      "nframes": 26,
      "resized_height": 336,
      "resized_width": 588,
      "min_pixels": 3136,
      "max_pixels": 197568
    },
    {
      "video": "{VIDEO_DIR}/tZ-QcbrBjaw.mp4",
      "video_start": 62.0,
      "video_end": 76.0,
      "nframes": 28,
      "resized_height": 336,
      "resized_width": 588,
      "min_pixels": 3136,
      "max_pixels": 197568
    },
    {
      "video": "{VIDEO_DIR}/tZ-QcbrBjaw.mp4",
      "video_start": 86.0,
      "video_end": 96.0,
      "nframes": 20,
      "resized_height": 336,
      "resized_width": 588,
      "min_pixels": 3136,
      "max_pixels": 197568
    }
  ],
  "clue_intervals": [[11.0, 24.0], [62.0, 76.0], [86.0, 96.0]],
  "experiment_arm": "clue_opsd",
  "response_format": "reasoning",
  "supervision_contract": {
    "kind": "clue-privileged-on-policy-self-distillation",
    "gold_answer_in_student_target": false,
    "gold_answer_in_reward": false,
    "gold_answer_in_teacher_prompt": false,
    "teacher_view": "dataset-clue-interval"
  }
}
```

证据片段通过同一源文件路径和起止秒读取，不要求事先导出三个新MP4。`USE_AUDIO_IN_VIDEO=1`，视频及其区间内音频共同进入模型，因此这些行没有单独的`audios`/`teacher_audios`字段。时间范围通过媒体配置生效，不额外把证据标注解释写到Clue teacher文本里。

这个样本的Full输入是240帧、280×560，视觉24,000 token；Clue teacher是三段共74帧、336×588，视觉9,324 token。两者使用同一动态分配规则，但短证据输入可获得更高的每帧空间预算：本例student平均100 token/帧、Clue teacher平均126 token/帧。因而新版Clue条件同时体现证据聚焦以及该规则下的分辨率分配。

实际JSONL入口（相对`dynamic_budget_v1/`）：

| 方法      | 正式训练文件                              |
| --------- | ----------------------------------------- |
| SFT       | `sft/formal/data/sft.jsonl`             |
| 标准OPSD  | `opsd/formal/data/reasoning.jsonl`      |
| Clue-OPSD | `clue_opsd/formal/data/reasoning.jsonl` |

每个文件5,000行，`gate/data/`下的32条仅用于新预算验证，不替代正式训练集。

### 7.6 一次OPSD训练更新究竟做什么

1. 从JSONL取出`messages + videos`，student用vLLM生成一段分析和最终选项，记录原始completion token IDs。
2. 构造student序列：Full输入＋刚生成的completion；构造teacher序列：`teacher_prompt + teacher_videos`＋**同一份completion token IDs**。不重新生成teacher答案，也不把student的末尾字母替换成标答。
3. Student与teacher分别前向。输入prompt长度、视频token数可以不同，但用于蒸馏的completion token必须逐个一致；损失只覆盖有效completion位置，包括分析文本和最终答案，而非只蒸最后一个字母。
4. 每个位置取teacher分数最高的20个词表token，收集student在同一组token上的logits，双方在该20维集合内重新归一化，计算JSD（beta=0.5），再对有效位置归约。这里的top-20描述蒸馏损失范围；它与rollout采样的`top_k=20`是两个独立设置。
5. 梯度只更新student的LoRA。Teacher与student共享冻结基座，teacher使用单独维护的LoRA EMA权重，初始化与student一致；每次optimizer step后更新：`teacher_lora = 0.95 × teacher_lora + 0.05 × student_lora`。两个OPSD臂都使用此规则，区别在teacher输入。

例如student生成“某段分析＋`<answer>C</answer>`”，而标准OPSD的标答是B：teacher仍在这段C轨迹的各前缀上给出分布监督。C不是训练真值，B也不会被直接写成student的SFT target。相比之下，SFT直接对数据中存好的解释＋B计算交叉熵，不执行上述on-policy生成与teacher重放。

对应实现入口：本项目的`prepare_dynamic_budget_training.py`生成字段与媒体预算；`run_dynamic_budget_training.py`设置训练参数；远端ms-swift的`swift/rlhf_trainers/gkd_helpers.py`构造两路编码，`gkd_loss.py`完成top-k JSD，`gkd_trainer.py`维护EMA teacher。`reasoning_pilot_audit_plugin.py`检查两路completion token一致性。

## 8. 此前版本的训练与测试结果

### 8.1 V1：全词表蒸馏 / Transformers rollout

运行版本为`reasoning_choice_full_5k_v1`。Choice-Clue-OPSD、Reasoning-Clue-OPSD和choice-only SFT均已完成全量5,000条、157步；LoRA r16/alpha32、全局batch32。两个OPSD臂使用同一个无标答证据teacher，区别是student生成选项还是生成分析＋选项；SFT只监督标答字母。媒体预算为2 FPS、最多768帧、每帧max_pixels28,672。该版本蒸馏为全词表，rollout后端为Transformers。

同一视频隔离OmniVideo-Test 128题的结果：

| 模型 / 训练输出        | Choice评测     | Reasoning评测  |
| ---------------------- | -------------- | -------------- |
| Base                   | 56/128，43.75% | 59/128，46.09% |
| Clue-OPSD，choice-only | 57/128，44.53% | 58/128，45.31% |
| Clue-OPSD，reasoning   | 57/128，44.53% | 59/128，46.09% |
| SFT，choice-only       | 57/128，44.53% | 52/128，40.63% |

八组均完整评测128题，无解析失败。Reasoning训练的Clue-OPSD在reasoning评测上与Base持平；旧choice-only SFT在reasoning评测上低于Base。

### 8.2 V2：top-20蒸馏 / vLLM reasoning rollout

运行版本为`reasoning_rank_top20_v1`。四组均已完成5,000条、157步，完整checkpoint-157已保存。保留V1的低分辨率输入预算，改用vLLM rollout和teacher top-20蒸馏，比较teacher特权类型与LoRA容量。四组训练均生成分析＋答案。

| 模型      | LoRA r / alpha | Choice评测     | Reasoning评测            |
| --------- | -------------- | -------------- | ------------------------ |
| Base      | —             | 56/128，43.75% | 59/128，46.09%           |
| Clue-OPSD | 16 / 32        | 58/128，45.31% | 53/128，41.41%           |
| Clue-OPSD | 64 / 128       | 57/128，44.53% | 51/128，39.84%           |
| 标准OPSD  | 16 / 32        | 56/128，43.75% | 54/128，42.19%           |
| 标准OPSD  | 64 / 128       | 58/128，45.31% | **62/128，48.44%** |

标准OPSD r64的reasoning成绩比同格式Base多3题，提升2.34个百分点。Clue r64与标准OPSD r64的reasoning各有1题无法解析，均计错。增大rank改善了本轮标准OPSD的结果，但没有改善Clue-OPSD。V1和V2同时改变rollout后端与蒸馏范围，不作为只改变rank的对照。

### 8.3 V2外部benchmark测试

使用V2的四个checkpoint-157，计划加Base对照。Daily-Omni为完整1,197题，OmniVideoBench为完整1,000题；每组均测choice和reasoning，使用Full音视频、greedy decoding及同一旧版低分辨率预算。

截至本次核验，完成的整体结果为：

| 模型          | Daily-Omni choice | Daily-Omni reasoning | OmniVideoBench |
| ------------- | ----------------- | -------------------- | -------------- |
| Base          | 待完成            | 待完成               | 待完成         |
| Clue-OPSD r16 | 734/1,197，61.32% | 待完成               | 待完成         |
| Clue-OPSD r64 | 734/1,197，61.32% | 待完成               | 待完成         |
| 标准OPSD r16  | 737/1,197，61.57% | 待完成               | 待完成         |
| 标准OPSD r64  | 737/1,197，61.57% | 待完成               | 待完成         |

已完成的四组全部可解析。Daily-Omni发布集合中的同输入、同标答重复题保留原始题目数逐条计分。Reasoning评测曾因跨卡汇总超时中断，现使用各卡独立分片推理、CPU合并继续；没有完整汇总的任务仍标为待完成。Base尚未完成，当前不能从这张表得出外部benchmark相对Base的训练增益。

### 8.4 结果口径与当前结论

- 5k原子视角实验中的**91.04%**是使用训练标答、逐case选择teacher视角得到的成绩，证明候选监督的互补空间；它不是训练后student的测试准确率。
- 旧低分辨率训练目前最好的128题reasoning结果是标准OPSD r64的48.44%，相对Base多3题；Clue-OPSD尚未显示一致的student收益。
- 当前空间预算偏低，是接下来需要验证的因素，尚未证明它就是旧版增益不足的原因。新版统一提高输入预算，并加入真正带解释监督的SFT对照。
- 新版动态预算三组处于等待状态；被停止的低分辨率reasoning SFT、GRPO不作为已完成baseline，也没有加入上述结果表。此前448条pilot不替代全量5k结果。

本次更新重新校验了V1八组、V2八组以及Daily-Omni四组已完成评测的结果、标签和输入文件SHA-256。后续成绩继续更新本节，保持不同输入预算分表展示。

## 9. 产物索引

- 选定清单：`gap5000/omnivideo_100k_train.gap5000.canonical.jsonl`。
- 清单SHA256：`21f5d82d810e39f98f1db24bb2f365bcb640d6f587884d11e98607469c3a232e`。
- 筛选实现：[select_omnivideo_gap_5000.py](../scripts/select_omnivideo_gap_5000.py)。
- 原子实验设置：[ATOMIC_GAP5000_EXPERIMENT.md](ATOMIC_GAP5000_EXPERIMENT.md)。
- E系列特权视角实验：[E_SERIES_GAP5000_EXPERIMENT.md](E_SERIES_GAP5000_EXPERIMENT.md)。
- E系列 case-level 图：[omnivideo_view_complementarity.png](../artifacts/omnivideo_view_complementarity.png)（同时提供 [SVG](../artifacts/omnivideo_view_complementarity.svg)）。
- 历史实验记录：[EXPERIMENTS.md](EXPERIMENTS.md)。
- 本轮远端结果根：`/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/omnivideo_score_20260906/exact_f1_2fps_768/`。其中`gap5000`保存筛选结果，`robustness_aggregate_v1/report`保存稳健性结果，`atomic_gap5000_v1/report`保存原子实验，`training_matrix_gap5000`保存四个5k训练输入 arm。`reasoning_choice_full_5k_v1`和`reasoning_rank_top20_v1`分别保存V1、V2全量训练及128题结果。
- 最新动态预算设置：[DYNAMIC_VIDEO_BUDGET_TRAINING.md](DYNAMIC_VIDEO_BUDGET_TRAINING.md)，远端根为`/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/dynamic_budget_v1`。
- V1设置与过程：[REASONING_CHOICE_FULL_5K.md](REASONING_CHOICE_FULL_5K.md)。
- V2设置与结果：[REASONING_RANK_TOP20.md](REASONING_RANK_TOP20.md)。
- 外部评测：[EXTERNAL_BENCHMARK_EVALUATION_20260910.md](EXTERNAL_BENCHMARK_EVALUATION_20260910.md)，远端根为`/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/external_benchmarks_20260910`；`<arm>/...`保存旧choice结果，`<arm>_independent/...`保存独立分片接续结果。
