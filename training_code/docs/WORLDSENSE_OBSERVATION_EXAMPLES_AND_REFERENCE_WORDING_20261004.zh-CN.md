# Observation 真实案例与中文翻译：429 条“参考答案口吻”从哪里来？

更新：2026-10-04。

## 1. 先回答你的怀疑

**“旧 observation 里有很多 `provided reference`、`reference answer`、`verified correct answer`，所以模型照抄出来”——这个具体解释不符合实际统计。**

- 1,453 条实际训练用 observation 正文中，这三个完整短语的命中数是 **0**；对应的原始 observation 也是 **0**。
- 但 teacher prompt 的固定模板每题都写了 `Verified correct answer:`。带 observation 的版本还加入 `Reference evidence observation:`、`auxiliary reference`、`the reference`，并再强调一次 `the verified correct answer`。
- 旧 observation 本身大量保留“检查确认了……”“这与答案一致”以及少量“我将提交这些区间”的标注口吻。它们虽然不含上述完整短语，仍可能与模板一起，让教师更倾向于写“依据参考材料解释答案”。
- 更直接的证据是：**这种措辞在训练中的学生生成回答里已经逐步增多，不是评测时才突然出现。**

因此，目前较有证据支持的解释是：**teacher 的参考材料提示方式，与 observation 的内容和标注口吻共同改变了蒸馏信号；学生逐渐学会了“我得到了参考答案”的说法。**但现有实验还不能把模板措辞和 observation 正文各自的因果贡献分离出来。

### 本文说的“目前 observation”是哪一版？

为解释你引用的 429 条输出，本文分析的是 **2026-09-30 正式训练实际使用的 1,453 条旧 observation**，而不是最近 50 题试跑／23 题重跑的新候选。后来的新候选没有回填到这批模型的训练中。

本次核对了训练 JSONL 的 SHA256，与该次启动配置记录的哈希一致；同时确认 SFT 的 analysis 与 CLUE 的 `teacher_observation` 逐条相同，LoRA／全参数 SFT 的监督消息也相同。

## 2. Observation 通常是什么样？

它不是统一格式的“纯视频事实”。常见形式是：**时间位置＋看见／听见的内容＋对答案的解释**；有些还保留标注 agent 的检查过程、选项分析、犹豫和提交意图。

长度按英文空白分词 `str.split()` 统计：

| 项目 | 数值 |
| --- | ---: |
| 条数 | 1,453 |
| 平均长度 | 68.3 词 |
| 中位数 | 57 词 |
| 最短／最长 | 20／658 词 |
| 超过 120 词 | 73 条，约 5.0% |

这些正文在当时没有按 120 词截断。SFT 完整监督 observation；CLUE 将完整 observation 放入 teacher 输入，同时要求**生成的 analysis**最多 120 词。输入 observation 长度与生成 analysis 长度不是同一件事。

下面前五例展示训练正文的完整英文与完整中文翻译；第六例太长，明确标为节选。**翻译忠实保留原文的判断和矛盾，不代表本次已重新核实视频。**

### Case 1：高达模型：比较接近我们需要的“事实＋简短归纳”

- ID：`YZGrxsiE::task1`；正文 52 词。
- 问题：她为什么喜欢动画版的高达模型？
- 数据集 gold：**它更结实。**
- clue 区间：`[[75.0, 83.0]]` 秒。
- 原始训练行：[CLUE 第 1 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_openqa_thinking_full_observation_20260930/data/clue_openqa_thinking.jsonl:1)／[SFT 第 1 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_sft_lora_20260930/data/sft_observation_openqa.jsonl:1)。

**训练使用的 observation 原文：**

```text
The host explicitly states her reasons for liking the 'Animever' (anime version) of the Gundam model between 75s and 83s. She mentions it is 'chunkier', the 'most solid suit that I have', and has 'metal pieces in the skirt and the shoulders'. These attributes align with the concept of being more rugged.
```

**中文翻译：**

> 主持人在 75～83 秒明确说出了自己喜欢这款高达模型“Animever”（动画版本）的理由。她提到它“更厚实”，是“我拥有的最结实的机体”，而且“裙甲和肩部有金属部件”。这些特征与“更结实”的含义相符。

**怎样理解这条：**

这条先给时间，再给听到的话，最后做简短归纳。它没有声称“系统提供了参考答案”。从文字结构看比较适合作为回答的依据，但本次只是摘录和翻译，没有重新听看视频来验证这些引语。

### Case 2：致谢者：从“匹配选项 B”机械改成“匹配答案文字”

- ID：`eapYeyoW::task1`；正文 42 词。
- 问题：视频中是谁说了感谢大家欢迎他们的话？
- 数据集 gold：**一位穿黑色长袖衬衫的女性。**
- clue 区间：`[[32.0, 35.0]]` 秒。
- 原始训练行：[CLUE 第 7 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_openqa_thinking_full_observation_20260930/data/clue_openqa_thinking.jsonl:7)／[SFT 第 7 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_sft_lora_20260930/data/sft_observation_openqa.jsonl:7)。

**训练使用的 observation 原文：**

```text
The inspection confirms that the woman at the podium, Nayo Harvin, says 'thank y'all for being so welcoming to us' between 00:32 and 00:35. She is wearing a black long-sleeve shirt. This matches the answer "A woman wearing a black long-sleeve shirt".
```

**中文翻译：**

> 检查确认，站在讲台前的女性 Nayo Harvin 在 00:32～00:35 说了“谢谢大家这么热情地欢迎我们”。她穿着黑色长袖衬衫。这与答案“一位穿黑色长袖衬衫的女性”一致。

**怎样理解这条：**

原始 observation 的结尾是 `This matches option B.`，即“这与选项 B 一致”。转换脚本把它改为 `This matches the answer "A woman wearing a black long-sleeve shirt".`。

这解释了旧标注中一种常见口吻的来源：原本是在定位选择题证据，改成开放题时只替换了字母，没有重新写成纯粹的音视频事实说明。这里虽然没有 `reference answer` 这两个词，却仍保留“把事实与某个已知答案对齐”的表达。

### Case 3：说话人：observation 指向白衣女性，gold 却是红裙女性

- ID：`JYBseFUa::task1`；正文 50 词。
- 问题：刚才是谁说了“What does that mean?”（“那是什么意思？”）？
- 数据集 gold：**一位穿红裙子的短发女性。**
- clue 区间：`[[15.5, 18.0]]` 秒。
- 原始训练行：[CLUE 第 305 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_openqa_thinking_full_observation_20260930/data/clue_openqa_thinking.jsonl:305)／[SFT 第 305 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_sft_lora_20260930/data/sft_observation_openqa.jsonl:305)。

**训练使用的 observation 原文：**

```text
The inspect view confirms the line 'What does that mean?' is spoken around 16-17s by a seated woman with an afro-textured hairstyle wearing a white top/jumpsuit, immediately after the short-haired woman in red delivers her line. This matches the answer "A woman with an afro wearing a white jumpsuit"'s description.
```

**中文翻译：**

> 复看片段确认，在约 16～17 秒，“那是什么意思？”这句话是一位坐着、留着蓬松卷发、穿白色上衣／连体衣的女性说的，紧接在那位穿红衣的短发女性说话之后。这与答案“一位留着蓬松卷发、穿白色连体衣的女性”的描述一致。

**怎样理解这条：**

**这条实际训练 observation 与同一行 gold 的人物描述冲突。**旧标注写的是 `option B`，转换后照着 B 的内容换成“白衣女性”；同一训练样本的标准答案仍是“短发红裙女性”。

SFT 因此同时监督了“分析说白衣女性”和“最终答案说红裙女性”；含 observation 的 CLUE 则把两种相冲突的信息一起交给 teacher。后者可能在已有分析和 gold 之间作语言上的协调，而没有真正解决谁说了哪句话。

这是可以从文本直接确认的内部矛盾，但不能据此裁定原视频正确答案究竟是 A、B 还是 C。前一篇 agentic 案例文档已进一步展示这题的旧查看轨迹和新盲答冲突。原文中的 `"'s description` 也是机械替换后留下的生硬语法。

### Case 4：游戏时间：把选项取舍和标注动作写进 observation

- ID：`CpXAoeKK::task0`；正文 99 词。
- 问题：游戏里，男子控制角色飞向橙红色龙时，游戏内时间是多少？
- 数据集 gold：**上午 6:15。**
- clue 区间：`[[83.0, 87.0]]` 秒。
- 原始训练行：[CLUE 第 84 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_openqa_thinking_full_observation_20260930/data/clue_openqa_thinking.jsonl:84)／[SFT 第 84 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_sft_lora_20260930/data/sft_observation_openqa.jsonl:84)。

**训练使用的 observation 原文：**

```text
The inspection confirms that Link jumps off the cliff and deploys his paraglider to fly toward the orange-red dragon at approximately 84 seconds into the video. At this exact moment, the in-game clock on the bottom-right HUD reads '06:10 AM'. The time then advances to '06:15 AM' shortly after he begins gliding. Since '06:10 AM' is not an option, but '06:15 AM' is the first available option corresponding to the start of the sustained flight/approach phase shown in the subsequent frames, I will select the interval covering the jump and initial glide where the clock transitions to 06:15 AM.
```

**中文翻译：**

> 检查确认，林克在视频约 84 秒时跳下悬崖，展开滑翔伞，飞向橙红色的龙。就在这一刻，右下角游戏界面上的时钟显示“上午 06:10”。开始滑翔后不久，时间才变为“上午 06:15”。由于“上午 06:10”不在选项中，而“上午 06:15”是后续画面所示持续飞行／接近阶段开始时第一个可选的时间，我会选择涵盖起跳和最初滑翔、时钟变到 06:15 的区间。

**怎样理解这条：**

这段混在一起的内容有三类：画面时间读数、为匹配选择题选项而作的解释、标注 agent 决定选择哪个区间的动作。

它没有被清理成开放题事实，因为旧清洗规则只处理 `option A/B/C/D` 这类固定格式，无法处理“这个时间不在选项中”这样的句子。最后一句 `I will select the interval...` 是标注者的操作说明，不是最终回答应学习的事实。这里引用的是**旧训练版**，不是后续质量重跑为这题生成的新候选。

### Case 5：快速装填器：有 reference 一词，但不是“参考答案”

- ID：`QMaVZkmC::task2`；正文 133 词。
- 问题：视频中的快速装填器是什么材料制成的？
- 数据集 gold：**金属。**
- clue 区间：`[[138.0, 145.0]]` 秒。
- 原始训练行：[CLUE 第 169 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_openqa_thinking_full_observation_20260930/data/clue_openqa_thinking.jsonl:169)／[SFT 第 169 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_sft_lora_20260930/data/sft_observation_openqa.jsonl:169)。

**训练使用的 observation 原文：**

```text
The question asks for the material of the 'quick reload device'. In the video, at approximately 02:18-02:24, the presenter unboxes and displays a blister-packed item he identifies as a 'Six millimeter Pro Shop speed loader'. Speed loaders are commonly referred to as quick reload devices in airsoft contexts. Although he does not explicitly state the material in this short clip, the visual evidence of the packaged product is the primary reference for this item. Another potential candidate is the 'another speed loader' mentioned around 03:10 with the Blaser sniper, but the Pro Shop one is more distinctly introduced as a separate accessory. Given the options (Plastic, Polymer, Metal, High molecular), and typical construction of such items, the visual inspection of this specific segment provides the necessary context to identify the object in question.
```

**中文翻译：**

> 问题问的是“快速装填器”的材料。在视频约 02:18～02:24，介绍者拆开包装并展示了一件吸塑包装的物品，称其为“Six millimeter Pro Shop 快速装填器”。在气枪运动语境中，speed loader 通常就是快速装填装置。虽然他在这段短片里没有明确说出材料，但这个包装产品的画面是判断该物品的主要参考依据。另一个可能的对象，是约 03:10 与 Blaser 狙击枪一起提到的“另一个快速装填器”，不过 Pro Shop 这件物品作为独立配件被介绍得更明确。结合选项（塑料、聚合物、金属、高分子材料）以及此类物品的常见结构，对这一片段的视觉检查提供了辨认对象所需的背景。

**怎样理解这条：**

这里的 `primary reference` 指“主要参考依据”，不是模型声称被告知了标准答案。因此不能把所有含 `reference` 的 observation 都当成泄漏参考答案口吻。

更值得关注的是证据充分性：这段承认没有听到材料说明，最后又借助选项和常见结构作判断，却没有具体说出哪一项视觉特征足以确认“金属”。它能帮助找到物品，但文字本身没有充分论证最终材料答案。

### Case 6：空间关系——把 658 词的反复推敲也当成了 observation

- ID：`yhHaTuLg::task0`；[训练第 97 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_openqa_thinking_full_observation_20260930/data/clue_openqa_thinking.jsonl:97)。
- 问题：穿豹纹衣服的女性，相对于三位男性位于哪里？
- gold：女性位于三名男性的正前方。
- 这条 observation 完整长度为 **658 词**，未截断。以下是三个真实段落的节选，中间段落省略；完整原文保存在原始训练行与本次 `audit.json` 中。

**节选一：**

```text
Wait, looking at the options again: Front, Right, Left, Behind. Since they are not in the same shot, is it possible the question refers to a specific composite image or poster *within* the scene? The synopsis mentions a movie poster next to Lucy. Does the poster feature the men? The description says the poster features Freddie Mercury.
```

> 等一下，再看一遍选项：前方、右方、左方、后方。既然他们不在同一个镜头里，问题是否指的是场景内部某张合成图或海报？概要提到 Lucy 旁边有一张电影海报。海报里有那几名男性吗？描述说海报上是 Freddie Mercury。

**节选二：**

```text
Let's assume the question is flawed or relies on a detail I can't see (like a reflection). But I must submit intervals. The evidence for the *existence* of both groups is throughout. The specific moments defining their separate spaces are the cuts.
```

> 假设题目有问题，或者依赖我看不到的细节，比如倒影。但我必须提交时间区间。两组人物“存在”的证据贯穿视频；界定他们各自空间的具体时刻，就是镜头切换处。

**节选三：**

```text
Re-reading the prompt: 'Locate the minimal time intervals... that contain the evidence needed to answer.' If the evidence is that they are NOT in the same place, the intervals showing each separately are the evidence.
```

> 重新读提示：“定位包含回答所需证据的最短时间区间……”如果证据是他们并不在同一地方，那么分别展示两组人物的片段就是证据。

**问题在哪里？**

这段不仅描述视频，还在复述标注任务、猜测数据集规律、反复检查选项、讨论如何提交区间。它甚至坦承没有共同画面、关系不明确，但训练最终答案仍给出了确定的“正前方”。这使“证据支持了什么”与“必须写出什么答案”脱节。

在 SFT 中，这些句子直接成为要模仿的目标文本；在 CLUE 中，它们进入 teacher 的参考上下文，可能影响老师对学生续写的偏好。两种路径不同，不能说 CLUE 在逐字监督这段 observation。

## 3. 全量统计：究竟是 observation 正文，还是 prompt 模板里有这些词？

### 3.1 与原报告保持一致的精确统计

匹配规则是**不区分大小写的连续子串**：`provided reference|reference answer|verified correct answer`。一条文本命中多个短语仍只计一条。不是语义分类，也不会把所有近义表达都算进来。

| 检查对象 | 文本条数 | 至少含一个上述短语 |
| --- | ---: | ---: |
| `merged.evidence.jsonl` 全部原 observation | 3,172 | **0** |
| 其中参与训练的原 observation | 1,453 | **0** |
| 实际训练用 observation 正文（选项字母替换后） | 1,453 | **0** |
| SFT 完整 assistant 目标（analysis＋answer） | 1,453 | **0** |
| CLUE student 训练 prompt | 1,453 | **0** |
| CLUE teacher 完整 prompt，含 observation | 1,453 | **1,453** |
| CLUE teacher 完整 prompt，不含 observation | 1,453 | **1,453** |
| 两个 benchmark 的评测 user prompt | 1,000 | **0** |

两版 teacher 命中的是 `verified correct answer`：含 observation 版本每题 **2 次**，不含 observation 版本每题 **1 次**。`provided reference` 与 `reference answer` 这两个完整短语在两版 teacher 输入中也都是 0。

因此，后来的大量 `provided reference` 更像是模型把“provided clue”“auxiliary reference”“correct answer”等上下文重新组合成一种回答口吻，而不是从 observation 正文整句复制。

### 3.2 但 observation 确实有其他标注口吻

| 正文中的词语／句式 | 命中条数 | 怎么解释 |
| --- | ---: | --- |
| `inspect`／`inspection`／`inspected` | 1,084 | 常描述 agent 的查看过程；不表示这些事实一定错误 |
| `option(s)`／`choice(s)` | 126 | 仍有选项词残留；不保证每条都在引用答案选项 |
| “matches / aligns with / corresponds to / supports … answer/option/choice”的限定正则 | 172 | 保留把证据与答案对齐的表达 |
| “I will select/submit/provide”“I have enough/sufficient evidence”“I must submit” | 83 | 保留取证和提交任务的语气 |
| `Wait`／`Let's reconsider/look`／`Re-reading` | 41 | 有停顿、反复推敲或重新读提示的表述 |
| 单独的 `reference` 一词 | 8 | 多数需看上下文；Case 5 就不是“参考答案” |

这些类别可重叠，不能相加当成“坏数据总数”，也不是人工逐条确认的问题标签。完整正则与命中 ID 均保存在审计 JSON 中。

## 4. 固定 teacher 模板如何把 observation 包装成参考材料？

当时实际使用的模板是：

```text
Verified correct answer: [正确答案内容]
Reference evidence observation:
[这里插入完整 observation]
Use this observation as an auxiliary reference together with the provided clue video and audio. Base your analysis on the available evidence and the verified correct answer. Do not treat the reference as the student response or simply copy it.
```

中文是：

> 已核实的正确答案：[答案内容]。
>
> 参考证据观察说明：[observation]。
>
> 请结合给出的 clue 音视频，把这段观察说明作为辅助参考。依据现有证据和已核实的正确答案进行分析。不要把参考内容当成学生的回答，也不要直接复制它。

这里有三个需要区分的事实：

1. **“verified”当时是我们写进提示的称呼，不是这批训练数据已经通过了后来新增的双重核验或人工审核。**Case 3 的内部冲突说明，不能仅凭这个词信任所有内容。
2. “不要直接复制”只要求避免照抄，没有明确要求“最终回答不要声称用户提供了参考答案”。模型可能把它改写成“根据所给参考答案……”而非直接复制 observation。
3. 不含 observation 的 teacher 同样有 `Verified correct answer:`，却没有出现相同规模的问题。因此，**不能把原因简单归结为这一个标签**。增加 observation 的正文、参考材料说明和它们与 gold 的相互影响，需要分别检查。

新增包装文字的实现位于 `training_code/scripts/add_worldsense_teacher_observation.py`；旧正文清洗位于 `prepare_worldsense_observation_sft.py`。后者主要将 `option/choice + A–D` 替换为选项内容，192 行发生过替换，没有完成事实核验、全文重写或去除所有标注过程话语。

## 5. 学生训练时已经出现了吗？——已经出现，而且逐步增多

核对 `completions.jsonl` 的记录代码后，可以确认下面统计的是**学生实际生成回答**，不是 teacher prompt，也不是教师生成的范文。

只统计 `outputs/formal`，排除 smoke。按保存 checkpoint 的三个阶段分组，每段各有 1,440 条回答日志：

| 正式训练阶段 | 含 observation 的 CLUE | 不含 observation 的 CLUE |
| --- | ---: | ---: |
| step 1–45，第 1 轮阶段 | 1 / 1,440（0.07%） | 0 / 1,440 |
| step 46–90，第 2 轮阶段 | 188 / 1,440（13.06%） | 0 / 1,440 |
| step 91–135，第 3 轮阶段 | 668 / 1,440（46.39%） | 2 / 1,440（0.14%） |

这里是生成记录的条数，不能当成不重复问题数，也不能将日志条数与“1,453 道训练题”直接等同。含 observation 版本在保存日志中的最早命中是 **step 45**。

两个具体原例：

**step 45，高尔夫挥杆计数。学生写：**

```text
To evaluate the reference answer, we focus on identifying the distinct times the club is swung through its full arc.
```

> 为了评估参考答案，我们重点辨认球杆完整挥动的各次动作。

该学生 prompt 只有音视频、问题和输出格式要求，没有参考答案。它却把自己置于“评估一个参考答案”的角色。

**step 52，红裙女性登场动作。学生在 answer 中写：**

```text
The answer to the question is that the women in red dresses performed the 'dance action' of crossing their arms, which is described in the provided reference solution.
```

> 答案是，穿红裙子的女性做了交叉双臂的“舞蹈动作”，这在给出的参考解答中有描述。

同样，学生输入中并未提供这份“参考解答”。从这些记录看，问题在训练中已经形成，后续评测沿用了这种回答口吻。

## 6. 为什么 observation 没有这些完整短语，学生仍能学出来？

可以把 teacher 看成一个拿到额外材料的答题者：它有 clue、gold 和 observation。student 只有全片与问题。二者接着**同一段学生已经生成的前缀**预测下一个 token。

```text
学生：完整音视频 + 问题 → 生成一段回答
                       ↓ 使用同一段回答前缀
教师：clue + gold + observation + 参考材料提示 → 给下一 token 的概率
                       ↓
训练：让学生的概率分布接近教师的概率分布
```

如果教师在某些前缀后更倾向续写“reference”“provided”等词，这种语言偏好也会进入蒸馏信号。损失函数没有天然区分“正确的音视频推理能力”和“教师拿着答案解释的说话方式”。学生可能学到两者，也可能过多学到后者。

**这是依据实现和现象提出的机制解释，不是已经逐 token 验证的因果结论。**本次没有保存可用于完整追溯这 429 条表达来源的教师 token 概率轨迹；不能声称已经测量了某一句前缀下教师究竟给了多大的“reference”概率。

本次配置是 `lmbda=1.0`、`sft_alpha=0`、全词表 JSD、CPU EMA teacher。它没有先生成一份完整 teacher 范文再让学生逐字照抄，也没有将 observation 作为 CLUE 的固定交叉熵目标。EMA 与学生沿自身输出继续训练可能维持这种倾向，但这里也没有单独做实验量化这个因素。

SFT 是另一个训练机制：虽然使用相同 observation 正文，它只直接监督 analysis＋gold 自然语言答案，不使用上述 teacher 参考材料包装。三轮 SFT 的本次 benchmark 回答都没有命中这三个短语。这个对照进一步削弱了“都是 observation 原文本身含大量同样短语”的解释，但不能证明 observation 对 CLUE 毫无影响。

## 7. 重新核对你引用的 429 条

范围：**含 observation 的 CLUE 第 3 轮模型，OmniVideoBench 500 题＋DailyOmni 500 题，共 1,000 条回答**。429 是命中回答条数，不是短语出现次数，也不是 429 道不同的训练题。

| 匹配项目 | 命中回答数 |
| --- | ---: |
| `provided reference` | 394 |
| `reference answer` | 67 |
| `verified correct answer` | 2 |
| 三项去重后的并集 | **429** |

一条回答可以含多个短语，所以 394＋67＋2 不能当成并集。按 benchmark 拆分是 OmniVideoBench 218 条、DailyOmni 211 条。

同一规则应用到各轮 checkpoint：

| 模型 | 第 1 轮 | 第 2 轮 | 第 3 轮 |
| --- | ---: | ---: | ---: |
| LoRA SFT | 0 / 1,000 | 0 / 1,000 | 0 / 1,000 |
| 全参数 SFT | 0 / 1,000 | 0 / 1,000 | 0 / 1,000 |
| CLUE 含 observation | 1 / 1,000 | 317 / 1,000 | 429 / 1,000 |
| CLUE 不含 observation | 0 / 1,000 | 0 / 1,000 | 1 / 1,000 |

原模型为 0 / 1,000。以上都来自你指定的 **v3 保存回答**，不是后来其他评测任务的结果。

429 条中，按 v3 的明确选项评分：**136 条选项正确、185 条选项错误、108 条没有可解析的唯一选项**。所以这些措辞不能直接当作错误标签；一个模型可能选对了字母，却仍声称自己看到了实际未提供的参考答案。

例如 `dailyomni_0431` 的第 3 轮含 observation 模型回答：

```text
<analysis>
At the beginning, there is only audio, no mention of the correct answer. However, the provided reference answer is B and the explanation is consistent with the observation made in the speech and vision of the clip. The correct answer of "B" is provided for the correct option without additional references. This answer is consistent with the provided correct answer.
</analysis>
<answer>B</answer>
```

中文翻译：

> 开头只有音频，没有提到正确答案。不过，给出的参考答案是 B，其解释与片段中的语音和视觉观察一致。正确答案“B”已经作为正确选项给出，不需要额外参考。这一答案与所提供的正确答案一致。
>
> 最终答案：B。

它的 B 恰好与 gold 相同，但评测输入只提供问题、选项和音视频；没有向模型提供标准答案。**字母得分正确，不等于这段自称“收到参考答案”的解释真实。**仅凭这段回复也不能断言它完全没有使用音视频。

## 8. 应该改什么？

### 8.1 清理 observation，需要改内容和写法

- 将“可见／可听事实”“必要推断”“仍未解决的疑点”分开保存。
- 去除 `I will submit/select intervals`、重新阅读标注指令、猜测选项规律等操作性话语。
- 不把“这与参考答案一致”当作证据；应该说明哪个画面、哪句话支持结论。
- observation 与 gold 冲突时进入复核，不通过把措辞改得更顺就直接投入训练。
- 长 observation 先整理为简洁、有证据的文本，再核验；不能只截断尾部，否则可能删掉关键限定或冲突提示。

Case 2 如果事实已经核实，可以保留“32–35 秒，穿黑色长袖的女性在讲台前说出致谢的话”，去掉“检查确认”和“匹配答案”的元话语。Case 3 则必须先解决人物归属，不能单靠文字润色修好。

### 8.2 Teacher 提示也要调整，但替换几个词不保证解决

仍可保留你要求的 golden clue、golden answer 和 observation，明确让 teacher 完成的是**可独立阅读的证据分析与最终回答**，并要求最终回复不要提“已给出的答案／参考解答／标注人员”。“verified”只有在确实通过规定核验时才有对应含义。

仅把 `Reference` 改名为另一个词，模型仍可能学到“借助答案解释答案”的方式。因此要检查真实学生 rollout，而不只检查模板文字是否删干净。

### 8.3 怎样验证到底是谁导致的？

建议在相同基础模型、样本顺序、学习率、生成与多媒体设置下，分开比较：

| 对照 | Observation 正文 | Teacher 外层包装 |
| --- | --- | --- |
| 基线 | 原正文 | 原 reference／verified 包装 |
| 只改正文 | 经过核验的事实型正文 | 原包装 |
| 只改包装 | 原正文 | 去除参考答案口吻、要求独立作答的包装 |
| 两者都改 | 经过核验的事实型正文 | 改后的包装 |

同时保留不含 observation 的对照。记录措辞比例、最终答案完整率、选项正确率、分析事实性；仅让这三个短语消失，不代表模型真的学会理解音视频。

本次没有启动上述新训练或修改现有数据。当前证据可以定位风险来源、否定“正文里很多同样短语”的说法，但不足以宣称已经证明唯一原因。

## 9. 可复核文件与统计口径

- 审计脚本：[audit.py](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_wording_audit_20261004/audit.py)。
- 全量统计、六例原文、命中 ID 与输入文件哈希：[audit.json](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_wording_audit_20261004/audit.json)。
- 含 observation 的训练中命中回答：[clue_obs_matching_rollouts.jsonl](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_wording_audit_20261004/clue_obs_matching_rollouts.jsonl)。
- 不含 observation 的对应记录：[clue_noobs_matching_rollouts.jsonl](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_wording_audit_20261004/clue_noobs_matching_rollouts.jsonl)。
- 各 checkpoint 的评测命中回答：[matching_evaluation_outputs.jsonl](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_wording_audit_20261004/matching_evaluation_outputs.jsonl)。
- 你指定的原报告：[v3 RESULTS.zh-CN.md](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_generalization_mcq500_npu96_20261003/rescore_explicit_formats_v3/RESULTS.zh-CN.md)。

复现命令：

```bash
cd /opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd
python training_runs/worldsense_observation_wording_audit_20261004/audit.py
```

审计只读取原始训练数据、正式训练 completion 日志和 v3 保存回答；写入独立分析目录。没有调用标注 API，也没有改动训练数据、模型或评测分数。
