# WorldSense 训练效果分析：适合初学者的详细说明

更新日期：2026-10-03。本文依据本项目已经完成的训练日志、训练目标文本和逐题评测输出整理。

## 先看结论

本次不能把所有结果简单概括成“模型什么也没学到”。实际有三类问题：

| 对象 | 已经观察到什么 | 应当怎样理解 |
| --- | --- | --- |
| SFT | 回答格式明显改善，训练 loss 下降，但答题正确率没有稳定提升 | 模型学会了训练文本中的格式和表达；这些变化还没有转化为可靠的答题收益。 |
| CLUE-OPSD | 开放式回答中大量重复分析，直到停止生成也没有最终答案 | 模型的长回答生成行为出了问题，许多题没有完成回答。 |
| 开放式评分 | 有明确的错判，包括把正确事件顺序判错 | 当前开放式分数需要复核，不能据几个百分点的差距精确判断模型优劣。 |

本文重点回答：

1. 为什么会没有最终答案？
2. 什么叫重复分析？它与正常的详细分析有什么区别？
3. 什么是贪婪解码？为什么它可能让重复更明显？
4. 什么叫长度漂移？训练日志中有什么迹象？
5. 为什么 loss 下降，答题效果却没有改善？
6. 训练与评测的音视频设置究竟哪里不同？

**证据范围：**四项新实验均训练了 3 个 epoch；评测包括原模型和四项实验的第 1/2/3 轮结果。每个模型评测同一组 518 道 WorldSense 留出题，来自 371 个未参与训练的视频。本文中的“含 observation 的 CLUE”与“不含 observation 的 CLUE”均为本轮全参数训练，不是早期 LoRA CLUE 实验。

**建议阅读顺序：**先看第 2～3 节的实际回答，再看第 4～7 节理解“缺答案、重复、贪婪解码、长度漂移”；第 8～9 节解释训练目标的问题，第 10～11 节解释多媒体和长度参数。第 12～14 节说明评分局限、结论和排查顺序。

---

## 1. 先理解模型是怎样把回答写出来的

### 1.1 token 是什么？

token 是模型处理文字的基本单位。它可以是一个单词、单词的一部分、一个标点，或其他字符组合。

所以：

- 120 个英文词不等于 120 个 token。
- `<analysis>`、`</analysis>`、`<answer>` 等标签也会占用 token，而且一个标签可能对应多个 token。
- “最多生成 768 tokens”不是“最多写 768 个英文词”。

正常的 120 词英文分析，加上短答案和标签，通常只需要几百个 token。具体数量由实际文字和 tokenizer 决定。

### 1.2 模型不是一次性写好整篇回答

模型每次预测下一个 token，再把它加入已有回答中，继续预测。

```text
视频 + 音频 + 问题 + 提示要求
              ↓
预测下一个 token
              ↓
加入已有回答
              ↓
根据“输入 + 已写出的回答”再次预测下一个 token
              ↓
……不断重复，直到满足停止条件
```

这意味着，模型后面的文字会受到自己前面文字的影响。如果前面进入了某种重复句式，后面可能继续沿着这种句式写下去。

### 1.3 本次希望模型完成的回答是什么样？

以“出现了几次掌声？”为例，下面是**为了说明格式而编写的示意回答，不是实际模型输出**：

```text
<analysis>
I identify four separate applause segments in the audio.
They are separated by speech or singing, so they should be counted
as four distinct occurrences rather than one continuous segment.
</analysis>
<answer>Four times.</answer>
```

它需要完成三个动作：

1. 写与题目有关的分析；
2. 结束分析，生成 `</analysis>`；
3. 进入 `<answer>`，给出明确答案。

**本文所示“没有最终答案”是指始终没有进入 `<answer>` 并提交答案。**写了很多分析，甚至分析中出现了正确数字，也不等于已经提交了最终答案。评分程序还会把多个答案标签造成的歧义、仅输出一个选项字母等情况判为无效答案。

---

## 2. 实际示例一：分析中提到了 six，却没有回答“六种”

### 2.1 这道题问什么？

- 样本：`AHvABhxR::task0`
- 原题：`How many types of sports events are shown in the video?`
- 中文：视频中展示了多少种体育赛事？
- 标准答案：`Six.`，即六种。

### 2.2 含 observation 的 CLUE，第 3 轮怎样回答？

以下为实际回答的节选；`...` 表示省略原文中间部分。

```text
<analysis>
The video showcases a variety of activities, including a firefighter
competition, a bodybuilding event, and a rescue operation.
...
The reference text provides additional context, stating that the
competition includes "six events through five categories," ...
```

可以看到，分析已经提到了 `six events`。但它没有结束分析并提交 `Six.`，接下来反复写：

```text
The reference text also mentions the "2020 Gyeonggi Province
Firefighters Competition," which is consistent with the video's content.

The reference text further specifies that the competition includes
"firefighting and rescue activities" and "bodybuilding,"
which matches the observed activities in the video.
```

上述两种句式各出现了 **11 次**。回答最后停在：

```text
The reference text also mentions the "2020 Gyeonggi Province
Firefighters Competition," which is consistent with
```

实际检查结果：

| 检查项 | 结果 |
| --- | --- |
| 分析词数 | 529 个英文词，远超提示要求的 120 词 |
| 是否出现 `</analysis>` | 没有 |
| 是否出现 `<answer>` | 没有 |
| 是否有最终答案 | 没有 |
| 输出重新分词长度 | 768 tokens |
| 结尾是否完整 | 不完整，停在半句话 |

**这不是只漏写了答案的闭合标签，而是根本没有进入最终答案部分。**

读者可以从分析中猜出“六种”，但模型没有按要求给出最终答案。评分程序也不会从一大段分析中挑选一个数字代替它的最终结论。

另外，评测输入中没有提供 reference text、golden answer 或 observation。模型却反复声称“reference text 提到了……”。可以确认它生成了这种表达，但不能把它说的 reference 当成评测时真正提供的材料。这可能反映了训练表达方式被带到了评测中，具体机制还需要验证。

来源：[含 observation 的实际输出，第 1 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_eval_20261001/openqa/clue_obs_epoch3/scored.jsonl:1)。

### 2.3 不含 observation 的 CLUE，同题也重复

它反复写：

```text
The video also includes a man performing a high jump,
a man running, and a man performing a long jump.
```

这句话完整出现 **31 次**。最后停在：

```text
The video also includes a man performing a high jump, a
```

整个分析为 **666 词**，重新分词也是 **768 tokens**，没有 `</analysis>`，也没有 `<answer>`。

这说明重复问题并不只存在于“teacher 加了 observation”的实验中。

来源：[不含 observation 的实际输出，第 1 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_eval_20261001/openqa/clue_noobs_epoch3/scored.jsonl:1)。

---

## 3. 实际示例二：题目要求数掌声，模型一直描述舞台

### 3.1 这道题问什么？

- 样本：`AOOFZaPS::task0`
- 原题：`How many times did applause occur?`
- 中文：出现了几次鼓掌声？
- 标准答案：`Four times.`，即四次。

### 3.2 CLUE 的实际回答

不含 observation 的 CLUE 第 3 轮回答如下，省略了后续大量相同句子：

```text
<analysis>
The video shows a stage performance with a man and a woman.
The audience is clapping and cheering, and the man is speaking
into a microphone. The woman is singing, and the audience is
clapping and cheering.

The man is speaking into a microphone, and the woman is singing.
The audience is clapping and cheering.
The man is speaking into a microphone, and the woman is singing.
The audience is clapping and cheering.
...
```

其中：

- “The man is speaking ... woman is singing.”完整出现 **33 次**；
- “The audience is clapping and cheering.”完整出现 **32 次**；
- 分析共 **628 词、768 tokens**；
- 结尾停在 `The audience is`；
- 没有生成最终答案。

**重复写了 32 次“观众鼓掌”，不表示视频真的有 32 次掌声。**这些是回答文本自身的重复，不能拿来做事件计数。

正常分析应该识别不同掌声片段、区分连续掌声和相互分离的掌声，再给出次数。当前回答只在描述“谁在讲话、谁在唱歌、观众在鼓掌”，没有完成计数。

来源：[实际输出，第 3 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_eval_20261001/openqa/clue_noobs_epoch3/scored.jsonl:3)。

### 3.3 SFT 同题：有最终答案，但内容错误

LoRA SFT 第 3 轮的完整实际回答是：

```text
<analysis>
The inspection confirms two distinct applause segments:
one at approximately 10-12 seconds (after the man finishes speaking)
and another at around 20-22 seconds (after the woman finishes singing).
These are the only two instances of applause visible in the video.
</analysis>
<answer>Two.</answer>
```

标准答案是四次，它回答两次。所以这是**答案完整、内容错误**。

这里的时间点是模型声称的证据，我们没有重新看视频核验，不能把这些时间点当成已经确认的事实。

来源：[SFT 实际输出，第 3 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_eval_20261001/openqa/sft_lora_epoch3/scored.jsonl:3)。

### 3.4 原模型同题：有答案，只是漏了闭合标签

原模型结束分析后写了：

```text
<answer>
Applause occurred multiple times during the performance.
```

它没有写 `</answer>`，但已经进入答案部分，并写了内容。当前评分会提取这段内容，继续判断语义。

它的问题是只说“多次”，没有给出题目要求的具体次数。因此判错的依据是答案不够明确，而不是单纯漏了闭合标签。

来源：[原模型实际输出，第 3 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_eval_20261001/openqa/base/scored.jsonl:3)。

### 3.5 请区分这三种失败

| 情况 | 例子 | 说明 |
| --- | --- | --- |
| 有最终答案，内容错误 | `<answer>Two.</answer>`，标准答案为 Four times | 模型完成回答，但判断错了。 |
| 有最终答案，内容不充分 | `<answer>Multiple times.`，没有闭合标签 | 可以评分，但没有回答具体次数。 |
| 没有最终答案 | `<analysis>` 内一直重复，始终无 `<answer>` | 模型没有完成回答。 |

修复这三种问题需要不同的方法。只改善答案提取规则，无法让一个从未写出答案的回答变成完整作答。

---

## 4. 为什么输出会在分析中停止？

### 4.1 生成有停止条件

常见停止条件包括：

1. 模型生成了结束 token，通常称为 EOS；
2. 命中了配置的其他停止条件；
3. 达到了允许生成的最大 token 数。

本次 CLUE 训练最多生成 **512 tokens**；开放式评测最多生成 **768 tokens**。

这个额度包括分析、答案、标签、换行等整个输出。**代码没有自动替最终答案预留一段额度。**如果模型把额度全花在分析上，系统到上限时就停止，不会额外允许它补写答案。

```text
正常输出：
分析 → 结束分析 → 最终答案 → 结束生成

当前大量错误输出：
分析 → 重复描述 → 重复描述 → …… → 用尽生成额度
                                              ↓
                                     最终答案尚未生成
```

上面的真实例子重新分词恰好是 768 tokens，而且停在半句话，与耗尽生成上限相符。保留的评测结果没有记录原始 `finish_reason`，所以这里的判断依据是输出长度和内容；不能声称直接读到了 `finish_reason=length`。

### 4.2 提示里写“120词以内”，为什么还会写到几百词？

提示是给模型的文字要求。模型需要学会遵守它，生成器不会因此自动计数到第 120 个词，然后强制切换到答案。

本次代码没有实现：

- 到一定分析长度时强制闭合分析；
- 为答案保留生成额度；
- 针对超过 120 词或缺答案设置独立奖励、惩罚。

因此“提示要求 120 词”和“代码保证最多 120 词”是两件不同的事。目前实现的是前者。

### 4.3 增大生成上限，能解决吗？

如果模型只是差几个 token 就能写完答案，适当增加上限有帮助。但上面例子已经重复同一句话几十次，问题是没有顺利切换到答案阶段。

仅提高上限可能让它继续重复。是否能恢复答案需要实验验证；优先应该检查重复路径和回答完整性，而不能只根据“缺答案”就不断增加上限。

---

## 5. 什么是“重复分析”？

### 5.1 正常的详细分析会不断增加有用信息

例如计数题的分析可以依次指出：

1. 在哪些时间段出现事件；
2. 哪些事件属于同一次连续发生；
3. 为什么最终计数是四次。

这些文字帮助回答题目。

### 5.2 当前的重复分析没有推进解题

“男人在讲话，女人在唱歌，观众在鼓掌”出现一次可以提供场景信息。连续出现几十次，没有增加新的时间点、事件区分或数量结论，却消耗了生成额度。

这就是本文所说的重复分析。它发生在**同一个回答的文本内部**；不是日志把一份回答打印了几十遍，也不是评测程序把同一道题重新运行了几十次。

### 5.3 怎样做统计？

诊断脚本检查长分析中重复出现的连续 20 词片段：至少 60 个词，且重复片段比例达到 40% 时，标记为重复分析。

这个规则用于发现长段落循环。它不是对所有分析事实正确性的判断，也可能漏掉只使用不同措辞的重复。

第三轮的结果：

| 模型 | 重复分析被标记题数 | 没有有效最终答案的题数 |
| --- | ---: | ---: |
| 原模型 | 1 | 3 |
| LoRA SFT | 0 | 8 |
| 全参数 SFT | 0 | 0 |
| CLUE，含 observation | 332 | 349 |
| CLUE，不含 observation | 395 | 412 |

两个计数不是完全相同的集合：有些重复回答最后仍能给答案，也有些缺答案的回答没有达到上述重复阈值。

来源：[生成诊断汇总](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_eval_20261001/analysis_diagnostics_summary.json)，[诊断脚本](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_code/scripts/diagnose_worldsense_openqa_outputs.py:24)。

---

## 6. 贪婪解码是什么意思？

### 6.1 模型会给下一个 token 一组概率

在每一个生成位置，模型都会计算词表中各个 token 的概率。解码方法决定“根据这些概率，实际选择哪一个 token”。

下面是**教学用的假设概率，不是从本次模型日志中读取的概率**。为便于理解，把部分候选 token 按作用描述：

| 下一步候选 | 假设概率 |
| --- | ---: |
| `The`，准备开始另一句场景描述 | 45% |
| 开始闭合分析标签的 token | 30% |
| 其他 token 合计 | 25% |

### 6.2 贪婪解码：每一步都选当前概率最高的 token

上例会选择 `The`。然后模型基于新增的 `The` 再预测下一步，继续选当前概率最高的 token。

```text
计算下一 token 的概率
        ↓
选概率最大的 token
        ↓
写入回答，继续下一步
```

它不会提前比较“把这一段完整写完后，哪一篇回答最好”，也不会额外检查“刚才是否已经说过同一句话”。

“贪婪”是算法名称，意思是每一步做当前最优的局部选择；不是模型有某种情绪或意图。

本次评测设置 `temperature=0`，对应框架中的贪婪生成。这里不是数学上真的把概率公式除以零，而是让框架使用确定性的最大概率选择。

### 6.3 随机采样：按概率抽取 token

如果采用概率采样，上例除了可能选 `The`，也可能抽到结束分析的 token，从而走入另一条生成路径。

本次 CLUE 训练 rollout 使用：

```text
temperature = 1.0
top_p = 1.0
top_k = -1
```

也就是按当前模型分布进行随机采样，没有通过 top-p/top-k 缩小候选范围。

这里的 `temperature` 控制分布的集中程度。温度越低通常越偏向高概率 token；本次训练的同一参数还用于蒸馏损失的 logits 缩放。

### 6.4 为什么贪婪解码可能把重复放大？

假设模型写完一句场景描述后，最高概率的下一步又是开始类似描述。贪婪解码就会继续描述。新描述加入上下文后，相同续写又可能成为最高概率，于是形成循环：

```text
描述画面
   ↓
最高概率续写：再描述同一画面
   ↓
已有回答更像这种描述模板
   ↓
最高概率续写：继续同一模板
   ↓
……
```

随机采样有机会走到其他路径，例如结束分析；但它也可能生成错误文字。**改成采样不保证答案正确。贪婪解码也不必然重复：原模型和 SFT 使用同样的贪婪评测，大多数回答并未循环。**

目前能确认的是：训练和评测的解码方法不同，CLUE 在当前贪婪评测下大量重复。是否主要由这种差异触发，需要对同一 checkpoint、同一批题、同样媒体输入，只改变解码方法来验证。

---

## 7. “长度漂移”是什么意思？

长度漂移指：随着训练推进，模型生成的回答逐渐变长，或越来越偏离期望的长度范围。

它不表示修改了 `max_completion_length`。本次各轮训练的上限一直是 512 tokens；变化的是模型在这个固定上限内实际生成的长度。

### 7.1 训练日志中的变化

每项 CLUE 实验共记录 4,320 条训练 rollout，每轮 1,440 条。按实际记录统计：

| 实验 | epoch | 平均生成 tokens | 到 512-token 上限的比例 | analysis 超过 120 词的比例 | 没有 `<answer>` 起点的比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 含 observation | 1 | 201 | 1.39% | 38.13% | 2.43% |
| 含 observation | 2 | 272 | 5.83% | 60.63% | 3.47% |
| 含 observation | 3 | 297 | 8.61% | 61.04% | 5.21% |
| 不含 observation | 1 | 150 | 0.07% | 16.04% | 0.56% |
| 不含 observation | 2 | 182 | 0.35% | 34.93% | 1.39% |
| 不含 observation | 3 | 218 | 1.88% | 48.54% | 2.29% |

例如，含 observation 的训练平均长度从约 201 增长到 297 tokens，同时超长、截断、缺答案比例上升。这是可以从训练中观察到的长度漂移。

**长回答本身不等于错误。**问题是本任务要求简短分析，而变长伴随着更多超限和缺答案，已经偏离预期。

### 7.2 评测中的失败进一步增加

| 模型 | 第 1 轮缺有效答案 | 第 2 轮 | 第 3 轮 |
| --- | ---: | ---: | ---: |
| CLUE，含 observation | 217/518 | 282/518 | 349/518 |
| CLUE，不含 observation | 146/518 | 322/518 | 412/518 |

第三轮分别有 **67.37%**、**79.54%** 的题没有有效最终答案。

但需要区分：训练采样中的严重重复，比留出集贪婪评测少得多。按相同长片段重复规则，4,320 条训练 rollout 中，含 observation 仅 7 条、不含 observation 仅 4 条被标记。因此可以说训练已有长度和完整性恶化迹象，不能说训练中早已发生了与评测同等程度的大面积循环。

来源：两项训练的 [含 observation 日志](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_openqa_thinking_full_observation_20260930/outputs/formal/v0-20260930-165152/logging.jsonl)、[不含 observation 日志](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_clue_full_no_observation_npu96_w10w11_20260930/outputs/formal/v0-20260930-182502/logging.jsonl)，以及同目录的 `completions.jsonl`。

### 7.3 训练过程中 student 也大量反复分析了吗？

**保存的训练回答中已经出现少量严重循环，但没有出现与评测相当的大面积逐字重复。**这次重新用与评测完全相同的答案/分析提取器及连续 20 词重复规则，检查了全部保存的训练输出：

| 实验 | 第 1 轮严重重复 | 第 2 轮 | 第 3 轮 | 全部训练输出中的比例 | 第 3 轮留出评测中的比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 含 observation | 4/1440 | 2/1440 | 1/1440 | 7/4320，0.16% | 332/518，64.09% |
| 不含 observation | 0/1440 | 0/1440 | 4/1440 | 4/4320，0.09% | 395/518，76.25% |

两项 `completions.jsonl` 都保存了 135 个 step，每个 step 32 条回答。框架把全部 16 个 rank 的生成文本汇总后写入，没有只抽几个回答。4,320 是生成回答条数，包含不同 epoch 重复处理训练题，不是 4,320 道独立题。

例如，不含 observation 的第 94 步真实训练回答反复出现：

```text
The man in white is wearing a white shirt and white trousers,
and he is the main character in the scene.
The man in white is standing behind the man in white.
...
```

最后仍没有答案。来源：[实际 student 训练生成，第 94 行中的第 2 条回答](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_clue_full_no_observation_npu96_w10w11_20260930/outputs/formal/v0-20260930-182502/completions.jsonl:94)。

因此，训练中已经有类似错误样例。更明显的总体预警是长度和有效答案比例：含 observation 每轮无有效答案为 45、70、108 条；不含 observation 为 15、41、61 条，每轮分母均为 1,440。“无有效答案”还包括多个答案标签等歧义，因此不能把这个数字与前表“根本没有 `<answer>` 起点”的数字混用。

含 observation 的严重重复计数并没有逐轮增加，所以不能说训练中“重复率一路上升”。逐轮增加的是回答长度及无效答案，而大量循环主要在当前留出集贪婪评测中暴露。

统计说明：以上 120 词比例已经改为使用当前评测同一套提取规则。之前采用较宽松正则的统计与此有小幅差异；严重重复 7/4、平均生成长度、截断比例及缺 `<answer>` 起点的统计未变化。

### 7.4 为什么训练与评测的重复数量相差这么大？

比较两列时，除了重复规则相同，其他条件并不完全相同：

| 条件 | 保存的训练回答 | 本次评测回答 |
| --- | --- | --- |
| 模型参数 | 训练过程中的不同 step，持续变化 | 固定的第 3 轮 checkpoint |
| 题目 | 参与训练的题目 | 未参与训练的留出题 |
| 生成方式 | 温度 1 的随机采样 | 温度 0 的贪婪解码 |
| 视觉空间预算 | 较高的动态预算 | 较低的固定预算 |
| 输出 token 上限 | 512 | 768 |

因此，这张表确证了“当前评测比训练记录严重得多”，但不能单凭两列就断言全部差异都由贪婪解码造成。

一种符合现有证据、仍待对照验证的解释是：训练使模型越来越倾向延长分析，形成了不稳定的续写行为；训练随机采样没有频繁走入长段循环，到了不同题目、不同媒体预算的贪婪评测下，重复路径被连续选择并放大。更长的输出上限也让循环有更多空间持续。

还应避免一个误解：当前 JSD 训练不是把学生写出的每个重复句子直接当作 golden answer 学习。它仍比较教师与学生的概率分布，教师有机会纠错。关键尚未验证的是：教师在这些错误前缀上，是否确实提供了足够强的停止、切换到答案或纠正内容的反馈。第 8 节继续解释这个问题。

---

## 8. CLUE 的 loss 下降，为什么没有保证答案变好？

### 8.1 先分清“教师知道答案”和“直接训练学生输出正确答案”

当前 CLUE 设置：

```text
lmbda = 1.0
sft_alpha = 0
beta = 0.5
clue_ema_alpha = 0.05
```

学生自行生成所有训练回答。教师收到 golden clue 和 golden answer；其中一项实验还收到 observation。

接下来，教师在学生已经生成的同一串回答前缀上，计算下一 token 的概率。损失比较教师、学生的完整词表概率分布，实际是 beta=0.5 的对称 JSD。

```text
学生输入：完整视频/音频 + 问题
                 ↓
学生生成一串回答 y
                 ↓
对回答 y 的各个位置：

学生：完整视频/音频 + 问题 + y 的前面部分 → 下一 token 分布
教师：clue片段 + 问题 + golden answer
      + 可选 observation + 同样的 y 前缀 → 下一 token 分布
                 ↓
让两组下一 token 分布更接近
```

因此，golden answer 是影响教师判断的条件信息。当前没有再单独计算“学生在标准答案序列上的交叉熵”。教师也不是先独立写出一篇经过检查的正确分析，再让学生照着全文学习。

### 8.2 分布接近，不等于已经完成正确回答

假设学生已经写出很多重复场景描述。在这种前缀下，如果教师也比较倾向继续描述，那么两者的下一 token 分布可以很接近，JSD 就会较小。

这是解释损失与效果脱节的一种可能机制。**我们尚未保存和分析教师在这些坏前缀上的完整分布，所以不能断言教师已经被证明在鼓励这些循环。**

但可以确定：当前 loss 没有直接给出“此题最终答案正确 1 分、错误 0 分”的反馈，也没有独立的回答完整性奖励。

两项训练首步到末步 loss：

- 含 observation：`0.0691 → 0.0142`；
- 不含 observation：`0.0501 → 0.0118`。

日志记录的蒸馏损失下降。但各步使用不同批次，EMA 教师也随训练变化，因此这不是同一批题、固定教师条件下的对照。它们不能代替最终答案正确率、缺答案率和重复率。

### 8.3 教师还会随学生变化

当前使用 EMA 教师，即学生参数的平滑平均：

```text
新教师参数 = 0.95 × 旧教师参数 + 0.05 × 最新学生参数
```

教师不会因一次学生更新而完全改变，但也不是始终固定的模型。它是否逐渐跟随了不理想的生成行为，需要进一步验证。

纯学生生成的 on-policy 蒸馏本身是 GKD 允许的研究设置。当前结果不能证明这类方法必然无效；需要检查本次教师反馈、回答约束和训练/评测分布是否合适。原论文也使用温度 1 的训练采样及贪婪或采样评测，因此温度差异本身不能直接认定为代码错误。[GKD 论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/5be69a584901a26c521c2b51e40a4c20-Paper-Conference.pdf)。

### 8.4 是不是代码根本没有训练“停止”？

审计没有发现“所有 EOS 都被屏蔽”或已确认的回答位置错位问题：

- 自然生成的 EOS 会参与蒸馏；
- 每个被监督位置比较的是整个词表，其中包括结束相关的候选 token；
- 触达生成上限但没有 EOS 的回答，不会自动补 EOS、闭合标签或答案；
- 缺答案的 rollout 没有因为缺答案而被剔除。

所以需要检查停止行为是否学得有效，不能把它解释成代码完全没有任何停止信号。

实现来源：[全参数 EMA 与 JSD 入口](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_code/scripts/worldsense_full_gkd_entry.py:123)。

---

## 9. SFT 为什么学会了格式，却没有稳定提高正确率？

### 9.1 当前监督文字怎样生成？

当前 SFT 把证据定位阶段的 observation 放进 `<analysis>`，再把正确选项内容放进 `<answer>`。

observation 是当时的标注模型提交的说明。它可能包含片段观察、猜测、摘要信息、选项比较和下一步检查计划。它没有经过“是否能推出 golden answer”的完整一致性审核。

生成脚本检查文本非空、标注状态为 submitted，并改写一部分显式选项字母引用；随后直接拼接 analysis 和 golden answer。

### 9.2 实际训练目标内部存在冲突

样本 `tQqMFbqn::task1` 问视频中有多少个靶子。下面是实际目标的节选：

```text
<analysis>
...
Count: Far left: 2. Next block: 3. Next block: 3.
Far right: 3. Total = 11. This matches the answer "Eleven".
...
</analysis>
<answer>Thirteen.</answer>
```

分析说 11，最终答案说 13。同一段监督要求模型模仿这两种互相矛盾的结论。

另两例：

| 训练样本 | analysis | answer |
| --- | --- | --- |
| `hFSjJhYE::task0` | 红衣人员在旗帜前方，明确排除左右关系 | 在旗帜左边 |
| `zetwUmDH::task0` | 顺序明确为 a → b → c → d | b → a → c → d |

这些是确定的文本内部矛盾。还没有重新观看视频确认哪边符合证据，所以本文没有认定 golden label 必然错误。

来源：[实际 SFT 目标，第 218 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_sft_lora_20260930/data/sft_observation_openqa.jsonl:218)；另两例在同文件第 322、1008 行。LoRA 与全参数 SFT 使用相同目标数据。

### 9.3 普通交叉熵不会自动审核分析是否合理

SFT 训练时，把目标回答的前面部分作为上下文，让模型预测目标中的下一个 token。预测越符合监督文字，交叉熵越低。

对于上面的矛盾文本，训练仍要求它在分析位置预测 `Eleven`，在答案位置预测 `Thirteen`。模型即使把这段文字学得非常熟，也不能说明它学会了从视频正确数靶子。

因此：**loss 下降说明更会拟合当前目标文字；目标文字是否可靠，决定这种拟合能否帮助实际解题。**

### 9.4 大部分目标 token 来自 analysis

对 1,453 条实际 SFT 目标重新分词：

| 内容 | 平均 tokens | 占完整 assistant 目标累计 tokens 的比例 |
| --- | ---: | ---: |
| analysis 正文 | 99.46 | 86.03% |
| golden answer 正文 | 6.50 | 5.63% |
| 完整目标，含标签等 | 115.61 | 100% |

当前没有对最终答案单独加权。增加 observation 确实让监督更长，但主要增加了分析文字。

这些是文本 token 占比，**不能直接说答案只贡献了 5.63% 的梯度**。不同 token 的预测难度、loss 和梯度可能不同。

### 9.5 模型确实学会了训练文本，但理解收益不足

| SFT | 第 1 轮平均训练 loss | 第 2 轮 | 第 3 轮 |
| --- | ---: | ---: | ---: |
| LoRA | 1.850 | 1.542 | 1.483 |
| 全参数 | 1.507 | 0.763 | 0.370 |

全参数 SFT 第三轮训练 token 准确率约 89.1%。开放式评测的完整格式率也从原模型 20.46% 提升到 100%；LoRA SFT 达到 98.46%。

#### 训练 token 准确率到底是什么？

它统计：**在给定输入和标准回答前缀的条件下，模型概率最高的下一个 token，有多少次恰好等于监督文本中的下一个 token。**

```text
标准目标：<analysis> ... Total = eleven ... </analysis><answer>Thirteen.</answer>

训练预测 answer 中的 Thirteen 时：
输入视频、题目，以及前面整段标准 analysis 都已经给定。
模型不必先独立生成一段正确分析，再来回答。
```

这个过程叫 teacher forcing。程序对 logits 取 `argmax`，把预测与向后错开一个位置的 labels 对齐，忽略 `labels=-100` 的位置，再统计相等比例。本次 `acc_strategy=token`，所以统计的是 token，而不是整道题或整段回答。普通 SFT 的问题、视频和音频输入不是这里要模仿的目标文字；主要统计 assistant 监督文字，包括分析、答案及受监督的格式/结束符。

**教学例子：**一段目标有 100 个受监督 token，99 个预测正确，唯一错误是把最终答案的 `Thirteen` 预测成 `Eleven`。那么 token 准确率仍有 99%，这道题的答案却是错的。这个 100-token 例子不是实际日志中的某条样本。

另外，训练使用标准前缀，实际评测使用模型自己写出的前缀。评测时前面一个错误可能影响后面的回答，而训练 token 准确率没有模拟这种完整的自由生成过程。

| SFT | 第 1 轮日志 token_acc 均值 | 第 2 轮 | 第 3 轮 |
| --- | ---: | ---: | ---: |
| LoRA | 58.24% | 62.41% | 63.37% |
| 全参数 | 63.16% | 77.88% | 89.07% |

这里是各轮训练日志记录值的算术均值，并非重新汇总全部 token 后计算的加权准确率。它能说明模型更会预测当前训练文本，不能解释为“89.1% 的 WorldSense 题目答对”。

实现依据：本机 [准确率计算代码](/opt/huawei/dataset/hyl_ulan/ylhu/third_party/ms-swift/swift/metrics/acc.py:10) 和 [训练器调用](/opt/huawei/dataset/hyl_ulan/ylhu/third_party/ms-swift/swift/trainers/mixin.py:1164)。

#### 完整格式率到底是什么？

它统计：**518 道评测题中，多少道输出通过当前的回答结构检查。**这里的检查要求一个非空且闭合的 `<analysis>...</analysis>`，随后是一个非空且闭合的 `<answer>...</answer>`，答案不能只是一枚 A～D 选项字母，也不能出现多个答案块。

```text
<analysis>The video shows eleven targets.</analysis>
<answer>Eleven.</answer>
```

即便正确答案是 Thirteen，上面仍是“格式完整、答案错误”。完整格式检查也不验证分析是否真实，或是否在 120 词以内。

- 原模型：106/518，20.46%。多数格式不完整案例只是省略了 `</answer>`；有效最终答案实际上有 515/518，因此不能说原模型只有 20.46% 的题给出了答案。
- LoRA SFT 第 3 轮：510/518，98.46%。
- 全参数 SFT 第 3 轮：518/518，100%。

所以，这个指标主要体现“能否按指定结构交卷”。答案语义正确率体现“交上来的答案是否正确”。两者应分别报告。实现见 [输出提取与格式检查](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_code/scripts/score_worldsense_openqa.py:61)。

这说明模型学习到了格式和目标表达。但当前选择题没有显示稳定收益，开放式分数又有裁判误判，不能据此宣称整体理解能力已经提高。

训练只有 train loss，没有独立 validation：`split_dataset_ratio=0`、`eval_strategy=no`。过拟合是合理的待验证解释，但目前缺少验证集曲线，不能把它当成唯一已证实原因。

实现来源：[SFT 数据生成脚本](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_code/scripts/prepare_worldsense_observation_sft.py:33)。

### 9.6 SFT 是否也有 analysis 越训越长的问题？

**目前没有证据支持 SFT 出现了 CLUE 那样持续、明显的长度漂移。全参数 SFT 第 3 轮有轻微变长，LoRA SFT 有少数超长尾部，需要继续监控。**

先区分两种记录：普通 SFT 训练拟合的是固定监督文本，没有像 CLUE 一样每步保存学生自由生成的 rollout。因此，不能根据 SFT train loss、token_acc 或某个 batch 的目标长度，直接声称模型在训练中越生成越长。

能够比较的是不同 checkpoint 在相同题目、相同生成设置下的实际回答。下面使用旧版评测：每个模型同一批 518 题、贪婪解码、生成上限 768 tokens，统计 analysis 的英文词数；包含未闭合 analysis，避免漏掉长回答。

| 模型 | analysis 平均词数 | 中位数 | 第 95 百分位词数 | 超过 120 词 | 近生成上限题数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原模型 | 59.65 | 57 | 94 | 5/518 | 3/518 |
| LoRA SFT 第 1 轮 | 62.30 | 57 | 91 | 5/518 | 5/518 |
| LoRA SFT 第 2 轮 | 55.24 | 52 | 83 | 5/518 | 5/518 |
| LoRA SFT 第 3 轮 | 57.27 | 51 | 88 | 8/518 | 7/518 |
| 全参数 SFT 第 1 轮 | 59.21 | 53 | 97 | 10/518 | 4/518 |
| 全参数 SFT 第 2 轮 | 57.37 | 53 | 90 | 6/518 | 0/518 |
| 全参数 SFT 第 3 轮 | 63.61 | 58 | 104 | 15/518 | 0/518 |

第 95 百分位表示约 95% 的题，其分析不超过该词数。近上限依据解码文本重新分词后至少 766 tokens，属于近似检查，不是直接读取 `finish_reason`。

可以看到：

1. LoRA 的均值和中位数没有逐轮增长。第 3 轮中位数反而比第 1 轮小；不能把少数超长回答当成整体长度漂移。
2. 全参数第 3 轮比第 2 轮长一些，超 120 词的题从 6 增至 15，但平均只有约 64 词，全部题仍有完整答案，没有出现近 768-token 上限的题。
3. SFT 第 3 轮两种模型的严重重复均为 0/518。旧评测中少量 SFT 超长回答不等于 CLUE 的大规模循环。

这轮长度结果为何仍有风险？训练提示只说 concise，没有给 120 词硬要求，且保留全部 observation；实际训练 analysis 有 73/1453 条超过 120 词，最长 658 词。模型可能模仿长说明、检查过程和工具叙述。评测却要求最多 120 词，这使少数样本更容易超长。它是合理解释，尚不能据此确认每个超长输出的具体成因。

应在固定验证题上每轮记录平均、中位数、P95、超 120 词比例、近上限比例、缺答案率和重复率；不要只看平均长度。新一轮使用训练媒体预算和随机解码的开放式结果尚在运行，不能把上表旧协议与新协议混在一起判断漂移。

数据来源：[旧版逐模型长度诊断](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_eval_20261001/analysis_diagnostics_summary.json)，以及各模型的 `openqa/<模型名>/scored.jsonl`。

### 9.7 没有大量重复，SFT 为什么仍没有稳定收益？

**能正常结束回答，是一个条件；能从音视频得出正确答案，是另一个条件。**第 3.3 节的 SFT 回答已经说明这一点：它格式正常、有最终答案，却把掌声次数答错。

| 证据或风险 | 为什么可能限制答题收益 | 目前能下的结论 |
| --- | --- | --- |
| analysis 与 answer 存在真实文本冲突 | 交叉熵要求模型同时模仿矛盾结论；不会自动判断哪一个符合视频 | 数据质量问题已确认，但不能据此把全部失败都归因于标注 |
| analysis 占目标文字 token 的约 86%，答案约 5.6% | 大量监督用于学叙述和格式；总体 loss/准确率不能代表关键答案位置 | 应单独看 answer loss/准确率；比例不等于梯度贡献比例 |
| observation 来自证据定位流程 | 工具检查、计划、猜测和摘要不是天然可靠的解题分析；部分最终状态还会看错 | 需要重新生成并核验 analysis，而非直接拼接已有文本 |
| 只用训练 loss，没有独立验证集 | 对训练文本的拟合提升，可能伴随泛化停滞；不能用最后一轮作为“最佳轮” | 过拟合是待验证解释，现有记录不能证明它是唯一原因 |
| 旧评测媒体细节少于训练 | 数字、OCR、次数和短动作等证据可能看不清；训练中学到的细节线索未必还能使用 | 正在以统一媒体设置重评测，不能提前给各因素分配责任比例 |
| 旧开放式评分存在误判 | 部分同义词、排序和数字答案被误判，会掩盖真实收益 | 应先修尺子，再比较训练方法 |

一个新核实的例子是 `HuRReUbF::task1`：视频 47.5 秒处比分为 22:24，48.25～48.6 秒的最后画面已经变为 28:24；golden answer 是 28:24，但训练 observation 写的是 22:24。即使区间看过、Gold 筛选模型答对，也不保证用于 SFT 的 observation 正确。视频核验截图和完整标注质量分析见 [任务 B 报告](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_code/docs/WORLDSENSE_ANNOTATION_QUALITY_PIPELINE_TASK_B_20261003.zh-CN.md)。

当前重新评测的第 3 轮选择题中，原模型为 235/518，LoRA SFT 为 247/518，全参数 SFT 为 232/518。LoRA 有小幅正向变化，不能再概括为“SFT 完全没效果”；但单一随机种子、尚未完成的各轮开放式结果，也不足以宣布稳定提升。这些结果使用新的媒体/解码/判分协议，不能与第 13 节旧表直接相减来归因。

### 9.8 具体怎样改进 SFT？

**第一步：先修数据与评测。**审核分析和答案是否一致，检查关键计数、末帧状态、方向及事件顺序。把 observation 改写成可核实、与题目相关的简短分析；不保留“下一步调用工具”“选项比较”等标注过程。选项中的事件符号必须有题内映射，没有映射的题先隔离。建议 analysis 先控制在 30～80 词、最多 120 词；超长文本应按证据重写和复核，不能直接截掉前 120 词后拼一个答案。

**第二步：让答案位置获得明确监督。**当前默认按受监督 token 计算交叉熵。可以尝试按区段分别取均值：

```text
loss = answer 区段平均交叉熵
     + β × analysis 区段平均交叉熵
     + γ × 格式标签及结束符的平均交叉熵
```

例如先试 β=0.5、γ=0.1，属于待验证实验建议。按区段取均值能避免长 analysis 仅因 token 多而在总量上占优势；但加权不能修复矛盾标签，也不保证准确率提高。还应分别记录 answer loss、answer token_acc，最终仍以自由生成的答题正确率决定效果。

**第三步：建立真正的验证与选模机制。**从开发数据按视频 ID 划出独立验证集，避免同一视频的不同题跨训练和验证。每轮自由生成回答，用固定媒体预算、提示、长度和评分规则测答案正确率、完整率、超长率、缺答案率和重复率。按验证集表现选 checkpoint，而不是只看 train loss 或默认选第 3 轮；不要把最终留出评测集反复用作调参集。

**第四步：逐项做小规模对照。**先比较原目标与“清理后 analysis+answer”，再在清理后的数据上单独比较默认 loss 与区段加权 loss。另做 answer-only 诊断，观察去掉标准 analysis 前缀后能否更好学习视频到答案的映射。保持样本、视频预算、提示、训练步数和评测集一致；改变输出形式本身也是实验因素，应单独说明。若清理后验证收益仍很快停滞，再测试较小学习率、较早 checkpoint 或更谨慎的解冻范围，不优先追加 epoch。

以上为改进方案。本次没有据此改动训练配置或启动新 SFT；现有评测继续执行。

---

## 10. 训练和评测的多媒体设置不一样吗？

**是的。学生都接收完整时间范围的视频和音频，但视觉空间预算明显不同。**还应区分学生完整视频与教师 clue 片段：教师只看 clue 是实验设计本身，不是漏输入。

### 10.1 四种设置分别控制什么？

| 设置 | 易懂解释 |
| --- | --- |
| 视频时间范围 | 例如看 0～80 秒，或只看 15～20 秒片段。 |
| 抽帧数量 / fps | 从这个时间范围选多少张画面。fps=2 表示目标约每秒选 2 帧；还会受帧数上限和处理器规则影响。 |
| 每帧分辨率 / max_pixels | 每张抽出的画面有多少空间细节。宽高越小，细字、小物体等通常越难分辨。 |
| 视觉 token 预算 | 把抽出的画面编码成模型输入后，占多少 token。帧数和分辨率都影响它。 |

“完整视频”指覆盖完整时间范围，**不是读取原视频每一帧、也不是保留原始高清分辨率**。

### 10.2 实际参数对比

以下训练列描述学生的完整视频输入；教师 clue 输入是另一个视图。

| 项目 | 本轮训练 | 本轮选择题 / 开放式评测 |
| --- | --- | --- |
| 目标时间采样 | fps=2，动态确定 nframes，上限 300 帧 | fps=2，上限 768 帧 |
| 平均抽帧数 | 166.37 | 约 173.07 |
| 每帧 max_pixels | 156,800～197,568，均值约 182,442 | 28,672 |
| 常见帧尺寸 | 280×560 或 336×588 | 112×224，占 431/518 题 |
| 平均视觉 tokens | 保存预算约 18,688 | 推算约 2,749 |
| 音频随视频输入 | 开启 | 开启 |

训练表中的 300 帧是配置上限。当前动态算法还受视觉预算限制：最多 24,000 个视觉 tokens，每帧至少预留 100 个，所以本轮实际训练样本最多用了 240 帧。训练实际帧数为 64～240；部分较长视频实际采样率低于目标 2 fps。这些帧仍在完整时间范围内均匀抽取。

训练视觉 tokens 来自每条样本保存的动态预算；评测帧数、尺寸和 tokens 依据冻结元数据与当前处理器 resize/temporal merge 规则推算，**不是逐次 forward 保存的实测**。

注意：28,672 是一个分辨率上限，不表示实际每张画面都恰好有 28,672 像素。处理器还会按宽高比和尺寸对齐规则调整，所以常见实际尺寸是 112×224。

### 10.3 抽出的帧数差不多，为什么视觉 tokens 差这么多？

因为每一帧变小了。

例如，336×588 的画面有 197,568 个像素；112×224 有 25,088 个像素。后者空间像素约为前者的 1/7.9。

可以把它理解成：训练时看一张较清楚的照片，评测时看同一类场景的缩略图。时间上仍能看到很多帧，但每帧的细字、表情、器具细节可能被缩小。

视觉 token 数也受帧数、网格与时间合并规则影响，因此不能单纯用像素比精确推出整体 token 比；实际预算汇总显示评测视觉 tokens 显著减少。

用这个模型的网格和时间合并方式，可以做一个便于理解的预算估算：

```text
视觉 tokens ≈ (帧数 ÷ 2) × (高度 ÷ 28) × (宽度 ÷ 28)

如果都输入 200 帧：
280×560 的图片：约 100 × 10 × 20 = 20,000 tokens
112×224 的图片：约 100 ×  4 ×  8 =  3,200 tokens
```

这是固定帧数下的示例，不是本次某道题的实际输入统计。音频读取范围不会因为图片缩小而按这个比例缩短。

### 10.4 音频是不是没有输入？

训练与评测都设置 `USE_AUDIO_IN_VIDEO=1`，音频处理按视频时间范围读取，并重采样到 16kHz。

- SFT 和 CLUE 学生训练输入：从 0 秒到完整视频时长，本轮训练视频均在 300 秒以内。
- 评测输入：默认从 0 秒读到视频结束，留出视频也限制在 300 秒以内。
- CLUE 教师：只读 golden clue 对应区间的视频和音频。

CLUE 首批输入审计记录了学生和教师的非空 `input_features`，可以确认实际传入了音频特征。这个检查每个 rank 只做第一次，不能代替逐条确认音频是否静音、掩码是否有效或语义识别是否正确。

日志中某些特征张量有固定 padding 尺寸，不能仅凭这个形状就认为每道题都输入了 300 秒真实音频。

### 10.5 这种差异会影响什么结论？

所有模型在本次评测中使用同一套媒体输入预算。因此，结果可以描述“在这套低空间预算的评测条件下，各模型表现如何”。

但训练用了明显更大的空间预算，当前评测不能充分说明模型在与训练相近的输入条件下能达到什么效果。

低分辨率可能影响小物体、计数、文字、表情等问题。它是否导致训练模型相对原模型表现不佳，需要把评测预算提高到与训练接近后，对同一小批题做比较。

这个对照中，原模型和所有被比较的训练模型都需要一起使用新的媒体预算，才能继续保持可比性。

**它不能单独解释 CLUE 大量循环。**因为原模型与 SFT 使用同样的评测媒体预算，重复远少于 CLUE；还存在解码、训练目标、样本分布等因素。

来源：[训练样本中的媒体预算](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_sft_lora_20260930/data/sft_observation_openqa.jsonl:1)、[评测媒体输入](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_eval_20261001/data/worldsense.openqa.jsonl:1)。

---

## 11. 几个“长度”参数不要混淆

| 参数 / 概念 | 本轮设置 | 控制什么 |
| --- | --- | --- |
| 视频最长时长 | 300 秒以内 | 多媒体原始时间长度，不是回答长度。 |
| 抽帧数量 / 帧数上限 | 训练使用动态 nframes，配置上限 300；评测 max_frames=768 | 视频抽帧数上限，不是生成文字 token 数；本轮训练实际最多 240 帧。 |
| max_pixels | 训练动态较高；评测 28,672 | 单帧空间预算，不是模型最大上下文。 |
| max_length | 32,768 | 训练/编码序列长度设置，需要容纳音视频表示、提示及回答等；还配合动态预算和预留检查。 |
| max_completion_length | CLUE 训练 512 | 一次训练 rollout 最多生成的 token 数。 |
| max_new_tokens | 开放式评测 768；选择题评测 8 | 一次评测最多新生成的 token 数。 |
| analysis 的 120 words | CLUE 与开放式评测提示要求 | 对分析部分的英文词数要求，目前没有硬性生成约束。 |

因此，`max_length=32768` 不表示模型可以在这次评测里输出 32,768 tokens。开放式评测实际输出仍受 `max_new_tokens=768` 限制。

SFT 训练提示只要求 concise analysis，没有明确 120 词上限，目标保留了完整 observation；评测则要求最多 120 词。这是另一项提示/目标分布差异。

---

## 12. 评分错误为什么也会影响“训练没效果”的判断？

### 12.1 当前只评最终答案的语义，不评整段分析的真实性

当前流程：提取 `<answer>` → 标准化精确匹配 → 必要时由固定原始 Qwen2.5-Omni-7B 比较题目、标准答案和最终答案，输出 YES/NO。

裁判不看视频，也不看 candidate analysis。每题得 0 或 1 分，分母始终为 518。分析格式和词数另外报告。

所以，即使一段分析写得很像认真推理，也不能仅凭这种形式认为它真实可靠。

### 12.2 实际示例：正确事件顺序被判错

样本 `IWWlNWAT::task0`，题目列出了五个动作，a～e 是动作编号：

```text
a. Throwing something at a hat.
b. Handing a book to the other hand.
c. Putting on a hat.
d. Writing on the book.
e. Bumping fists.
```

标准答案：`acbde.`，即 a → c → b → d → e。

| 模型 | 实际最终顺序 | 当前裁判判分 | 与参考是否一致 |
| --- | --- | --- | --- |
| 原模型 | c → b → d → a → e | YES | 不一致，错判为对 |
| LoRA SFT 第 3 轮 | a → c → b → d → e | NO | 一致，错判为错 |

SFT 实际最终答案是：

```text
<answer>a, c, b, d, e.</answer>
```

这和 `acbde.` 表达相同顺序。普通文本标准化未把它们识别为同一事件排列，交给语言模型裁判后又发生误判。对这种题应该提取完整事件序列后用规则比较。

这里的 a～e 是题目里的事件标签，不是模型只回答了一个选择题选项字母。

来源：[原模型同题输出，第 83 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_eval_20261001/openqa/base/scored.jsonl:83)、[SFT 同题输出，第 83 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_eval_20261001/openqa/sft_lora_epoch3/scored.jsonl:83)。

### 12.3 66/66 校验为什么没发现这些问题？

校验取同一批 33 道题，每题给裁判一次标准答案原文、一次错误选项原文，共 66 条。13 项模型评测重复的是这同一套探针。

这能检查一些明显的接受/拒绝行为，但没有充分覆盖真实回答的改写、错序、遗漏关键事实等情况。校验通过不能证明所有真实预测都判得准确。

保守排序子集审计中，24 条可解析预测已有 7 处明确误判；这个子集不是随机抽样，不能外推整体误判率。

因此，现有开放式小幅分差需要复核。CLUE 的大量缺答案则可直接从原始输出确认，不依赖裁判正确与否。

---

## 13. 当前结果应该怎样总结？

第三轮结果：

| 模型 | 选择题正确率 | 开放式当前裁判分（待复核） | 无有效最终答案 |
| --- | ---: | ---: | ---: |
| 原模型 | 45.75% | 23.36% | 3/518 |
| LoRA SFT | 43.05% | 21.62% | 8/518 |
| 全参数 SFT | 44.02% | 23.36% | 0/518 |
| CLUE，含 observation | 44.40% | 6.56% | 349/518 |
| CLUE，不含 observation | 45.95% | 5.60% | 412/518 |

选择题与开放式题提供的信息不同：选择题提供候选答案并要求选一个字母；开放式需要自行组织分析和答案。不能直接把两种模式的百分比相减，认为差值全是分析造成的。

### 已确认

- CLUE 在本轮开放式评测中大量重复分析，没有最终答案，且随 epoch 加重。
- CLUE 训练中平均生成长度、超 120 词比例及缺答案比例上升；SFT 的逐轮长度证据见第 9.6 节，不能混用。
- 部分 SFT analysis 与 golden answer 互相矛盾。
- SFT 回答格式明显改善，训练 loss 显著下降，但未显示稳定答题收益。
- 训练/评测的解码方式和视觉空间预算不同。
- 开放式裁判存在明确误判。

### 仍需实验验证

- 贪婪解码是否是触发 CLUE 循环的主要因素。
- 教师在坏前缀上是否提供了不足或错误的纠错反馈。
- EMA 教师是否跟随了不理想的学生行为。
- 媒体预算变化对各模型正确率的具体影响。
- SFT 有多少问题来自过拟合，有多少来自目标噪声或分布变化。

训练/评测筛选分布也不同：训练 Tier A/B/D 为 279/393/781；评测为 9/11/498。评测 96.14% 为 Tier D。Tier 是按 clue 相对完整视频的模型增益定义，不是人工难度等级，不能直接说 D 更难。

---

## 14. 下一步应怎样有顺序地排查？

### 第一步：把评测尺子校准

对数字、事件排序等可规则判断的答案，优先用规则；对真实语义改写做人工抽查。分别报告答案正确性、回答完整率、重复率和格式率。

目的：先保证分数能可靠反映实际回答，避免正确回答被错判。

### 第二步：用小批题区分解码和媒体因素

对同一 checkpoint、同一批视频和问题，分别做：

1. 保持媒体输入不变，仅比较贪婪解码与训练使用的采样方式；
2. 保持解码不变，仅比较当前低空间预算与接近训练的动态媒体预算。

每次只改变一个因素，记录正确率、重复率、缺答案率和长度。

目的：确定“是哪一项变化触发问题”，而不是同时改多个参数后无法解释结果。

### 第三步：清理训练目标

逐条检查 observation 是否与 golden answer 一致，删除或重新审核猜测、选项比较、检查计划和明显矛盾。保留能由视频/音频支持、与最终答案一致的分析。

目的：让更长的监督提供可靠证据和推理，而不是更多不一致的文字。

### 第四步：让训练指标覆盖真正关心的结果

建立独立验证集，监控最终答案正确率、回答完整率和重复率。SFT 分开统计 analysis/answer loss，并考虑给答案单独权重；CLUE 检查教师反馈和缺答案轨迹的处理。

一个实现细节：当前 Swift 仅在非 STUDENT 分支加入 `sft_alpha × CE`。所以保持 `lmbda=1`，单把 `sft_alpha` 改成正数，并不会自动得到 golden answer 的直接监督。若以后希望纯学生 rollout 加上 gold 锚点，需要显式实现额外的 gold 序列损失。

**本文提出排查顺序，不代表这些改动已经执行或已经被证明能提高效果。**

---

## 15. 一句话记住每个概念

| 概念 | 一句话解释 |
| --- | --- |
| 没有最终答案 | 模型还在分析里，停止时仍未写出 `<answer>` 和答案内容。 |
| 重复分析 | 相同描述不断出现，却没有增加解决题目所需的信息。 |
| 贪婪解码 | 每一步都选择当前概率最高的 token。 |
| 长度漂移 | 上限不变，但模型随着训练越来越倾向生成长回答。 |
| loss 下降 | 模型越来越符合当前训练目标；还需要确认目标质量及验证集效果。 |
| 多媒体预算差异 | 时间覆盖、帧数、分辨率和输入 token 数都需要分别检查。 |
| 教师有 golden answer | 教师知道参考答案，但学生是否得到有效纠错，还取决于训练目标和教师反馈。 |

**本次最需要优先解决的三个问题：评分可靠性、监督文本一致性、CLUE 的回答完整性。**

## 16. 相关代码和结果入口

- [全部模型的评测结果表](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_eval_20261001/comparison/table.csv)
- [开放式固定提示与生成上限](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_eval_20261001/data/openqa_manifest.json)
- [开放式评分代码](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_code/scripts/score_worldsense_openqa.py)
- [生成质量诊断代码](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_code/scripts/diagnose_worldsense_openqa_outputs.py)
- [本轮 CLUE 全参数启动配置](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_code/scripts/run_worldsense_openqa_thinking_full_npu_w1w2.sh)
- [本轮 SFT 目标生成代码](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_code/scripts/prepare_worldsense_observation_sft.py)
- [全参数 SFT 训练日志](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_sft_full_npu96_w6w7_20260930/outputs/formal/v0-20260930-180035/logging.jsonl)
- [LoRA SFT 训练日志](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_sft_lora_20260930/outputs/formal/v0-20260930-164521/logging.jsonl)
