# WorldSense：3 组真实 SFT 与 CLUE-OPSD 训练样本

本文从本次 observation+answer SFT 和开放题+thinking CLUE-OPSD 的正式训练 JSONL 中直接摘录，同一道题并排展示两种训练方式，便于比较。英文 prompt 与 SFT 监督回答均保持原文，没有改写、缩写或补造模型回答。

SFT 的 LoRA 与全参数实验，在这 3 道题上的 `messages` 和 `videos` 完全一致。CLUE-OPSD 的两个全参数实验使用相同的 student prompt，teacher 分为“golden clue + golden answer + observation”和“golden clue + golden answer”两个版本，下面均完整列出。

这些代码块是数据文件中的内容层 prompt。模型模板添加的角色标记、可能的默认 system 内容，以及多媒体编码后的 token 序列没有在本文展开。`<video>` 是多媒体占位符，实际文件和时间区间由 `videos` 或 `teacher_videos` 提供；golden clue 并不是额外复制到 prompt 里的文字。训练配置启用了视频内音频。

| 数据部分 | 模型能看到什么 | 用途 |
| --- | --- | --- |
| SFT user prompt | 完整视频/音频 + 问题 + 输出要求 | 输入 |
| SFT assistant target | observation 改写后的 analysis + 正确选项内容 | 固定监督目标，使用 teacher forcing |
| CLUE student prompt | 完整视频/音频 + 问题 + thinking 要求 | 学生据此生成回答 |
| CLUE teacher prompt（带 observation） | clue 片段 + 标准答案 + observation + thinking 要求 | 教师在学生生成的相同前缀上提供分布监督 |
| CLUE teacher prompt（不带 observation） | clue 片段 + 标准答案 + thinking 要求 | 对照实验的教师输入 |

这两个 CLUE 实验采用 `lmbda=1.0、sft_alpha=0`，沿学生实际生成的回答训练。因此不能把下面的 SFT assistant target 当成 CLUE 的固定训练 completion，也不能把 teacher prompt 中的 observation 当成教师实际生成的回答。学生每一步生成的内容会变化，本文只展示数据中真实保存的 prompt。

本批 SFT 提示要求 concise analysis，没有明确 120 词限制；CLUE 提示明确要求最多 120 个英文词，学生生成上限为 512 tokens。120 词与 512 tokens 是两个不同单位。


以下每段英文原文后附中文翻译，仅供阅读对照；实际训练数据仍为英文。翻译保留 `<video>`、`<analysis>`、`<answer>` 等标签，以及原文中“用英语、最多 120 个英文词”的要求。原始数据中的措辞残留、数字写法和分析与答案冲突也按原样保留；“已核实的正确答案”是对 prompt 字段名的翻译，不代表本次翻译进行了新的核验。

## Case 1：故事推断：Winnie 下一步会做什么？

- `case_id`：`AAWgrzYx::task0`
- 标准答案：`Withdraw money from an ATM.`
- 视频：[原视频](/opt/huawei/dataset/hyl_ulan/ylhu/LongOmniRL/benchmarks/WorldSense/videos/AAWgrzYx.mp4)
- SFT 原始行：[第 102 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_sft_lora_20260930/data/sft_observation_openqa.jsonl:102)
- CLUE 带 observation 原始行：[第 102 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_openqa_thinking_full_observation_20260930/data/clue_openqa_thinking.jsonl:102)
- CLUE 不带 observation 原始行：[第 102 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_clue_full_no_observation_npu96_w10w11_20260930/data/clue_openqa_thinking.jsonl:102)

### 多媒体输入

SFT 与 CLUE student 使用的完整视频视图：

| 视图编号 | 原视频区间（秒） | 抽样帧数 | 帧尺寸（高 × 宽） |
| --- | --- | ---: | --- |
| 1 | 0.0～131.36 | 240 | 280 × 560 |

CLUE teacher 使用的 golden clue 视图，按表中顺序对应 teacher prompt 的各个 `<video>`：

| 视图编号 | 原视频区间（秒） | 抽样帧数 | 帧尺寸（高 × 宽） |
| --- | --- | ---: | --- |
| 1 | 60.0～75.0 | 30 | 336 × 588 |
| 2 | 94.0～100.0 | 12 | 336 × 588 |

### SFT：user prompt 原文

```text
<video>
Question: What is Winnie most likely to do next?
Provide a concise analysis of the video and audio evidence relevant to the question inside <analysis>...</analysis>, then give the answer in natural language inside <answer>...</answer>. Do not output an option letter or refer to answer options.
```

**中文翻译：**

```text
<video>
问题：Winnie 接下来最有可能做什么？
请在 <analysis>...</analysis> 中简要分析与问题相关的视频和音频证据，然后在 <answer>...</answer> 中用自然语言给出答案。不要输出选项字母，也不要提及答案选项。
```

### SFT：assistant 监督答案原文

```text
<analysis>
The video is a narrated story about Winnie. The question asks what she is 'most likely to do next'. Based on the narrative flow established in the inspection of 60s-75s, Winnie leaves a clothing store because she lacks cash and explicitly heads to an ATM. The subsequent scenes (94s+) show her at the ATM entering her PIN. Therefore, the immediate action following her decision to leave the shop and go to the machine is withdrawing money. This aligns with the answer "Withdraw money from an ATM".
</analysis>
<answer>Withdraw money from an ATM.</answer>
```

**中文翻译：**

```text
<analysis>
视频讲述了一个关于 Winnie 的故事，并配有旁白。问题问的是她“接下来最有可能做什么”。根据检查 60～75 秒片段时确认的故事发展，Winnie 因为现金不足而离开服装店，并明确前往自动取款机。随后（94 秒之后）的画面显示，她正在自动取款机前输入密码。因此，她决定离开商店并前往取款机之后，紧接着的行动就是取钱。这与答案“从自动取款机取钱”一致。
</analysis>
<answer>从自动取款机取钱。</answer>
```

### CLUE-OPSD：student prompt 原文

```text
<video>
Question: What is Winnie most likely to do next?
Briefly analyze the video and audio evidence in English using at most 120 words. Write your analysis inside <analysis>...</analysis>, then give the answer in natural language inside <answer>...</answer>. Do not output an option letter.
```

**中文翻译：**

```text
<video>
问题：Winnie 接下来最有可能做什么？
请用英语简要分析视频和音频证据，最多使用 120 个英文词。将分析写在 <analysis>...</analysis> 中，然后在 <answer>...</answer> 中用自然语言给出答案。不要输出选项字母。
```

### CLUE-OPSD：teacher prompt 原文（带 observation）

```text
<video>
<video>
Question: What is Winnie most likely to do next?
Verified correct answer: Withdraw money from an ATM.
Reference evidence observation:
The video is a narrated story about Winnie. The question asks what she is 'most likely to do next'. Based on the narrative flow established in the inspection of 60s-75s, Winnie leaves a clothing store because she lacks cash and explicitly heads to an ATM. The subsequent scenes (94s+) show her at the ATM entering her PIN. Therefore, the immediate action following her decision to leave the shop and go to the machine is withdrawing money. This aligns with the answer "Withdraw money from an ATM".
Use this observation as an auxiliary reference together with the provided clue video and audio. Base your analysis on the available evidence and the verified correct answer. Do not treat the reference as the student response or simply copy it.
Briefly analyze the video and audio evidence in English using at most 120 words. Write your analysis inside <analysis>...</analysis>, then give the answer in natural language inside <answer>...</answer>. Do not output an option letter.
```

**中文翻译：**

```text
<video>
<video>
问题：Winnie 接下来最有可能做什么？
已核实的正确答案：从自动取款机取钱。
参考证据观察说明：
视频讲述了一个关于 Winnie 的故事，并配有旁白。问题问的是她“接下来最有可能做什么”。根据检查 60～75 秒片段时确认的故事发展，Winnie 因为现金不足而离开服装店，并明确前往自动取款机。随后（94 秒之后）的画面显示，她正在自动取款机前输入密码。因此，她决定离开商店并前往取款机之后，紧接着的行动就是取钱。这与答案“从自动取款机取钱”一致。
请结合提供的线索视频和音频，将这段观察说明作为辅助参考。依据现有证据和已核实的正确答案进行分析。不要把参考内容当作学生的回答，也不要直接照抄。
请用英语简要分析视频和音频证据，最多使用 120 个英文词。将分析写在 <analysis>...</analysis> 中，然后在 <answer>...</answer> 中用自然语言给出答案。不要输出选项字母。
```

### CLUE-OPSD：teacher prompt 原文（不带 observation）

```text
<video>
<video>
Question: What is Winnie most likely to do next?
Verified correct answer: Withdraw money from an ATM.
Briefly analyze the video and audio evidence in English using at most 120 words. Write your analysis inside <analysis>...</analysis>, then give the answer in natural language inside <answer>...</answer>. Do not output an option letter.
```

**中文翻译：**

```text
<video>
<video>
问题：Winnie 接下来最有可能做什么？
已核实的正确答案：从自动取款机取钱。
请用英语简要分析视频和音频证据，最多使用 120 个英文词。将分析写在 <analysis>...</analysis> 中，然后在 <answer>...</answer> 中用自然语言给出答案。不要输出选项字母。
```

这个案例的 teacher prompt 开头有两个 `<video>`，分别对应 60～75 秒、94～100 秒两个片段；student 和 SFT 则各用一个完整视频视图。

## Case 2：画面文字识别：维修人员的联系电话

- `case_id`：`ABXtrFIK::task0`
- 标准答案：`755716155.`
- 视频：[原视频](/opt/huawei/dataset/hyl_ulan/ylhu/LongOmniRL/benchmarks/WorldSense/videos/ABXtrFIK.mp4)
- SFT 原始行：[第 1243 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_sft_lora_20260930/data/sft_observation_openqa.jsonl:1243)
- CLUE 带 observation 原始行：[第 1243 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_openqa_thinking_full_observation_20260930/data/clue_openqa_thinking.jsonl:1243)
- CLUE 不带 observation 原始行：[第 1243 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_clue_full_no_observation_npu96_w10w11_20260930/data/clue_openqa_thinking.jsonl:1243)

### 多媒体输入

SFT 与 CLUE student 使用的完整视频视图：

| 视图编号 | 原视频区间（秒） | 抽样帧数 | 帧尺寸（高 × 宽） |
| --- | --- | ---: | --- |
| 1 | 0.0～242.6 | 240 | 280 × 560 |

CLUE teacher 使用的 golden clue 视图，按表中顺序对应 teacher prompt 的各个 `<video>`：

| 视图编号 | 原视频区间（秒） | 抽样帧数 | 帧尺寸（高 × 宽） |
| --- | --- | ---: | --- |
| 1 | 238.0～242.0 | 8 | 336 × 588 |

### SFT：user prompt 原文

```text
<video>
Question: Which of the following phone numbers can be used to contact the mechanic shown in the video?
Provide a concise analysis of the video and audio evidence relevant to the question inside <analysis>...</analysis>, then give the answer in natural language inside <answer>...</answer>. Do not output an option letter or refer to answer options.
```

**中文翻译：**

```text
<video>
问题：以下哪个电话号码可以用来联系视频中的维修人员？
请在 <analysis>...</analysis> 中简要分析与问题相关的视频和音频证据，然后在 <answer>...</answer> 中用自然语言给出答案。不要输出选项字母，也不要提及答案选项。
```

### SFT：assistant 监督答案原文

```text
<analysis>
The inspection confirms the phone number displayed on the end card is '07 5571 6155'. This matches the answer "755716155".
</analysis>
<answer>755716155.</answer>
```

**中文翻译：**

```text
<analysis>
检查确认，片尾画面上显示的电话号码是“07 5571 6155”。这与答案“755716155”一致。
</analysis>
<answer>755716155.</answer>
```

### CLUE-OPSD：student prompt 原文

```text
<video>
Question: Which of the following phone numbers can be used to contact the mechanic shown in the video?
Briefly analyze the video and audio evidence in English using at most 120 words. Write your analysis inside <analysis>...</analysis>, then give the answer in natural language inside <answer>...</answer>. Do not output an option letter.
```

**中文翻译：**

```text
<video>
问题：以下哪个电话号码可以用来联系视频中的维修人员？
请用英语简要分析视频和音频证据，最多使用 120 个英文词。将分析写在 <analysis>...</analysis> 中，然后在 <answer>...</answer> 中用自然语言给出答案。不要输出选项字母。
```

### CLUE-OPSD：teacher prompt 原文（带 observation）

```text
<video>
Question: Which of the following phone numbers can be used to contact the mechanic shown in the video?
Verified correct answer: 755716155.
Reference evidence observation:
The inspection confirms the phone number displayed on the end card is '07 5571 6155'. This matches the answer "755716155".
Use this observation as an auxiliary reference together with the provided clue video and audio. Base your analysis on the available evidence and the verified correct answer. Do not treat the reference as the student response or simply copy it.
Briefly analyze the video and audio evidence in English using at most 120 words. Write your analysis inside <analysis>...</analysis>, then give the answer in natural language inside <answer>...</answer>. Do not output an option letter.
```

**中文翻译：**

```text
<video>
问题：以下哪个电话号码可以用来联系视频中的维修人员？
已核实的正确答案：755716155.
参考证据观察说明：
检查确认，片尾画面上显示的电话号码是“07 5571 6155”。这与答案“755716155”一致。
请结合提供的线索视频和音频，将这段观察说明作为辅助参考。依据现有证据和已核实的正确答案进行分析。不要把参考内容当作学生的回答，也不要直接照抄。
请用英语简要分析视频和音频证据，最多使用 120 个英文词。将分析写在 <analysis>...</analysis> 中，然后在 <answer>...</answer> 中用自然语言给出答案。不要输出选项字母。
```

### CLUE-OPSD：teacher prompt 原文（不带 observation）

```text
<video>
Question: Which of the following phone numbers can be used to contact the mechanic shown in the video?
Verified correct answer: 755716155.
Briefly analyze the video and audio evidence in English using at most 120 words. Write your analysis inside <analysis>...</analysis>, then give the answer in natural language inside <answer>...</answer>. Do not output an option letter.
```

**中文翻译：**

```text
<video>
问题：以下哪个电话号码可以用来联系视频中的维修人员？
已核实的正确答案：755716155.
请用英语简要分析视频和音频证据，最多使用 120 个英文词。将分析写在 <analysis>...</analysis> 中，然后在 <answer>...</answer> 中用自然语言给出答案。不要输出选项字母。
```

原问题仍保留 “Which of the following” 的选择题措辞，但 prompt 中已经没有列出选项。电话号码及其标点按原数据保留。这里展示的是现有训练数据，不能把措辞残留理解为已完成开放题质量核验。

## Case 3：最终比分：包含 analysis 与答案冲突的实际样本

- `case_id`：`HuRReUbF::task1`
- 标准答案：`28 to 24.`
- 视频：[原视频](/opt/huawei/dataset/hyl_ulan/ylhu/LongOmniRL/benchmarks/WorldSense/videos/HuRReUbF.mp4)
- SFT 原始行：[第 1282 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_sft_lora_20260930/data/sft_observation_openqa.jsonl:1282)
- CLUE 带 observation 原始行：[第 1282 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_openqa_thinking_full_observation_20260930/data/clue_openqa_thinking.jsonl:1282)
- CLUE 不带 observation 原始行：[第 1282 行](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_clue_full_no_observation_npu96_w10w11_20260930/data/clue_openqa_thinking.jsonl:1282)

### 多媒体输入

SFT 与 CLUE student 使用的完整视频视图：

| 视图编号 | 原视频区间（秒） | 抽样帧数 | 帧尺寸（高 × 宽） |
| --- | --- | ---: | --- |
| 1 | 0.0～48.715382048715384 | 96 | 336 × 588 |

CLUE teacher 使用的 golden clue 视图，按表中顺序对应 teacher prompt 的各个 `<video>`：

| 视图编号 | 原视频区间（秒） | 抽样帧数 | 帧尺寸（高 × 宽） |
| --- | --- | ---: | --- |
| 1 | 46.0～48.715 | 4 | 336 × 588 |

### SFT：user prompt 原文

```text
<video>
Question: What is the final score shown in the video?
Provide a concise analysis of the video and audio evidence relevant to the question inside <analysis>...</analysis>, then give the answer in natural language inside <answer>...</answer>. Do not output an option letter or refer to answer options.
```

**中文翻译：**

```text
<video>
问题：视频中显示的最终比分是多少？
请在 <analysis>...</analysis> 中简要分析与问题相关的视频和音频证据，然后在 <answer>...</answer> 中用自然语言给出答案。不要输出选项字母，也不要提及答案选项。
```

### SFT：assistant 监督答案原文

```text
<analysis>
The video shows a football game. The score updates throughout the clip. At the end of the inspected segment (around 47-48s), the score graphic clearly displays 'TEXAS 22 MARYLAND 24'. This matches the answer "22 to 24".
</analysis>
<answer>28 to 24.</answer>
```

**中文翻译：**

```text
<analysis>
视频展示了一场美式橄榄球比赛。比分在片段播放过程中不断更新。在所检查片段的末尾（约 47～48 秒），比分图明确显示“TEXAS 22 MARYLAND 24”（得克萨斯队 22 分，马里兰队 24 分）。这与答案“22 比 24”一致。
</analysis>
<answer>28 比 24。</answer>
```

### CLUE-OPSD：student prompt 原文

```text
<video>
Question: What is the final score shown in the video?
Briefly analyze the video and audio evidence in English using at most 120 words. Write your analysis inside <analysis>...</analysis>, then give the answer in natural language inside <answer>...</answer>. Do not output an option letter.
```

**中文翻译：**

```text
<video>
问题：视频中显示的最终比分是多少？
请用英语简要分析视频和音频证据，最多使用 120 个英文词。将分析写在 <analysis>...</analysis> 中，然后在 <answer>...</answer> 中用自然语言给出答案。不要输出选项字母。
```

### CLUE-OPSD：teacher prompt 原文（带 observation）

```text
<video>
Question: What is the final score shown in the video?
Verified correct answer: 28 to 24.
Reference evidence observation:
The video shows a football game. The score updates throughout the clip. At the end of the inspected segment (around 47-48s), the score graphic clearly displays 'TEXAS 22 MARYLAND 24'. This matches the answer "22 to 24".
Use this observation as an auxiliary reference together with the provided clue video and audio. Base your analysis on the available evidence and the verified correct answer. Do not treat the reference as the student response or simply copy it.
Briefly analyze the video and audio evidence in English using at most 120 words. Write your analysis inside <analysis>...</analysis>, then give the answer in natural language inside <answer>...</answer>. Do not output an option letter.
```

**中文翻译：**

```text
<video>
问题：视频中显示的最终比分是多少？
已核实的正确答案：28 比 24。
参考证据观察说明：
视频展示了一场美式橄榄球比赛。比分在片段播放过程中不断更新。在所检查片段的末尾（约 47～48 秒），比分图明确显示“TEXAS 22 MARYLAND 24”（得克萨斯队 22 分，马里兰队 24 分）。这与答案“22 比 24”一致。
请结合提供的线索视频和音频，将这段观察说明作为辅助参考。依据现有证据和已核实的正确答案进行分析。不要把参考内容当作学生的回答，也不要直接照抄。
请用英语简要分析视频和音频证据，最多使用 120 个英文词。将分析写在 <analysis>...</analysis> 中，然后在 <answer>...</answer> 中用自然语言给出答案。不要输出选项字母。
```

### CLUE-OPSD：teacher prompt 原文（不带 observation）

```text
<video>
Question: What is the final score shown in the video?
Verified correct answer: 28 to 24.
Briefly analyze the video and audio evidence in English using at most 120 words. Write your analysis inside <analysis>...</analysis>, then give the answer in natural language inside <answer>...</answer>. Do not output an option letter.
```

**中文翻译：**

```text
<video>
问题：视频中显示的最终比分是多少？
已核实的正确答案：28 比 24。
请用英语简要分析视频和音频证据，最多使用 120 个英文词。将分析写在 <analysis>...</analysis> 中，然后在 <answer>...</answer> 中用自然语言给出答案。不要输出选项字母。
```

**本例有已确认的监督文本冲突：**SFT analysis 写 `22 to 24`，assistant answer 却是 `28 to 24`；带 observation 的 teacher prompt 同时包含这两个不一致的信息。不带 observation 的 teacher prompt 只保留 `Verified correct answer: 28 to 24.`。这些原文没有在本次导出中修正。

此前视频复核发现：较早的 47.5 秒画面为 22:24，随后最后画面更新为 28:24。具体截图与核验说明见 [标注质量报告](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_code/docs/WORLDSENSE_ANNOTATION_QUALITY_PIPELINE_TASK_B_20261003.zh-CN.md)。

## 数据来源与核对方式

导出时逐项核对了三个 case 在四份数据中的 ID、原始行号、SFT 的 messages/videos 一致性、CLUE 两个版本的 student prompt/媒体输入一致性，以及 teacher 的 `<video>` 数量与 clue 片段数。本文不是模型模板编码后的完整 token dump。

| 数据版本 | 原始文件 | SHA-256 |
| --- | --- | --- |
| sft_lora | [sft_observation_openqa.jsonl](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_sft_lora_20260930/data/sft_observation_openqa.jsonl) | `baeeeb22d689b76276ef34a5da786100add977318f072c9e8a0a0821b09f99d2` |
| sft_full | [sft_observation_openqa.jsonl](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_observation_sft_full_npu96_w6w7_20260930/data/sft_observation_openqa.jsonl) | `baeeeb22d689b76276ef34a5da786100add977318f072c9e8a0a0821b09f99d2` |
| clue_obs | [clue_openqa_thinking.jsonl](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_openqa_thinking_full_observation_20260930/data/clue_openqa_thinking.jsonl) | `6dd306cbe334e7d1c857c1315e1adce0f9445cdb9477469bd7890ef56a705606` |
| clue_noobs | [clue_openqa_thinking.jsonl](/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_clue_full_no_observation_npu96_w10w11_20260930/data/clue_openqa_thinking.jsonl) | `0c529cb2fa4e804f0ef16598850506086aa6176c1b4660520ac283a38d6fc9d9` |
