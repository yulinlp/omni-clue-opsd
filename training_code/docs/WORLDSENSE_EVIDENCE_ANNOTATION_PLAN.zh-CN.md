# WorldSense 证据区间标注方案（用于 CLUE-OPSD / 多原子视角 OPSD 的特权教师证据）

> 状态：草案 v1（2026-09-21）
> 目标读者：后续实现该 pipeline 的 agent / 工程同学
> 相关文档：`docs/SESSION_HANDOFF_20260917.zh-CN.md`（CLUE-OPSD 训练契约）、
> `docs/OMNIVIDEO_5K_EXPERIMENT_SUMMARY (2).md`（原子视角 E 系列）

---

## 0. 一句话目标

给 WorldSense 的 **1,662 个视频 / 3,172 道题**（数据中无证据区间标注）生成
**高精度、可审计的证据时间区间（evidence intervals）**，作为 `teacher` 的特权输入，
接入现有 CLUE-OPSD 的 `clue_opsd` 数据契约，从而在 WorldSense 上开展 OPSD 训练。

---

## 1. 现状核查（已实测，非估计）

### 1.1 数据

| 项目 | 实测值 | 来源 |
| --- | --- | --- |
| 视频数 | 1,662 | `worldsense_qa.json` 顶层 key 数 |
| 题目数 | 3,172（task0=1662, task1=994, task2=422, task3=85, task4=9） | 同上 |
| 每题字段 | `task_domain`, `task_type`, `question`, `answer`(字母), `candidates` | `task0..task4` |
| 视频字段 | `video_id`, `video_duration`(如 `"60s"`), `duration`, `domain`, `sub_category`, `audio_class`, **`video_caption`（整段详细描述）** | 同上 |
| 时长 | 15s – 656s，均值 140.7s；**198 条 > 300s** | `video_duration` 统计 |
| audio_class | Speech 1442 / Event 1037 / Music 940（多标签） | 同上 |
| domain | Daily Life 348, Tech & Science 249, Music 242, Sports 211, Film & TV 193, Culture & Politics 166, Performance 144, Games 109 | 同上 |
| 任务类型 | Object Counting 205, Spatial Relation 197, Attribute Recognition 181, Event Sorting 171, Temporal Localization 169, Action Counting 165, Causal Reasoning 151, Text and Diagram 134, Human Interaction 132, Human-object Interaction 127, Event Recognition 122, Attribute Reasoning 121, **Audio Source Localization 120, Audio Recognition 116**, Temporal Prediction 110 等 | `task_type` 统计 |
| 视频文件 | 11 个 zip（`worldsense_videos_0..10.zip`），共 1,662 个 mp4，**尚未解压**（`asset/` 为空） | `unzip -l` |
| 媒体规格（抽检 1 条） | h264 640×360 + **aac 44.1kHz 立体声** | `ffprobe` |

**关键结论：**
1. 数据自带 `video_caption`（整段视频的细粒度文字描述）与 `audio_class`，
   是很强的**弱先验**，可用于候选生成与交叉校验，但不能当作证据区间。
2. `audio_class` 三分（语音/事件/音乐）意味着**证据模态分布差异极大**：
   - `Speech`（1442）多半需要语音/对话时间点 → 需要 ASR 或语音理解；
   - `Music`（940）可能需要乐段/歌词区间 → 需要音频事件/音频理解；
   - `Event`（1037）多为环境声/动作声 → 需要音频事件定位；
   - `Audio Source Localization` / `Audio Recognition` 两类题（共 236 题）**必须**音频侧定位。
3. 198 条视频 > 300s，而 Qwen2.5-Omni 音频前端上限约 300s（现有
   `dynamic_budget.py` 也按 `min(duration, 300)` 处理），**长视频必须分块**。
4. 现有 OmniVideo-100K 目标格式是 `analysis.designated_segments`
   字符串（`"[00:14 - 00:28]\n[00:35 - 00:46]"`），经
   `src/omni_opsd/data/common.py::parse_time_ranges` 解析成
   `clue_intervals: [[start, end], ...]`。**新标注直接产出同一格式**即可复用整条
   数据/训练链路。

### 1.2 可用模型盘点（`/share/home/ylhu/models` 及相关目录）

| 类别 | 模型 | 路径 | 能力 | 建议角色 |
| --- | --- | --- | --- | --- |
| **Omni（视+音+文）** | **Qwen3-Omni-30B-A3B-Instruct** | `models/Qwen3-Omni-30B-A3B-Instruct` | thinker 含 vision+audio，指令强 | ⭐ 主力：候选生成 + 充分性验证 |
| Omni | Qwen2.5-Omni-7B | `models/Qwen2.5-Omni-7B` | 视+音+文，工程最熟 | 跨模型一致性验证（与训练 student 同族） |
| Omni | Qwen2.5-Omni-3B | `models/Qwen2.5-Omni-3B` | 视+音+文，弱 | 消融 / 快速预筛（可选） |
| Omni | Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 | `models/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` | 视+音，含 `sound_context_token` | 第三方视角仲裁（可选） |
| **VL only** | Qwen3.6-35B-A3B-FP8 | `models/Qwen3.6-35B-A3B-FP8` | 含 `video_token_id`，MoE，FP8 | 视觉窗口打分 / 视觉候选 |
| VL only | gemma-4-12B-it | `models/gemma-4-12B-it` | `gemma4_unified`（带 vision） | 视觉候选多样性 |
| VL only | Qwen3.5-9B / 4B / 2B | `models/Qwen3.5-*` | 均带 `vision_config` | 轻量窗口打分 |
| **ASR** | **Qwen3-ASR-1.7B** | `/share/home/ylhu/zmlong/Qwen3-ASR-1.7B`（配套 env `qwen3-asr`） | 语音转写 | 语音题定位（带时间戳转写） |
| ASR | faster-whisper（small） | env `lightomni` 内 `faster_whisper 1.2.1` | 转写 | 备选 / 交叉校验 |
| **音频事件/情感** | emotion2vec_plus_large | `models/emotion2vec_plus_large` | 情感/事件 embedding | 情感类题目音频侧候选 |
| OCR | DeepSeek-OCR | `models/DeepSeek-OCR` | 文字/图文 | `Text and Diagram Understanding` 题辅助 |
| 文本 LLM | Qwen3-14B / Qwen3-8B / Llama-3.1-8B | `models/...` | 纯文本 | caption 对齐、区间解析、融合决策 |

**可复用代码：**
- `src/omni_opsd/temporal/evidence.py`：`EvidenceSpan` / `SampleEvidence` 数据结构；
- `src/omni_opsd/data/common.py::parse_time_ranges`：多种时间戳格式解析；
- `src/omni_opsd/temporal/views.py` / `data/atomic_views.py`：把区间渲染成
  Qwen-Omni 媒体描述符（`video_start/video_end/nframes/...`），即 G/H/T/S/A/V 视角；
- `src/omni_opsd/data/dynamic_budget.py`：预算/上下文约束（32k、音频 300s 上限）；
- `src/omni_opsd/temporal/consistency.py`：跨视角一致性/多数投票；
- `scripts/repair_omnivideo_oe5k_clue_frame_ranges.py`：区间帧数 clamp 修复（接入训练前必跑）。

---

## 2. 设计原则

1. **Recall → Precision 两段式**：候选生成阶段宁多勿漏；验证阶段宁缺毋滥。
2. **时间戳必须"落地"**：不信任模型直接说出的秒数。候选区间以
   **滑动窗口逐窗打分** + **验证阶段边界收缩/扩张**来锚定，避免幻觉时间戳。
3. **充分性优先（answer-conditioned sufficiency）**：一个区间被接受的最低标准是
   "仅用该区间（视频+同步音频）就能让强 teacher 答对本题"。因为这正是 OPSD 里
   teacher 的实际使用方式。
4. **跨模型/跨族一致性**：至少两个不同模型族（如 Qwen3-Omni-30B 与
   Qwen2.5-Omni-7B）对同一区间判定一致才进入最终集；分歧样本进入人工复核。
5. **答案只用于"验证/选择"，不进入教师 prompt**：与现有 CLUE-OPSD 契约一致
   （`gold_answer_in_teacher_prompt=False`）。标注阶段允许用金标答案做
   sufficiency 判定，但必须记录 `answer_conditioned=true`，并额外产出
   "不看答案"的弱标注变体用于消融与可部署路由研究。
6. **可复现、可审计**：每个区间记录模型、prompt 版本、采样参数、分数、
   来源阶段、是否人工复核；全流程版本化 + SHA256 + 审计报告。
7. **与现有训练契约对齐**：输出即 `[HH:MM:SS - HH:MM:SS]` 字符串 /
   `[[start,end],...]`，可直接喂给 `prepare_*` 生成 `clue_opsd` JSONL。

---

## 3. Pipeline 总体架构

```text
┌──────────────────────── Phase 0：准备 ────────────────────────┐
│ 解压 1662 mp4 → 探测时长/帧率/音轨 → 扁平化 3172 道题          │
│ question_id = f"{video_id}::{taskN}"，长视频(>300s)切片登记     │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────── Phase 1：候选生成（recall-first）─────────────┐
│ P1 Omni 粗扫（Qwen3-Omni-30B，整段视频低帧率，多次采样）        │
│ P2 视觉滑窗打分（Qwen3.6-35B / gemma-4 / Qwen3.5，2–4s 窗）    │
│ P3 音频侧候选（Qwen3-ASR 转写+时间戳 / omni 音频窗 / emotion2vec）│
│ P4 Caption 对齐（LLM 从 video_caption 抽元素→粗定位）           │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────── Phase 2：候选融合与规范化 ────────────────────┐
│ IoU 聚类去重 → 每簇取并集 → 约束检查（时长/帧数/覆盖比/上限）    │
│ 输出每题的 top-K 候选假设（含来源与分数）                       │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────── Phase 3：验证与选择（precision-first）────────┐
│ V1 充分性：仅区间媒体 + 题面 → teacher 必须答对（正确概率/margin）│
│ V2 最小性：边界收缩二分，找"最小充分区间"                       │
│ V3 必要性：全视频去掉区间 → 答案概率应下降（抽样/候选级）        │
│ V4 跨模型一致性：第二模型族复验，一致才接受                     │
│ V5 稳定性：同模型 n 次采样投票一致                              │
│ → 选择：满足条件的最小/置信最高区间；否则标 review / no_evidence │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────── Phase 4：审计与人工抽检 ──────────────────────┐
│ 自动指标（充分率/一致率/长度/覆盖/分类型分布）                  │
│ 分层人工抽检 200–300 题（task_type × audio_class × 时长桶）     │
│ IoU/P/R 验收门限 + 失败模式回流改 prompt                       │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────── Phase 5：物化与接入训练 ──────────────────────┐
│ 输出 worldsense_evidence.jsonl（含 designated_segments 兼容字段）│
│ 新增 prepare_worldsense.py（复用 dynamic_budget + clue 契约）  │
│ 帧修复 preflight → 预算审计 → 复用 CLUE-OPSD launcher          │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. 各阶段详细设计

### Phase 0：准备

1. 解压 11 个 zip 到 `data/WorldSense/videos/<video_id>.mp4`（保持与
   OmniVideo-100K 相同的目录约定，便于复用 `video_root/videos/` 逻辑）。
2. 用 `decord`/`ffprobe` 探测每条视频：`n_frames`、`avg_fps`、`duration`、
   是否有音轨、音频采样率，写 `data/WorldSense/media_index.json`。
   **注意**：`video_duration` 字段（如 `"60s"`）与真实时长可能不一致，
   一律以探测值为准。
3. 扁平化题目：
   ```json
   {
     "question_id": "dvOkwKAs::task0",
     "video_id": "dvOkwKAs",
     "task_index": 0,
     "task_domain": "Understanding",
     "task_type": "Spatial Relation",
     "question": "...",
     "candidates": ["A. ...", "B. ...", "C. ...", "D. ..."],
     "answer_letter": "B",
     "video_caption": "...",
     "audio_class": ["Music"],
     "domain": "Music",
     "duration_s": 499.2,
     "needs_audio": true,
     "chunk_plan": [[0, 300], [300, 499.2]]
   }
   ```
4. 按 `task_type` 映射到**证据形态模板**（决定后续提示词与约束），例如：
   - `Temporal Localization` / `Event Sorting`：允许 2–4 个离散窗口，总时长上限更宽；
   - `Spatial Relation` / `Attribute Recognition`：单窗口为主（>=2s），偏视觉；
   - `Audio Source Localization` / `Audio Recognition`：必须包含音频证据，窗口>=1.5s；
   - `Object Counting` / `Action Counting`：单窗口 + 覆盖完整计数过程；
   - `Text and Diagram`：窗口需覆盖可读画面（必要时调用 OCR 校验）；
   - `Music` 类题目：叠加歌词/乐段候选。

### Phase 1：候选生成（recall-first）

#### P1 Omni 粗扫（主候选）
- 模型：`Qwen3-Omni-30B-A3B-Instruct`（主）+ `Qwen2.5-Omni-7B`（多样性）。
- 输入：整段视频（低帧率，如 1 FPS、短边 224，控制 token）+ 题目 + 选项 +
  `video_caption` 作为"视频梗概"提示（明确告知"以下是视频梗概，请据此定位证据"）。
- 采样：temperature 0.7，n=3（不同随机种子），提高召回。
- 输出 JSON（强约束）：
  ```json
  {"evidence": [{"start": 12.5, "end": 20.0, "modality": "visual|audio|both",
                 "reason": "...", "confidence": 0.0-1.0}],
   "evidence_absent": false}
  ```
- 解析失败自动重试（最多 2 次，附带上一次输出让其修正）。

#### P2 视觉滑窗打分（时间戳锚定）
- 把视频切成 2–4s 滑窗（步长=窗长/2），每窗渲染一个 clip。
- 视觉模型（`Qwen3.6-35B-A3B-FP8` 优先，`gemma-4-12B-it` / `Qwen3.5-9B` 做补充）
  对每窗二分类/三分类：
  `{"contains_evidence": "yes|no|uncertain", "confidence": 0.0-1.0}`；
  提示词给出题目 + 选项（不给答案）+ "该窗口是否包含回答该问题所需的视觉证据"。
- 合并连续 `yes` 窗口为候选区间（允许 1 窗空洞由 `uncertain` 桥接）。
- **作用**：把时间戳锚定到真实窗口边界，同时提供与 omni 不同的证据视角。
- 批量策略：同一视频的窗可批量推理；窗口裁剪优先用 decord 直读时间范围，避免落盘。

#### P3 音频侧候选
- **ASR 路径**（`Speech` 类必做）：用 `Qwen3-ASR-1.7B`（或 faster-whisper）
  产出带时间戳的转写；用题目关键词/命名实体/数字（如人名、数字、地点）匹配片段，
  产出候选区间与文本证据。
- **音频事件路径**（`Event`/`Music`）：
  - 用 omni 模型对音频轨做 5–10s 窗打分（"该窗口是否包含与问题相关的关键声音"）；
  - `emotion2vec_plus_large` 对情感/情绪题产出高显著度时间段；
  - `Music` 题的歌词/人声段可复用 ASR。
- 输出同样归一化为 `[[start,end],...]` + `modality=audio` + `reason`（转写片段）。

#### P4 Caption 对齐（弱先验）
- `video_caption` 是按时间顺序写的整段描述。用 Qwen3-14B（纯文本）：
  1) 把 caption 切成有序事件句；2) 判断哪几句与题目证据相关；
  3) 输出相关句子的**相对顺序位置**（而非秒数）。
- 再与 P2 的窗口序列做序列对齐（如 DTW/最长公共子序列），得到粗略区间。
- 该路径**仅作补充与校验**，权重低于 P1/P2/P3。

### Phase 2：候选融合与规范化

1. 汇总某题所有候选区间，按 IoU ≥ 0.5 聚类；每簇取并集，记录来源集合、
   各来源分数、模态标签。
2. 规范化约束（可配置，默认按 `dynamic_budget` 与训练契约）：
   - 区间必须落在 `[0, duration]` 内，且 `end - start ≥ 1.0s`（保证 2 FPS 下 ≥2 帧）；
   - 单题最多 `max_spans=4`，总时长 ≤ `min(60s, 40% × duration)`（Temporal/Event
     Sorting 类放宽到 6 段 / 50%）；
   - 相邻区间间隔 < 0.5s 时合并；
   - 对齐到帧边界（保留 0.01s 精度即可，最终由帧修复脚本 clamp）。
3. 输出每题 top-K（默认 K=4）候选假设，附 `{source, score, modality}`。

### Phase 3：验证与选择（precision-first，本方案的质量核心）

#### V1 充分性验证（sufficiency，必做）
- 教师模型：`Qwen3-Omni-30B-A3B-Instruct`（主判定），
  `Qwen2.5-Omni-7B`（跨族复验）。
- 对每个候选区间 I：
  - 只喂 I 的视频+同步音频（用 `atomic_views` 的裁剪描述符）；
  - 题面不变，**不提供答案**；
  - 记录：`correct`（是否选中金标字母）、`p_true`（金标选项概率）、
    `margin`（最优-次优概率差）、`entropy`。
- 若模型输出格式异常，重试 2 次，仍失败记 `parse_error`。

#### V2 最小性（minimality，必做但可近似）
- 对 V1 通过的区间做**边界收缩**：左/右边界各自按 25% 步长收缩，
  每次收缩后重跑 V1；保留仍满足充分性的最小长度版本（贪心，最多 4 轮）。
- 目的：去掉"顺手多看的上下文"，让特权信息更精确，降低 teacher 泄漏噪声。

#### V3 必要性（necessity，抽样/候选级，成本高）
- 从全视频中**挖掉**区间 I（其余保留）再让 teacher 答题；
  若仍答对且 `p_true` 不降，说明 I 可能不必要 → 降级。
- 仅对最终候选执行；可只对"高分但来源单一"的样本执行以控制成本。

#### V4 跨模型一致性（必做）
- 让 `Qwen2.5-Omni-7B` 对同一区间做 V1；若两模型都 `correct`，接受；
- 若一个 `correct`、一个 `wrong` → `review`（人工或第三方模型仲裁，
  可用 Nemotron-Omni-30B）；
- 若两者都 `wrong` → 该候选丢弃（该题若无任何通过候选，进入兜底策略）。

#### V5 稳定性（可选）
- 同一模型对同一区间在 temperature 0.2 下采样 3 次，要求至少 2/3 一致。

#### 选择策略（每题）
1. 优先选"充分 + 跨模型一致 + 最短"的区间；
2. 若有多个都不满足，选 `p_true/margin` 最高的候选并标 `low_confidence`；
3. 若全部候选失败：标记 `no_evidence`，**不进入训练集**（或按实验设计单独分组，
   避免给 teacher 喂错误特权信息）；
4. 对 `Temporal Localization` / `Event Sorting` 类，允许多区间联合验证
   （把多个窗口一起喂给 teacher）。

### Phase 4：审计与人工抽检

#### 4.1 自动指标（全量）
| 指标 | 定义 | 目标 |
| --- | --- | --- |
| sufficiency_rate | 主 teacher 仅凭区间答对的比例 | ≥ 0.95 |
| cross_model_rate | 两个模型族都答对的比例 | ≥ 0.90 |
| answer_margin_gain | 区间输入相对全视频输入的 margin 提升 | > 0 |
| modal_accuracy | 自动预测的证据模态 vs `audio_class` 的一致性 | ≥ 0.85 |
| interval_length | 区间总长/占比分布 | 见约束 |
| no_evidence_rate | 无通过候选的比例 | ≤ 0.05（超过需排查） |
| parse_error_rate | 输出解析失败比例 | ≤ 0.01 |

#### 4.2 分层人工抽检（质量门）
- 抽样 200–300 题，分层维度：`task_type`（重点覆盖 Audio*、Temporal*、Text&Diagram）
  × `audio_class` × 时长桶（<60s / 60–300s / >300s）。
- 每题由 2 名标注者独立判定：①区间是否覆盖关键证据；②区间是否含无关内容；
  ③是否存在更短充分区间。分歧由第 3 人仲裁。
- 计算：与人工区间的 temporal IoU（目标中位数 ≥ 0.5）、
  evidence precision ≥ 0.9、recall ≥ 0.8、accept rate ≥ 0.9。
- **未达门限则回到 Phase 1/3 调整 prompt 与参数，重新标注受影响子集**。

#### 4.3 泄漏与合规审计
- 检查区间内是否包含"直接写出答案的屏幕文字"——若该文字本身是题目要求的
  证据（如 `Text and Diagram` 题），属于合法证据；但需在标注中记录
  `answer_text_visible=true`，供训练分析时分层。
- 记录 `answer_conditioned`（标注过程是否使用了金标答案）；训练用的
  teacher prompt 永远不含答案（与现有契约一致）。
- 输出 `worldsense_evidence_audit.md`，含全部指标、失败案例与人工抽检记录。

### Phase 5：物化与接入训练

1. **标注产物**：
   ```
   data/worldsense/evidence/worldsense_evidence.jsonl
   ```
   每行：
   ```json
   {
     "question_id": "dvOkwKAs::task0",
     "video_id": "dvOkwKAs",
     "duration": 499.2,
     "clue_intervals": [[12.5, 20.0]],
     "designated_segments": "[00:12 - 00:20]",
     "evidence_modality": "both",
     "sufficiency": {"qwen3_omni_30b": {"correct": true, "p_true": 0.83, "margin": 0.51},
                      "qwen25_omni_7b": {"correct": true, "p_true": 0.71}},
     "provenance": ["P1:qwen3-omni-30b", "P2:qwen3.6-35b", "P3:qwen3-asr"],
     "answer_conditioned": true,
     "review_status": "accepted|review|no_evidence",
     "version": "worldsense_evidence_v1"
   }
   ```
   同时输出 `worldsense_evidence.no_answer_variant.jsonl`（不使用答案做选择的
   弱监督版本，用于消融）。
2. **训练数据生成**：新增 `scripts/prepare_worldsense.py`（可大量复制
   `scripts/prepare_omnivideo_oe5k.py` 的结构）：
   - 读取 QA + evidence，构造 `sft / answer_free / opsd / clue_opsd / labels`；
   - 复用 `dynamic_budget_for` / `dynamic_clue_budget_for` / `dynamic_video_spec` /
     `_clue_video_specs`，保持与 OmniVideo OE-5K 完全一致的
     `sampling_contract`（2 FPS、32k 上下文、音频开启、`use_audio_in_video=True`）；
   - 过滤 `no_evidence` 行；长视频按 `min(duration,300)` 的音频预算规则处理；
   - 跑 `repair_omnivideo_oe5k_clue_frame_ranges.py` 做帧 preflight；
   - 写 manifest（含 `visual_budget_cap`、`dropped_no_evidence`、SHA256）。
3. **训练**：直接复用
   `scripts/run_omnivideo_oe5k_clue_opsd_gpu05_v15k.sh` 的模式，做一个
   `run_worldsense_clue_opsd_*.sh`（改 dataset/tag/root/port），
   并按 handoff「8k→24k 接续步骤」的审计流程先做 1-step 显存 gate。

---

## 5. 工程实现建议

### 5.1 代码布局（沿用现有 `src/omni_opsd` 约定）

```text
src/omni_opsd/worldsense/
  __init__.py
  schema.py        # QuestionRecord / CandidateInterval / EvidenceRecord 数据类
  probe.py         # zip 解压、decord/ffprobe 媒体探测、chunk_plan
  propose.py       # P1 omni 粗扫（含 JSON 强约束与重试）
  windows.py       # P2 视觉滑窗生成与打分、连续窗合并
  audio_side.py    # P3 ASR/音频事件/emotion2vec 候选
  caption_align.py # P4 caption → 有序句 → 序列对齐
  fuse.py          # Phase 2 聚类、约束、top-K
  verify.py        # V1–V5（充分性/最小性/必要性/一致性/稳定性）
  audit.py         # 自动指标 + 人工抽检抽样清单
  materialize.py   # Phase 5 输出 evidence / designated_segments
scripts/annotate_worldsense_evidence.py   # 命令行编排（stage 可选、可断点续跑）
scripts/prepare_worldsense.py             # 训练数据生成（复制 OE-5K 生成器）
```

### 5.2 执行与并发

- 分阶段可独立运行，落盘中间结果（`*.candidates.jsonl` / `*.verified.jsonl`），
  支持按 `question_id` 断点续跑与分片（`sharding.py` 已有分片工具可参考）。
- 推理服务优先用 vLLM（单模型常驻，批处理）；**必须验证音频不被丢弃**
  （见 handoff 的 vLLM 音频问题）：在正式标注前跑
  `USE_AUDIO_IN_VIDEO=1` 的编码/推理自检，`vllm_drop_audio=0`。
  若 vLLM 音频路径不可靠，退回 Transformers 批推理（慢但一致）。
- 长视频（>300s）：按 `chunk_plan` 分块送音频模型；视频侧不切（低帧率整段），
  但 P2 滑窗天然按时间片处理。
- 建议并发模型：Qwen3-Omni-30B（2 卡 TP2 或 1 卡 FP8）、
  Qwen2.5-Omni-7B（1 卡）、Qwen3.6-35B/vision（1 卡）、ASR（1 卡）。

### 5.3 成本估算（粗算，需实测校正）

- 3172 题。
- P1：3172 × 3 次整段低帧率推理 ≈ 9,500 次；
- P2：按平均 140s/视频、3s 窗、50% 重叠 ≈ 90 窗/视频 → 1,662×90 ≈ 150k 窗推理（批量可摊薄）；
- P3：ASR 1,662 次（可按需只跑 Speech/Music）+ 音频窗约 30k 次；
- V1/V4：每候选 2 模型 × (3172×~2 候选) ≈ 12,700 次（含最小性收缩后更多，约 2–3×）。
- 在 4×A100 上、两个 30B 级模型 + 一个 7B + 一个 VL 并发，**预计 2–4 天**
  可完成全量；人工抽检 2–3 人日。

---

## 6. 风险与缓解

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| 模型幻觉时间戳 | 区间错位，teacher 学到噪声 | P2 滑窗锚定 + V2 边界收缩 + 帧修复 clamp |
| 音频证据被忽略 | Audio* 类题目质量差 | 强制 P3 音频路径；对 `audio_class` 含 Speech/Music/Event 的题，验证必须包含音频输入；单独统计 Audio* 类通过率 |
| 长视频 >300s 音频上限 | 音频截断，证据漏标 | chunk_plan 分块；区间若落在 chunck 边界外需二次核验 |
| 单一模型偏置（如偏向 caption 描述） | 系统性错误 | 跨模型族一致性（Qwen3-Omni vs Qwen2.5-Omni vs VL-only）+ 人工抽检 |
| 用答案选区间导致"答案泄漏式"标注 | teacher 特权的语义偏移 | `answer_conditioned` 标记 + 产出 no-answer 变体 + 论文中明确区分（与 handoff 多原子视角设计一致） |
| vLLM 丢音频 | 音频题标注退化 | 标注前自检；必要时全部走 Transformers |
| 成本/排期超预期 | 阻塞训练计划 | 分阶段交付：先做 300–500 题 pilot 校准质量与成本，再全量 |
| GPU 被训练占用 | 争抢资源 | 标注与训练错峰；使用 gpu02/gpu04 等账号（见 `/share/home/ylhu/EvoEmbedding/GPU_ACCOUNT_ACCESS.private.md`，勿在日志/代码中暴露密码） |

---

## 7. 交付物与里程碑

| 阶段 | 交付物 | 验收 |
| --- | --- | --- |
| M0 准备 | `media_index.json`、扁平化 `worldsense_questions.jsonl` | 1662 视频可解码、3172 题齐全 |
| M1 候选 | `worldsense.candidates.jsonl` | 每题 ≥1 候选的覆盖率 ≥ 0.98 |
| M2 融合 | `worldsense.fused.jsonl`（top-K + 约束检查） | 全部区间通过物理约束 |
| M3 验证 | `worldsense.verified.jsonl` | sufficiency ≥0.95、cross-model ≥0.90 |
| M4 审计 | `worldsense_evidence.jsonl` + `worldsense_evidence_audit.md` + 人工抽检记录 | 人工 precision ≥0.9、recall ≥0.8、IoU 中位数 ≥0.5 |
| M5 接入 | `scripts/prepare_worldsense.py` + `run_worldsense_clue_opsd_*.sh` + manifest/审计 | 1-step gate 不 OOM、预算审计通过 |

**Pilot 建议**：先选 300–500 题（覆盖全部 `task_type` 与 `audio_class`）跑通 M1–M4，
用人工抽检校准 prompt 和门限，再决定是否全量。这样可以把质量风险与成本风险前置。

---

## 8. 附录

### 8.1 Prompt 模板（P1 候选生成，示意）

```text
You are an evidence-localization annotator.
You will see a full video (with audio), a question, and its options.
Video synopsis (may be imperfect): {video_caption}

Question: {question}
Options:
{A. ...}
{B. ...}
...

Task: identify the minimal time intervals in the video that contain the evidence
needed to answer the question. Rules:
- Return 1-4 intervals, each >= 1.0 second, within [0, {duration}] seconds.
- If the question depends on sound (speech/music/event), include the matching audio interval.
- Do NOT write the answer or any explanation of the answer.
- Output valid JSON only:
{"evidence": [{"start": 12.5, "end": 20.0, "modality": "visual|audio|both", "reason": "...", "confidence": 0.9}],
 "evidence_absent": false}
```

### 8.2 Prompt 模板（V1 充分性验证，示意）

```text
You will see only a short clip (video + synchronized audio) and a question.
Question: {question}
Options:
{A. ...}
...
Reply with exactly one option letter in <answer>...</answer> and nothing else.
```

### 8.3 与现有 `clue_opsd` 字段映射

| WorldSense 标注字段 | 训练 JSONL 字段 | 说明 |
| --- | --- | --- |
| `clue_intervals` | `clue_intervals` | `[[start,end],...]` |
| `designated_segments` | （中间产物） | `"[00:12 - 00:20]"`，供 `parse_time_ranges` |
| `evidence_modality` | `metadata.evidence_modality` | 分析用 |
| — | `teacher_videos` | 由 `_clue_video_specs` 生成，每区间独立采样 |
| — | `dynamic_teacher_budget` | 由 `dynamic_clue_budget_for` 生成 |
| — | `sampling_contract` | 固定 `use_audio_in_video=True`、`teacher_view="dataset-clue-interval"` |

### 8.4 待确认事项（实现前需拍板）

1. 是否接受"用金标答案做 sufficiency 选择"（本方案默认接受，并记录标记、另出 no-answer 变体）？
2. 人工抽检的执行者与预算（2–3 人日）？
3. 标注模型优先级：Qwen3-Omni-30B 为主是否与现有 vLLM/Transformers 环境兼容
   （需先做一次含音频的编码自检）？
4. Pilot 规模与任务子集（建议 300–500 题，全类型覆盖）。
5. 是否把标注产出的 no-answer 变体也纳入正式训练矩阵（对应 handoff 的
   Fixed/Confidence/Mixture 路由实验）。

---

## 9. 视频解压记录（2026-09-21 已完成）

- 解压目标：`/share/home/ylhu/Light-Omni/data/omni_benchmarks/WorldSense/videos/<video_id>.mp4`
  （与 OmniVideo-100K 相同的 `videos/` 目录约定，便于后续复用
  `video_root/videos/` 的路径解析逻辑）。
- 来源：`worldsense_videos_0.zip` ~ `worldsense_videos_10.zip`（共 11 个）。
- 结果：**1,662 个 mp4 全部解压成功，17 GB**。
- 完整性校验：QA 的 1,662 个 `video_id` 与 mp4 文件名 **一一对应，零缺失、零多余**。
- 抽检媒体规格（`zwkbEZwG.mp4`）：h264 640×360 + aac 44.1kHz 双声道，
  真实时长 499.2s（注意与 `video_duration` 字段可能不一致，一律以探测值为准）。

---

## 10. 附录 B：WorldSense 中文示例题（5 道，覆盖不同证据形态）

> 以下 5 道题从 3,172 道题中挑选，覆盖 5 种典型证据类型，用于了解数据风格与
> 本方案为何需要多模态、多粒度的证据标注。
> 原文（英文）+ 中文翻译对照；答案用 **加粗** 标出。

### 例 1｜音频/语音证据（Object Counting）

- `video_id`: `NBzSgbJK`（task0）
- 元数据：domain=Performance / Talks；时长 212s；audio_class=Speech + Event
- 原文 Q：**How many countries are mentioned in the video?**
- 中文 Q：**视频中提到了多少个国家？**
- 选项：
  - A. Three. → 三个
  - B. Five. → 五个
  - C. Four. → 四个
  - D. One. → 一个
- 答案：**C（四个）**
- 视频梗概（caption 中译）：一只手在白板上作画：先画出一栋带三根柱子的建筑，旁边基座上标着 "MAN" 的人头和标着 "FOX" 的狐狸，旁有气泡写着 "WE HAVE ONE & ONLY LIFE"；红字写着 "WHEN WE TALK OF BUILDING AN EMPATHIC CIVILISATION"。接着画出人形并写上 "WE ARE HOMO-EMPATHICUS"，红字提问 "HOW DOES CONSCIOUSNESS CHANGE IN HISTORY?"；人物头顶的圆圈标着 "APPS"、"FOOTBALL"、"COFFEE"。随后又画出持斧的 "MEDIEVAL SERF（中世纪农奴）" 与 "MODERN MAN（现代人）"，头顶圆圈标 "FOOD"、"SOCIAL POSITION"、"GOD"；最左侧画着驼背大嘴的 "FORAGER/HUNTER（采集狩猎者）"，头顶标 "BELONGING"、"SEX" 等。
- 证据特征：**答案藏在讲者口播里（提到了 4 个国家）**，属于音频/语音证据，
  必须依赖 ASR 或音频理解来定位时间点；画面（白板动画）本身不能给出答案。

### 例 2｜空间关系（Spatial Relation）

- `video_id`: `dvOkwKAs`（task0）
- 元数据：domain=Music / Covers；时长 60s；audio_class=Music
- 原文 Q：**What is the position of the metal roller door relative to the woman wearing white in the video?**
- 中文 Q：**视频中金属卷帘门相对于白衣女子的位置在哪里？**
- 选项：
  - A. To the left of the woman wearing white. → 在白衣女子的左侧
  - B. To the right of the woman wearing white. → 在白衣女子的右侧
  - C. In front of the woman wearing white. → 在白衣女子的前方
  - D. Behind the woman wearing white. → 在白衣女子的后方
- 答案：**B（在她右侧）**
- 视频梗概（caption 中译）：一男一女身穿白色短袖衬衫、面向镜头弹奏古筝；女子衬衫胸前有小 logo、腰系红带，两人右手戴拨片拨弦；背景是绘有天体与有机图案的彩色壁画墙，**女子身后是一扇只能部分看见的、关闭的金属门**；镜头向右摇，露出另外两名同样弹古筝的女子（一人系浅青色腰带、一人系红腰带），背景可见部分楼梯；镜头固定，四人继续演奏。
- 证据特征：**纯视觉空间关系**，关键证据在固定机位画面中（女子与门的位置关系），
  通常只需 1 个短窗口 + 中等空间分辨率。

### 例 3｜乐器计数（Audio Source Localization）

- `video_id`: `TGqYCmQM`（task0）
- 元数据：domain=Music / Music Videos；时长 60s；audio_class=Music
- 原文 Q：**How many instruments are in the video?**
- 中文 Q：**视频中有多少件乐器？**
- 选项：
  - A. Three. → 三件
  - B. Two. → 两件
  - C. One. → 一件
  - D. Four. → 四件
- 答案：**B（两件）**
- 视频梗概（caption 中译）：一名年轻女子和一名男子面向镜头拉小提琴；女子在画面左
  侧，男子在右侧、略微在一架黑色谱架之后；女子穿浅色碎花短袖连衣裙（蓝紫花色），
  男子穿浅蓝色短袖衬衫和浅色长裤，戴眼镜；两人都把琴夹在下颌处、用琴弓拉弦；背景
  是浅棕色木饰面墙，画面中央有一扇小方格窗，室内光线柔和温暖；两人全程持续演奏。
- 证据特征：本题虽归类于 **Audio Source Localization**，但答案（2 把小提琴）主要
  由视觉计数确认，音频可作为辅助校验（两个声部/音色）。标注时应同时保留视觉与
  音频候选，避免把"音源定位"类题目一律当成纯音频题。

### 例 4｜时间定位（Temporal Localization）

- `video_id`: `UYkFSXsh`（task1）
- 元数据：domain=Culture & Politics / Politics；时长 119s；audio_class=Speech + Event
- 原文 Q：**At what point in the video do the audience members cheer?**
- 中文 Q：**观众在视频的哪个时间点欢呼？**
- 选项：
  - A. In the middle of the video. → 视频中段
  - B. Throughout the video. → 全程
  - C. At the end of the video. → 视频结尾
  - D. At the beginning of the video. → 视频开头
- 答案：**D（视频开头）**
- 视频梗概（caption 中译）：视频以浅色背景开场，顶部中央是 logo，其下是 YouTube
  订阅按钮，屏幕底部有文字条；随后画面切到带多彩边框的场景：中央一名男子手持话筒，
  身后是白色背景板（顶部有文字），右侧有一棵圣诞树，底部出现文字横幅与多个社交媒体
  图标；接着切到坐着的人群观众；再切回男子讲话并做手势、指向镜头继续说话；期间底部
  不断浮现各种广告横幅；最后以男子站立面向镜头的画面结束。
- 证据特征：**音频事件的时间定位**（观众欢呼声出现在开头），必须靠音频事件检测
  或 ASR（欢呼不在转写文本里，属于非语音事件）来定位；这是 `audio_class=Event`
  类题目的典型代表。

### 例 5｜文字/图文识别（Text and Diagram Understanding）

- `video_id`: `FnOIAada`（task0）
- 元数据：domain=Tech & Science / Auto；时长 24s；audio_class=Speech + Event
- 原文 Q：**What is the number on the rear wing of the red vehicle at the start of the video?**
- 中文 Q：**视频开头红色车辆尾翼上的数字是多少？**
- 选项：
  - A. 56.
  - B. 50.
  - C. 58.
  - D. 59.
- 答案：**B（50）**
- 视频梗概（caption 中译）：改装赛车在椭圆形沥青赛道上高速行驶；画面中出现车群里
  不同位置的多辆赛车（红、蓝、白、黑、黄及多种涂装），车辆彼此贴得很近；背景中
  短暂出现一面绿旗和一名穿安全背心的男子；镜头切到赛车在起跑线上列队，再切回比赛
  画面；视频结尾，一名金发、戴墨镜的女子被（画面外的人）递上一顶红色棒球帽。
- 证据特征：**阅读小目标上的数字**，关键证据只在视频开头若干秒内、且字号很小，
  需要**高空间分辨率**（甚至 OCR 辅助）；这正是原子视角 S（highres）要解决的场景。

### 示例小结（对标注 pipeline 的启示）

| 例 | 主要证据模态 | 时间粒度 | 空间要求 | 需要的标注能力 |
| --- | --- | --- | --- | --- |
| 1 国家计数 | 语音（口播） | 多个离散点 | 低 | ASR / 音频理解 |
| 2 卷帘门位置 | 视觉（空间） | 1 个短窗 | 中 | VL 空间关系 |
| 3 乐器计数 | 视觉为主 + 音频校验 | 1 个短窗 | 中 | VL 计数 + 音源校验 |
| 4 观众欢呼 | 音频事件 | 1 个精确时间段 | 低 | 音频事件定位 |
| 5 尾翼数字 | 视觉（小字） | 1 个精确短窗 | **高** | OCR / 高分辨率视觉 |

这 5 例说明：**证据模态在样本间差异极大**（语音 / 音乐 / 事件 / 空间 / 文字），
且时间粒度与空间分辨率需求各不相同——这正是本方案采用
"多路径候选生成（P1–P4）+ 逐样本充分性验证（V1–V4）+ 跨模型一致性"的原因；
也说明不能只用单一大模型一次性输出时间戳，否则例 1、例 4、例 5 这类题会系统性失败。

---

## 11. 附录 C：自适应证据定位 Agent（tool-loop）设计

### 11.1 先回答一个关键前提

**模型不能从 prompt 文本里自己改采样率/分辨率/音频开关/视频区间**——
这些都是 `qwen_omni_utils` 的结构化媒体参数（`fps`|`nframes`、
`resized_height/width`、`min_pixels/max_pixels`、`video_start/video_end`）
和 template/env 级开关（`USE_AUDIO_IN_VIDEO`），必须在**调用前**由外部代码写入。
LLM 只看到 token 化之后的输入，prompt 里的文字无法回头修改预处理。

**所以"自适应"的正确实现方式 = tool-loop**：让模型输出"下一步想看什么"的
结构化请求，由 harness 重新裁段/改 fps/改分辨率/切音频，再调用模型。
模型负责决策，harness 负责 I/O 与预处理。

### 11.2 对"主 agent + 多 subagent"理解的修正

你的理解方向正确，但要精确化成三点：

1. **核心是 tool-call loop（ReAct 风格）**：
   `调用工具 → 等待返回 → 判断"还需探索 or 可以收尾" → 下一步`。
   这个循环是必须的，也是主体。
2. **"多个 subagent"不是必须，且不建议做成自由多 agent**。本任务推荐：
   **1 个 controller（主决策）+ 若干确定性专家工具（specialist tools）
   + 1 个确定性验收器（verifier）+ 失败时的 critic（复核）**。
   专家（视觉/音频/ASR/OCR）是"有固定输入输出契约的工具"，不是各自独立规划
   的 agent——这样才能审计、复现、控成本。
3. **"线索是否充分"不能由模型自述决定**。模型只负责*提议*区间；
   是否接受由**确定性充分性验证器**（Phase 3 的 V1/V4）裁决。
   模型说"够了"不算数，验证通过才算数。

一句话：**控制器提议、工具执行、验证器裁决、critic 只在失败时介入**。

### 11.3 Episode 结构（每题一个独立回合）

```text
Episode(question_id)
├─ Controller: Qwen3-Omni-30B-A3B-Instruct（主决策）
├─ Tools（确定性，全部带缓存与审计）
│  1. get_media_info()                     # duration/fps/has_audio/task_type/audio_class/caption
│  2. get_transcript(start,end)             # Qwen3-ASR-1.7B，带时间戳转写（缓存）
│  3. search_transcript(terms, window)      # 在转写里检索关键词/实体/数字
│  4. score_windows(start,end,win,step,modality)  # VL 或 omni 逐窗打分，返回连续高分段
│  5. ocr(start,end,max_pixels)             # DeepSeek-OCR，返回带时间戳的文本
│  6. inspect(start,end,fps|nframes,max_pixels,use_audio,focus)
│       # harness 渲染指定视角（复用 views.py / atomic_views.py），
│       # 就一个聚焦问题调用专家模型，返回 {answer,p_true,margin,observations}
│  7. verify(intervals)                     # 确定性 V1+V4：Qwen3-Omni-30B + Qwen2.5-Omni-7B
│  8. submit(intervals, rationale)          # 终止动作；仍需过 verify 才算接受
└─ Verifier（不由 controller 控制）: 充分性 + 跨模型一致性 + 物理约束
```

**关键点**：`inspect` 能让"模型自适应"落地——controller 自己决定
`start/end/fps/max_pixels/use_audio`，harness 按这些参数重渲染后调用模型。
媒体参数因此变成**可审计的结构化决策**，而不是不可控的自然语言。

### 11.4 Tool-loop 协议（两种实现，推荐第 2 种）

**实现 1：原生 tool-calling**
- 事实核查：Qwen3-Omni-30B 的 chat template **原生支持**
  `tools / tool_call / tool_response`（模板长度 6519，含三者）。
- 用 vLLM 起 OpenAI 兼容服务，开启 `--enable-auto-tool-choice`
  + tool parser，controller 直接产出 tool_calls。
- 局限：`tool_response` 只能回文本，**不能把视频/音频媒体塞回 tool 返回**；
  媒体重看必须由 harness 另开一次多模态调用。因此媒体型工具仍需
  "controller 给参数 → harness 渲染 → 再调用"的桥接。

**实现 2（推荐）：结构化 JSON 协议 + 显式 Python 编排**
- 每轮要求 controller 只输出**一个** action 的严格 JSON（见 11.5）。
- harness 解析 → 执行工具 → 把**紧凑文本 observation** 追加回上下文。
- 不依赖任何 agent 框架（环境里也没有装），完全用现有
  transformers/vLLM + `openai` 客户端即可；便于批处理、断点续跑、缓存与审计。
- 对批量标注（3172 题）这是唯一现实的选择。

### 11.5 Action / Observation Schema

```json
// controller 每轮输出恰好一个 JSON
{"action":"inspect","start":30.0,"end":40.0,"fps":4,"max_pixels":156800,
 "use_audio":true,"focus":"白板上是否出现国家名称？","why":"核对口播国家数"}
{"action":"search_transcript","terms":["country","nation","China","France","India"],"window":[0,212]}
{"action":"score_windows","start":0,"end":212,"win":3.0,"step":1.5,"modality":"visual"}
{"action":"ocr","start":5.0,"end":9.0,"max_pixels":313600}
{"action":"get_transcript","start":0,"end":212}
{"action":"verify","intervals":[[12.5,20.0]]}
{"action":"submit","intervals":[[12.5,20.0]],"rationale":"..."}
```

```json
// harness 返回的 observation（示例）
{"tool":"inspect","params":{...},
 "result":{"answer":"C","p_true":0.82,"margin":0.47,
           "notes":["whiteboard lists COUNTRY: CHINA/INDIA/BRAZIL/FRANCE"]},
 "cost":{"frames":40,"seconds":6.2},"cached":false}
```

### 11.6 循环策略与预算（防止失控）

| 项 | 默认 | 说明 |
| --- | --- | --- |
| `max_turns` | 10 | 含 submit/verify 轮 |
| `max_inspect` | 6 | 媒体重看次数上限 |
| `max_transcript` | 1 次 | 转写只跑一次，之后只用检索 |
| 同参去重 | 强制 | `(tool, params)` 命中缓存直接返回，不重复计费 |
| 无进展检测 | 连续 2 轮无新证据 → 强制 `verify/submit` | 避免打转 |
| 早停 | verify 通过即结束 | 最短路径优先 |
| 兜底阶梯 | caption 对齐候选 → 高分候选 → `review/no_evidence` | 绝不静默给错区间 |

### 11.7 终止判定（谁来决定"够了"）

```text
controller 认为够了 → submit(intervals)
        ↓
verifier（确定性，不可被 controller 绕过）
  1) 物理约束：区间在 [0,duration]、≥1.0s、段数/总长上限
  2) V1 充分性：Qwen3-Omni-30B 仅用该区间必须答对（p_true/margin 达标）
  3) V4 跨模型：Qwen2.5-Omni-7B 复验一致
  4) （抽样）V3 必要性：全视频去掉该区间后 p_true 应下降
        ↓
 通过 → accepted     未过 → 把失败原因作为新 observation 回到循环
                       预算耗尽仍未过 → review / no_evidence
```

### 11.8 伪代码

```python
def run_episode(q, controller, tools, verifier, budget):
    state = init_state(q)                      # 题面/选项/caption/媒体元数据
    for turn in range(budget.max_turns):
        action = controller.decide(state)      # 严格 JSON，失败重试≤2
        if action.type == "submit":
            v = verifier.check(q, action.intervals)      # V1+V4+物理约束
            if v.accepted:
                return accept(q, action.intervals, state.trace, v)
            state.observe("verify_failed", v)  # 失败原因回灌，继续探索
            continue
        obs = tools.execute(action, cache=True)          # inspect 内部会重渲染媒体
        state.observe(action, obs)
        if budget.exhausted(state) or state.no_progress():
            break
    return fallback(q, state)                  # 最优候选 / review / no_evidence

# 并行方式：题目之间并行（独立 episode），单 episode 内严格串行
run_batch(all_questions, workers=N)            # 每个 worker 独占模型副本/请求队列
```

### 11.9 "多 subagent" 何时值得引入

- **默认不引入**：视觉/音频/ASR/OCR 都作为工具调用，契约固定、结果可缓存。
- **只在两种场景引入 subagent**：
  1. **失败复核（critic）**：verify 失败或跨模型分歧时，交给第三方模型
     （如 Nemotron-Omni-30B）或人工队列独立复核，避免"同一模型自证"；
  2. **长视频分块并行**：>300s 的视频按 chunk 分给多个音频专家组并行处理，
     再由 controller 汇总（这是"任务并行"，不是"自由协商"）。
- 自由多 agent（各自规划、互相聊天）会带来不可复现、成本翻倍、错误累积，
  **不适用于要求质量与审计的标注流水线**。

### 11.10 与主 pipeline 的关系与落地清单

- 第 4 章的 P1–P4 从"固定阶段"降级为 **controller 可调用的工具集**；
  Phase 2/3 仍是确定性阶段，其中 verifier 同时作为 in-loop 工具与终审工具。
- 新增代码：`src/omni_opsd/worldsense/agent/`
  （`controller.py` / `tools.py` / `episode.py` / `budget.py` / `trace.py`）；
  复用 `temporal/views.py`、`data/atomic_views.py`（渲染视角）、
  `temporal/consistency.py`（跨模型投票）。
- 每个 episode 产出一份 `agent_trace.jsonl`（action/observation/成本/结果），
  便于审计与人工抽检定位问题。
- 成本控制：先跑 300–500 题 pilot，统计平均 turns / inspect 次数 / GPU 秒，
  再决定全量预算（预期多数题 2–4 轮即可早停）。

### 11.11 环境事实（已核查）

| 事项 | 结论 |
| --- | --- |
| Qwen3-Omni-30B tool-calling | ✅ chat template 含 `tools` / `tool_call` / `tool_response` |
| 现成 agent 框架 | ❌ 环境未安装（无 qwen-agent / langchain / autogen 等） |
| 可用的推理/客户端 | ✅ `ms_swift`、`transformers`、`openai`（可接 vLLM OpenAI API）、`vllm` |
| 推荐路线 | 结构化 JSON 协议 + 显式 Python 编排（不依赖外部 agent 框架） |
| 媒体重看 | 必须由 harness 渲染（`tool_response` 无法携带音视频） |
| 音频一致性 | 标注前必须验证 vLLM 不丢音频（`USE_AUDIO_IN_VIDEO=1`，`vllm_drop_audio=0`），否则退回 Transformers |

---

## 12. 实现状态：自适应证据定位（localization-only v1，2026-09-21）

> 本轮只搭建"证据定位"：tool-loop 探索 + 提交候选区间（含 observation）。
> 候选融合（Phase 2）与充分性验证（Phase 3）**尚未实现**，按用户要求后续再做。

### 12.1 代码清单（已写入仓库）

| 文件 | 作用 |
| --- | --- |
| `src/omni_opsd/worldsense/__init__.py` | 包出口 |
| `src/omni_opsd/worldsense/schema.py` | `QuestionRecord` / `Interval` / trace 事件数据结构 |
| `src/omni_opsd/worldsense/config.py` | `AgentConfig`：turn/inspect/时长/fps/pixels 等硬限制 |
| `src/omni_opsd/worldsense/media.py` | decord/ffprobe 探测（时长、fps、帧数、有无音轨）+ 探测缓存 |
| `src/omni_opsd/worldsense/dataset.py` | `worldsense_qa.json` 扁平化；**先过滤再探测**（避免误扫 1662 个视频） |
| `src/omni_opsd/worldsense/views.py` | 把模型请求渲染成 Qwen 结构化视频/音频描述符；interleaved/mixed 两种模式；旧视图上下文裁剪 |
| `src/omni_opsd/worldsense/prompts.py` | 主 agent 提示词（含全部元数据字段 + 证据类型指引 + JSON 协议） |
| `src/omni_opsd/worldsense/protocol.py` | 严格 JSON 解析（容忍代码块/前后缀文本）、action 校验 |
| `src/omni_opsd/worldsense/clients.py` | `MockOmniClient`（CPU 测试）与 `TransformersOmniClient`（Qwen3-Omni / Qwen2.5-Omni） |
| `src/omni_opsd/worldsense/episode.py` | 单题 tool-loop：inspect / get_media_info / submit、拒绝-重试、预算控制 |
| `src/omni_opsd/worldsense/runner.py` | 批量执行、断点续跑（按 question_id 跳过）、逐题 trace 落盘 |
| `scripts/annotate_worldsense_evidence.py` | CLI：`--mock` / `--dry-run` / 真实模型运行 |
| `scripts/run_worldsense_evidence_agent.sh` | GPU 启动脚本（自动配 ffmpeg/decord 环境变量） |
| `tests/test_worldsense_agent.py` | 11 个 CPU 单测 |

### 12.2 Prompt 关键升级（按用户要求）

系统提示词中的元数据块包含：`video_duration_seconds`（探测值，权威）、
`domain`、`sub_category`、`audio_class`、`task_domain`、`task_type`、
`video_synopsis`（caption，标注为弱提示）、`question`、`options`。

并加入**证据类型指引**：Temporal/Event Sorting → 多个短窗 + av；Text/Diagram →
提高 max_pixels；Audio Recognition / audio_class 含 Music → 用 audio/av 视图听；
含 Speech → 听口播；含 Event → 看 av 关键动作；不确定时先粗扫最有希望区域再收窄。

输出契约要求每个 action 都带 `observation`；`submit` 时**每个区间**都要带
自己的 `observation`（说明该区间为何是关键证据），并给出 `confidence`。
硬规则里明确禁止输出答案文本/选项字母。

### 12.3 已完成的验证（无 GPU）

1. **单元测试 11/11 通过**：元数据齐全性、JSON 解析容错、参数 clamp、混合模态渲染
   （silent 视图确实无音频）、上下文裁剪、inspect→submit 流程、非法 submit 拒绝后重试、
   预算耗尽、区间总量超限拒绝、runner 断点续跑 + trace 往返。
2. **真实 processor 冒烟**（Qwen3-Omni-30B processor，CPU）：
   - av-only → `use_audio_in_video=True`，`input_ids (1,1441)`、`video_grid_thw (1,3)`、
     `input_features (1,128,500)`、自动抽取 1 条音频；
   - mixed（av + silent） → `use_audio_in_video=False`，`video_grid_thw (2,3)`，
     av 视图带显式音频、silent 视图无音频。
3. **mock 端到端**：用真实 WorldSense 视频探测（如 `AAWgrzYx.mp4`：131.36s / 25fps /
   有音轨 / 640×360），4 轮完成 inspect→get_media_info→inspect→submit，
   输出 JSONL + trace（含 events/views/messages）正常。

### 12.4 运行方式

CPU 冒烟（可复现）：

```bash
PYTHONPATH=src python scripts/annotate_worldsense_evidence.py \
  --qa  /share/home/ylhu/Light-Omni/data/omni_benchmarks/WorldSense/worldsense_qa.json \
  --video-root /share/home/ylhu/Light-Omni/data/omni_benchmarks/WorldSense/videos \
  --output /tmp/ws_agent/evidence.jsonl --limit 3 --mock
```

GPU 真实运行（需空闲卡）：

```bash
OMNI_OPSD_AGENT_DEVICE=cuda:0 OMNI_OPSD_AGENT_LIMIT=5 \
  bash scripts/run_worldsense_evidence_agent.sh
```

运行环境要点：
- 必须把 `ffmpeg` 加入 PATH（脚本已自动加 `omniagent_gyh/bin`），否则
  `audioread.ffdec.NotInstalledError`；
- `FORCE_QWENVL_VIDEO_READER=decord`、`USE_AUDIO_IN_VIDEO=1`（脚本已设）；
- processor 提示 `cap_pixels_per_frame` 将在 transformers v5.22 改变默认行为，
  届时需显式设定以保持与参考实现一致；
- 断点续跑：同一 `--output` 重复执行会跳过已有 `question_id`，可安全重跑；
- 每题都会写一行 trace（`*.trace.jsonl`），记录 action、渲染参数、events 与消息，
  便于审计与人工抽检。

### 12.5 待办（下一步）

1. **GPU 实测**：当前 gpu05 被他人 sglang 占满且其余节点无 slurm 作业，
   待有空闲卡后先跑 5–10 题真实 Qwen3-Omni-30B，检查 JSON 协议遵循率、
   平均轮数、延迟与各区间的 observation 质量。
2. 视实测结果调 prompt（提高首轮命中率、减少无效 inspect）。
3. 再接入 Phase 2 候选融合与 Phase 3 充分性验证（V1/V4），复用现有 verifier 设计。

---

## 13. 正式标注运行与 V1 实现（2026-09-22）

### 13.1 崩溃根因与修复（关键）

首轮真实运行出现**间歇性原生段错误**（dmesg: `python[pid]: segfault at 0 ip 0`，
rc=139 / rc=1），模式为"每个进程第 1 题成功、下一题崩溃"，无法用于批量标注。

排查结论：

- 日志中的 `[NVBLAS] cublasXtSgemm failed with error=1` 是**干扰项**（可复现且不致命）；
- 真正的崩溃点在 **Qwen3-Omni 的 talker / code2wav 路径**。查看
  `modeling_qwen3_omni_moe.py` 的 `generate`：`generate_audio = return_audio and
  self.has_talker`，即只要 talker 驻留，某些路径就会走到 code2wav；
- 修复：对 `qwen3_omni_moe` 调用 `model.disable_talker()`（删除 talker+code2wav，
  `has_talker=False`），此后 `generate` **直接返回 thinker 结果**，完全跳过 talker。
- 修复后稳定性探针：**连续 5 个不同题目的多模态推理 5/5 通过，显存无增长**
  （alloc 恒定 59.2G/卡），输出质量良好（例："the number 50 ... rear wing at the
  start"、"two people playing violins"）。
- 注意：`disable_talker()` 对 **Qwen2.5-Omni 不适用**（2.5 的 generate 仍引用
  `self.talker` → AttributeError），因此仅在 Qwen3 分支调用。

### 13.2 正式标注设计（8 卡全用）

| 组件 | 实现 |
| --- | --- |
| 分片 | `scripts/make_worldsense_shards.py`：按视频时长降序轮转分配到 8 片，每片 396–397 题、约 930 分钟视频，负载均衡 |
| 媒体索引 | `scripts/build_worldsense_media_index.py`：16 进程并行探测 1662 个视频（时长/fps/帧数/音轨），一次构建、8 worker 只读加载（无竞态） |
| Worker | `run_agent_shard.sh`：每 worker 绑定 1 张 GPU（`CUDA_VISIBLE_DEVICES=i`、`--device auto`），`flock` 独占分片锁防双写 |
| 看护 | `supervise_shard.sh`：崩溃后按 question_id 断点续跑（最多 200 次尝试），rc=3（锁被占）直接退出 |
| 启动 | `launch_formal_annotation.sh`：一次拉起 8 个分片（GPU 0–7） |
| 监控 | `scripts/monitor_worldsense_annotation.py`：每 30 分钟检查各分片进度、心跳（evidence/supervisor 日志 mtime）、GPU 占用；心跳陈旧且对应 GPU 空闲时才重启分片 |
| 上下文保护 | 正式配置 `--max-view-seconds 60 --max-media-in-context 1`，限制单视图与上下文规模，防止长视频峰值失控 |
| 断行保护 | `runner._load_done` 在续跑时截断崩溃产生的半行 JSON，避免污染产物 |

分片目录：

```
output/worldsense_evidence_formal/
  media_index.json          # 预构建媒体索引
  shards_manifest.json      # 分片清单（题数/时长/ids 文件）
  shard{0..7}.ids           # 各分片 question_id 清单
  shard{0..7}/evidence.jsonl       # 定位结果（含 observation）
  shard{0..7}/evidence.trace.jsonl # 逐题完整 trace（actions/views/messages）
  monitor.log / monitor_status.json
```

启动后实测吞吐：**约 10 题/分钟**（8 卡并行），3172 题预计 **5–6 小时**完成；
状态分布示例：`submitted` 为主，少量 `no_evidence` / `budget_exhausted`
（这些行不进入后续验证/训练）。

### 13.3 V1 充分性验证实现

新增文件：

| 文件 | 作用 |
| --- | --- |
| `src/omni_opsd/worldsense/verify.py` | V1 核心：`VerifierConfig` / `SufficiencyResult` / 区间-only prompt / 字母解析 / 概率打分 / `verify_interval` / `verify_question` |
| `scripts/verify_worldsense_evidence.py` | V1 CLI：读定位产物 → 逐区间验证 → 输出每题的 correctness 与分布指标；`--mock` 可 CPU 运行 |
| `output/worldsense_agent_jobs/run_verify_shard.sh` | V1 单卡 worker（flock + 续跑） |
| `output/worldsense_agent_jobs/supervise_verify.sh` | V1 看护（崩溃续跑） |
| `output/worldsense_agent_jobs/launch_verify_all.sh` | V1 8 分片启动 |
| `tests/test_worldsense_verify.py` | 8 个 CPU 单测（字母解析、softmax、prompt 无答案泄漏、正确/错误判定、最优区间选择、无金标场景） |

关键设计：

- Teacher 只看到**候选区间媒体 + 题面 + 选项**，不提供答案；
- `generate_with_scores`（client 新增）用一次贪心生成同时得到答案文本与**首步 logits**，
  据此在选项字母 token 上重归一化得到 `p_true` / `p_max` / `margin` / `entropy`；
- 已验证 Qwen3 的 `output_scores` 会转发给 thinker，且 `disable_talker()` 后
  直接返回 thinker 结果（避免 talker 崩溃）；
- 每题汇总 `any_correct` / `n_correct` / `best_interval`（按 `p_true` 排序），
  供 Phase 2 融合与后续"置信路由"使用。

V1 单元测试与 mock 端到端均通过；**真实 GPU 运行待标注任务结束或 GPU 空闲后执行**：

```bash
# 在 gpu06 上（对应标注分片完成后）
OMNI_OPSD_SHARD_INDEX=i OMNI_OPSD_AGENT_CUDA_VISIBLE_DEVICES=i \
  bash output/worldsense_agent_jobs/supervise_verify.sh
# 或一次拉起 8 个分片
bash output/worldsense_agent_jobs/launch_verify_all.sh
```

### 13.4 下一步

1. 标注完成后跑 V1（8 卡并行），产出 `worldsense_verify_formal/shard*/verify.jsonl`；
2. 汇总 sufficiency_rate、各 task_type / audio_class 的通过率与分布指标；
3. 实现 Phase 2 候选融合（按 V1 通过的区间聚类/择一）与 V4 跨模型一致性；
4. 用通过 V1 的区间物化 `clue_opsd` 训练数据（复用 `prepare_*` 与预算/帧修复流程）。

---

## 14. 标注完成与 V1 正式运行（2026-09-22）

### 14.1 标注正式实验完成（3172/3172）

- 运行区间：00:21 → 约 05:00（**4.6 小时**，8 卡并行，约 11 题/分钟）。
- 结果分布：`submitted` **2833（89.3%）**、`no_evidence` 248（7.8%）、
  `budget_exhausted` 91（2.9%）。
- 区间质量：平均 **1.25 个区间/题**（434 题多区间），区间时长 min 1.0s /
  p50 **5.0s** / max 78s；`observation` 中位数 **245 字符**（91 条为空，均为
  `budget_exhausted` 行）；3073/3172 行带 `confidence`。
- 过程质量：平均 **3.13 轮**、**1.82 次 inspect**（上限 3）；协议解析错误仅
  29 行（0.9%）；586 次 submit 被规则拒绝后重试成功。
- 人工抽检 5 例：4/5 的 observation 与真实证据一致且时间点合理，例如
  `NBzSgbJK`（口播 4 国）精确定位到 180–185s 并列出 BRITAIN/GERMANY/
  AMERICA/FRANCE；`dvOkwKAs`（金属门在白衣女子右侧）定位 5–10s 且描述正确。
  1 例遗漏（`UYkFSXsh` 观众欢呼在开头，模型判为 no_evidence）。

### 14.2 编排器缺陷与修复（V1 首次启动失败）

- 现象：编排器报告 `verify-start(ok)`，但 3 小时 0 行产出、8 卡空转。
- 根因：编排器启动验证时**未创建 `verify_dir/shardN/` 子目录**，shell 重定向
  `> shardN/supervisor_restart.log` 失败，worker 立即退出；而 `echo started=$!`
  仍返回 0，造成"启动成功"的假象。
- 修复：`launch_shard()` 在远程命令前加 `mkdir -p`；同时本地补齐 8 个分片目录；
  旧的部分结果归档到 `output/worldsense_verify_formal_partial_v1/`。

### 14.3 V1 正式实验（8 卡）

- 08:31 启动，配置：`fps=2`、`max_pixels=156800`（默认）+
  **按任务类型自适应提升到 313600**（Text and Diagram / Attribute / Counting /
  Fine-grained 等小字与细粒度任务），`modality=av`，贪心生成 4 tokens。
- 实测吞吐约 **150 行/分钟**（8 卡），3172 题约 **20–40 分钟**完成；
  `p_true / margin / entropy` 全部正常产出（已验证 Qwen3 的 `output_scores`
  会转发给 thinker）。
- 早前低分辨率部分结果（810 题，已归档）的初步指标：
  - 问题级 `sufficiency_rate = 0.406`，区间级 `interval_accuracy = 0.370`；
  - 通过区间的 `best_p_true` 均值 0.845 / 中位数 0.917，`best_margin` 均值 0.728；
  - 分任务：Audio Recognition 0.61、Event Sorting 0.57、Audio Source
    Localization 0.53、Emotion Change 0.55 较高；Object Counting 0.22、
    Action Counting 0.16、Temporal Localization 0.28、Causal Reasoning 0.32 较低；
  - 分音频类型：Music 0.449、Event 0.396、Speech 0.393。
- 说明：该 0.406 是**单区间充分率**（raw 定位候选），不是最终数据集质量。
  按方案设计，未通过 V1 的题将进入 Phase 2 融合/高分辨率复验/人工复核，
  最终训练集只保留通过 V1 的区间。

### 14.4 监控

`scripts/worldsense_pipeline_monitor.py` 守护进程（每 30 分钟）：
- 两阶段状态机（annotate → verify → done），按分片推进；
- 心跳陈旧且对应 GPU 空闲时自动重启；
- 全部完成后自动生成 `output/worldsense_verify_formal/summary.json` 并退出。
