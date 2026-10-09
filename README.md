> **2026-10-09 交接更新：请先阅读 [HADOFF.md](HADOFF.md)。** 本页保留早期流程；最新 1000 题实验、结果、迁移步骤及未上传资产以交接文档为准。

# omni-clue-opsd — WorldSense 证据条件增益筛选与 CLUE-OPSD 训练交接文档

> **交接日期**：2026-09-25
> **交接内容**：① WorldSense 全量标注数据 ② Full/Gold 模型筛选结果与指标 ③ 筛选出的 1.5k 训练集 ④ 全部代码
> **上游背景**：OmniVideo-5K 的"证据条件增益"方法论（见 `training_code/docs/OMNIVIDEO_5K_EXPERIMENT_SUMMARY (2).md` 第 1/6 节）

---

## 目录

1. [项目目标与数据流](#1-项目目标与数据流)
2. [仓库结构与数据字典](#2-仓库结构与数据字典)
3. [环境与依赖](#3-环境与依赖)
4. [阶段一：WorldSense 全量标注（Qwen3.8-Omni-Flash API）](#4-阶段一worldsense-全量标注)
5. [阶段二：证据条件增益筛选（Full vs Gold，Qwen2.5-Omni-7B）](#5-阶段二证据条件增益筛选)
6. [多媒体参数设置（重点）](#6-多媒体参数设置重点)
7. [阶段三：训练（SFT / 全参 / 后续 OPSD / CLUE-OPSD）](#7-阶段三训练)
8. [端到端复现步骤](#8-端到端复现步骤)
9. [已知问题与后续工作](#9-已知问题与后续工作)

---

## 1. 项目目标与数据流

**研究目标**：将 **evidence-conditioned teacher**（能看到"黄金证据区间"的教师）的优势，通过蒸馏/自蒸馏转移给 **full-video student**（只能看整片的学生）。为此先在数据层面筛出**具有明显证据条件增益**的题目。

```
WorldSense 原始数据（3172 题 / 1662 视频）
        │  阶段一：Qwen3.8-Omni-Flash API 自动标注
        ▼
标注数据（每题：证据区间 + 分析 + 置信度）        ← data/annotation/
        │  阶段二：Qwen2.5-Omni-7B 冻结模型，Full（整片）vs Gold（证据段）
        ▼                                          配对打分：Δacc / Δp
筛选结果（3079 题分数 + 2717 题配对指标）          ← data/screening/
        │  排序选样（Δacc → Δp → Δlogp → P(GT)）
        ▼
选定集 selected_1500（1036 视频，+19.2pp gap）     ← data/selection/
        │  格式转换 + 帧数过滤
        ▼
训练 JSONL（1453 题，swift 动态预算格式）          ← data/sft/
        │  阶段三：SFT（LoRA 已跑通）/ OPSD / CLUE-OPSD
        ▼
        checkpoint…
```

**关键数字一览**

| 阶段 | 规模 | 结果 |
| --- | --- | --- |
| WorldSense 全量 | 3172 题 / 1662 视频（16–657s，中位 90s） | — |
| API 标注 | 3172 题全部完成 | **submitted 3118（98.3%）**；blocked 41；no_evidence 11；budget_exhausted 2 |
| 筛选用候选 | 3118 → **3079 题 / 1626 视频**（排除选项重复 17、区间不可用 22） | 媒体审计 0 隔离 |
| Full vs Gold（全部） | 3079 题 | Full **47.22%** vs Gold **50.31%**（**+3.09pp**，McNemar p=0.0001） |
| ≤300s 子集 | 2717 题 / 1436 视频 | Full 47.52% vs Gold 49.87%（+2.36pp） |
| **选定集 selected_1500_300s** | **1500 题 / 1036 视频** | Full **42.13%** → Gold **61.33%**，**+19.20pp**，平均 Δp +0.1664 |
| 训练数据 | 1453 题（去掉 <64 帧的 47 题） | swift 动态预算格式 |
| SFT-LoRA 训练 | 138 步（3 epochs） | 8h54m，loss 2.247 → 1.636，checkpoint@92/138 |

---

## 2. 仓库结构与数据字典

```
omni-clue-opsd/
├── README.md                     ← 本文档
├── training_code/                ← 标注 + 训练代码（原 OmniOPSD_training_code_20260907，已剔除 output/ 等大目录）
│   ├── src/omni_opsd/            ← worldsense 标注管线、dynamic_budget（已修 512 余量 bug）、数据/训练工具
│   ├── scripts/                  ← 标注/筛选辅助/训练启动脚本（含本次新增的 worldsense 相关脚本）
│   ├── configs/                  ← zero2.json、fsdp2*.json（训练切分配置）
│   ├── docs/                     ← 全部设计文档（含本项目的 WORLDSENSE_*、Train_Settings.md、OmniVideo 参考）
│   ├── tests/                    ← 单测（worldsense 标注/验证/API 后端）
│   └── patches/                  ← ms-swift 补丁
├── screening_code/               ← Full vs Gold 筛选代码（原父仓库相关部分）
│   ├── scripts/worldsense_gap/   ← 候选构建 / 预算 / 打分 / 选样 / 监控 / vLLM 服务
│   ├── scripts/*.py              ← 复用的 OmniVideo 打分协议（score_omnivideo_gap.py 等）
│   ├── select_omnivideo_gap_5000.py
│   └── src/omni_opsd/selection/  ← 媒体解码/编码（decode_video 增加了 min_pixels/max_pixels 参数）
└── data/
    ├── annotation/               ← 阶段一产物
    │   ├── merged.evidence.jsonl   3172 行：每题最终标注（见字段表）
    │   ├── captions.jsonl          2563 条：模型读到的"带时间戳 caption"（caption 缓存，含 key/耗时）
    │   ├── stats.json              标注统计（调用量/token/耗时分布）
    │   └── traces.tar.gz           32 分片的完整交互日志（每轮 action/inspect/submit，审计用，23.6MB 压缩）
    ├── screening/                ← 阶段二产物
    │   ├── candidates.jsonl        3079 行：筛选输入（question/choices/answer/evidence_spans/时长/分辨率）
    │   ├── full.jsonl              3079 行：Full（整片）打分（概率分布/预测/P(GT)/token 用量/帧数/网格）
    │   ├── gold.jsonl              3079 行：Gold（证据段）打分（同上）
    │   ├── per_question.jsonl      2717 行：**配对指标**（Δacc/Δp/Δlogp/tier/两级预测与概率）
    │   ├── summary.json            总体 + 分任务类型统计（Full/Gold 准确率、Δacc、Δp）
    │   ├── selection_summary_300s.json  选定集统计（1500 题、+19.20pp）
    │   └── audit_tier_c.jsonl      224 行：Δacc=−1（证据反而有害）的审计清单
    ├── selection/                ← 阶段二选样产物
    │   ├── selected_1500_300s.jsonl ★ 最终采用（1500 题/1036 视频，≤300s）
    │   ├── selected_1500.jsonl      旧版（未过滤 >300s，+23.3pp，口径不一致）
    │   └── selected_2000.jsonl      N=2000 版本（+17.5pp）
    └── sft/                      ← 阶段三训练数据
        ├── sft.jsonl               ★ 1453 行 swift SFT 格式（messages + videos + 动态预算）
        └── prepare_report.json     构建报告
```

### 核心字段说明

**`data/annotation/merged.evidence.jsonl`**（每题一行）

| 字段 | 含义 |
| --- | --- |
| `question_id` | `{video_id}::task{N}` |
| `status` | `submitted` / `blocked`(内容审核) / `no_evidence` / `budget_exhausted` |
| `clue_intervals` | **证据区间** `[[start, end], ...]`（秒，1–4 段） |
| `observation` | 模型给出证据的分析文本（约 400 字符） |
| `confidence` / `turns` / `inspect_calls` / `elapsed_s` | 自评置信度 / 交互轮数 / 媒体查看次数 / 耗时 |
| `media_views` | 模型实际查看过的视图（含 max_pixels/fps/modality，可审计） |

**`data/screening/per_question.jsonl`**（每题一行，配对指标）

| 字段 | 含义 |
| --- | --- |
| `delta_acc` | `1[Gold对] − 1[Full对] ∈ {+1,0,−1}` |
| `delta_p` | `P(GT|Gold) − P(GT|Full)`（正确答案概率差） |
| `delta_logp` | 正确答案对数概率差（排序同分时使用） |
| `tier` | **A**（Δacc=+1 强增益）/ **B**（Δacc=0 且 Δp≥0.1 概率增益）/ **C**（Δacc=−1 疑似标注问题）/ **D**（其余） |
| `pred_full/gold`、`correct_full/gold`、`p_gt_full/gold` | 两级预测与概率 |

**`data/sft/sft.jsonl`**（swift 训练格式）

```json
{
  "messages": [
    {"role":"user","content":"<video>\nQuestion: ...\nOptions:\nA. ...\n...\nBriefly analyze ... <answer>...</answer>"},
    {"role":"assistant","content":"{teacher 分析}\n<answer>{字母}</answer>"}
  ],
  "videos": [{"video":"/path.mp4","video_start":0.0,"video_end":189.2,"nframes":226,
              "resized_height":280,"resized_width":560,"min_pixels":3136,"max_pixels":156800}],
  "sampling_contract": {...}, "dynamic_student_budget": {...},
  "experiment_arm": "sft", "sft_target_source": "api_annotation_observation + gold_answer"
}
```

---

## 3. 环境与依赖

| 项 | 值 |
| --- | --- |
| Python 环境（训练/标注/vLLM） | `/share/home/ylhu/.conda/envs/vllm`（transformers 5.6.0 / vLLM 0.11.2 / torch 2.9.0） |
| 本地模型 | `/share/home/ylhu/models/Qwen2.5-Omni-7B`（筛选评分 + 训练基座） |
| 云端 API | Qwen3.8-Omni-Flash（DashScope，仅标注阶段用；密钥经环境变量注入，**勿入库**） |
| 训练框架 | ms-swift 4.6.0.dev0 + 自定义补丁（`training_code/patches/`，用 `scripts/apply_*_patches.sh` 应用） |
| ffmpeg | `/share/home/ylhu/.conda/envs/omniagent_gyh/bin/ffmpeg`（切片/转码用） |
| 硬件 | 4×A100-80GB（单节点） |

> ⚠️ **两个同名的 `omni_opsd` 包**：`training_code/src/omni_opsd`（标注/训练，含 `worldsense/`）与 `screening_code/src/omni_opsd`（筛选，含 `selection/`）。运行不同阶段时请把对应目录放在 `PYTHONPATH` 最前。

---

## 4. 阶段一：WorldSense 全量标注

**做什么**：让 Qwen3.8-Omni-Flash 观看整段视频（含音频），自主决定要看哪些片段，最终提交"能支撑问题答案"的证据区间。

**工作流（每题 3 个阶段）**：

1. **caption 阶段**：整片以源分辨率采样（≤1312 帧，约 2fps），让模型生成**带时间戳的整片描述**（缓存到 `captions.jsonl`，键 = `question_id|caption_v3|fps|px`，重复运行不重算）；
2. **规划 + inspect**：把 caption 注入上下文，模型可多次请求查看指定时间窗的高清片段（base64 内联）；
3. **submit**：提交 1–4 段证据区间 + 分析，或声明无证据。

**关键运行参数**（`training_code/scripts/annotate_worldsense_evidence.py`）：

| 参数 | 值 | 说明 |
| --- | --- | --- |
| caption 上限 | 8000 tokens | 截断自动重试（附格式提醒） |
| 最大轮数 / 最大 inspect | 12 / 6 | 预算耗尽则 `budget_exhausted` |
| 单窗最长 | 120s | 防上下文爆炸 |
| 切片上限 | **7MB**（base64 后 ≈9.3MB < 10MB 接口限制） | `--max-clip-mb 7` |
| 代理策略 | `size_aware` | 按请求像素抽帧，不超源分辨率 |
| 上下文媒体 | 最多 2 个视图 | `--max-media-in-context 2` |
| 并发 | 32 分片 × 1（实测 0 限流） | 全量 1.6 小时完成 |

**产物**：`data/annotation/`（3172 题，98.3% 提交成功；平均 159s/题、2.8 轮、1.70 次 inspect；共 4570 个区间，中位 8s）。

---

## 5. 阶段二：证据条件增益筛选

### 5.1 原理

用**同一个冻结模型**（Qwen2.5-Omni-7B）在两种输入下各答一次，只有媒体范围不同：

- **Full**：完整视频 + 完整音频
- **Gold**：标注的证据区间（多区间共享一份帧预算）+ 对应音频

打分方式：**首 token 的 A/B/C/D 选项字母概率归一化**（vLLM 的 `logprobs` + `allowed_token_ids` 精确取字母），预测 = argmax。
每题计算：`Δacc = 1[Gold对] − 1[Full对]`；`Δp = P(GT|Gold) − P(GT|Full)`。

### 5.2 完整流程与命令

```bash
cd screening_code
export PYTHONPATH=$PWD/src:$PWD:$PWD/scripts

# ① 候选构建（从标注结果 + QA 生成统一格式；预筛非法/重复选项/不可用区间）
python scripts/worldsense_gap/prepare_candidates.py \
  --annotation <repo>/data/annotation/merged.evidence.jsonl \
  --qa /path/to/worldsense_qa.json \
  --media-index /path/to/media_index.json \
  --video-root /path/to/WorldSense/videos \
  --output ../outputs/worldsense_gap/candidates.jsonl
# → 3079 题（排除 17 选项重复 + 22 区间不可用）

# ② 媒体审计（元数据 + 哈希，17 秒；产出每视频的帧时间戳索引）
python scripts/worldsense_gap/audit.py \
  --canonical ../outputs/worldsense_gap/candidates.jsonl \
  --output ../outputs/worldsense_gap/audit.json --workers 8
# → 0 隔离

# ③ 启动两个 vLLM 服务（Full / Gold 各一套；split 模式 = 视频帧 + 独立音频）
bash scripts/worldsense_gap/serve_vllm.sh 0 8091 outputs/.../vllm_full.log   # 节点 A
bash scripts/worldsense_gap/serve_vllm.sh 0 8092 outputs/.../vllm_gold.log   # 节点 B
# 探针：python scripts/check_omnivideo_gap_service.py --endpoint ... --require-mode split

# ④ 打分（每题两个视图；逐请求下发动态预算 max_pixels/fps；resume 安全）
python scripts/worldsense_gap/score.py \
  --canonical ../outputs/worldsense_gap/candidates.jsonl \
  --audit ../outputs/worldsense_gap/audit.json \
  --model-dir /path/to/Qwen2.5-Omni-7B \
  --endpoint http://<nodeA>:8091/v1/chat/completions \
  --output-dir ../outputs/worldsense_gap/full/full_run --views full --workers 12
# （Gold 同理，指向 8092）

# ⑤ 配对指标 + 分层 + 选样（N=1500，≤300s）
python scripts/worldsense_gap/select_gap.py \
  --run-dir ../outputs/worldsense_gap/full \
  --canonical ../outputs/worldsense_gap/candidates.jsonl \
  --output-dir ../outputs/worldsense_gap/metrics \
  --target-size 1500 --max-duration 300

# ⑥ 30 分钟监控（进度/准确率/失败/ETA）
python scripts/worldsense_gap/monitor.py --run-dir ../outputs/worldsense_gap/full --expected 3079 --interval 1800
```

### 5.3 筛选结果

| 数据集 | 题数 | Full | Gold | 差距 | Δp 均值 | Tier A/B/C |
| --- | --- | --- | --- | --- | --- | --- |
| 全部候选 | 3079 | 47.22% | 50.31% | **+3.09pp**（p=0.0001） | +0.0264 | 349/467/254 |
| ≤300s 子集 | 2717 | 47.52% | 49.87% | +2.36pp | +0.0197 | 288/404/224 |
| >300s（被排除） | 362 | 45.03% | 53.59% | +8.56pp（**失真，见下**） | +0.0764 | 61/63/30 |
| **选定 1500（≤300s）** | **1500** | **42.13%** | **61.33%** | **+19.20pp** | **+0.1664** | 288/404/— |

**选样规则**（与 OmniVideo 一致）：排序键 `Δacc ↓ → Δp ↓ → Δlogp ↓ → P(GT|Gold) ↓ → question_id ↑`，每视频最多 5 题，取前 N 条。N 与增益的取舍：N=500→+57.6pp / 1000→+28.8pp / **1500→+19.2pp** / 2000→+14.4pp（采样率越低增益越大）。

**为什么排除 >300s（362 题 / 190 视频，11.7%）**：Qwen2.5-Omni 的音频前端**硬截断在 300 秒**（实测：1500/5000/7500 tokens 对应 60/200/300s，400s 与 900s 仍只有 7500 tokens），因此对 >300s 的视频，"Full 条件"实际只看得到前 300s 音频；而当时的筛选预算按完整时长预留音频 token，**压低了下发分辨率 → Full 得分被无辜拉低 → gap 虚高（+8.56pp 不可信）**。过滤后口径干净。

### 5.4 期间的三个关键修复（都已入库）

1. **`dynamic_budget.py` 的 512 余量 bug**：原实现 `可用视觉 = 上下文 − 文本 − 音频`（未留安全余量），当视觉 cap 不再生效（**时长 > ~268s**）时合计必然逼近 32768，最终检查 `+512` 直接报错。修复：新增 `CONTEXT_HEADROOM_TOKENS = 512` 并提前扣除（full 版与 clue 版都改了）。
2. **音频 300s 封顶**：训练侧 `audio_seconds = min(duration, 300)` 是正确的（与实测一致）→ 筛选侧统一。
3. **`select.py` 命名冲突**：筛选目录里叫 `select.py` 会覆盖标准库 `select`（transformers 依赖 httpx 会崩）→ 改名 `select_gap.py`。

---

## 6. 多媒体参数设置（重点）

### 6.1 Qwen2.5-Omni-7B 的硬事实（从模型 config 读取，非估计）

| 参数 | 值 | 来源 |
| --- | --- | --- |
| 上下文长度 | **32,768** | `max_position_embeddings` |
| 视觉 patch / 合并 | 14×14 patch，2×2 merge → **28×28 像素 = 1 视觉 token** | `patch_size=14, merge_size=2` |
| **时间合并** | **相邻 2 帧合并为 1 个时间组** | `temporal_patch_size=2` |
| 音频 token 率 | **25 token/秒** | `position_id_per_seconds=25` |
| 音频前端上限 | **300 秒**（超出截断，实测） | `n_samples=4.8M = 300s@16kHz` |
| GQA | 4 个 KV 头（KV cache 很小：32k token ≈ 1.8GB） | `num_key_value_heads=4` |

### 6.2 动态预算（训练与筛选统一口径）

```
① 音频预算 = ceil(min(时长, 300) × 25) + 64        # 64 = 每个视频区间的边界预留
② 文本预留 = 2048；安全余量 = 512
③ 可用视觉 = min(24000, 32768 − 2048 − 512 − 音频预算)
④ 目标帧数 nominal = min(时长 × 2fps, 300帧)
   买得起帧数 affordable = floor(可用视觉 / 100)   # 每帧至少 100 token 的设计下限
   实际帧数 nframes = min(nominal, 300, affordable)   # 不够就减帧，仍覆盖全程
⑤ 分辨率网格：在 pair_budget = min(256, 可用视觉×2/nframes) 内，
   选最接近源宽高比的 28 对齐网格（不超源分辨率、不做上采样）
⑥ 总检查：文本 + 音频 + 视觉 + 512 ≤ 32768
```

**Token 公式**：

```text
视觉token = ceil(采样帧数 / 2) × (缩放后高/28) × (缩放后宽/28)   ← 两帧合并口径
音频token = ceil(秒数 × 25)
```

**各时长档实例**（源 640×360）：

| 时长 | 音频 tok | 帧数 | 网格 | 视觉 tok | 合计(+文本) |
| --- | --- | --- | --- | --- | --- |
| 60s | 1,564 | 120 | 12×21 | 15,120 | 19,244 |
| 129s | 3,289 | 240 | 10×20 | 24,000 | 29,849 |
| 268s | 6,764 | 234 | 10×20 | 23,400 | 32,724 |
| 300s | 7,564 | 226 | 10×20 | 22,600 | 32,724 |
| 657s | 7,564（音频封顶） | 226 | 10×20 | 22,600 | 32,724 |

**分辨率设计窗口**：每采样帧 100–128 视觉 token（对应 10×20 ~ 12×21 网格）；WorldSense 87% 的题源分辨率是 640×360 → 原生上限仅 ~150 tok/帧，**提高预算无法再显著提升清晰度**，只能加帧率。

**一致性要求（重要）**：筛选的打分口径 = 训练的数据口径。若后续要改预算（如 fps 2→4、cap 24000→28000），**必须重跑筛选**，否则测得的 gap 不迁移。

---

## 7. 阶段三：训练

### 7.1 已跑通：SFT-LoRA（Qwen2.5-Omni-7B）

| 参数 | 值 | 来源 |
| --- | --- | --- |
| 数据 | `data/sft/sft.jsonl`（1453 题 / 1003 视频） | 本仓库 |
| 微调 | **LoRA r=64, α=128, target_modules=all-linear** | `scripts/run_worldsense_sft_lora_gpu07.sh` |
| 硬件 | 4×A100-80GB，nproc=4 | 同上 |
| batch | per_device **1** × grad_accum **8** = 全局 **32** | 同参考 3B 验证配置 |
| 步数 | **138**（= 3 epochs × 1453/32≈46），每 **46 步**存一次 | `OMNI_OPSD_MAX_STEPS/SAVE_STEPS` |
| checkpoint | `save_total_limit=3`（三个 epoch 都保留） | `OMNI_OPSD_SAVE_TOTAL_LIMIT` |
| 优化 | LR **1e-5**，cosine，warmup_ratio 0.03，max_grad_norm **0**，bf16 | `run_gap5000_sft_cuda.sh` |
| 显存 | gradient_checkpointing + sdpa + `use_logits_to_keep`（省 ~20GB logits） | 同上 |
| 其他 | seed 20260904，dataloader_workers=0，no_dataset_shuffle，`USE_AUDIO_IN_VIDEO=1` | 同上 |
| **实测** | **8h54m 完成，232s/step，loss 2.247 → 1.636** | 运行日志 |

**启动命令**：

```bash
cd training_code
export OMNI_OPSD_DATASET=$PWD/../data/sft/sft.jsonl      # 或仓库内相对路径
bash scripts/run_worldsense_sft_lora_gpu07.sh            # 4 卡 LoRA
```

### 7.2 SFT 全参（7B）的现状 ⚠️

**尚未跑通**，遇到基础设施限制（已排除的尝试都记录在此，供接手人省时间）：

| 尝试 | 结果 | 根因 |
| --- | --- | --- |
| DeepSpeed ZeRO-2/3（标准方案） | ✗ | **环境未装 deepspeed**，且集群网络（pip 代理/镜像）不通，无法安装 |
| FSDP2 默认预设（`--fsdp fsdp2`） | ✗ | accelerate 崩溃：`'Tensor' object has no attribute 'device_mesh'`（视觉/音频塔不在 `TRANSFORMER_BASED_WRAP` 内，参数不是 DTensor） |
| FSDP2 关 efficient loading | ✗ | CUDA OOM：**视觉塔激活 ≈70GB**（90k patch × 32 层未重算：226 帧高清视频进入 ViT） |
| FSDP2 + CPU offload | ✗ | 仍 OOM（激活是主因，offload 只挪状态） |
| FSDP2 + SIZE_BASED_WRAP | ✗ | `fully_shard does not support ModuleList` |

**可行路径（按推荐序）**：
1. 修复网络后 `pip install deepspeed` → ZeRO-3 + CPU offload（7B 全参的标准解）；
2. 换 **3B 模型**跑全参（纯 DDP 即可：6GB 权重 + 6GB 梯度 + 24GB 优化器）；
3. FSDP2 + **降低输入预算**（固定低分辨率 28672 px/帧，激活降到 ~10GB）——但输入分布与筛选口径不一致；
4. 双节点 8 卡（需多节点配置）。

### 7.3 OPSD / CLUE-OPSD（参数草案，待补全）

来自 `training_code/scripts/run_video_odyssey_training_arm.sh`：

| 参数 | 值 |
| --- | --- |
| `--rlhf_type` | gkd |
| `--lmbda` / `--beta` / `--temperature` | 1.0 / 0.5（JSD） / 1.0 |
| `--sft_alpha` | 0 |
| `--max_completion_length` | 8 |
| `--gkd_logits_topk` | 100（teacher top-k 词表蒸馏） |
| rollout | `--use_vllm true --vllm_mode colocate --vllm_gpu_memory_utilization 0.30 --sleep_level 1`，top_p 1.0 / top_k 20 |
| EMA teacher | `clue_ema_alpha=0.05`（LoRA-shadow EMA，补丁实现）；可选 `full_ema_teacher` + `gold_ce_alpha=0.25` |
| LR | 2e-6（OPSD/Clue-OPSD，参考文档 §6.1） |
| teacher 输入 | **证据区间音视频**（不接收标答）；student 输入与 SFT 完全一致 |
| 数据准备 | `training_code/scripts/prepare_dynamic_budget_training.py`（会同时产出 sft/opsd/clue_opsd 三份 JSONL，需把输入换成 WorldSense 版） |

### 7.4 训练代码设置速查

| 文件 | 作用 |
| --- | --- |
| `scripts/run_worldsense_sft_lora_gpu07.sh` | **本次 LoRA 启动脚本**（数据/步数/save 设置都在这里） |
| `scripts/run_worldsense_sft_full_gpu07.sh` | 全参尝试脚本（FSDP2 配置可切换） |
| `scripts/run_gap5000_sft_cuda.sh` | 通用 SFT 入口（LoRA 参数默认 r64/α128/LR1e-5） |
| `scripts/run_gap5000_sft_3b_full_cuda.sh` | 3B 全参入口（batch1×accum8 + 解冻全部模块） |
| `scripts/run_video_odyssey_training_arm.sh` | **训练臂通用实现**（构造 swift 命令；支持 `OMNI_OPSD_DEEPSPEED` / `OMNI_OPSD_FSDP` / `OMNI_OPSD_SAVE_TOTAL_LIMIT` 钩子） |
| `scripts/prepare_worldsense_sft_data.py` | **本次新增**：selected_1500_300s → swift SFT JSONL（含动态预算与契约字段） |
| `src/omni_opsd/data/dynamic_budget.py` | 动态预算（已修 512 余量；训练/筛选唯一权威实现） |
| `configs/fsdp2*.json` | FSDP2 配置存档（默认/关 efficient loading/CPU offload/SIZE_BASED_WRAP） |

---

## 8. 端到端复现步骤

```bash
# 0) 环境
conda activate /share/home/ylhu/.conda/envs/vllm
cd training_code && bash scripts/apply_ms_swift_patches.sh   # 应用 ms-swift 补丁

# 1) 标注（需要 DashScope API key 环境变量；32 并发约 1.6h）
python scripts/annotate_worldsense_evidence.py \
  --qa <WorldSense QA json> --video-root <videos> \
  --question-id-file <ids> --backend api --api-model qwen3.8-omni-flash \
  --caption-store <captions.jsonl> --max-clip-mb 7 --proxy-policy size_aware \
  --output <evidence.jsonl> --trace <trace.jsonl>

# 2) 筛选（见 §5.2 的六步：候选 → 审计 → vLLM 服务 → 打分 → 选样 → 监控）

# 3) 训练数据准备
python scripts/prepare_worldsense_sft_data.py \
  --selected ../data/selection/selected_1500_300s.jsonl \
  --annotation ../data/annotation/merged.evidence.jsonl \
  --qa <WorldSense QA json> --output data/worldsense_gap/sft/formal/data/sft.jsonl
# 注意：构建后需过滤 frames_per_video_input < 64 的样本（本仓库 data/sft/sft.jsonl 已是过滤版）

# 4) SFT-LoRA 训练
bash scripts/run_worldsense_sft_lora_gpu07.sh
```

---

## 9. 已知问题与后续工作

| # | 问题 | 状态/建议 |
| --- | --- | --- |
| 1 | **7B 全参训练未跑通**（deepspeed 缺失 + FSDP2 不兼容） | 修网络装 deepspeed，或换 3B/降预算（见 §7.2） |
| 2 | **Tier C 审计未完成**（224 题 Δacc=−1） | 任务分布：Temporal Localization 30、Causal Reasoning 23…；极端案例 P(GT) 由 1.00 崩到 0.01，大概率标注错误；建议人工复核 `data/screening/audit_tier_c.jsonl` |
| 3 | 47 题（<64 帧短视频）被训练数据过滤 | 如需保留可提高这些题的目标 fps（偏离统一口径） |
| 4 | LoRA 第 1 epoch checkpoint 因 `save_total_limit=2` 被轮转删除 | 已把默认改 3；如需重跑注意设置 |
| 5 | 音频 300s 截断导致 >300s 题被排除 | 若希望覆盖长视频，需改用分块音频/多段输入方案（超出当前模型前端能力） |
| 6 | 筛选与训练的 87% 题源分辨率仅 640×360 | 提高预算对清晰度无益（原生上限 ~150 tok/帧）；改 fps 前须重跑筛选 |
| 7 | 复数次运行可复现性 | 筛选为贪心/温度 1 采样，首 token 概率确定；但 API 标注存在 run-to-run 差异（pilot 实测 83% 同结果） |
| 8 | `select_omnivideo_gap_5000.py` 为 OmniVideo 原版 | WorldSense 版是 `screening_code/scripts/worldsense_gap/select_gap.py` |

---

## 附：关键文档索引（在 `training_code/docs/`）

| 文档 | 内容 |
| --- | --- |
| `WORLDSENSE_EVIDENCE_GAP_SCREENING.zh-CN.md` | 筛选方案全文（本文 §5/6 的详细版） |
| `Train_Settings.md` | 训练参数与设置汇总（SFT/OPSD/Clue-OPSD） |
| `NPU_TRAINING_LAUNCH_GUIDE.zh-CN.md` | 当前 ModelArts 环境的 SFT / CLUE-OPSD 手动启动命令与核心脚本入门说明 |
| `WORLDSENSE_ANNOTATION_AND_V1_WALKTHROUGH.zh-CN.md` | 标注流程与 V1 验证实战 |
| `WORLDSENSE_EVIDENCE_ANNOTATION_PLAN.zh-CN.md` | 标注方案设计 |
| `OMNI_MULTIMODAL_NOTES.zh-CN.md` | 多模态入门笔记（token/patch/base64/ffmpeg 基础） |
| `OMNIVIDEO_5K_EXPERIMENT_SUMMARY (2).md` | 上游方法论文档（第 1/6 节为筛选与训练协议） |
