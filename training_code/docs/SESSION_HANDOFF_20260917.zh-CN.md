# Omni-OPSD Session Handoff（2026-09-17）

本文档记录本 session 完成的 5k_oe 实验、训练脚本、数据文件、输出目录和已知限制，供后续 session 直接接续。项目根目录是：

~~~
/share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907
~~~

## 当前结论

三个 3B 训练都已经完成，最终评测也已经完成，训练日志没有新的致命错误。

| 方法 | GPU/账号 | 训练步数 | 最终 checkpoint | OmniVideoBench |
| --- | --- | ---: | --- | ---: |
| SFT | gpu04 四卡，xysui | 157/157 | `checkpoint-157` | 350/1000 = 35.0% |
| 普通 OPSD | gpu05 四卡，ylhu | 157/157 | `checkpoint-157` | 340/1000 = 34.0% |
| CLUE-OPSD | gpu07 六卡 `1,3,4,5,6,7`，wxzhao | 141/141 | `checkpoint-141` | 342/1000 = 34.2% |

本轮使用的模型均为：

~~~
/share/home/ylhu/models/Qwen2.5-Omni-3B
~~~

## 数据集和最终 prompt

训练集是从 OmniVideo-100K `train_oe_70k.jsonl` 按固定 ID 清单生成的 5,000 条 open-ended 数据。ID 清单为：

~~~
docs/omnivideo_oe_5k_ids.json.gz
~~~

数据生成脚本是：

~~~
scripts/prepare_omnivideo_oe_5k.py
~~~

最终使用的 prompt 是 non-reasoning direct-answer 模式：

~~~
<video>
Question: {question}
Answer the question directly using the relevant video and audio evidence. Do not provide a step-by-step reasoning trace or meta-commentary. Give a concise, self-contained answer with the key supporting evidence.
~~~

该 prompt 在 `scripts/prepare_omnivideo_oe_5k.py` 的 `_user_prompt()` 中生成。训练数据和配置说明见：

~~~
data/omnivideo_oe_5k/README.md
data/omnivideo_oe_5k/manifest.json
~~~

主要数据文件：

~~~
data/omnivideo_oe_5k/omnivideo_oe_5k.sft.jsonl
data/omnivideo_oe_5k/omnivideo_oe_5k.opsd.jsonl
data/omnivideo_oe_5k/omnivideo_oe_5k.clue_opsd.jsonl
data/omnivideo_oe_5k/omnivideo_oe_5k.answer_free.jsonl
~~~

数据 manifest 中的动态预算为：2 FPS、32,768 context、视觉预算上限 8,000、每条 student 视频 80 帧、音频开启。OmniVideoBench 的项目内评测文件为：

~~~
data/OmniVideoBench/omnivideobench.answer_free.jsonl
data/OmniVideoBench/omnivideobench.labels.jsonl
~~~

评测脚本会检查 1,000 条数据、ID 完整性和答案泄漏。

### 当前多媒体输入契约与视觉预算（CLUE-OPSD restart2）

2026-09-18 在 gpu05 四卡启动的 CLUE-OPSD restart2 使用的是过滤后的 4,703 条 OE-5K 数据：

~~~
data/omnivideo_oe_5k_clue_only_repaired/omnivideo_oe_5k.clue_opsd.jsonl
~~~

当前多媒体配置以每条数据中的 `sampling_contract` 和 `dynamic_*_budget` 为准：

- `target_fps=2.0`，student 对完整视频做均匀采样，当前数据行固定为 `nframes=80`。这里的 80 是 80 张视频图像帧，不是 80 秒；音频是独立输入。
- `max_frames_per_view=300` 是上限，实际 80 帧由动态视觉预算和数据契约共同确定。teacher 只读取 golden clue 区间，各区间分别采样，所有区间合计最多 80 帧；例如一条样本的分配为 `14+16+16+16+18=80`。
- 视频空间尺寸约为 `280×560`，`min_pixels=3136`，`max_pixels=156800`；当前动态视觉预算为 `8000` tokens，上限也是 `8000`。
- 上下文窗口按 `32768` tokens 计算，文本预留 `2048` tokens，最大 completion 为 `512` tokens。视觉和音频预算会随视频时长或 clue 区间总时长变化，避免把媒体输入无界地塞入上下文。
- student 的音频覆盖完整视频；teacher 的音频与其 golden clue 区间同步，只保留这些区间对应的音频。音频预算按 `ceil(音频秒数×25)+64` 计算。例如 118 秒的完整视频对应约 3,014 个音频 tokens；5 个 clue 区间合计 61 秒时约为 1,845 个音频 tokens。
- 当前使用 `USE_AUDIO_IN_VIDEO=1`、`video_reader=decord`、本地视频直读和 `attn_impl=sdpa`。当前启动配置见：

~~~
output/omnivideo_oe5k_clue_opsd_3b_full_ema_gpu05_restart2_20260918/launch_config.txt
output/omnivideo_oe5k_clue_opsd_3b_full_ema_gpu05_restart2_20260918/clue_opsd/v0-20260918-121827/args.json
~~~

### vLLM rollout 丢弃音频的问题及修复

旧版普通 OPSD 使用 vLLM rollout，并在启动脚本中设置了 `OMNI_OPSD_VLLM_DROP_AUDIO=1`。Qwen-Omni 的该 vLLM 占位符路径在当时的版本中无法稳定处理音视频同时输入，于是 rollout 生成 completion 时跳过了音频，只使用视觉输入；而后续 Transformers 的 student/teacher loss forward 仍使用 `use_audio_in_video=True` 的音频输入。严格来说，旧实验中首先丢音频的是 rollout student 路径，不是 loss 阶段的 Transformers teacher forward；如果 teacher 也部署成 vLLM 服务，则 teacher 请求同样会丢音频。

这会造成同一条训练样本在“生成 completion”和“计算蒸馏 loss”时的模态不一致：rollout 没听到声音，loss 又要求模型在音视频输入上拟合该 completion 和 teacher 分布。对于歌词、语音、环境声等依赖音频的题目，privileged teacher 的信息优势会被削弱，甚至可能把错误的无音频 completion 蒸馏回 student。

当前 CLUE-OPSD restart2 已明确修复为：

- `OMNI_OPSD_USE_VLLM=false`，rollout、student forward 和 EMA teacher 都走 Transformers 路径；
- `OMNI_OPSD_VLLM_DROP_AUDIO=0`；
- `USE_AUDIO_IN_VIDEO=1`，student 使用完整视频音频，teacher 使用 clue 区间对应音频；
- 训练启动日志应出现 `use_audio_in_video=True`，且不应再出现 `Setting omni_opsd_vllm_drop_audio: True`。

因此，当前 restart2 不再具有旧 OPSD 的 vLLM 音频丢弃问题。若未来重新启用 vLLM，必须先验证 vLLM 请求、completion 生成和 loss forward 三处都保留音频，并把 `vllm_drop_audio` 设为 0；否则该实验不能与当前音频一致的 Transformers 配置直接比较。

### 视觉预算配置偏低的问题

需要特别区分“代码支持的动态预算”和“当前 OE-5K 数据实际物化的预算”。通用实现 `src/omni_opsd/data/dynamic_budget.py` 的视觉预算上限默认为 24,000，实验总结第 6.2 节也要求：

~~~text
available_visual = min(24000, 32768 - 2048 - audio_budget)
target_fps = 2
max_total_frames = 300
每帧约 100–128 个视觉 token
~~~

但 `scripts/prepare_omnivideo_oe_5k.py` 为 3B full-parameter + full-EMA 训练设置了 `--visual-budget-cap` 默认值 8,000，并将其标注为硬件安全预算。当前 `data/omnivideo_oe_5k/manifest.json` 和 `data/omnivideo_oe_5k_clue_only_repaired/manifest.json` 均确认实际物化值为 8,000。

这会产生以下结果：

- student 的 `available_visual` 始终被 8,000 封顶；由于预算策略按每帧至少 100 个视觉 token，`budget_frames=8000/100=80`。
- 当前训练视频时长为 60–180 秒，2 FPS 的名义帧数为 120–300，因此所有 student 行都被压成固定 80 帧，student 视觉 token 没有真正随样本变化。
- 以 118 秒样本为例，文本预留 2,048、音频预算 3,014、视觉 8,000、预留余量 512，总计约 13,574 tokens，只占 32,768 上下文的约 41%。当前 4,703 条 student 数据平均占用约 13,423 tokens，平均上下文利用率约 41%。
- teacher 因为只看 clue 区间而有所变化，但当前视觉 token 平均约 7,032、最大仍为 8,000，平均总输入约 10,584 tokens，也没有接近上下文窗口上限。

因此，当前 8,000-token 版本是为了降低 full-parameter EMA 训练的显存风险而采用的保守工程覆盖值，不是上下文窗口允许的最大视觉预算，也不符合实验总结第 6/7 章所描述的 24,000-token 正式高预算设置。项目中 `data/omnivideo_oe_5k/README.md` 的旧文字还曾写成 16,000，但实际 JSONL 和 manifest 的权威值是 8,000，应以后者为准。

若要执行文档要求的高预算实验，不能只修改训练启动参数，因为帧数、缩放尺寸和预算已经写入 JSONL。应新建版本目录，以 `--visual-budget-cap 24000` 重新生成数据，再对最重样本做实际 Transformers 编码、完整训练步和峰值显存 gate；如果 24,000 不稳定，可先测 16,000。预计 24,000-token 版本的 student 视觉预算约为 15,120–24,000，帧数约为 120–240，之后才能判断更高视觉信息量是否改善训练效果。

### 从 8,000 根本切换到 24,000 visual tokens 的接续步骤

下面是后续 agent 应执行的完整切换流程。核心原则是：`visual_budget_cap` 会在数据物化阶段决定每条样本的 `nframes`、空间缩放和 `dynamic_*_budget`；训练启动时再改 `max_length`、`max_frames_per_view` 或 manifest 文字，都不会增加已经写入 JSONL 的视觉 token。24k 实验必须重新生成一套带新媒体描述符的 JSONL，并从 base checkpoint 重新开始，不能接着 8k checkpoint 恢复。

#### 1. 固化新的版本目录并重新生成数据

不要覆盖当前 `data/omnivideo_oe_5k/` 或 `data/omnivideo_oe_5k_clue_only_repaired/`，否则无法复现实验。建议使用独立目录，例如：

~~~bash
cd /share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907
NEW_ROOT=data/omnivideo_oe_5k_v24k

PYTHONPATH=src python scripts/prepare_omnivideo_oe_5k.py \
  --annotation /share/home/ylhu/Light-Omni/data/omni_benchmarks/OmniVideo-100K/train_oe_70k.jsonl \
  --ids docs/omnivideo_oe_5k_ids.json.gz \
  --video-root /share/home/ylhu/Light-Omni/data/omni_benchmarks/OmniVideo-100K \
  --output-dir "$NEW_ROOT" \
  --visual-budget-cap 24000
~~~

这一步会同时写出 `sft`、`opsd`、`clue_opsd`、`answer_free` 和 labels 等数据。必须检查新目录的 manifest：

~~~bash
python - <<'PY'
import json
from pathlib import Path
p = Path("data/omnivideo_oe_5k_v24k/manifest.json")
m = json.loads(p.read_text())
print(json.dumps({k: m.get(k) for k in ("rows", "annotated_clue_rows", "full_video_fallback_rows", "visual_budget_cap")}, ensure_ascii=False, indent=2))
assert m.get("visual_budget_cap") == 24000
assert m.get("rows") == 5000
PY
~~~

如果希望把 24k 设为以后脚本的默认值，也要同步修改 `scripts/prepare_omnivideo_oe_5k.py` 中的参数：

~~~python
parser.add_argument("--visual-budget-cap", type=int, default=24_000,
                    help="dynamic visual budget cap for the 3B full-EMA run")
~~~

但是，修改默认值本身不会修改任何旧 JSONL；仍需用上面的命令重新物化数据。`src/omni_opsd/data/dynamic_budget.py` 的默认上限本来就是 24,000，因此通常不需要改它；真正造成当前 8k 数据的是生成器显式设置的 8,000 覆盖值。`data/omnivideo_oe_5k/README.md` 中旧的 16,000/8,000 说明也应改成 24,000，并注明当前 8k 目录仍保持历史不变。

#### 2. CLUE 数据仍然只使用 golden clue，并重新做视频帧预检

生成器会保留没有时间戳的 297 条 fallback 行。当前正式 CLUE 实验的契约是只喂 golden clue 区间，所以应从新生成的 `clue_opsd` 文件中过滤出有 `clue_intervals` 的 4,703 条，再运行 decoder frame repair。示例：

~~~bash
NEW_ROOT=data/omnivideo_oe_5k_v24k
CLUE_ROOT=data/omnivideo_oe_5k_v24k_clue_only_repaired
mkdir -p "$CLUE_ROOT"

python - "$NEW_ROOT/omnivideo_oe_5k.clue_opsd.jsonl" "$CLUE_ROOT/omnivideo_oe_5k.clue_opsd.unrepaired.jsonl" <<'PY'
import json, sys
from pathlib import Path
src, dst = map(Path, sys.argv[1:])
rows = []
for line in src.open(encoding="utf-8"):
    if not line.strip():
        continue
    row = json.loads(line)
    # The 297 rows without timestamp annotations must not silently become
    # full-video teachers in the clue-only experiment.
    if row.get("clue_intervals"):
        rows.append(row)
assert len(rows) == 4703, len(rows)
with dst.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("clue-only rows:", len(rows))
PY

PYTHONPATH=src python scripts/repair_omnivideo_oe5k_clue_frame_ranges.py \
  --input "$CLUE_ROOT/omnivideo_oe_5k.clue_opsd.unrepaired.jsonl" \
  --output "$CLUE_ROOT/omnivideo_oe_5k.clue_opsd.jsonl" \
  --report "$CLUE_ROOT/frame_repair_report.json" \
  --workers 8
~~~

修复脚本会用 decord 探测真实视频帧数，把每个 clue 区间的 `nframes` clamp 到安全的偶数，并在极短相邻区间需要时合并区间；它不会把 teacher 改回完整视频。修复后应核对 `len(teacher_videos) == len(clue_intervals)`、所有 teacher `nframes >= 2` 且为偶数，并把新目录的 `manifest.json` 写明 `visual_budget_cap=24000`、`rows=4703`、`dropped_no_golden_clue=297` 和 repair report 路径。

#### 3. 在训练前审计 24k 是否真的写入了每行数据

不要只看 README。以下审计必须直接读取将要传给训练器的 JSONL；它能发现“代码改了但仍误用旧数据”的情况：

~~~bash
python - <<'PY'
import json, statistics
from pathlib import Path
p = Path("data/omnivideo_oe_5k_v24k_clue_only_repaired/omnivideo_oe_5k.clue_opsd.jsonl")
rows = [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]
assert len(rows) == 4703
student = [r["dynamic_student_budget"] for r in rows]
teacher = [r["dynamic_teacher_budget"] for r in rows]
assert all(r["sampling_contract"]["use_audio_in_video"] is True for r in rows)
assert all(int(r["sampling_contract"]["dynamic_video_budget"]) == 24000 for r in rows)
assert max(int(r["visual_budget_cap"]) for r in student) == 24000
assert max(int(r["max_checked_tokens"]) for r in student + teacher) <= 32768
print("student visual tokens min/max/mean:",
      min(r["visual_budget_tokens"] for r in student),
      max(r["visual_budget_tokens"] for r in student),
      round(statistics.mean(r["visual_budget_tokens"] for r in student), 1))
print("student frames min/max:", min(r["nframes"] for r in student), max(r["nframes"] for r in student))
print("teacher visual tokens min/max:",
      min(r["visual_budget_tokens"] for r in teacher), max(r["visual_budget_tokens"] for r in teacher))
PY
~~~

24k 版本在当前 2 FPS、32,768 context、文本预留 2,048、最大 completion 512 的契约下，student 通常应从原来的固定 80 帧变成约 120–240 帧；视觉 token 会随视频长宽和音频预算变化，常见范围约为 15,120–24,000。每条样本必须满足 `text + audio + visual + 512 <= 32768`。若审计仍显示 `nframes=80`、`visual_budget_tokens=8000` 或 `visual_budget_cap=8000`，说明训练仍在使用旧目录，不能启动正式实验。

#### 4. 做实际编码和显存 gate，再启动 full-parameter 训练

24k 只在纸面上满足上下文限制，不代表 full-parameter EMA 的峰值显存一定可用。应先从审计结果中选 `max_checked_tokens` 最大的 student/teacher 行，使用当前训练环境做 Transformers 编码 smoke test：

~~~bash
USE_AUDIO_IN_VIDEO=1 FORCE_QWENVL_VIDEO_READER=decord \
PYTHONPATH=src python scripts/debug_opsd_encode.py \
  --model /share/home/ylhu/models/Qwen2.5-Omni-3B \
  --dataset data/omnivideo_oe_5k_v24k_clue_only_repaired/omnivideo_oe_5k.clue_opsd.jsonl \
  --index <heavy-row-index> --max-length 32768 --mode train \
  --truncation-strategy raise
~~~

然后用 4 卡、`per_device_train_batch_size=1`、`gradient_accumulation_steps=8` 做 1 个正式训练 step 的 gate，确认四卡都有显存分配、没有 `CUDA out of memory`、没有 `smart_nframes`/decord EOF、没有 `max_checked_tokens` 超限。24k 训练必须保留这些设置：

~~~text
4 GPUs × per_device_train_batch_size 1 × gradient_accumulation_steps 8 = global batch 32
num_train_epochs=1, max_length=32768, USE_AUDIO_IN_VIDEO=1
OMNI_OPSD_USE_VLLM=false, OMNI_OPSD_VLLM_DROP_AUDIO=0
OMNI_OPSD_FULL_EMA_TEACHER=true, OMNI_OPSD_FULL_EMA_OFFLOAD=true
~~~

若 gate OOM，先记录峰值和失败样本，降低到 16k 做对照；不要通过把 `max_length` 改小来掩盖 24k 输入被截断。只有 gate 通过后才运行 1 epoch，并从 `/share/home/ylhu/models/Qwen2.5-Omni-3B` 初始化新的输出目录。

#### 5. 复制并切换版本化 launcher

当前 restart2 launcher 硬编码的是旧 8k repaired 数据：

~~~text
scripts/run_omnivideo_oe5k_clue_opsd_gpu05_restart2.sh
OMNI_OPSD_DATASET=.../data/omnivideo_oe_5k_clue_only_repaired/omnivideo_oe_5k.clue_opsd.jsonl
~~~

不要直接改正在使用的 launcher。复制成 `scripts/run_omnivideo_oe5k_clue_opsd_gpu05_v24k.sh`，至少修改 `OMNI_OPSD_DATASET`、实验 tag、输出 root、log/MASTER_PORT；把 dataset 指向：

~~~text
data/omnivideo_oe_5k_v24k_clue_only_repaired/omnivideo_oe_5k.clue_opsd.jsonl
~~~

其余音频一致性、full-parameter EMA、诊断日志、验证集和 checkpoint 设置保持与 restart2 相同。按照有效 global batch 重新核算 `max_steps`；当前 4,703 行、90% train split、global batch 32 的 1 epoch 是 133 steps，但换 split 或换 5,000 行 arm 后不能盲目沿用 133。新输出目录必须保存 `launch_config.txt`、数据 manifest、JSONL SHA256、峰值显存、诊断日志和中间 checkpoint。

#### 6. 最容易犯的错误

- 只改 `max_length=32768` 或 `max_frames_per_view=300` 不会提高视觉预算；二者只是上下文/上限，真正的 per-row `nframes` 已写在 JSONL。
- 只改 `src/omni_opsd/data/dynamic_budget.py` 的默认值，或只改 manifest/README，也不会重写旧数据。
- 生成了 24k 的原始 CLUE 文件却继续把 launcher 指到 `data/omnivideo_oe_5k_clue_only_repaired/`，实际仍是 8k。
- 24k 数据和 8k checkpoint 的输入分布不同。为保持比较有效，24k 实验应从 base 重新训练，而不是从 8k 的 `checkpoint-*` 恢复；只有同一预算、同一数据契约且 checkpoint 中包含完整 EMA 状态时才允许 resume。
- `teacher` 的 clue 预算和 `student` 的完整视频预算不必相等，但两者都必须满足同一个 32,768 context/headroom 约束，并且都保持 `use_audio_in_video=True`。

## 三种训练模式

### 1. SFT：直接用 gold answer 做 teacher forcing

SFT 的 `messages` 是上面的 full-video direct-answer prompt，assistant target 是数据中的 open-ended gold answer。SFT 没有 on-policy rollout，也没有 OPSD teacher；本轮使用 full-parameter fine-tuning，LLM、ViT 和 aligner 都没有冻结。

实际启动脚本：

~~~
output/launch_omnivideo_oe5k_sft_gpu04.sh
~~~

该脚本调用的通用 3B SFT 脚本：

~~~
scripts/run_gap5000_sft_3b_full_cuda.sh
~~~

底层统一训练入口：

~~~
scripts/run_training_arm_a100.sh sft
~~~

实际参数：

~~~
CUDA_VISIBLE_DEVICES=0,1,2,3
per_device_train_batch_size=1
gradient_accumulation_steps=8
global_batch_size=4*1*8=32
num_train_epochs=1
max_steps=157
save_steps=25
learning_rate=1e-5
max_length=32768
torch_dtype=bfloat16
attn_impl=sdpa
USE_AUDIO_IN_VIDEO=1
~~~

训练输出：

~~~
output/omnivideo_oe5k_sft_3b_full_gpu04_20260915_oe5k_sft_3b_full_gpu04_retry2/
~~~

最终模型：

~~~
output/omnivideo_oe5k_sft_3b_full_gpu04_20260915_oe5k_sft_3b_full_gpu04_retry2/sft/v0-20260915-151958/checkpoint-157
~~~

训练日志和指标：

~~~
.../sft.log
.../sft/v0-20260915-151958/logging.jsonl
~~~

### 2. 普通 OPSD：full-video privileged-answer teacher

Student 看到完整视频和上面的 direct-answer prompt，不接收 gold answer。`_opsd_training_row()` 会把正确答案追加到 `teacher_prompt`，teacher 看到同一完整视频和 privileged answer，然后对 student 的 on-policy completion 做 GKD/JSD 蒸馏。当前普通 OPSD 使用 full-parameter student；`opsd_ema_alpha=0`，没有额外的 LoRA shadow EMA teacher。

数据构造逻辑仍在：

~~~
scripts/prepare_omnivideo_oe_5k.py
~~~

实际启动脚本：

~~~
output/launch_omnivideo_oe5k_opsd_gpu05_repaired_20260916.sh
~~~

该脚本调用：

~~~
scripts/run_gap5000_opsd_3b_full_cuda.sh
scripts/run_training_arm_a100.sh opsd
~~~

实际数据和参数：

~~~
dataset=data/omnivideo_oe_5k_opsd_repaired/omnivideo_oe_5k.opsd.jsonl
CUDA_VISIBLE_DEVICES=0,1,2,3
per_device_train_batch_size=1
gradient_accumulation_steps=8
global_batch_size=32
num_train_epochs=1
max_steps=157
save_steps=25
learning_rate=2e-6
tuner_type=full
freeze_llm=false
freeze_vit=false
freeze_aligner=false
use_vllm=true
vllm_mode=colocate
vllm_tensor_parallel_size=2
gkd_logits_topk=20
rollout_top_p=0.95
rollout_top_k=20
USE_AUDIO_IN_VIDEO=1
~~~

训练输出：

~~~
output/omnivideo_oe5k_opsd_3b_full_gpu05_20260916_oe5k_opsd_3b_full_gpu05_repaired_v1/
~~~

最终模型：

~~~
output/omnivideo_oe5k_opsd_3b_full_gpu05_20260916_oe5k_opsd_3b_full_gpu05_repaired_v1/opsd/v0-20260916-120504/checkpoint-157
~~~

重要限制：普通 OPSD 的 vLLM rollout 记录为 `vllm_drop_audio=1`，因为当前 vLLM 版本的 Qwen-Omni 音视频占位符路径存在兼容性问题；训练/teacher 侧仍启用音频。这一项是普通 OPSD 的已知模态不一致点。后续若研究效果，应优先改成 Transformers rollout 或修复 vLLM 音频路径，再重新做对照。

### 3. CLUE-OPSD：golden clue teacher + full-parameter EMA

Student 看到完整视频和上面的 direct-answer prompt。Teacher 只看到数据提供的 golden clue 视频区间，prompt 中不加入 gold answer。该设计由 `_clue_opsd_training_row()` 生成；teacher 的 `teacher_videos` 使用同一源 MP4 的多个时间区间，student 和 teacher 的文本 prompt 不包含答案。

最终运行使用过滤后的 4,703 条数据：没有 golden clue 区间的 297 条被排除。

~~~
data/omnivideo_oe_5k_clue_only_repaired/omnivideo_oe_5k.clue_opsd.jsonl
~~~

实际启动脚本：

~~~
output/launch_omnivideo_oe5k_clue_opsd_gpu07_resume100_20260916.sh
~~~

该脚本调用：

~~~
scripts/run_omnivideo_oe_5k_clue_opsd_3b_full_cuda.sh
scripts/run_gap5000_clue_opsd_3b_full_cuda.sh
scripts/run_gap5000_clue_opsd_cuda.sh
scripts/run_training_arm_a100.sh clue_opsd
~~~

实际参数：

~~~
CUDA_VISIBLE_DEVICES=1,3,4,5,6,7
nproc=6
per_device_train_batch_size=1
gradient_accumulation_steps=5
global_batch_size=30
num_train_epochs=1
max_steps=141
save_steps=25
learning_rate=2e-6
tuner_type=full
freeze_llm=false
freeze_vit=false
freeze_aligner=false
full_ema_teacher=true
full_ema_alpha=0.05
full_ema_offload=true
gold_ce_alpha=0
use_vllm=false
vllm_drop_audio=0
USE_AUDIO_IN_VIDEO=1
log_completions=true
diag_enabled=true
diag_top_k=20
diag_frequency=1
~~~

六张卡无法用整数梯度累积得到 32 的全局 batch，因此该 launcher 显式设置 `OMNI_OPSD_ALLOW_NON_32_GLOBAL_BATCH=1` 并保留原运行的 6×1×5=30 配置，以保证从 checkpoint 恢复时 scheduler/data-loader 语义不变。

训练最初在旧目录的 step 119 附近因为 long MP4 经 BytesIO 解码触发 decord EOF 退出。修复后从有效的 `checkpoint-100` 恢复到新目录：

~~~
旧保存点：
output/omnivideo_oe5k_clue_opsd_3b_full_ema_gpu07_6gpu_20260915_repaired_v1/clue_opsd/v0-20260915-194006/checkpoint-100

最终恢复目录：
output/omnivideo_oe5k_clue_opsd_3b_full_ema_gpu07_6gpu_20260916_resume100_repaired_v2/
~~~

最终模型：

~~~
output/omnivideo_oe5k_clue_opsd_3b_full_ema_gpu07_6gpu_20260916_resume100_repaired_v2/clue_opsd/v0-20260916-222515/checkpoint-141
~~~

CLUE 训练日志还包含：

~~~
.../clue_opsd/v0-20260916-222515/logging.jsonl
.../clue_opsd/v0-20260916-222515/completions.jsonl
.../clue_opsd.log
~~~

EMA shadow 没有写入旧 checkpoint；因此恢复时 EMA teacher 从 checkpoint-100 的 student 权重重新初始化，之后按 alpha=0.05 更新。student 的模型、optimizer、scheduler、RNG 等 checkpoint 状态正常恢复。

## 视频修复和解码注意事项

普通 OPSD 使用修复后的数据：

~~~
data/omnivideo_oe_5k_opsd_repaired/omnivideo_oe_5k.opsd.jsonl
~~~

其中视频 `7dyWNjop7jg.mp4` 被 ffmpeg 重编码为 H.264/AAC/faststart，修复记录在：

~~~
data/omnivideo_oe_5k_opsd_repaired/manifest.json
~~~

CLUE 修复后的数据和报告：

~~~
data/omnivideo_oe_5k_clue_only_repaired/omnivideo_oe_5k.clue_opsd.jsonl
data/omnivideo_oe_5k_clue_only_repaired/frame_repair_report.json
data/omnivideo_oe_5k_clue_only_repaired/decode_preflight_resume_vllm.json
~~~

最终 CLUE preflight 结果是 `passed=true`、2,361 个 unique videos、`decode_failures=[]`。训练 launcher 设置了：

~~~
OMNI_OPSD_DIRECT_LOCAL_VIDEO=1
DECORD_EOF_RETRY_MAX=20480
DECORD_NUM_THREADS=1
FORCE_QWENVL_VIDEO_READER=decord
~~~

这些设置避免把本地 MP4 转成 BytesIO 后并发 seek 到视频尾部造成 EOF。`mmco: unref short failure` 和 librosa FutureWarning 在日志中出现过，但没有导致本轮训练失败。

## 评测脚本和结果

训练完成后的统一评测 wrapper：

~~~
output/eval_oe5k_after_training.sh
~~~

它根据最终 checkpoint 自动设置模型路径，然后调用：

~~~
scripts/run_omnivideobench_eval.sh
~~~

评测使用 1,000 条 answer-free OmniVideoBench 数据，音频开启、直接读取本地视频、每个评测进程 batch=1。当前三份结果：

~~~
output/omnivideobench_oe5k_sft_3b_gpu04_20260915/summary.json
output/omnivideobench_oe5k_opsd_3b_gpu05_20260916_repaired_v1/summary.json
output/omnivideobench_oe5k_clue_opsd_3b_gpu07_20260916_resume100_repaired_v2/summary.json
~~~

`summary.json` 中 `parse_failures=0`，三组都完整解析 1,000 条。结果是训练后模型在固定 OmniVideoBench 评测集上的结果，不等同于此前初始 SFT session 的 base/7B 对照结果；初始 SFT 的问题和上下文分析见：

~~~
docs/SFT_ANALYSIS_20260910.zh-CN.md
~~~

## 监控和后续接续

训练与评测的统一监控脚本：

~~~
output/monitor_three_trainings_hourly_and_eval.py
~~~

当前 daemon 的 PID 文件和日志：

~~~
output/monitor_three_trainings_hourly_and_eval.pid
output/monitor_three_trainings_hourly_and_eval.log
~~~

监控间隔为 3600 秒，`RUNS` 已更新为本轮最终目录；它会在检测到最终 checkpoint 后自动启动评测，并检查 `summary.json`。本 session 结束时三个训练和三个评测均已完成，后续 session 可以停止该监控 daemon，或把 `RUNS` 改成新的实验目录后复用。

相关分析和设计资料：

~~~
docs/SFT_HANDOFF_20260908.zh-CN.md
docs/SFT_ANALYSIS_20260910.zh-CN.md
docs/OMNIVIDEO_5K_EXPERIMENT_SUMMARY (2).md
docs/CLUE_OPSD_DIAGNOSTICS_CODEX_PROMPT.md
docs/2604.13016v2.pdf
~~~

代码工作区存在较多历史实验产生的未提交/未跟踪文件；后续 session 不要执行全仓库清理或 reset，先查看具体路径和输出目录再修改。GPU 账号密码只允许从以下私有文件读取，handoff 不记录密码：

~~~
/share/home/ylhu/EvoEmbedding/GPU_ACCOUNT_ACCESS.private.md
~~~

## OmniVideo-Test 505 条评测与训练失败原因分析（2026-09-18）

### 评测结果和可比性

本次四个模型使用同一份 505 条 OmniVideo-Test、同一份标签、相同的 Transformers 推理脚本、相同的音频设置和 greedy `max_new_tokens=8`。四组 `parse_failures=0`，因此结果差异不是由于输出解析失败或评测样本不一致造成的。

| 模型 | 正确数 | 准确率 | 相对 Base |
| --- | ---: | ---: | ---: |
| 原始 3B | 238/505 | 47.13% | — |
| SFT 后 3B | 233/505 | 46.14% | -0.99pp |
| OPSD 后 3B | 233/505 | 46.14% | -0.99pp |
| CLUE-OPSD 后 3B | 232/505 | 45.94% | -1.19pp |

详细结果位于：

~~~
output/omnivideo_test_505_matrix_gpu05_20260917/base/summary.json
output/omnivideo_test_505_matrix_gpu05_20260917/sft/summary.json
output/omnivideo_test_505_matrix_gpu05_20260917/opsd/summary.json
output/omnivideo_test_505_matrix_gpu05_20260917/clue_opsd/summary.json
~~~

按同一题目配对比较，SFT 有 20 条由错变对、25 条由对变错；OPSD 有 3 条由错变对、8 条由对变错；CLUE-OPSD 有 1 条由错变对、7 条由对变错。SFT 改变了 65/505 条输出，OPSD 只改变 20/505 条，CLUE-OPSD 只改变 12/505 条。也就是说，SFT 发生了明显的行为漂移，但方向不利；OPSD 和 CLUE-OPSD 的训练信号太弱，几乎没有改变 MCQ 决策策略。505 条测试集的单次比例标准误约为 2.2 个百分点，所以目前不能把约 1 个百分点的差异单独解释为统计显著的能力下降，但也没有观察到可靠提升。

### 训练目标和评测目标不一致

OE-5K 训练是开放式 direct-answer 问答，训练 prompt 要求输出包含视频/音频证据的自由文本答案；OmniVideo-Test 则要求从四个选项中输出一个字母。训练集的 gold answer 平均约 79 个词，且没有统一的 A/B/C/D 标签。

因此：

- SFT 优化的是自由文本答案的 teacher-forcing CE，而不是选项字母 CE；
- OPSD 优化的是 student 与 privileged teacher 的输出分布 JSD，而不是正确选项准确率；
- CLUE-OPSD 的 teacher 既没有看到答案，也只看到 clue 区间；
- 训练 prompt 和评测 prompt 的回答格式、输出长度和监督目标都不同。

这解释了为什么 SFT 的训练 loss 可以下降但 MCQ 分数下降，也解释了为什么 OPSD/CLUE-OPSD 的低 JSD 不能被当作答案能力提高。论文关于 OPD 失败的核心诊断同样指出，蒸馏 loss 下降只表示 student 拟合 teacher；如果 teacher advantage 不为正，蒸馏会退化为弱信号或自蒸馏。

### SFT 参数和 loss 曲线

参数和日志：

~~~
output/omnivideo_oe5k_sft_3b_full_gpu04_20260915_oe5k_sft_3b_full_gpu04_retry2/sft/v0-20260915-151958/args.json
output/omnivideo_oe5k_sft_3b_full_gpu04_20260915_oe5k_sft_3b_full_gpu04_retry2/sft/v0-20260915-151958/logging.jsonl
~~~

主要配置是 full-parameter、4 卡、`batch=1`、`gradient_accumulation_steps=8`，有效 batch size 为 32，`learning_rate=1e-5`，1 epoch、157 个 optimizer steps、`max_length=32768`。LLM、ViT 和 aligner 均未冻结，实际约 4.703B 参数可训练。

loss 从 2.5669 降到 1.8635，平均 1.9819；teacher-forcing token accuracy 从 0.481 上升到 0.553。这证明模型确实在拟合 OE-5K 的自由文本答案，但不是训练集答案正确率。该 run 使用 `eval_strategy=no`、`split_dataset_ratio=0`，没有验证集，也没有根据验证指标选择 checkpoint；`max_grad_norm=0` 表示没有梯度裁剪。对 full-parameter 多模态模型来说，`1e-5` 偏激进，配合单 epoch、无验证和无裁剪，容易造成回答格式漂移或能力遗忘。

### OPSD 参数、loss 和实现问题

参数和日志：

~~~
output/omnivideo_oe5k_opsd_3b_full_gpu05_20260916_oe5k_opsd_3b_full_gpu05_repaired_v1/opsd/v0-20260916-120504/args.json
output/omnivideo_oe5k_opsd_3b_full_gpu05_20260916_oe5k_opsd_3b_full_gpu05_repaired_v1/opsd/v0-20260916-120504/logging.jsonl
output/omnivideo_oe5k_opsd_3b_full_gpu05_20260916_oe5k_opsd_3b_full_gpu05_repaired_v1/opsd.log
~~~

OPSD 使用 full-parameter、4 卡、有效 batch size 32、`learning_rate=2e-6`、1 epoch、157 steps、`use_vllm=true`，没有验证集、completion 日志或诊断指标，也没有 full-parameter EMA teacher。总 loss 从 0.0761 降到 0.0398，`clue/jsd_loss` 从 0.0666 降到 0.0551，并在后半段约 0.05 附近平台。该曲线只说明 student 更接近 teacher 分布，不能说明 teacher 给出了正确答案。

`gold_ce_alpha=0.25` 在这次 OE-5K OPSD 中实际上没有生效。`ms-swift/swift/rlhf_trainers/gkd_trainer.py` 的 `_gold_answer_token_loss()` 只接受单个 `A/B/C/D` 选项并寻找相应的 answer token；OPSD 数据行没有 `gold_answer` 字段，长篇答案只存在于 `teacher_prompt`，所以辅助 CE 返回空值。

更严重的是，OPSD 日志明确记录：

~~~
Using environment variable `OMNI_OPSD_VLLM_DROP_AUDIO`,
Setting omni_opsd_vllm_drop_audio: True.
~~~

数据契约和 Transformers 训练 forward 都设置了 `use_audio_in_video=True`；严格来说，旧 OPSD 中明确丢弃音频的是 vLLM rollout（用于生成 student completion）的路径，而不是 loss 阶段的 Transformers teacher forward。于是 completion 是由“无音频的 rollout student”生成的，随后 student/teacher loss forward 却使用带音频的输入，形成 rollout 与训练输入的模态不一致；如果将 teacher 也部署为 vLLM 服务，则 teacher 请求同样会丢音频。这个问题会削弱 privileged teacher 的有效信息，属于本次 OPSD 最重要的实现问题之一。

### 旧 CLUE-OPSD 参数、EMA 恢复和诊断指标

参数和日志：

~~~
output/omnivideo_oe5k_clue_opsd_3b_full_ema_gpu07_6gpu_20260916_resume100_repaired_v2/clue_opsd/v0-20260916-222515/args.json
output/omnivideo_oe5k_clue_opsd_3b_full_ema_gpu07_6gpu_20260916_resume100_repaired_v2/clue_opsd/v0-20260916-222515/logging.jsonl
~~~

旧 CLUE 使用 6 卡、`batch=1`、梯度累积 5，有效 batch size 是 30，不是 32；数据只有 4,703 条，297 条没有 golden clue 区间而被排除。它配置了 `full_ema_teacher=true`、`full_ema_alpha=0.05`，但从旧 `checkpoint-100` 恢复时，该 checkpoint 没有保存 `full_ema_teacher.pt`。因此 EMA teacher 的历史状态没有恢复，旧结果不是完整、连续的 full-parameter EMA 实验。

恢复后的可见训练段中，loss 约为 0.0286 到 0.0393，JSD 约为 0.0226 到 0.0618；验证 loss 在 step 125 和最终 step 141 分别为 0.0316 和 0.0303。诊断指标为：`overlap_ratio` 平均约 0.838，`overlap_kl` 平均约 0.136，`overlap_adv_paper` 平均约 -0.0121，`empty_overlap_rate` 约为 0.00018。

高 overlap 加上持续为负的 teacher advantage 表明 teacher 没有稳定优于 student。CLUE teacher 只看到 golden clue 区间，不接收答案，音频预算也低于 full-video student；它实际上不是一个更强的 privileged teacher。该条件下使用 EMA 只能平滑一个弱 teacher，不能创造额外答案信息。

### 训练集测试结果缺口

目前没有这三个正式 run 的统一训练集 generation/accuracy 报告：

- SFT 只有 teacher-forcing `token_acc`，没有自由文本答案 exact match 或 judge 分数；
- OPSD 没有 completion 日志，只有 JSD/loss；
- 旧 CLUE 有 completion 和分布诊断，但没有统一的自由文本答案正确率；
- 现有 A/B/C/D 评测器不能直接评测 OE-5K 的长篇自由文本答案。

因此不能把 SFT 的 token accuracy 当作训练集答题正确率，也不能把 OPSD/CLUE 的 JSD 当作答案准确率。已有的 1,000 条 OmniVideoBench 结果是独立的 held-out MCQ 评测，不是训练集测试：Base 34.5%，SFT 35.0%，OPSD 34.0%，CLUE-OPSD 34.2%。训练集与 OmniVideo-Test/OmniVideoBench 没有视频或 case ID 重叠，未发现数据泄漏。

### 日志异常的影响判断

日志中出现过 `mmco: unref short failure`、librosa FutureWarning、`Could not estimate the number of tokens`、vLLM usage reporter 的 `JSONDecodeError`，以及少量 decord EOF 后回退到 torchvision。这些 run 没有出现 NaN、OOM 或训练中途停止，loss 也没有发散，因此目前没有证据认为这些非致命日志是分数下降的主要原因。它们仍应在后续实验中固定，以提高可重复性。

综合原因优先级如下：

1. 训练是开放式答案目标，评测是 MCQ 字母目标，缺少直接选项监督；
2. OPSD 的 gold-answer CE 对 OE-5K 实际无效；
3. OPSD vLLM rollout 丢音频，teacher/student 模态输入不一致；
4. OPSD 没有 full EMA、验证集和诊断日志，无法确认 teacher 有效；
5. 旧 CLUE 恢复时没有恢复 EMA teacher 状态；
6. CLUE teacher 只看 clue 区间且没有答案，诊断显示 teacher advantage 为负；
7. SFT 的 full tuning、`1e-5` 学习率、无梯度裁剪和无验证集选择导致行为漂移；
8. 训练和评测的视觉采样契约不同，增加了输入分布偏移。

## 以 CLUE-OPSD 为基础的多原子视角多教师蒸馏：核心思想与研究方案

> 本节记录这项工作的核心方法构想和下一步实验方案。它建立在当前 CLUE-OPSD 的“student 看完整视频、teacher 看特权证据片段”结构上，但“多原子视角、逐样本 teacher 选择和多教师融合”尚未纳入前面所列旧版 3B 训练结果，不能把本节的 oracle 数字当成训练后 student 分数。

### 方法动机

视频中的正确答案往往依赖某一类局部证据：一个短暂动作需要更密集的时间采样，画面中的小物体需要更高空间分辨率，歌词或环境声需要精确的音频区间，事件因果或顺序问题则可能需要证据前后的时间上下文。把所有视频都用同一种输入方式交给 teacher，会把这些样本相关的差异平均掉。

CLUE-OPSD 已经体现了第一步思想：student 使用完整音视频，teacher 只使用数据集提供的 golden clue 区间；teacher 的输出分布通过相同的 student completion 蒸馏给 student。它的问题在于只使用一个固定的 clue 视角。当 clue 区间过短、漏掉前后上下文、空间细节不足或证据主要存在于音频时，teacher 本身可能不如 full-video student，`overlap_adv_paper` 就会变成负值，蒸馏只是在要求 student 拟合一个较弱的分布。

多原子视角多教师方法把“证据位置”和“证据使用方式”拆成可解释的原子因素，针对同一道题构造多个 teacher 输入。每个 teacher 只改变一个主要因素，因而可以回答两个问题：哪些证据变换能够提供有效监督，以及不同视角是否在不同样本上互相补充。student 训练完成后只需要完整视频推理，额外的视角 teacher 只存在于训练阶段。

### 原子视角的定义

每个视角使用相同的问题、选项、模型和评分协议，只改变媒体输入或采样方式。当前 5k 原子实验已经形成了六个适合作为 teacher 候选的视角：

| 视角 | 输入构造 | 主要改变因素 | 作用 |
| --- | --- | --- | --- |
| G / `E2_G_exact` | 只输入标注的 Gold 视觉区间，并保留同一区间音频 | 证据定位 | 核心精确证据 teacher |
| H / `E4_H_halo3` | Gold 区间前后各扩展约 3 秒，再输入同步音视频 | 时间上下文 | 防止关键动作发生在标注边界外 |
| T / `E7_T_8fps` | 保持 Gold 区间和空间预算，提高到 8 FPS | 时间采样密度 | 捕获短暂动作、顺序和运动变化 |
| S / `E8_S_highres` | 保持 Gold 区间、2 FPS 和音频，提高空间像素预算 | 空间分辨率 | 捕获文字、小物体和细粒度外观 |
| A / `E13_A_audio_exact` | Gold 视觉区间加独立的精确 Gold 音频输入 | 音频模态和时间定位 | 处理歌词、声音和口语证据 |
| V / `E12_gold_v` | 只输入 Gold 视觉，关闭原生视频音频 | 音频移除 | 音频作用的对照，也保留部分视觉互补性 |

这里的 Gold 表示证据位置特权，不表示把正确答案写进 teacher prompt。标准 OPSD 的“把正确选项写进 teacher prompt”应作为独立的 answer-privileged baseline；多原子视角实验要研究的是证据输入本身的互补性。所有视角都必须使用相同的 Qwen-Omni 预处理和音频协议，禁止某个 teacher 在 rollout 中静默使用 `vllm_drop_audio=1`。

原子视角不是简单地把视频裁短。5k 实验中，Full 音视频 A0 为 56.66%，精确证据音视频 A3 为 80.88%，与证据时长匹配但均匀抽取的 A4 只有 55.52%。因此，主要收益来自证据位置和模态选择，而不是少看了几秒视频。E 系列进一步显示，精确 Gold G 为 78.80%，高分辨率 S 为 80.36%，关闭音频的 V 只有 57.80%；不同视角的收益随任务和样本变化，不能只按整体平均分决定 teacher。

### 多教师的两层结构

第一层是“同一模型、不同证据输入”的视角 teacher。为了把增益归因于证据而不是模型参数，建议所有视角共享同一个 frozen initial teacher 或同一个 full-parameter EMA teacher。多教师不应使用多个未经对齐的模型 checkpoint，否则无法区分视角差异和参数差异。

第二层是“逐样本 teacher 路由”。对训练样本 (i)，令 (y_i) 为真实选项，(v) 为候选视角，teacher 在该视角下给出正确选项概率 (p_t^v(y_i|x_i))。训练阶段可用标签做离线选择：

```text
v_i* = argmax_v p_t^v(y_i | x_i, view_v)
```

被选视角的 teacher prompt 仍然不包含 (y_i)；真实答案只用于离线判断哪个证据视角对该样本更可靠。这一点很重要：如果把答案也写入 teacher prompt，就无法判断收益来自证据还是答案泄漏。

在不希望每条样本只使用一个 teacher 时，可以保留多个 teacher 分布并按质量加权：

```text
w_i,v = softmax((p_t^v(y_i) - b_i) / tau_select)
q_i = sum_v w_i,v * q_i,v
```

其中 `b_i` 可以是 full-video baseline 的正确答案概率，`q_i,v` 是视角 teacher 在 student completion 各位置上的词表分布，`tau_select` 控制选择的尖锐程度。若某个视角只在少数样本上有独立价值，它仍可得到较高的逐样本权重，而不会因为整体平均准确率较低被删除。

多教师训练应至少保留三种模式：

1. **固定 teacher**：所有样本固定使用 G 或 S，测量单一视角的基线；
2. **oracle 路由**：用训练答案选择 (v_i^*)，提供可达到的 teacher 上限；
3. **多教师融合或可部署路由**：按 teacher 的校准置信度、答案 margin、熵和历史可靠性加权，不使用测试答案。

当前 5k 原子实验中，固定 Highres 视角的准确率为 80.36%，用真实答案对 G/H/T/S/V/A 六视角逐 case 选择为 91.04%，而按模型最高置信度选择只有 76.40%。91.04% 是标签指导的 teacher 监督上限，不是 student 评测结果；它说明不同视角确实存在互补监督空间，也说明简单地选择“最自信的 teacher”是不可靠的。相对固定 Highres，六视角 oracle 额外救回 535 道错题，同时只引入 1 道反向错误，净增 534 道。

### 一次多教师 OPSD 更新

student 对每个样本仍然只接收完整音视频和统一的 reasoning/answer prompt，并生成一份 on-policy completion。所有 teacher 使用这同一份 completion token IDs 重算分布，不重新生成各自的答案。这样，teacher 之间比较的是“在同一条 student 轨迹上如何评价下一个 token”，而不是把不同 teacher 生成的文本直接拼接在一起。

推荐的更新流程如下：

1. 读取样本的 Full student 输入和候选视角配置；
2. student 用完整音视频生成分析和 `<answer>字母</answer>`；
3. 对每个候选视角，把相应证据视频/音频和相同 completion 送入 teacher；
4. 计算每个 teacher 的正确答案概率、答案 margin、熵和 top-k 分布；
5. 依据固定、oracle 或可部署路由选择一个 teacher，或融合多个 teacher 分布；
6. 在相同 completion 位置计算 student 与 teacher 的 JSD/KL 蒸馏损失；
7. 对最终 `<answer>` 区域加入直接 gold CE，并可对答案 token 提高权重；
8. optimizer step 后更新 full-parameter EMA teacher，并保存 EMA 权重、路由统计和 optimizer 状态。

一个可实现的联合损失是：

```text
L = lambda_jsd * JSD(p_student, q_multi_teacher)
  + alpha_answer * CE(p_student(answer_tokens), gold_answer)
  + gamma_sft * CE(p_student, gold_completion)
  + eta_consistency * L_view_consistency
```

其中 `q_multi_teacher` 可以是单个被选 teacher 的分布，也可以是多个视角的加权分布；`alpha_answer` 只作用于最终答案区域，避免长分析文本的 token 数量淹没答案监督；`L_view_consistency` 用于约束不同 teacher 对明确证据位置的分布不要出现无法解释的冲突。蒸馏仍必须只覆盖有效 completion token，不能把媒体 token、prompt token 或 padding 纳入 loss。

如果使用 top-k JSD，所有 teacher 必须在同一温度和同一 completion 位置计算 top-k，并记录完整词表归一化的 teacher/student 熵。对于 answer region，建议同时保留 full-vocab answer-token CE，不能只依赖 teacher top-k，因为真实答案可能不在 teacher 的 top-k 交集里。

### 为什么这能修复固定 CLUE-OPSD 的弱点

固定 CLUE-OPSD 假设“golden clue 一定比 Full 输入更适合 teacher”。实际数据中这个假设不总成立：有的题需要 clue 区间，有的题需要前后 3 秒上下文，有的题需要高时间采样，有的题需要高分辨率，有的题主要依赖音频。固定 clue teacher 在这些样本上可能给出比 student 更差的分布，导致 `overlap_ratio` 高而 `overlap_adv_paper` 为负。

多原子视角方法不强迫所有样本接受同一种证据裁剪，而是让 teacher 的特权形式随样本变化：

- 需要定位的题选择 G；
- 证据边界不完整的题选择 H；
- 快速动作或事件顺序题选择 T；
- 文字、小目标和细节题选择 S；
- 声音、歌词或口语题选择 A；
- 仅视觉证据有价值或音频质量不稳定时保留 V 作为对照。

这样 teacher advantage 的来源变成“为当前样本找到更合适的证据视角”，而不是“默认局部 clue 总是优于完整视频”。如果所有候选 teacher 都与 student 高度重叠且没有正 advantage，应降低该样本的蒸馏权重，或者只使用 gold answer CE，避免把无信息的 JSD 梯度写入 student。

### 必须记录的统计量

多教师实验不能只记录总 loss。每个样本和每个视角至少应保存：

- teacher 的选项准确率、正确答案概率、答案 margin、熵；
- student 的对应指标以及 `teacher_advantage`；
- `overlap_ratio`、`overlap_kl`、`overlap_adv_paper`、entropy gap；
- 视角选择频率、每类任务的选择频率和平均权重；
- 相对固定 Highres 的 rescue count、独有正确数和错误转换数；
- gold answer CE、最终 answer-token loss、student/teacher answer-token accuracy；
- teacher 全部错误、候选分布冲突、answer token 不在 top-k 的比例；
- 使用 oracle 路由、置信度路由和随机路由时的 selection regret；
- 按视频聚类 bootstrap 的准确率和置信区间。

尤其要把“teacher 选择准确率”与“student 训练后准确率”分开报告。91.04% 这类标签指导选择结果只能说明候选 teacher 有潜在互补信息，不能证明 student 已经学会了这些信息。

### 推荐的实验矩阵

为了把视角互补性、答案监督和训练实现问题分开，后续实验应使用相同 student 初始化、相同训练样本顺序、相同有效 batch、相同视频预算和相同独立按视频划分的验证集：

| 实验 | Teacher | 目的 |
| --- | --- | --- |
| Base | 无训练 | 原模型基线 |
| Fixed-G OPSD | 固定 G | 与旧 CLUE-OPSD 的直接比较 |
| Fixed-S OPSD | 固定 S | 测试高分辨率证据 teacher |
| Random-view OPSD | 随机 G/H/T/S/V/A | 区分视角多样性和有依据选择 |
| Oracle-route OPSD | 训练标签选择最佳视角 | 多教师可达到的上限 |
| Confidence-route OPSD | 不看答案的校准路由 | 可部署路由对照 |
| Mixture OPSD | 多视角加权分布 | 测试分布融合是否比硬选择稳定 |
| +Answer CE | 上述各项加 answer-token CE | 验证直接答案监督的贡献 |
| Frozen/EMA teacher | 固定初始 teacher 或连续 EMA teacher | 排除 teacher 漂移和恢复错误 |

正式比较时，rollout 和 teacher 必须统一音频设置，关闭 `vllm_drop_audio`；所有 full-parameter EMA 状态必须在每个中间 checkpoint 中保存并恢复；训练期间同时评估验证集 answer accuracy 和 teacher advantage。只有当 teacher advantage 为正、answer-token loss 下降、验证集提升且独立测试集提升同时出现时，才能认为多视角蒸馏真正把证据使用能力转移给了 Full-video student。
