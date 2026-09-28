# Train_Settings — 训练参数与设置汇总

> 目的：集中记录本项目所有训练臂（**SFT / OPSD / Clue-OPSD**）的完整参数与设置，供复现、审查与实验对照。
> 数据来源：训练代码逐行提取（含默认值），文件路径与变量名均可追溯。
>
> 版本：v1（2026-09-24）｜SFT 已完整记录；OPSD / Clue-OPSD 见 §2/§3（草案，待补全）
> 本次训练目标：**Qwen2.5-Omni-7B**，数据集 = **WorldSense 证据增益筛选集 N=1500**

---

## 1. SFT（监督微调）

### 1.1 调用链与框架

```
scripts/run_gap5000_sft_cuda.sh              # 7B 主入口（formal / gate 两种模式）
  └─ scripts/run_training_arm_a100.sh sft    # CUDA/A100 适配层（设备数校验）
       └─ scripts/run_video_odyssey_training_arm.sh   # 构造并执行 ms-swift 命令
            └─ swift sft <common_args>
```

| 项 | 值 |
| --- | --- |
| 训练框架 | **ms-swift**（`third_party/ms-swift`，含自定义补丁） |
| 补丁脚本 | `scripts/apply_ms_swift_patches.sh`、`scripts/apply_qwen_omni_utils_patches.sh` |
| Python 环境 | `/share/home/ylhu/.conda/envs/vllm`（vLLM 0.11.2 / transformers 5.6.0） |
| 入口环境变量 | `OMNI_OPSD_ENV`、`OMNI_OPSD_MODEL`、`OMNI_OPSD_DATASET`、`OMNI_OPSD_*` 系列 |

### 1.2 模型与微调方式

| 参数 | 值 | 来源（脚本默认） |
| --- | --- | --- |
| 基座模型 | `/share/home/ylhu/models/Qwen2.5-Omni-7B` | `run_gap5000_sft_cuda.sh:11` |
| tuner_type | `lora` | `run_video_odyssey_training_arm.sh:27` |
| **lora_rank** | **64** | `run_gap5000_sft_cuda.sh:70`（脚本内默认 16） |
| **lora_alpha** | **128** | `run_gap5000_sft_cuda.sh:71`（脚本内默认 32） |
| **target_modules** | **all-linear** | `run_video_odyssey_training_arm.sh:599` |
| 全参模式（可选） | `OMNI_OPSD_TUNER_TYPE=full` + `FREEZE_LLM/VIT/ALIGNER=false` | `run_gap5000_sft_3b_full_cuda.sh` |

> 说明：ms-swift 多模态默认冻结 vision tower / aligner；**全参微调必须显式解冻**（LoRA 模式无此问题）。

### 1.3 优化器与调度（完整参数表）

| 参数 | 值 | 来源 |
| --- | --- | --- |
| GPU 数 / nproc_per_node | **4**（`CUDA_VISIBLE_DEVICES=0,1,2,3`） | `run_gap5000_sft_cuda.sh:12,52` |
| per_device_train_batch_size | **2** | `run_gap5000_sft_cuda.sh:68` |
| gradient_accumulation_steps | **4** | `run_gap5000_sft_cuda.sh:69` |
| **全局 batch size** | **32**（4×2×4，脚本自动校验） | `launch_config.txt` 生成逻辑 |
| max_steps | **157**（formal；= 5000/32 ≈ 1 epoch）；gate 模式 = 1 | `run_gap5000_sft_cuda.sh:33` |
| num_train_epochs | **1** | `:66` |
| **learning_rate** | **1e-5** | `:72` |
| lr_scheduler_type | **cosine** | `run_video_odyssey_training_arm.sh:559` |
| warmup_ratio | **0.03** | `:560` |
| max_grad_norm | **0**（关闭梯度裁剪） | `:565` + `run_gap5000_sft_cuda.sh:75` |
| torch_dtype | **bfloat16** | `:554` |
| gradient_checkpointing | **true** | `:566` |
| attn_impl | **sdpa** | `:567` + `run_gap5000_sft_cuda.sh:76` |
| use_logits_to_keep | **true**（仅计算最后一 token 的 logits，省显存） | `:591` 条件分支 |
| seed | **20260904** | `:85` |
| save_steps / save_total_limit | **25 / 2** | `:561,562` |
| logging_steps | **1** | `:563` |
| report_to | **none** | `:573` |
| eval_strategy | **no**（训练中不评测） | `:65` |
| split_dataset_ratio | **0**（不自动切分验证集） | `:64` |
| dataset_num_proc | **1** | `:568` |
| dataloader_num_workers | **0** | `:569` |
| dataloader_persistent_workers | **false** | `:570` |
| no_dataset_shuffle | **true**（不打乱） | `:571` |
| resume_from_checkpoint | 空（可选） | `:17` |

### 1.4 多模态输入与动态预算

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `USE_AUDIO_IN_VIDEO` | **1** | 视频自带音轨（joint 音视频输入） |
| `FORCE_QWENVL_VIDEO_READER` | **decord** | 视频解码后端 |
| `MAX_NUM_WORKERS_FETCH_VIDEO` | **1** | 视频读取并发 |
| min_pixels | **3136** | 单帧最小像素 |
| max_length | **32768** | 总上下文 |

**动态视频预算（`src/omni_opsd/data/dynamic_budget.py` 常量）**：

| 常量 | 值 |
| --- | --- |
| `CONTEXT_TOKENS` | 32,768 |
| `TEXT_RESERVE_TOKENS` | 2,048 |
| `VISUAL_BUDGET_CAP` | 24,000 |
| `MAX_TOTAL_FRAMES` | 300 |
| `TARGET_FPS` | 2.0 |
| `MIN/MAX_VISUAL_TOKENS_PER_FRAME` | 100 / 128 |
| `AUDIO_TOKENS_PER_SECOND` | **25** |
| `VIDEO_INTERVAL_OVERHEAD` | 64（每个视频区间额外预留） |
| `PATCH_FACTOR` | 28 |
| 音频时长封顶 | `audio_seconds = min(duration, 300.0)`（Qwen2.5-Omni 音频前端上限） |

预算规则：`视觉预算 = min(24000, 32768 − 2048 − 音频预算)`；帧数目标 2 FPS、上限 300；**多区间共享同一份预算**（按区间时长比例分配，偶数帧）；数据中显式保存 `nframes`、`resized_height/width`（28 对齐）。

### 1.5 训练数据格式（SFT）

JSONL 每行（示例结构）：

```json
{
  "messages": [
    {"role": "user", "content": "<video>\nQuestion: ...\nOptions:\nA. ...\nB. ...\n...\nBriefly analyze the video and audio evidence ... <answer>...</answer>"},
    {"role": "assistant", "content": "{evidence explanation}\n<answer>{letter}</answer>"}
  ],
  "videos": [{
    "video": "/path/to/video.mp4",
    "video_start": "0.0", "video_end": "129.0",
    "nframes": "240", "resized_height": "280", "resized_width": "560",
    "min_pixels": "3136", "max_pixels": "156800"
  }],
  "prompt_id": "...", "case_id": "...", "video_id": "...",
  "response_format": "reasoning",
  "dynamic_student_budget": { "context_tokens": "32768", "audio_budget_tokens": "...", "visual_budget_tokens": "24000", "nframes": "240", "spatial_grid": ["10","20"], ... }
}
```

- 数据准备脚本：`scripts/prepare_dynamic_budget_training.py`（同时产出 SFT / OPSD / CLUE-OPSD 三份 JSONL）
- assistant 监督 = 数据集已有证据解释（`metadata.connections`）+ `<answer>字母</answer>`
- **无 QA 改写**（沿用原始问题与选项）

### 1.6 环境变量与运行细节

| 变量 | 值 |
| --- | --- |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | 1 / 1 |
| `TOKENIZERS_PARALLELISM` | false |
| `MASTER_ADDR` / `MASTER_PORT` | 127.0.0.1 / 29911（SFT 入口默认） |
| 输出 | `output/gap5000_sft_cuda_<tag>/sft/`（含 `launch_config.txt`、`sft.log`） |

### 1.7 本次 WorldSense 落地方案（N=1500）

| 项 | 值 |
| --- | --- |
| 数据集 | `outputs/worldsense_gap/metrics/selected_1500.jsonl`（**1500 题 / 1081 视频**） |
| 集合质量 | Full 40.8% → Gold 64.1%，**+23.27pp**，平均 Δp +0.2019 |
| Tier 构成 | A（Δacc=+1）349 ｜ B（Δacc=0 且 Δp≥0.1）467 ｜ D 684 |
| 预计 steps | 1500 / 32 ≈ **47 steps/epoch**（1 epoch） |
| 数据准备 | **需新增** WorldSense 版 `prepare_dynamic_budget_training.py`（复用同一动态预算与 prompt 模板，输入换成 selected_1500 + WorldSense QA） |
| 训练/评测划分 | 待定：WorldSense 无官方 train/test；建议按**视频隔离**留出 held-out（或沿用参考方案的同预算 Base 对照） |

### 1.8 与筛选管线的一致性核对（重要）

| 项 | 筛选管线（本次已跑） | 训练管线（本文件） | 状态 |
| --- | --- | --- | --- |
| 视觉预算公式 | `min(24000, 32768−2048−音频)` | 同 | ✅ 一致 |
| 帧率/帧上限 | 2 FPS / 300 帧 | 同 | ✅ |
| 网格对齐 | 28 对齐、按源宽高比、不超源分辨率 | 同 | ✅ |
| 多区间预算 | 共享一份、按时长比例 | 同 | ✅ |
| **音频预算** | 用**完整时长** × 25 | **封顶 300s** × 25 | ⚠️ **不一致**：>300s 视频筛选侧高估音频 token（如 657s：16425 vs 7500）→ 建议对齐后再训练/对照 |

---

## 2. OPSD（待补全）

> 占位。以下为从代码中提取的**已知参数（草案）**，待你确认/补全后定稿。

| 参数 | 值（草案） | 来源 |
| --- | --- | --- |
| 训练入口 | `scripts/run_gap5000_opsd_3b_full_cuda.sh`（3B）/ 7B 待确认 | `scripts/` |
| rlhf_type | **gkd** | `run_video_odyssey_training_arm.sh:680` |
| learning_rate | **2e-6**（OPSD/Clue-OPSD） | 参考文档 §6.1 |
| lmbda | 1.0 | `:684` |
| beta | 0.5（JSD） | `:685` |
| temperature | 1.0 | `:686` |
| sft_alpha | 0 | `:689` |
| max_completion_length | 8 | `:690` + `:37` |
| gkd_logits_topk | 100（top-20 词表蒸馏待确认） | `:691` + `:38` |
| rollout top_p / top_k | 1.0 / 20 | `:39,40` |
| use_vllm / vllm_mode | true / colocate | `:41,54` |
| vllm_gpu_memory_utilization | 0.30 | `:41` |
| vllm_tensor_parallel_size | 1 | `:55` |
| sleep_level | 1 | `:42` |
| opsd_ema_alpha | 0.0（默认关闭）/ 0.05（EMA 阴影） | `:59,58` |
| offload_model / offload_optimizer | 视配置（rollout 期间优化器状态移 CPU） | `:68,69` |
| diag_*（诊断） | enabled=false / top_k=20 / temperature=1.0 / frequency=1 / chunk_size=256 | `:74-78` |
| LoRA-shadow EMA | `scripts/apply_ms_swift_patches.sh` 注入 | 补丁 |

## 3. Clue-OPSD（待补全）

> 占位。已知草案（同上表公共部分），Clue 专属：

| 参数 | 值（草案） | 来源 |
| --- | --- | --- |
| clue_ema_alpha | **0.05**（默认；`allow_non_ema_clue=0` 时启用） | `:58,70` |
| full_ema_teacher | false（可选 true） | `:60` |
| full_ema_alpha / full_ema_offload | 0.05 / false | `:61,62` |
| gold_ce_alpha | 0.25（full-EMA teacher 时） | `:63` |
| teacher 输入 | **证据区间（Clue）音视频**，不接收标答 | 方案 §2 |
| student 输入 | Full 音视频（与 SFT 逐条一致） | 方案 §6.1 |
| 训练数据 | `data/gap5000/dynamic_budget_v1/clue_opsd/.../reasoning.jsonl` | `prepare_dynamic_budget_training.py` |
| EMA teacher 更新 | LoRA-shadow EMA（补丁实现） | `apply_ms_swift_patches.sh` |

---

## 附：关键代码位置索引

| 内容 | 路径 |
| --- | --- |
| SFT 7B 入口 | `scripts/run_gap5000_sft_cuda.sh` |
| SFT 3B 全参入口 | `scripts/run_gap5000_sft_3b_full_cuda.sh` |
| 训练臂通用实现 | `scripts/run_video_odyssey_training_arm.sh` |
| A100 适配层 | `scripts/run_training_arm_a100.sh` |
| 动态预算常量 | `src/omni_opsd/data/dynamic_budget.py` |
| 数据准备 | `scripts/prepare_dynamic_budget_training.py` |
| ms-swift 补丁 | `scripts/apply_ms_swift_patches.sh`、`scripts/apply_qwen_omni_utils_patches.sh` |
| 参考实验总结 | `docs/OMNIVIDEO_5K_EXPERIMENT_SUMMARY (2).md`（§6.1 训练设置、§7 prompt） |
| WorldSense 筛选产物 | `outputs/worldsense_gap/metrics/selected_1500.jsonl` |
| WorldSense 筛选方案 | `docs/WORLDSENSE_EVIDENCE_GAP_SCREENING.zh-CN.md` |
