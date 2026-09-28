# 动态视频预算版 SFT / 标准 OPSD / Clue-OPSD 设置

本文档记录 `OMNIVIDEO_5K_EXPERIMENT_SUMMARY (2).md` 第 6、7 章在当前工作区的可执行入口。当前动态预算流包含 SFT、标准 OPSD 和固定证据 Clue-OPSD；旧版四臂矩阵保持不变。

## 数据与媒体预算

`scripts/prepare_dynamic_budget_training.py` 从冻结清单
`data/gap5000/gap5000.canonical.jsonl` 生成三条新数据流：

- SFT：`data/gap5000/dynamic_budget_v1/sft/formal/data/sft.jsonl`
- 标准 OPSD：`data/gap5000/dynamic_budget_v1/opsd/formal/data/reasoning.jsonl`
- Clue-OPSD：`data/gap5000/dynamic_budget_v1/clue_opsd/formal/data/reasoning.jsonl`

每条数据显式保存 `nframes`、`resized_height`、`resized_width`、音频开关和预算审计字段；32 条 gate 数据位于各自的 `gate/data/` 目录。SFT 的 assistant 目标是完整 `metadata.connections` 解释加换行后的 `<answer>{gold}</answer>`。OPSD 的 `messages` 只有 student user prompt，标答只在 `teacher_prompt` 中，teacher 使用同一完整视频和音频，并与 student 复用 on-policy completion token IDs。Clue-OPSD 的 student `videos` 与标准 OPSD 完全相同，teacher 只读取 `clue_intervals` 对应的多个结构化时间区间，不接收标答；所有 clue 区间共享一个视觉和音频预算。

预算规则为上下文 32,768、文本预留 2,048、音频 25 token/秒加视频区间 64 token、视觉上限 24,000、目标 2 FPS、最多 300 帧、每个采样帧 100–128 个视觉 token。当前清单经实际生成得到 5,000 行、1,471 个视频、帧数 120–240、视觉 token 15,120–24,000，平均 21,452.19；首条 129 秒 854×480 样本为 240 帧、280×560、24,000 token。每条 row 的 context preflight 保留 512 token 余量。

## 训练参数

| 参数 | SFT | 标准 OPSD | Clue-OPSD |
| --- | ---: | ---: |
| 模型 | Qwen2.5-Omni-7B | Qwen2.5-Omni-7B |
| GPU / rank | 4 | 4 |
| `per_device_train_batch_size` | 2 | 2 |
| `gradient_accumulation_steps` | 4 | 4 |
| 全局 batch | 32 | 32 |
| `max_steps` | 157（约 1 epoch） | 157（约 1 epoch） |
| LoRA `rank / alpha` | 64 / 128 | 64 / 128 |
| 学习率 | 1e-5 | 2e-6 | 2e-6 |
| scheduler / warmup | cosine / 0.03 | cosine / 0.03 |
| dtype / attention | BF16 / SDPA | BF16 / SDPA |
| `max_grad_norm` | 0 | 0 |
| 音视频输入 | `USE_AUDIO_IN_VIDEO=1` | `USE_AUDIO_IN_VIDEO=1` | `USE_AUDIO_IN_VIDEO=1` |

OPSD rollout 使用 vLLM colocate、TP=2、temperature=1、top-k=20、top-p=0.95、最多 512 token。蒸馏使用 teacher top-20 词表、JSD beta=0.5、temperature=1、`lmbda=1`、`sft_alpha=0`。本地 ms-swift HF GKD trainer 已加入 `opsd_ema_alpha=0.05` 的 LoRA-shadow EMA teacher：每个 optimizer step 后按 `teacher = 0.95 * teacher + 0.05 * student` 更新，teacher forward 临时使用 EMA LoRA；SFT 不启用该字段。

Clue-OPSD 使用同样的 rollout、top-20 JSD 和 LoRA r64/alpha128 设置，并将 `clue_ema_alpha=0.05` 传给本地 HF GKD trainer。该字段维护独立的 LoRA-shadow EMA teacher；teacher forward 临时替换为 EMA LoRA，并使用数据集证据区间对应的音视频。vLLM 0.11.x 的 Qwen2.5-Omni rollout 存在音视频占位符冲突时，启动器可记录 `OMNI_OPSD_VLLM_DROP_AUDIO=1`：这只影响 vLLM 生成阶段，Transformers student/teacher 训练前向仍按 `USE_AUDIO_IN_VIDEO=1` 编码。

## 启动入口

gate（只做 32 条输入和一步检查）：

```bash
bash scripts/run_gap5000_sft_cuda.sh gate
OMNI_OPSD_ENV=/share/home/ylhu/.conda/envs/vllm bash scripts/run_gap5000_opsd_cuda.sh smoke
```

正式 SFT：

```bash
bash scripts/run_gap5000_sft_cuda.sh formal
```

正式标准 OPSD：

```bash
bash scripts/run_gap5000_opsd_cuda.sh formal
```

正式 Clue-OPSD：

```bash
bash scripts/run_gap5000_clue_opsd_cuda.sh formal
```

需要顺序执行两条正式实验时使用 `scripts/run_gap5000_formal_cuda.sh`；它默认使用 4 张卡，先 SFT 后 OPSD，并分别设置 SFT 学习率 1e-5 与 OPSD 学习率 2e-6。正式 OPSD 默认使用 `/share/home/ylhu/.conda/envs/vllm`，因为该环境包含 vLLM；SFT 也使用同一环境以保证 `swift` CLI 与多媒体依赖一致。启动器会自动把可用的 ffmpeg 加入 PATH。

## 验证

已通过：

- 动态预算单元测试和旧数据构造测试：9 项通过；
- SFT / OPSD gate 数据的 32 行完整性、音频契约、媒体路径、帧数和 batch 预检；
- Clue-OPSD 动态 gate 数据的多段 `teacher_videos`、共享预算、音频契约、媒体路径、帧数和 batch 预检；
- Qwen2.5-Omni-7B 首条动态视频在 vLLM 模式下的模板编码，未发生 context 或 `nframes`/`fps` 冲突；
- `swift rlhf --help` 已识别 `--opsd_ema_alpha`、`--clue_ema_alpha`、`--vllm_tensor_parallel_size`、LoRA r/alpha 和 epoch 参数。

生成统计、文件 SHA-256 和完整训练参数记录在 `data/gap5000/dynamic_budget_v1/dynamic_budget_summary.json`。
