# Qwen2.5-Omni-3B/7B SFT 效果与时序上下文分析

日期：2026-09-10

## 结论

3B 的 SFT 训练确实完成了 1 个 epoch，但本次训练存在一个明确的音频配置错配，并且 LoRA 的有效参数更新幅度很小。因此，SFT 后在 OmniVideoBench 上的分数几乎不变是可以解释的。

当前评测结果如下：

| 模型 | OmniVideoBench 结果 |
|---|---:|
| Qwen2.5-Omni-3B 原模型 | 340/1000 = 34.0% |
| Qwen2.5-Omni-3B SFT | 347/1000 = 34.7%，提升 0.7 个百分点 |
| Qwen2.5-Omni-7B 原模型 | 359/1000 = 35.9% |
| Qwen2.5-Omni-7B SFT | 355/1000 = 35.5%，下降 0.4 个百分点 |

结果文件：

- [3B 原模型评测结果](</share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907/output/omnivideobench_qwen25_omni3b_base_20260909/summary.json>)
- [3B SFT 评测结果](</share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907/output/omnivideobench_qwen25_omni3b_sft_checkpoint157_20260909/summary.json>)
- [7B 原模型评测结果](</share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907/output/omnivideobench_qwen25_omni7b_base_20260909/summary.json>)
- [7B SFT 评测结果](</share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907/output/omnivideobench_qwen25_omni7b_sft_checkpoint175_20260909/summary.json>)

## 1. 3B SFT 训练日志检查

[3B 训练启动配置](</share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907/output/qwen25_omni3b_sft_1epoch_gpu02_20260908_193353/launch_config.txt>) 和 [训练日志](</share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907/output/qwen25_omni3b_sft_1epoch_gpu02_20260908_193353/launcher.log>) 显示：

- 训练数据 5000 条；
- 使用 4 张卡；
- `per_device_train_batch_size=1`；
- `gradient_accumulation_steps=8`；
- 有效 batch size 为 `4×1×8=32`；
- `num_train_epochs=1`；
- `global_step=157/157`，`epoch=1`；
- 最终训练 loss 为 0.3706；
- 没有 OOM、NaN 或中途终止；
- 没有验证集，`val_dataset=None`，没有验证 loss 和 best checkpoint 选择。

训练日志中有重复的 H264 `mmco: unref short failure` 解码警告，但训练仍完整完成 157 个优化步，目前没有证据表明训练因这些警告而跳过或异常退出。

训练样本的监督目标主要是单个选项字母 `A/B/C/D`。因此训练 token accuracy 平均约 0.852，更多反映选项输出和部分答案先验的学习，不能单独证明视频理解能力已经提升。

## 2. 3B 训练与评测存在音频配置错配

这是目前最明确的训练问题。

3B 启动配置文件中记录了 `use_audio_in_video=1`，但 [3B 启动脚本](</share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907/scripts/run_qwen25_omni3b_sft_one_epoch.sh:70>)没有导出 `USE_AUDIO_IN_VIDEO=1`。ms-swift 因而使用默认值，实际训练日志显示：

```text
Setting use_audio_in_video: False
```

证据见 [3B launcher.log](</share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907/output/qwen25_omni3b_sft_1epoch_gpu02_20260908_193353/launcher.log:940>)。

评测脚本则明确执行了：

```bash
export USE_AUDIO_IN_VIDEO=1
```

见 [OmniVideoBench 评测脚本](</share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907/scripts/run_omnivideobench_eval.sh:132>)。

因此 3B 的实际流程是：

```text
SFT：视频 + 文本
评测：视频 + 音频 + 文本
```

这会造成训练和评测的模态分布不一致。交接文档中写的 `use_audio_in_video=true` 是训练契约，不能替代本次 3B 运行时日志；本次 3B 运行时日志才是实际行为的依据。7B 正式 SFT 日志显示 `use_audio_in_video=True`，所以 7B 的小幅下降不能归因于这个特定错误。

## 3. LoRA 更新幅度较小

3B 使用 LoRA，视觉编码器和 aligner 都被冻结，只有约 0.538% 的参数参与训练。训练参数包括：

- LoRA rank=16；
- LoRA alpha=32；
- target modules=`all-linear`，实际目标主要是 thinker 的 q/k/v/o、up/down/gate 投影；
- learning rate=`2e-6`；
- cosine learning-rate schedule；
- 总共只有 157 个优化步，末尾学习率降为 0。

把 checkpoint-157 的 LoRA A/B 参数合成为有效权重增量后，252 个目标矩阵的相对更新量约为：

- `||ΔW||/||W||` 中位数：`1.97×10^-4`；
- 最大值：`5.13×10^-4`。

这说明 adapter 确实发生了更新，但相对于基座模型的改动很小。结合只训练一个 epoch、没有验证集、监督目标很短，SFT 后只出现零点几百分点的变化是符合日志表现的。

## 4. 训练数据时长、采样 FPS 和分辨率

训练文件为 [omnivideo_100k_train.sft.jsonl](</share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907/data/gap5000/training_matrix/omnivideo_100k_train.sft.jsonl>)。

按每条样本的 `video_start/video_end` 统计：

- 最短片段：60 秒；
- 中位数：100 秒；
- 最长片段：180 秒，即 3 分钟；
- 1471 个唯一视频，5000 条训练样本。

这里的 3 分钟是训练使用的视频片段长度，不一定代表原始 MP4 文件本身只有 3 分钟。

训练视频采样契约是：

- 目标采样 FPS：2.0；
- `max_frames=768`，这是上限，不是每条样本实际使用的帧数；
- 3 分钟视频理论上约采样 360 帧；
- 按实际解码统计，训练唯一视频的最大采样帧数约为 358 帧；
- 采样帧数会受原视频总帧数和偶数帧约束影响，因此实际 FPS 可能略低于 2.0。

分辨率不是固定值，而是动态调整：

- `min_pixels=3136`；
- `max_pixels=28672`；
- 宽高通常调整为 28 的倍数；
- 保持视频原始宽高比。

最长训练样例的实际视觉输入约为 `224×112`，面积约 25,088 像素。这个尺寸只是一个样例，不能作为所有训练视频的固定分辨率。

## 5. OmniVideoBench 的视频采样情况

[OmniVideoBench data.parquet](</share/home/ylhu/datasets/OmniVideoBench/data.parquet>) 共 1000 条评测样本，统计结果为：

- 最短：5 秒；
- 中位数：292 秒，即 4 分 52 秒；
- 最长：1956 秒，即 32 分 36 秒；
- 663 条超过 3 分钟；
- 491 条超过 5 分钟；
- 400 条达到 768 帧上限。

评测同样请求 2 FPS，但长视频受到 768 帧上限限制。例如最长的 32 分 36 秒视频实际只有约 0.393 FPS 的稀疏采样，其视觉输入约为：

```text
768 帧
112×224 分辨率
video_grid_thw=[384, 8, 16]
视觉 token=12,288
```

因此训练和评测存在明显的时长分布差异：

```text
训练：最多 3 分钟，约 2 FPS
评测：最长 32 分钟，长视频被压缩到最多 768 帧
```

这会影响评测绝对准确率，也会削弱短视频 SFT 对长视频评测的迁移效果。

## 6. 3B 和 7B 的上下文窗口

两个模型的 thinker 文本配置均为：

- `max_position_embeddings=32768`；
- `sliding_window=32768`；
- 没有启用 sliding window；
- 训练和推理均设置 `max_length=32768`。

配置文件：

- [Qwen2.5-Omni-3B/config.json](</share/home/ylhu/models/Qwen2.5-Omni-3B/config.json>)
- [Qwen2.5-Omni-7B/config.json](</share/home/ylhu/models/Qwen2.5-Omni-7B/config.json>)

因此 3B 和 7B 的名义多模态序列上下文窗口都是 32768 tokens。

需要区分帧数和视觉 token 数。Qwen-Omni 使用：

```text
temporal_patch_size=2
patch_size=14
spatial_merge_size=2
```

视觉 token 数约为：

```text
(T/2) × (H/14) × (W/14) / 4
```

实际最大值约为：

| 场景 | 视觉 token |
|---|---:|
| 训练最长样例 | 约 5,728 |
| 评测最长样例 | 12,288 |
| 按 768 帧和最大像素面积估计的理论上限 | 不超过约 13,824 |

因此，训练和评测的普通输入序列长度都没有因为视频帧数超过 32768。

评测启用了音频。音频预处理最大处理 300 秒，最多约产生 7,499 个音频 token。即使按最坏情况估计：

```text
视觉 13,824 + 音频 7,499 + 文本和特殊 token < 32,768
```

所以没有发生普通意义上的上下文 token 溢出。

## 7. 长视频仍存在两类上下文风险

### 7.1 音频超过 300 秒会被截断

3B 评测使用音频，但 Qwen-Omni 音频前端配置为 `chunk_length=300` 秒。超过 5 分钟的音频不会完整进入模型。如果问题所需证据发生在 5 分钟以后，模型可能无法利用对应的音频信息。

评测标注中有一部分 reasoning step 的时间点超过 300 秒，因此这不是纯理论问题。

### 7.2 时间 RoPE 位置可能超过 32768

这与普通输入 token 长度不同。Qwen-Omni 的时间位置编号按视频真实采样间隔计算：

- 训练最长片段：最大时间位置约 4.5k，低于 32768；
- 评测最长 32 分 36 秒视频：最大时间位置约 4.88 万，超过 32768。

也就是说：

```text
普通输入序列长度：没有超出 32768
长视频时间位置编号：可能超过 32768，属于 RoPE 位置外推风险
```

模型代码可能通过动态 RoPE 继续运行，但这些位置已经超出模型正常训练范围。按 `position_id_per_seconds=25` 粗略估计，约 21.8 分钟以上的视频开始进入这一风险区间，OmniVideoBench 中约有 39 条样本超过该长度。

## 8. 主要原因排序

1. **3B SFT 实际关闭了音频，但评测开启了音频。** 这是最明确的训练和评测不一致。
2. **更新步数少、学习率低且最终衰减到零。** 只有 157 个优化步，LoRA 相对权重更新非常小。
3. **训练和评测时长分布差异大。** 训练片段最多 3 分钟，评测有大量超过 3 分钟的视频，最长超过 32 分钟。
4. **长视频的 768 帧上限导致极低有效 FPS。** 这会损失长时间范围内的细节。
5. **训练目标只有一个选项字母。** 训练 loss 和 token accuracy 不能充分衡量视频视觉/音频 grounding。
6. **视觉编码器和 aligner 冻结。** LoRA 只作用于 thinker 语言模块，视觉和音频对齐能力基本没有适配。
7. **没有验证集。** 无法判断训练是否过拟合、哪个 checkpoint 最好，也无法据此调节训练策略。

## 9. 下一轮训练或评测建议

1. 在 3B SFT 脚本中显式加入 `export USE_AUDIO_IN_VIDEO=1`，并确认运行日志显示 `Setting use_audio_in_video: True`。
2. 保证训练和评测采用一致的音频策略。
3. 对超过约 20 分钟的视频使用分段评测或重新设计时间采样，避免时间位置大幅外推。
4. 对超过 300 秒的音频采用分段或时间窗口处理，避免只保留前 5 分钟。
5. 若仍要求只训练 1 个 epoch，应增加验证集，并重新评估学习率、LoRA 更新强度和可训练模块范围。
