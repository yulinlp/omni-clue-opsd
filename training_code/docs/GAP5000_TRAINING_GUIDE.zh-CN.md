# OmniVideo MCQ Gap5000 四组训练：完整性审查与操作步骤

审查日期：2026-09-07（参数更新：2026-09-08）。项目根目录：`/share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907`。

## 1. 结论：数据与本地入口已补齐，当前还不能直接宣布可以开训

原代码包**并非开箱即用**。本次已补齐指定 ID 数据准备、开训预检和遗漏的旧版抽帧工具，修正 CUDA 队列及 OPSD 启动问题，并实际生成四组各 5000 条 JSONL。

**目前仍缺少已验证的 ms-swift 功能版本和经过 GPU 冒烟验证的训练环境。** 包内指定的提交是 `960c5bf2cb070d1e3483ed93965f2e338d3ae93a`。本机找到的以下仓库不能直接替代：

- `/share/home/ylhu/Light-Omni/thirdparty/ms-swift`：没有指定提交，且训练器目录结构不同。
- `/share/home/ylhu/.cache/ms-swift-upstream`：当前提交为 `d2953c5c14ee4dd3082fdd6a21b10d3cd7a54cde`，没有本实验所需的 `clue_ema_alpha` GKD 教师实现。

访问 GitHub 核实指定提交时失败，因此**没有证实该提交可以从公开仓库获取，也没有证实它不存在**。必须取得原训练机器上的功能仓库、包含该提交的 Git bundle，或提供该功能分支的仓库地址。包内几个音视频补丁不包含完整的 LoRA-shadow EMA 教师实现，不能只对任意上游版本应用它们就认为依赖齐全。

当前已对两个本地公开 clone 恢复补丁中可以确定的部分：结构化视频请求、有界音视频解码，以及 `teacher_videos`/`teacher_audios` 的 teacher 媒体透传。EMA 权重更新仍未恢复。因而标准 OPSD 可以走当前 ms-swift 的动态 self-distillation teacher；原设计的 CLUE-OPSD EMA 实验仍需正确功能仓库，非 EMA CLUE 只能作为显式标记的媒体诊断。

本次没有安装或替换共享 Python 环境，也没有申请 GPU、启动正式训练或验证显存需求。下面步骤中的运行时安装方案需要在正确功能仓库到位后执行；只有四组单步测试成功，才能确认所选机器与环境可训练。

## 2. 实际数据审查结果

| 项目 | 结果 |
| --- | --- |
| 原始标注 | `/share/home/ylhu/Light-Omni/data/omni_benchmarks/OmniVideo-100K/train_mcq_30k.jsonl` |
| ID 文件 | 项目根目录下 `omnivideo_gap5000.aggregate.sample_ids.txt` |
| ID 数 / 唯一 ID 数 / 匹配标注数 | 5000 / 5000 / 5000 |
| 唯一视频数 | 1471 |
| 不含证据区间的选中样本 | 0 |
| 已存在的视频目录 | `/share/home/ylhu/Omni-OPSD/data/cache/omnivideo_5k/videos` |
| 视频文件缺失数 / 空文件数 | 0 / 0 |
| 视频总体积 | 14,559,602,790 字节，约 13.56 GiB |
| 容器头检查 | 1471 个视频均可打开，均有视频流和音频流 |
| 四组矩阵 | 各 5000 条，ID 顺序一致，学生提示词及视频描述一致 |

容器头检查不等于逐帧解码检查，也不保证所有证据时间点都能被目标解码器正常读取；实际训练仍需单步及后续运行验证。

任务分布：事件排序 1053、比较 864、假设推理 826、情感分析 740、因果推理 731、未来预测 511、总结 275。

原始数据目录下主要是标注与分卷压缩包；本机视频已在上述缓存目录解压。无需再次解压。新增脚本的 `--video-dir` 指向直接包含 `<video_id>.mp4` 的目录，不能多加一层 `videos/`。

已生成并可查看：

- `data/gap5000/gap5000.canonical.jsonl`：保留指定 ID 顺序的规范化训练数据。
- `data/gap5000/selection_audit.json`：样本数、分布、输入及输出 SHA-256。
- `data/gap5000/media_header_audit.json`：本次视频流与音频流检查结果。
- `data/gap5000/training_matrix/omnivideo_100k_train.{sft,opsd,clue_opsd,grpo}.jsonl`。
- `data/gap5000/training_matrix/training_matrix_summary.json`：四组输入一致性与监督隔离检查、文件校验和。

以上 JSONL 引用本机视频绝对路径。迁移机器后，应按新的视频目录重新构建，不要继续使用旧路径。

## 3. 本次补充和修正

| 文件 | 内容 |
| --- | --- |
| `scripts/prepare_omnivideo_selected.py` | 按给定 ID 精确选取，不重新抽样、不切分；拒绝重复 ID、漏匹配、缺视频、无效答案和缺失证据 |
| `src/omni_opsd/data/omnivideo_100k.py` | 支持先按 ID 过滤再解析，避免无关行影响选定子集 |
| `scripts/check_training_ready.py` | 按实验组检查关键补丁、模型分片、5000 条配对矩阵、运行时导入和 CUDA；CLUE 非 EMA 诊断可显式放宽 EMA 检查 |
| `scripts/run_video_odyssey_training_arm.sh` | OPSD 使用动态 teacher；GKD 默认采用学生 top-100 + 双方 tail mass；rollout 默认 `use_vllm=true`、`top_p=1.0`、`top_k=20`；CLUE-OPSD 默认检查 EMA、也支持显式的非 EMA 诊断；避免 `pipefail` 与提前结束的 `grep -q` 误判；支持指定 Python；显式 `--split_dataset_ratio 0` |
| `scripts/run_training_arm_a100.sh` | 同步实际 CUDA 设备数；CUDA 默认启用正常梯度裁剪、使用 SDPA |
| `scripts/run_omnivideo_baseline_queue.sh` | CUDA 环境使用 A100 入口、推导设备数、消除固定八卡及 NPU 标签影响 |
| `scripts/materialize_opsd_frame_lists.py` | 补回旧版测试需要的抽帧/音视频辅助接口；本次四组 F1 训练不调用此工具 |
| `tests/test_selected_omnivideo.py`、`tests/test_cuda_launchers.py` | 覆盖精确选样、错误输入拒绝、CUDA 设备传递与冒烟失败后跳过正式训练 |

全量本地测试：**28 passed**。这包含旧版音视频命令构造测试，未执行该可选工具的真实 ffmpeg 转码。四组主流程使用原视频结构化描述，不通过稀疏抽帧替代完整音视频输入。

## 4. 四种方法实际比较的内容

| 方法名 / 命令参数 | 学生输入 | 监督与教师 |
| --- | --- | --- |
| SFT / `sft` | 完整视频及音频、问题、选项 | 答案字母作为 assistant 训练目标 |
| OPSD / `opsd` | 同上 | 教师看同一个完整音视频，教师提示词额外提供正确答案 |
| CLUE-OPSD / `clue_opsd` | 同上 | 教师仅看证据区间的音视频，不提供正确答案 |
| GRPO / `grpo` | 同上 | 正确答案仅在 `solution` 中，由选择题奖励插件计算奖励 |

命令中写 `clue_opsd`，不是 `clue-opsd`。OPSD 的 teacher 视图可以使用当前策略的动态 self-distillation；`55cadb5` 的 LoRA-shadow EMA 扩展从提交标题看是为 CLUE-OPSD 增加的。当前公开 clone 没有该实现，CLUE 默认会被拦截；设置 `OMNI_OPSD_ALLOW_NON_EMA_CLUE=1` 才会运行明确标记的非 EMA 诊断。证据来自数据集生成流程的 `analysis.designated_segments`，不是本次新增的人工标注。

四组统一采用 2 FPS、每帧 3136–28672 像素、最多 768 帧、完整视频音频启用的 F1 数据设置。保持这一数据协议不变，避免各组各自降低帧数造成不可比。

## 5. 第一步：进入目录，固定数据与模型路径

以下所有命令均从本代码包根目录执行。硬件命令默认以单机 CUDA/A100 为目标；GPU 分配方式应沿用所在集群已有的作业流程。

```bash
cd /share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907
export OMNI_OPSD_PROJECT_ROOT="$PWD"
export OMNI_OPSD_MODEL=/share/home/ylhu/models/Qwen2.5-Omni-7B
export OMNI_OPSD_MATRIX_DATASET_ROOT="$PWD/data/gap5000/training_matrix"
export OMNI_OPSD_MS_SWIFT_ROOT="$PWD/third_party/ms-swift"
```

本机发现模型目录存在；预检会进一步检查权重索引中列出的分片。GPU 节点必须能访问同样的模型和视频路径。

## 6. 第二步：取得正确的 ms-swift 功能仓库（当前待满足）

优先从原运行环境取得包含指定提交的仓库。下面的 `MS_SWIFT_SOURCE` 是必须替换的占位符，可以是本地完整 Git 仓库路径、Git bundle 路径或功能仓库地址。

```bash
export MS_SWIFT_SOURCE=/实际路径/包含所需提交的ms-swift仓库或bundle
mkdir -p "$PWD/third_party"
git clone "$MS_SWIFT_SOURCE" "$OMNI_OPSD_MS_SWIFT_ROOT"
git -C "$OMNI_OPSD_MS_SWIFT_ROOT" checkout 960c5bf2cb070d1e3483ed93965f2e338d3ae93a
git -C "$OMNI_OPSD_MS_SWIFT_ROOT" rev-parse HEAD
```

如果目标目录已经存在，不要重复 clone 或覆盖；先核实来源与提交。若 checkout 失败，停在这里获取正确源码。不要替换为上游最新提交继续运行，也不要只新增 CLI 字段冒充 EMA 教师实现。

原 README 中的公开仓库 clone 步骤只有在确认公开远端确实包含该提交时才适用。

## 7. 第三步：安装独立训练环境与完整依赖

原代码包没有锁定完整的训练环境；`requirements-training.txt` 仅列项目补充依赖，并未安装 ms-swift 自身的 Transformers、TRL、PEFT 等依赖。最好同时取得原功能仓库使用的 `pip freeze`，优先复用已经验证的版本组合。

没有原环境锁文件时，下面是安装起点，**不是已经在本机验证的版本组合**。CUDA PyTorch 应按照 GPU 主机驱动与集群已有环境安装策略选择；下面的常规 pip 安装若不适合该主机，应替换为该主机已验证的 CUDA wheel 安装命令。不要安装 Ascend/torch_npu。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install torch torchvision
python -m pip install -e "$OMNI_OPSD_MS_SWIFT_ROOT"
python -m pip install -e '.[qwen25-training,test]'
python -m pip install -r requirements-training.txt
python -m pip check

export OMNI_OPSD_PYTHON_BIN="$(command -v python)"
export OMNI_OPSD_SWIFT_BIN="$(command -v swift)"
export OMNI_OPSD_PYTHON_DEPS="$PWD"
```

`python3` 必须是 3.10–3.12。`swift` 应来自刚激活的环境。需要 ffmpeg 可执行文件时，由该机器现有的软件管理方式安装；主流程还需要能够正常导入 PyAV、librosa 和 torchvision。

## 8. 第四步：应用必需补丁

对于当前公开 clone，使用项目脚本恢复结构化视频请求、有界音视频和非 EMA teacher 媒体透传；不要单独添加 `--clue_ema_alpha` CLI 字段。只有取得包含完整 EMA 实现的功能提交后，才运行 `ensure_ms_swift_clue_cli.sh`。qwen-omni-utils 的两个运行时补丁仍需单独应用。

```bash
bash scripts/apply_ms_swift_patches.sh "$OMNI_OPSD_MS_SWIFT_ROOT"
QWEN_UTILS_ROOT="$(python -c 'from pathlib import Path; import qwen_omni_utils; print(Path(qwen_omni_utils.__file__).resolve().parent.parent)')"
bash scripts/apply_qwen_omni_utils_patches.sh "$QWEN_UTILS_ROOT"

export PYTHONPATH="$PWD/src:$OMNI_OPSD_MS_SWIFT_ROOT"
python -m py_compile \
  "$OMNI_OPSD_MS_SWIFT_ROOT/swift/infer_engine/protocol.py" \
  "$OMNI_OPSD_MS_SWIFT_ROOT/swift/template/templates/qwen.py" \
  "$OMNI_OPSD_MS_SWIFT_ROOT/swift/rl_core/data.py" \
  "$OMNI_OPSD_MS_SWIFT_ROOT/swift/rlhf_trainers/gkd_helpers.py"
```

若补丁应用失败，先检查目标版本，不要忽略错误继续训练。CLUE 视频必须同时限制视频帧和音频时间区间，否则会改变教师可见的信息。

## 9. 第五步：构建或复核指定 5000 条数据

本机已经生成了输出；如果路径与输入没变，可直接查看审计 JSON 并进入下一步。需要重建时：

```bash
export PYTHONPATH="$PWD/src:$OMNI_OPSD_MS_SWIFT_ROOT"
python scripts/prepare_omnivideo_selected.py \
  --annotation /share/home/ylhu/Light-Omni/data/omni_benchmarks/OmniVideo-100K/train_mcq_30k.jsonl \
  --sample-ids "$PWD/omnivideo_gap5000.aggregate.sample_ids.txt" \
  --video-dir /share/home/ylhu/Omni-OPSD/data/cache/omnivideo_5k/videos \
  --expected-count 5000 \
  --output-dir "$PWD/data/gap5000"

python scripts/build_omnivideo_training_matrix.py \
  --canonical "$PWD/data/gap5000/gap5000.canonical.jsonl" \
  --output-dir "$OMNI_OPSD_MATRIX_DATASET_ROOT" \
  --fps 2 --max-frames 768 --min-pixels 3136 --max-pixels 28672 \
  --use-audio-in-video

cat data/gap5000/selection_audit.json
cat data/gap5000/training_matrix/training_matrix_summary.json
```

确认 `rows=5000`、`unique_video_ids=1471`、`student_inputs_identical=true`、`answer_isolated_by_contract=true`。不要使用 `prepare_omnivideo_100k.py` 默认的随机视频训练/验证切分代替这一步，否则训练集合会改变。

## 10. 第六步：在已分配的 GPU 节点上预检

以下以已分配八张卡为例；`CUDA_VISIBLE_DEVICES` 必须对应实际获分配的设备。如果调度器已经设置它，应保留调度器值。这里不提供脱离作业分配直接占卡的启动命令。

```bash
# 仅在八张卡确实全部分配给当前作业时使用这一示例：
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMNI_OPSD_GKD_SAFE_MODE=0
export OMNI_OPSD_GKD_MAX_GRAD_NORM=1.0
export OMNI_OPSD_ATTN_IMPL=sdpa
export USE_AUDIO_IN_VIDEO=1
# OPSD/CLUE-OPSD 的 GKD/rollout 参数（启动器也会使用这些默认值）
export OMNI_OPSD_GKD_LOGITS_TOPK=100
export OMNI_OPSD_ROLLOUT_TOP_P=1.0
export OMNI_OPSD_ROLLOUT_TOP_K=20
export OMNI_OPSD_USE_VLLM=true
export OMNI_OPSD_VLLM_MODE=colocate
# colocate 显存工程参数；可按实际 GPU 显存覆盖
export OMNI_OPSD_VLLM_GPU_MEMORY_UTILIZATION=0.30
export OMNI_OPSD_VLLM_SLEEP_LEVEL=1

python scripts/check_training_ready.py \
  --arm sft \
  --ms-swift-root "$OMNI_OPSD_MS_SWIFT_ROOT" \
  --model "$OMNI_OPSD_MODEL" \
  --matrix-dir "$OMNI_OPSD_MATRIX_DATASET_ROOT"

# 若检查非 EMA 的 CLUE-OPSD 诊断，使用：
python scripts/check_training_ready.py \
  --arm clue_opsd --allow-non-ema-clue \
  --ms-swift-root "$OMNI_OPSD_MS_SWIFT_ROOT" \
  --model "$OMNI_OPSD_MODEL" \
  --matrix-dir "$OMNI_OPSD_MATRIX_DATASET_ROOT"

PYTHONPATH=.:src python -m pytest -q
bash -n scripts/run_training_arm_a100.sh \
  scripts/run_video_odyssey_training_arm.sh \
  scripts/run_omnivideo_baseline_queue.sh
```

只有对应实验组的预检退出码为 0 才继续。`--static-only` 可在无 GPU 的登录节点上检查文件与功能标记，但不验证依赖导入或 CUDA。预检的源码标记检查不代替真实训练，也不证明补丁语义正确。SFT 不要求 EMA；当前公开 clone 的 CLUE-OPSD 只能在显式 `--allow-non-ema-clue` 下通过非 EMA 诊断预检。

## 11. 第七步：四组分别进行单步冒烟测试

建议先使用与正式训练相同的输入协议。以下默认八卡、每卡 batch=1、梯度累积=4；SFT/OPSD/CLUE-OPSD 的名义有效 batch 为 32。GRPO 每个问题生成 8 个候选，其生成批次与独立问题数由固定版本框架决定，不能直接把 32 都解释为不同问题。

```bash
export OMNI_OPSD_PER_DEVICE_TRAIN_BATCH_SIZE=1
export OMNI_OPSD_GRADIENT_ACCUMULATION_STEPS=4
export OMNI_OPSD_MAX_LENGTH=32768
export OMNI_OPSD_MAX_COMPLETION_LENGTH=8
export OMNI_OPSD_SEED=20260904
export OMNI_OPSD_MIN_PIXELS=3136
export MAX_NUM_WORKERS_FETCH_VIDEO=1
export OMNI_OPSD_MAX_STEPS=1
export OMNI_OPSD_SAVE_STEPS=1

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
export GAP5000_SMOKE_ROOT="$PWD/output/gap5000_smoke_$RUN_TAG"
mkdir -p "$GAP5000_SMOKE_ROOT"
set -o pipefail
for arm in sft opsd clue_opsd grpo; do
  export OMNI_OPSD_DATASET="$OMNI_OPSD_MATRIX_DATASET_ROOT/omnivideo_100k_train.$arm.jsonl"
  export OMNI_OPSD_OUTPUT_DIR="$GAP5000_SMOKE_ROOT/$arm"
  bash scripts/run_training_arm_a100.sh "$arm" 2>&1 | tee "$GAP5000_SMOKE_ROOT/$arm.log" || break
done
```

检查四组日志均完成一步反向传播与优化，loss 为有限值，并保存有效 checkpoint。仅看到 `validated dataset` 或进程启动不算成功。OPSD 应确认教师读取完整视频；只有原设计的 CLUE-OPSD EMA 运行才应确认 EMA 教师已启用，非 EMA CLUE 诊断不能作为 EMA 实验结论。检查 `RUN_CLASSIFICATION.txt` 的 `dataset_rows=5000`、`gkd_logits_topk=100`、`rollout_top_p=1.0`、`rollout_top_k=20`、`use_vllm=true`、`paper_exact=false`、`gkd_safe_mode=0` 与正确实验组名称。启用 vLLM 前，当前训练环境必须安装与 CUDA、PyTorch 和 ms-swift 兼容的 vLLM；启动器会在缺少该包时提前退出。

如果显存不足，先记录哪一阶段失败，处理解码并发、上下文长度、生成批次或硬件资源问题。不能将某一组改成无音频/稀疏帧后仍与原四组协议混为一谈。32768 上下文及八卡示例不代表已验证的显存保证。训练框架可能对过长输入拒绝、截断或重采样，正式训练前需检查日志，避免把清单 5000 条误当作全部样本均已有效处理。

## 12. 第八步：正式训练

### 方式 A：队列自动完成四组冒烟与正式训练

这是本项目已有队列支持的路径。它会再次对每组先做一步测试，失败的组会被跳过，其余组继续。全程顺序运行，每次一组使用当前分配的全部 GPU。

```bash
export OMNI_OPSD_QUEUE_ARMS=sft,opsd,clue_opsd,grpo
export OMNI_OPSD_FULL_MAX_STEPS=300
export OMNI_OPSD_FULL_SAVE_STEPS=25
export OMNI_OPSD_QUEUE_ROOT="$PWD/output/gap5000_queue_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OMNI_OPSD_QUEUE_ROOT"
set -o pipefail
bash scripts/run_omnivideo_baseline_queue.sh 2>&1 | tee "$OMNI_OPSD_QUEUE_ROOT/queue.log"
```

这里必须设置 `OMNI_OPSD_FULL_MAX_STEPS`；队列不使用单组入口的 `OMNI_OPSD_MAX_STEPS` 作为正式步数。不要为四组队列设置同一个恢复 checkpoint，四种方法应各自从同一基座开始。

300 是优化步数，不是样本数，也不保证恰好遍历 5000 条一次。SFT 等在有效 batch=32 时，300 步名义上处理约 9600 个样本位置，具体尾批与框架行为以日志为准。默认学习率 2e-6、LoRA rank=16、alpha=32、目标模块 `all-linear`、cosine 调度、warmup_ratio=0.03、BF16。冻结模块等框架默认值应从保存的实际训练参数中复核；当前代码包不是完整的原环境参数锁。

### 方式 B：只运行一个正式实验组

```bash
arm=opsd
export OMNI_OPSD_DATASET="$OMNI_OPSD_MATRIX_DATASET_ROOT/omnivideo_100k_train.$arm.jsonl"
export OMNI_OPSD_OUTPUT_DIR="$PWD/output/gap5000_${arm}_$(date +%Y%m%d_%H%M%S)"
export OMNI_OPSD_MAX_STEPS=300
export OMNI_OPSD_SAVE_STEPS=25
bash scripts/run_training_arm_a100.sh "$arm"
```

替换 `arm` 可运行其他组；正式四组应使用同一个初始模型，不要把 SFT 输出默认作为其他三组起点。若要比较 SFT 后的二阶段训练，需要单独定义另一组实验。

## 13. 第九步：检查产物、留存环境、断点恢复

队列输出结构：

```text
output/gap5000_queue_<时间>/
  QUEUE_CLASSIFICATION.txt
  completion_manifest.tsv
  logs/smoke.<arm>.log
  logs/full.<arm>.log
  smoke/<arm>/
  full/<arm>/
    RUN_CLASSIFICATION.txt
    exit_code.txt
    SUCCESS 或 FAILED
    （ms-swift 实际 checkpoint 可能位于自动生成的子目录中）
```

查看 `completion_manifest.tsv`，要求四组 `full` 均为 `SUCCESS`，然后检查各组 checkpoint 的 `adapter_config.json`、`adapter_model.safetensors`、`trainer_state.json` 及实际步数。不要只依赖外层 SUCCESS 标记来判断实验质量。

```bash
mkdir -p "$OMNI_OPSD_QUEUE_ROOT/environment"
python -m pip freeze > "$OMNI_OPSD_QUEUE_ROOT/environment/pip-freeze.txt"
git -C "$OMNI_OPSD_MS_SWIFT_ROOT" rev-parse HEAD > "$OMNI_OPSD_QUEUE_ROOT/environment/ms-swift-commit.txt"
git -C "$OMNI_OPSD_MS_SWIFT_ROOT" diff > "$OMNI_OPSD_QUEUE_ROOT/environment/ms-swift-local.patch"
cp data/gap5000/selection_audit.json "$OMNI_OPSD_QUEUE_ROOT/environment/"
cp data/gap5000/training_matrix/training_matrix_summary.json "$OMNI_OPSD_QUEUE_ROOT/environment/"
```

恢复某一组时使用单组入口，设置该组自己的 `OMNI_OPSD_RESUME_FROM_CHECKPOINT=/实际checkpoint目录`，并保持数据、方法、采样协议与原运行一致。不要复用已经带 `SUCCESS`/`FAILED`/`SKIPPED` 的队列目录；队列会拒绝覆盖。

## 14. 训练之后的评估范围

这 5000 条是指定训练集，不能作为独立测试成绩。应另行准备与训练视频不重叠的评估集，并对基座与四个 adapter 使用相同的完整音视频学生输入和解码规则。

包内 `run_video_odyssey_training_eval.sh` 是历史 VideoOdyssey 评估入口，默认 108 条、默认音频关闭；不能直接认为它已经配置为 OmniVideo 测试集。本次完成的是指定 5000 条的训练准备与启动说明，没有新增或宣称完成独立测试集构建。

## 15. 当前执行到哪里

已完成：精确选样、全部视频存在性与音视频流检查、四组实际数据构建、本地脚本修补、28 项测试及 shell 语法检查。

待完成：取得指定 ms-swift 功能源码、在 GPU 环境安装匹配依赖并应用补丁、预检通过、四组单步真实训练验证、正式训练。第 6 节的功能源码是当前首先需要解决的外部前提。
