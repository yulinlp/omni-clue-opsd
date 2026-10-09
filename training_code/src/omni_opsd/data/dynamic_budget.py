# -*- coding: utf-8 -*-
"""动态视频预算（Dynamic Budget）——SFT 与 OPSD/CLUE-OPSD 共用的“采样口径”实现。

一句话理解：
    给定一段视频（时长、分辨率）和 32768 的上下文窗口，算出：
    - 音频要占多少 token；
    - 视觉最多能占多少 token；
    - 应该抽多少帧（nframes）、把每帧缩放到多大（resized_height/width），
      才能既不超上下文、又尽量清晰。
    并把结果写进数据行，让训练时按同一口径读取视频，保证可审计、可复现。

为什么不直接用 Qwen 默认策略：
    Qwen 的默认视频策略（768 帧 / 28672 像素）会悄悄改变输入口径。
    这里把预算“物化”到每条数据的媒体描述里，训练器必须按写死的 nframes 与
    resized_* 执行，避免口径漂移。

token 估算公式（本文件的核心，务必记住）：
    视觉 token = (采样帧数 / 2) × (缩放后高/28) × (缩放后宽/28)
        解释：相邻 2 帧合并成 1 个时间组；每 28×28 像素 = 1 个视觉 token。
    音频 token = ceil(秒数 × 25)
        解释：Qwen2.5-Omni 音频前端 25 token/秒，且超过 300 秒会被截断。
    总检查：文本预留 + 音频 + 视觉 + 512 余量 ≤ 32768。
"""

# from __future__ import annotations：让类型注解在旧版 Python 上也能正常解析。
from __future__ import annotations

import math   # 取整/开方
import re     # 从 "640x360" 这类字符串里提取宽高
from typing import Any   # 类型注解


# ============================ 全局常量（预算口径） ============================
CONTEXT_TOKENS = 32_768              # 模型上下文长度（config.json 的 max_position_embeddings）
TEXT_RESERVE_TOKENS = 2_048          # 预留给问题/选项/模板等文本的 token
VISUAL_BUDGET_CAP = 24_000           # 视觉 token 的硬上限（设计上限，不是模型限制）
MAX_TOTAL_FRAMES = 300               # 最多抽 300 帧（即使视频很长也不再加帧）
TARGET_FPS = 2.0                     # 目标采样帧率：每秒 2 帧
MIN_VISUAL_TOKENS_PER_FRAME = 100    # 每帧至少给 100 个视觉 token（保证清晰度下限）
MAX_VISUAL_TOKENS_PER_FRAME = 128    # 每帧最多给 128 个视觉 token（控制总量）
AUDIO_TOKENS_PER_SECOND = 25         # 音频 25 token/秒
VIDEO_INTERVAL_OVERHEAD = 64         # 每个视频区间预留 64 token（媒体边界/模板开销）
PATCH_FACTOR = 28                    # 28 像素 = 1 视觉 token（patch 14 × merge 2）
CONTEXT_HEADROOM_TOKENS = 512        # 安全余量：预算是估算，必须留出 512 token 防超窗


def _even_floor(value: float | int) -> int:
    """向下取整到“最大的正偶数”（不足 2 就返回 2）。

    例：_even_floor(5.9) = 4；_even_floor(2.0) = 2；_even_floor(1) = 2。
    为什么需要偶数：相邻 2 帧合并为 1 个时间组，帧数必须是偶数。
    """
    result = int(math.floor(float(value)))   # 先向下取整
    result -= result % 2                     # 再抹掉奇数尾巴（% 2 余 1 就减 1）
    return max(2, result)                    # 至少返回 2


def _parse_resolution(value: Any) -> tuple[float, float]:
    """从 "640x360" / "640X360" / "640×360" 这类字符串里解析出 (宽, 高)。

    解析失败时返回默认 (16.0, 9.0)，即常见 16:9 比例，避免除零/崩溃。
    """
    # 正则：(\d+) 一组数字，[xX×] 分隔符，(\d+) 另一组数字。
    match = re.search(r"(\d+)\s*[xX×]\s*(\d+)", str(value or ""))
    if not match:
        return 16.0, 9.0
    width, height = float(match.group(1)), float(match.group(2))
    if width <= 0 or height <= 0:
        return 16.0, 9.0
    return width, height


def _choose_grid(*, aspect_ratio: float, pair_token_budget: int) -> tuple[int, int]:
    """为“一个时间组（2 帧）”选择 28 像素对齐的网格尺寸 (高网格数, 宽网格数)。

    返回的是“网格数”而不是像素：例如 (10, 20) 表示 10×28=280 高、20×28=560 宽。
    ``pair_token_budget``：一个时间组（两帧合并后）最多可用的视觉 token 数。

    选择标准（打分函数）：
      - 比例误差 ratio_error：网格宽高比与源视频宽高比越接近越好；
      - 预算误差 product_error：网格面积（= token 数）越接近预算越好；
      - 综合分 score = ratio_error + 4.0 × product_error，取最小者。
      - 权重 4.0 说明“尽量用满预算”比“比例完全一致”更重要一点；
        例如 16:9、预算 200 时，会选 10×20（正好 200）而不是比例更接近但只用了
        198 的 11×18。
    """
    if pair_token_budget < 4:
        raise ValueError(f"pair_token_budget is too small: {pair_token_budget}")
    aspect_ratio = max(float(aspect_ratio), 1e-6)   # 防止 0 或负数
    # 下限：至少 2×100=200 token 的“两帧空间”，避免选出每帧不足 100 token 的网格。
    lower_product = max(4, min(pair_token_budget, 2 * MIN_VISUAL_TOKENS_PER_FRAME))

    # candidates 收集所有候选：(综合分, 比例误差, -面积, 高网格, 宽网格)
    candidates: list[tuple[float, float, int, int, int]] = []
    # 枚举所有可能的高网格数（从 1 到预算值）。
    for height_grid in range(1, pair_token_budget + 1):
        max_width_grid = pair_token_budget // height_grid   # 该高度下最大宽网格数
        if max_width_grid < 1:
            continue
        # 理想宽度 = 宽高比 × 高网格数（小数）。
        ideal_width = aspect_ratio * height_grid
        # 只考察几个“有希望”的宽度：向下取整、向上取整、贴预算上限、上限-1、上限-2。
        # 用集合 { ... } 去重。这些边界值能覆盖“最接近比例”和“最接近预算”两类最优解。
        for width_grid in {
            max(1, min(max_width_grid, int(math.floor(ideal_width)))),
            max(1, min(max_width_grid, int(math.ceil(ideal_width)))),
            max_width_grid,
            max(1, max_width_grid - 1),
            max(1, max_width_grid - 2),
        }:
            product = height_grid * width_grid   # 该网格的 token 数（=面积）
            if product < lower_product or product > pair_token_budget:
                continue                          # 超出预算或太小，跳过
            # 比例误差用对数差：对大比例和小比例同样敏感，且天然非负。
            ratio_error = abs(math.log((width_grid / height_grid) / aspect_ratio))
            # 预算误差：没用满的比例（0 表示正好用满）。
            product_error = (pair_token_budget - product) / pair_token_budget
            # 综合打分：比例误差 + 4 倍预算误差（权重见上文说明）。
            score = ratio_error + 4.0 * product_error
            candidates.append((score, ratio_error, -product, height_grid, width_grid))
    if not candidates:
        # 兜底（极窄/极宽比例时可能一个候选都没有）：按面积开方给个近似网格。
        height_grid = max(1, int(math.sqrt(pair_token_budget / aspect_ratio)))
        width_grid = max(1, min(pair_token_budget // height_grid, round(aspect_ratio * height_grid)))
        return height_grid, width_grid
    # min(candidates) 按元组顺序比较：先比 score，再比 ratio_error、-product……
    _, _, _, height_grid, width_grid = min(candidates)
    return height_grid, width_grid


def dynamic_budget_for(
    row: dict[str, Any],
    *,
    context_tokens: int = CONTEXT_TOKENS,
    text_reserve_tokens: int = TEXT_RESERVE_TOKENS,
    visual_budget_cap: int = VISUAL_BUDGET_CAP,
    max_total_frames: int = MAX_TOTAL_FRAMES,
    target_fps: float = TARGET_FPS,
    audio_tokens_per_second: int = AUDIO_TOKENS_PER_SECOND,
    interval_overhead_tokens: int = VIDEO_INTERVAL_OVERHEAD,
) -> dict[str, Any]:
    """为“整片视频”（SFT / 普通 OPSD 学生）计算一份确定性的视听预算。

    输入 row 至少包含：
        duration             ：视频时长（秒）
        metadata.resolution  ：源分辨率字符串，如 "640x360"
    返回一个字典：nframes / resized_height / resized_width / 各类 token 统计等。
    """
    # ---- 读取时长并校验 -------------------------------------------------------
    duration = float(row.get("duration", 0.0))
    if duration <= 0:
        raise ValueError(f"invalid video duration: {duration}")
    if target_fps <= 0 or max_total_frames < 2 or max_total_frames % 2:
        raise ValueError("target_fps must be positive and max_total_frames must be even")

    # ---- 源分辨率与宽高比 -----------------------------------------------------
    metadata = row.get("metadata") or {}
    source_width, source_height = _parse_resolution(metadata.get("resolution"))
    aspect_ratio = source_width / source_height

    # ---- ① 音频预算：最多按 300 秒算（模型音频前端上限），再加区间开销 ---------
    # Qwen2.5-Omni 的音频前端硬截断在 300 秒；超出部分听不到，所以按 min(时长,300)。
    audio_seconds = min(duration, 300.0)
    audio_budget = int(math.ceil(audio_seconds * audio_tokens_per_second)) + interval_overhead_tokens

    # ---- ② 可用视觉预算 = min(上限, 上下文 − 文本预留 − 512 余量 − 音频) -------
    # 512 余量必须“提前扣掉”，否则长视频（>268 秒）会算出刚好贴着 32768 的预算，
    # 最终检查必然报错（这是历史上修过的 bug）。
    available_visual = min(
        int(visual_budget_cap),
        int(context_tokens) - int(text_reserve_tokens) - CONTEXT_HEADROOM_TOKENS - audio_budget,
    )
    if available_visual <= 0:
        raise ValueError(
            f"no visual budget remains: context={context_tokens}, text={text_reserve_tokens}, "
            f"audio={audio_budget}"
        )

    # ---- ③ 帧数：先按 2fps 理想帧数，再看“买得起多少帧” ------------------------
    # nominal_frames：理想帧数 = min(时长×2, 300)，且取偶数。
    nominal_frames = _even_floor(min(duration * target_fps, float(max_total_frames)))
    # budget_frames：按“每帧至少 100 token”折算，视觉预算最多能买多少帧。
    budget_frames = _even_floor(available_visual / MIN_VISUAL_TOKENS_PER_FRAME)
    # 实际帧数 = 三者最小（理想、上限、买得起），再强制偶数。
    nframes = min(nominal_frames, max_total_frames, budget_frames)
    nframes = max(2, nframes - nframes % 2)

    # ---- ④ 每个时间组（2 帧）的 token 预算 ------------------------------------
    # 每帧最多 128 token → 一组最多 256；同时不超过“预算/组数”。
    pair_token_budget = min(
        MAX_VISUAL_TOKENS_PER_FRAME * 2,
        int((available_visual * 2) // nframes),
    )
    pair_token_budget = max(4, pair_token_budget)
    # 选网格（返回网格数，不是像素）。
    height_grid, width_grid = _choose_grid(
        aspect_ratio=aspect_ratio,
        pair_token_budget=pair_token_budget,
    )
    # 视觉 token 总数 = 时间组数 × 每组网格面积。
    visual_tokens = (nframes // 2) * height_grid * width_grid
    # 网格数 × 28 = 缩放后的像素尺寸。
    resized_height = height_grid * PATCH_FACTOR
    resized_width = width_grid * PATCH_FACTOR
    total_reserved = text_reserve_tokens + audio_budget + visual_tokens

    # ---- ⑤ 最终检查：文本 + 音频 + 视觉 + 512 ≤ 上下文 -------------------------
    # 这是“预检”：真正的 tokenizer 编码由训练冒烟闸门验证。
    if total_reserved + CONTEXT_HEADROOM_TOKENS > context_tokens:
        raise ValueError(
            f"dynamic budget exceeds context headroom: reserved={total_reserved}, "
            f"context={context_tokens}"
        )

    # ---- 返回预算字典（这些字段会被写进数据行，训练时按此执行） -----------------
    return {
        "version": "dynamic_video_budget_v1",          # 口径版本号
        "context_tokens": int(context_tokens),
        "text_reserve_tokens": int(text_reserve_tokens),
        "audio_seconds": float(audio_seconds),         # 实际计入的音频秒数（≤300）
        "audio_budget_tokens": int(audio_budget),
        "interval_overhead_tokens": int(interval_overhead_tokens),
        "visual_budget_tokens": int(visual_tokens),
        "visual_budget_cap": int(visual_budget_cap),
        "target_fps": float(target_fps),
        "max_total_frames": int(max_total_frames),
        "nframes": int(nframes),                       # 实际抽帧数
        "source_resolution": f"{int(source_width)}x{int(source_height)}",
        "source_aspect_ratio": float(aspect_ratio),
        "resized_height": int(resized_height),         # 缩放后高度（像素）
        "resized_width": int(resized_width),           # 缩放后宽度（像素）
        "spatial_grid": [int(height_grid), int(width_grid)],   # 网格数（非像素）
        "visual_tokens_per_sampled_frame": float(visual_tokens / nframes),
        "visual_tokens_per_temporal_pair": int(height_grid * width_grid),
        "reserved_tokens": int(total_reserved),        # 文本+音频+视觉
        "max_checked_tokens": int(total_reserved + CONTEXT_HEADROOM_TOKENS),
    }


def dynamic_clue_budget_for(
    row: dict[str, Any],
    spans: list[list[float]] | tuple[tuple[float, float], ...],
    *,
    context_tokens: int = CONTEXT_TOKENS,
    text_reserve_tokens: int = TEXT_RESERVE_TOKENS,
    visual_budget_cap: int = VISUAL_BUDGET_CAP,
    max_total_frames: int = MAX_TOTAL_FRAMES,
    target_fps: float = TARGET_FPS,
    audio_tokens_per_second: int = AUDIO_TOKENS_PER_SECOND,
    interval_overhead_tokens: int = VIDEO_INTERVAL_OVERHEAD,
) -> dict[str, Any]:
    """为“多个证据区间”（CLUE-OPSD 教师）计算一份**共享**的视听预算。

    与整片版本的关键区别：
      - 所有证据段共享同一个上下文窗口（不是每段各给 24k token）；
      - 音频预算按“证据总时长”计算，每个区间额外加 64 token 边界开销；
      - 空间网格只选一次，所有区间用同一套缩放尺寸；
      - 帧数至少为 2×区间数（每段至少 2 帧，否则时间组合并不成立）。

    这样教师看到的是“几段证据视频”，但预算可审计、不会撑爆上下文。
    """
    if not spans:
        raise ValueError(f"sample {row.get('sample_id')} has no clue intervals")
    duration = float(row.get("duration", 0.0))
    if duration <= 0:
        raise ValueError(f"invalid video duration: {duration}")
    if target_fps <= 0 or max_total_frames < 2 or max_total_frames % 2:
        raise ValueError("target_fps must be positive and max_total_frames must be even")

    metadata = row.get("metadata") or {}
    source_width, source_height = _parse_resolution(metadata.get("resolution"))
    aspect_ratio = source_width / source_height

    # ---- 规范化区间：裁剪到 [0, duration]，并检查每段时长 > 0 -------------------
    normalized_spans: list[tuple[float, float]] = []
    for raw_start, raw_end in spans:
        start = max(0.0, min(duration, float(raw_start)))
        end = max(0.0, min(duration, float(raw_end)))
        if end <= start:
            raise ValueError(f"invalid clue interval for {row.get('sample_id')}: {raw_start, raw_end}")
        normalized_spans.append((start, end))

    # ---- ① 音频预算：按“证据段总时长”算（≤300s），每段再加 64 token 开销 ------
    evidence_duration = sum(end - start for start, end in normalized_spans)
    audio_seconds = min(evidence_duration, 300.0)
    audio_budget = (
        int(math.ceil(audio_seconds * audio_tokens_per_second))
        + len(normalized_spans) * int(interval_overhead_tokens)
    )
    # ---- ② 可用视觉预算（与整片版相同的扣减逻辑） ------------------------------
    available_visual = min(
        int(visual_budget_cap),
        int(context_tokens) - int(text_reserve_tokens) - CONTEXT_HEADROOM_TOKENS - audio_budget,
    )
    if available_visual <= 0:
        raise ValueError(
            f"no clue visual budget remains: context={context_tokens}, text={text_reserve_tokens}, "
            f"audio={audio_budget}"
        )

    # ---- ③ 帧数：按证据总时长算理想帧数，再受预算限制 --------------------------
    nominal_frames = _even_floor(min(evidence_duration * target_fps, float(max_total_frames)))
    budget_frames = _even_floor(available_visual / MIN_VISUAL_TOKENS_PER_FRAME)
    nframes = min(nominal_frames, max_total_frames, budget_frames)
    # 每段至少 2 帧，否则 qwen-omni 的“2 帧合并”无法成立。
    if nframes < 2 * len(normalized_spans):
        raise ValueError(
            f"clue frame budget too small for {len(normalized_spans)} intervals: {nframes}"
        )
    nframes = max(2, nframes - nframes % 2)

    # ---- ④ 选一次网格（所有区间共用） ------------------------------------------
    pair_token_budget = min(
        MAX_VISUAL_TOKENS_PER_FRAME * 2,
        int((available_visual * 2) // nframes),
    )
    pair_token_budget = max(4, pair_token_budget)
    height_grid, width_grid = _choose_grid(
        aspect_ratio=aspect_ratio,
        pair_token_budget=pair_token_budget,
    )
    visual_tokens = (nframes // 2) * height_grid * width_grid
    resized_height = height_grid * PATCH_FACTOR
    resized_width = width_grid * PATCH_FACTOR
    total_reserved = text_reserve_tokens + audio_budget + visual_tokens
    if total_reserved + CONTEXT_HEADROOM_TOKENS > context_tokens:
        raise ValueError(
            f"clue dynamic budget exceeds context headroom: reserved={total_reserved}, "
            f"context={context_tokens}"
        )

    # ---- 返回预算字典（比整片版多记录区间数量、证据总时长、区间列表） -----------
    return {
        "version": "dynamic_video_budget_v1",
        "context_tokens": int(context_tokens),
        "text_reserve_tokens": int(text_reserve_tokens),
        "audio_seconds": float(audio_seconds),
        "audio_budget_tokens": int(audio_budget),
        "interval_overhead_tokens": int(interval_overhead_tokens),
        "interval_count": len(normalized_spans),
        "evidence_duration": float(evidence_duration),
        "visual_budget_tokens": int(visual_tokens),
        "visual_budget_cap": int(visual_budget_cap),
        "target_fps": float(target_fps),
        "max_total_frames": int(max_total_frames),
        "nframes": int(nframes),
        "source_resolution": f"{int(source_width)}x{int(source_height)}",
        "source_aspect_ratio": float(aspect_ratio),
        "resized_height": int(resized_height),
        "resized_width": int(resized_width),
        "spatial_grid": [int(height_grid), int(width_grid)],
        "visual_tokens_per_sampled_frame": float(visual_tokens / nframes),
        "visual_tokens_per_temporal_pair": int(height_grid * width_grid),
        "reserved_tokens": int(total_reserved),
        "max_checked_tokens": int(total_reserved + CONTEXT_HEADROOM_TOKENS),
        "clue_intervals": [[float(start), float(end)] for start, end in normalized_spans],
    }


def dynamic_video_spec(
    row: dict[str, Any],
    budget: dict[str, Any],
    *,
    min_pixels: int = 3_136,
) -> dict[str, Any]:
    """把预算字典变成 qwen-omni 认识的“结构化视频描述”。

    row 需要包含：duration（时长）、video_path（视频路径）。
    返回的字典会直接写进数据行的 videos 列表。
    """
    duration = float(row["duration"])
    return {
        "video": str(row["video_path"]),          # 视频文件路径
        "video_start": 0.0,                       # 从第 0 秒开始（整片）
        "video_end": duration,                    # 到结尾
        # 注意：qwen-omni-utils 里 nframes 与 fps 互斥，只能给其中一个。
        "nframes": int(budget["nframes"]),
        "resized_height": int(budget["resized_height"]),
        "resized_width": int(budget["resized_width"]),
        "min_pixels": int(min_pixels),            # 每帧像素下限
        "max_pixels": int(budget["resized_height"] * budget["resized_width"]),
    }


def dynamic_sampling_contract(
    budget: dict[str, Any],
    *,
    use_audio_in_video: bool = True,
    teacher_view: str = "full-video-uniform",
) -> dict[str, Any]:
    """生成可审计的“采样契约”字段（写进数据行，启动前检查也读它）。

    启动脚本会逐行检查 sampling_contract，例如 use_audio_in_video 是否与运行时
    环境变量一致、frames_per_video_input 是否与 videos[].nframes 一致。
    """
    return {
        "fps": float(budget["target_fps"]),
        "target_fps": float(budget["target_fps"]),
        "min_pixels": 3_136,
        "max_frames_per_view": int(budget["max_total_frames"]),
        "frames_per_video_input": int(budget["nframes"]),
        "max_pixels": int(budget["resized_height"] * budget["resized_width"]),
        "resized_height": int(budget["resized_height"]),
        "resized_width": int(budget["resized_width"]),
        "student_view": "full-video-uniform",       # 学生看均匀采样的全片
        "teacher_view": teacher_view,               # 教师视角（整片或证据段）
        "teacher_frame_cap_total": int(budget["nframes"]),
        "use_audio_in_video": bool(use_audio_in_video),
        "dynamic_budget_version": budget["version"],
        "dynamic_video_budget": int(budget["visual_budget_tokens"]),
        "dynamic_audio_budget": int(budget["audio_budget_tokens"]),
        "dynamic_text_reserve": int(budget["text_reserve_tokens"]),
        "dynamic_reserved_tokens": int(budget["reserved_tokens"]),
    }
