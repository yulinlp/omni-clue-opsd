"""Dynamic full-video budgets used by the current SFT and OPSD experiments.

The budget is deliberately materialized into every media descriptor.  Qwen's
Omni video utility accepts ``nframes`` and ``resized_height``/
``resized_width`` directly, which makes the context calculation auditable and
avoids silently falling back to the old 768-frame/28,672-pixel policy.
"""

from __future__ import annotations

import math
import re
from typing import Any


CONTEXT_TOKENS = 32_768
TEXT_RESERVE_TOKENS = 2_048
VISUAL_BUDGET_CAP = 24_000
MAX_TOTAL_FRAMES = 300
TARGET_FPS = 2.0
MIN_VISUAL_TOKENS_PER_FRAME = 100
MAX_VISUAL_TOKENS_PER_FRAME = 128
AUDIO_TOKENS_PER_SECOND = 25
VIDEO_INTERVAL_OVERHEAD = 64
PATCH_FACTOR = 28
CONTEXT_HEADROOM_TOKENS = 512


def _even_floor(value: float | int) -> int:
    """Return the largest positive even integer not greater than ``value``."""

    result = int(math.floor(float(value)))
    result -= result % 2
    return max(2, result)


def _parse_resolution(value: Any) -> tuple[float, float]:
    match = re.search(r"(\d+)\s*[xX×]\s*(\d+)", str(value or ""))
    if not match:
        return 16.0, 9.0
    width, height = float(match.group(1)), float(match.group(2))
    if width <= 0 or height <= 0:
        return 16.0, 9.0
    return width, height


def _choose_grid(*, aspect_ratio: float, pair_token_budget: int) -> tuple[int, int]:
    """Choose 28-pixel grid dimensions close to the source aspect ratio.

    ``pair_token_budget`` is the spatial token budget for two temporal frames.
    We prefer a grid close to the source ratio while retaining at least 80% of
    the available budget.  The dimensions are returned in grid units, not
    pixels.  The budget term has a slightly higher weight than the aspect
    term: this makes the documented 16:9/200-token case resolve to 10x20
    (24,000 visual tokens at 240 frames), rather than silently dropping to
    11x18 (23,760 tokens).
    """

    if pair_token_budget < 4:
        raise ValueError(f"pair_token_budget is too small: {pair_token_budget}")
    aspect_ratio = max(float(aspect_ratio), 1e-6)
    # The lower bound is also the selected 100-token/frame floor expressed in
    # two-frame spatial units.  For the current 2 FPS/24k policy the pair
    # budget is never below 200, so this excludes otherwise valid-looking
    # 96-token/frame grids near an aspect-ratio boundary.
    lower_product = max(4, min(pair_token_budget, 2 * MIN_VISUAL_TOKENS_PER_FRAME))
    candidates: list[tuple[float, float, int, int, int]] = []
    for height_grid in range(1, pair_token_budget + 1):
        max_width_grid = pair_token_budget // height_grid
        if max_width_grid < 1:
            continue
        # Include the nearest ratio widths and the nearest budget boundary.
        # The latter is necessary for common 16:9 inputs: 10x20 is the
        # closest full-budget grid at a 200-token pair budget, although 20 is
        # not the nearest integer to 1.78*10.
        ideal_width = aspect_ratio * height_grid
        for width_grid in {
            max(1, min(max_width_grid, int(math.floor(ideal_width)))),
            max(1, min(max_width_grid, int(math.ceil(ideal_width)))),
            max_width_grid,
            max(1, max_width_grid - 1),
            max(1, max_width_grid - 2),
        }:
            product = height_grid * width_grid
            if product < lower_product or product > pair_token_budget:
                continue
            ratio_error = abs(math.log((width_grid / height_grid) / aspect_ratio))
            product_error = (pair_token_budget - product) / pair_token_budget
            # Use almost all of the budget while keeping the source ratio.
            # The product term is deliberately strong enough to prefer 10x20
            # over 11x18 at budget 200, while 12x21 remains preferable to a
            # badly shaped 16x16 grid at budget 256.
            score = ratio_error + 4.0 * product_error
            candidates.append((score, ratio_error, -product, height_grid, width_grid))
    if not candidates:
        # This fallback is only reachable for unusually narrow source ratios.
        height_grid = max(1, int(math.sqrt(pair_token_budget / aspect_ratio)))
        width_grid = max(1, min(pair_token_budget // height_grid, round(aspect_ratio * height_grid)))
        return height_grid, width_grid
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
    """Allocate a deterministic full-video visual/audio budget for one row."""

    duration = float(row.get("duration", 0.0))
    if duration <= 0:
        raise ValueError(f"invalid video duration: {duration}")
    if target_fps <= 0 or max_total_frames < 2 or max_total_frames % 2:
        raise ValueError("target_fps must be positive and max_total_frames must be even")

    metadata = row.get("metadata") or {}
    source_width, source_height = _parse_resolution(metadata.get("resolution"))
    aspect_ratio = source_width / source_height

    # Qwen2.5-Omni's audio frontend is bounded at 300 seconds.  The extra
    # interval overhead reserves room for the media/template boundary tokens.
    audio_seconds = min(duration, 300.0)
    audio_budget = int(math.ceil(audio_seconds * audio_tokens_per_second)) + interval_overhead_tokens
    # Reserve the 512-token template/media headroom up front.  Without this the
    # grid selection uses ~all of the remaining context and the final check
    # below rejects any video long enough for the visual cap to stop binding
    # (>~268 s), which crashed the WorldSense data preparation.
    available_visual = min(
        int(visual_budget_cap),
        int(context_tokens) - int(text_reserve_tokens) - CONTEXT_HEADROOM_TOKENS - audio_budget,
    )
    if available_visual <= 0:
        raise ValueError(
            f"no visual budget remains: context={context_tokens}, text={text_reserve_tokens}, "
            f"audio={audio_budget}"
        )

    nominal_frames = _even_floor(min(duration * target_fps, float(max_total_frames)))
    # At least 100 visual tokens per sampled frame is the selected policy. If
    # the nominal 2 FPS count cannot fit, uniformly reduce the frame count.
    budget_frames = _even_floor(available_visual / MIN_VISUAL_TOKENS_PER_FRAME)
    nframes = min(nominal_frames, max_total_frames, budget_frames)
    nframes = max(2, nframes - nframes % 2)

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

    # Keep 512 tokens of headroom beyond the configured text reserve. This is
    # a metadata/preflight check; actual tokenizer encoding is performed by
    # the training smoke gate before a formal run.
    if total_reserved + CONTEXT_HEADROOM_TOKENS > context_tokens:
        raise ValueError(
            f"dynamic budget exceeds context headroom: reserved={total_reserved}, "
            f"context={context_tokens}"
        )

    return {
        "version": "dynamic_video_budget_v1",
        "context_tokens": int(context_tokens),
        "text_reserve_tokens": int(text_reserve_tokens),
        "audio_seconds": float(audio_seconds),
        "audio_budget_tokens": int(audio_budget),
        "interval_overhead_tokens": int(interval_overhead_tokens),
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
    """Allocate one shared visual/audio budget across clue intervals.

    CLUE-OPSD must not give every evidence interval a fresh 24k-token budget.
    The intervals share one context window; the audio and 64-token boundary
    reserve are computed over the concatenated evidence duration, while the
    spatial grid is selected once for all intervals.  This is the dynamic
    policy described in the experiment summary and keeps the teacher media
    auditable without materialising clipped MP4 files.
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
    normalized_spans: list[tuple[float, float]] = []
    for raw_start, raw_end in spans:
        start = max(0.0, min(duration, float(raw_start)))
        end = max(0.0, min(duration, float(raw_end)))
        if end <= start:
            raise ValueError(f"invalid clue interval for {row.get('sample_id')}: {raw_start, raw_end}")
        normalized_spans.append((start, end))

    evidence_duration = sum(end - start for start, end in normalized_spans)
    audio_seconds = min(evidence_duration, 300.0)
    audio_budget = (
        int(math.ceil(audio_seconds * audio_tokens_per_second))
        + len(normalized_spans) * int(interval_overhead_tokens)
    )
    # Reserve the 512-token template/media headroom up front.  Without this the
    # grid selection uses ~all of the remaining context and the final check
    # below rejects any video long enough for the visual cap to stop binding
    # (>~268 s), which crashed the WorldSense data preparation.
    available_visual = min(
        int(visual_budget_cap),
        int(context_tokens) - int(text_reserve_tokens) - CONTEXT_HEADROOM_TOKENS - audio_budget,
    )
    if available_visual <= 0:
        raise ValueError(
            f"no clue visual budget remains: context={context_tokens}, text={text_reserve_tokens}, "
            f"audio={audio_budget}"
        )

    nominal_frames = _even_floor(min(evidence_duration * target_fps, float(max_total_frames)))
    budget_frames = _even_floor(available_visual / MIN_VISUAL_TOKENS_PER_FRAME)
    nframes = min(nominal_frames, max_total_frames, budget_frames)
    # At least two frames per interval keeps each structured video mapping
    # valid for qwen-omni's temporal merge.  The frozen gap5000 evidence set
    # satisfies this under the 2 FPS policy.
    if nframes < 2 * len(normalized_spans):
        raise ValueError(
            f"clue frame budget too small for {len(normalized_spans)} intervals: {nframes}"
        )
    nframes = max(2, nframes - nframes % 2)

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
    """Create a Qwen-Omni structured full-video descriptor."""

    duration = float(row["duration"])
    return {
        "video": str(row["video_path"]),
        "video_start": 0.0,
        "video_end": duration,
        # ``nframes`` and ``fps`` are mutually exclusive in qwen-omni-utils.
        "nframes": int(budget["nframes"]),
        "resized_height": int(budget["resized_height"]),
        "resized_width": int(budget["resized_width"]),
        "min_pixels": int(min_pixels),
        "max_pixels": int(budget["resized_height"] * budget["resized_width"]),
    }


def dynamic_sampling_contract(
    budget: dict[str, Any],
    *,
    use_audio_in_video: bool = True,
    teacher_view: str = "full-video-uniform",
) -> dict[str, Any]:
    """Return the auditable sampling fields consumed by the launch preflight."""

    return {
        "fps": float(budget["target_fps"]),
        "target_fps": float(budget["target_fps"]),
        "min_pixels": 3_136,
        "max_frames_per_view": int(budget["max_total_frames"]),
        "frames_per_video_input": int(budget["nframes"]),
        "max_pixels": int(budget["resized_height"] * budget["resized_width"]),
        "resized_height": int(budget["resized_height"]),
        "resized_width": int(budget["resized_width"]),
        "student_view": "full-video-uniform",
        "teacher_view": teacher_view,
        "teacher_frame_cap_total": int(budget["nframes"]),
        "use_audio_in_video": bool(use_audio_in_video),
        "dynamic_budget_version": budget["version"],
        "dynamic_video_budget": int(budget["visual_budget_tokens"]),
        "dynamic_audio_budget": int(budget["audio_budget_tokens"]),
        "dynamic_text_reserve": int(budget["text_reserve_tokens"]),
        "dynamic_reserved_tokens": int(budget["reserved_tokens"]),
    }
