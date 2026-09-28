"""Deterministic Qwen3-Omni frame plans without model or dataset imports."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def probe_video(video_path: str | Path) -> dict[str, float | int | str]:
    try:
        from decord import VideoReader, cpu
    except ImportError as error:
        raise RuntimeError("video probing requires the media extra: pip install -e '.[media]'") from error
    path = Path(video_path).resolve()
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    count = len(reader)
    fps = float(reader.get_avg_fps())
    if count < 2 or not math.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"invalid video metadata for {path}: count={count}, fps={fps}")
    return {
        "video_path": str(path),
        "source_frame_count": count,
        "source_fps": fps,
        "source_duration": count / fps,
        "decoder": "decord",
    }


def clamp_intervals(
    intervals: list[list[float]] | list[tuple[float, float]],
    duration: float,
    *,
    halo_before: float = 0.0,
    halo_after: float = 0.0,
    merge: bool = True,
) -> list[list[float]]:
    clipped = []
    for raw_start, raw_end in intervals:
        start = max(0.0, min(duration, float(raw_start) - halo_before))
        end = max(0.0, min(duration, float(raw_end) + halo_after))
        if end > start:
            clipped.append([start, end])
    clipped.sort()
    if not merge:
        return clipped
    merged: list[list[float]] = []
    for start, end in clipped:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def allocate_span_frame_counts(
    intervals: list[list[float]] | list[tuple[float, float]],
    *,
    nominal_fps: float,
    max_total_frames: int,
    min_frames_per_span: int = 2,
) -> list[int]:
    if not intervals or nominal_fps <= 0 or max_total_frames <= 0:
        raise ValueError("invalid evidence sampling configuration")
    if min_frames_per_span % 2 or max_total_frames % 2:
        raise ValueError("Qwen3-Omni frame allocations must be even")
    durations = [float(end) - float(start) for start, end in intervals]
    if any(value <= 0 for value in durations):
        raise ValueError(f"invalid intervals: {intervals!r}")
    if len(intervals) * min_frames_per_span > max_total_frames:
        raise ValueError("too many evidence spans for the frame budget")
    desired = [
        max(min_frames_per_span, int(2 * math.ceil(duration * nominal_fps / 2.0)))
        for duration in durations
    ]
    if sum(desired) <= max_total_frames:
        return desired
    base_pairs = min_frames_per_span // 2
    available = max_total_frames // 2 - len(intervals) * base_pairs
    weights = np.asarray(durations, dtype=np.float64)
    raw = weights / weights.sum() * available
    pairs = np.floor(raw).astype(np.int64)
    order = sorted(range(len(intervals)), key=lambda index: (-(raw[index] - pairs[index]), index))
    for index in order[: int(available - pairs.sum())]:
        pairs[index] += 1
    return [int(2 * (base_pairs + value)) for value in pairs]


def _even_full_count(duration: float, source_count: int, fps: float, max_frames: int) -> int:
    count = min(source_count, max_frames, max(2, int(math.floor(duration * fps))))
    if count % 2:
        count -= 1
    if count < 2:
        raise RuntimeError("Qwen3-Omni requires an even frame count of at least two")
    return count


def build_plan(
    record: dict[str, Any],
    metadata: dict[str, Any],
    *,
    kind: str,
    fps: float,
    max_frames: int,
    halo_before: float = 0.0,
    halo_after: float = 0.0,
) -> dict[str, Any]:
    source_fps = float(metadata["source_fps"])
    source_count = int(metadata["source_frame_count"])
    duration = float(metadata["source_duration"])
    record_id = record.get("id", record.get("sample_id", "unknown"))
    intervals_original = record.get("clue_intervals", record.get("evidence_spans", []))
    if source_fps <= 0 or source_count < 2 or duration <= 0:
        raise RuntimeError(f"invalid source metadata for {record_id}: {metadata}")
    if fps <= 0 or max_frames < 2:
        raise ValueError("fps must be positive and max_frames must be at least two")
    sampling_fps = min(float(fps), source_fps)
    if kind == "full":
        count = _even_full_count(duration, source_count, sampling_fps, max_frames)
        indices = np.linspace(0, source_count - 1, count).round().astype(int).tolist()
        intervals = [[0.0, duration]]
        indices_by_span = [indices]
        timestamps_by_span = [[index / source_fps for index in indices]]
        effective_fps = count / duration
    elif kind in {"gold", "gold_halo", "gold_dense", "gold_dense_think"}:
        intervals = clamp_intervals(
            [list(item) for item in intervals_original], duration,
            halo_before=halo_before, halo_after=halo_after, merge=True,
        )
        if not intervals:
            raise RuntimeError(f"no non-empty evidence intervals for {record_id}")
        counts = allocate_span_frame_counts(
            intervals,
            nominal_fps=sampling_fps,
            max_total_frames=max_frames if max_frames % 2 == 0 else max_frames - 1,
        )
        indices_by_span, timestamps_by_span = [], []
        actual_counts = []
        for (start, end), count in zip(intervals, counts):
            first = max(0, int(math.ceil(start * source_fps - 1e-9)))
            last = min(source_count - 1, int(math.floor(end * source_fps + 1e-9)))
            available = max(0, last - first + 1)
            actual_count = min(count, available)
            if actual_count % 2:
                actual_count -= 1
            if actual_count < 2:
                raise RuntimeError(
                    f"evidence span {start:.6f}-{end:.6f}s for {record_id} contains "
                    "fewer than two distinct native frames; refusing frame duplication"
                )
            indices = np.linspace(first, last, actual_count).round().astype(int).tolist()
            if len(set(indices)) != len(indices):
                raise RuntimeError("sampling plan would duplicate native frames")
            indices_by_span.append(indices)
            timestamps_by_span.append([index / source_fps for index in indices])
            actual_counts.append(actual_count)
        effective_fps = sum(actual_counts) / sum(end - start for start, end in intervals)
    else:
        raise ValueError(f"unknown plan kind: {kind}")
    total_frames = sum(map(len, indices_by_span))
    if total_frames < 2 or total_frames % 2:
        raise RuntimeError(f"invalid Qwen3 frame count for {record_id}: {total_frames}")
    return {
        "kind": kind,
        "requested_fps": float(fps),
        "model_fps": sampling_fps,
        "native_fps_capped": bool(float(fps) > source_fps),
        "effective_fps": float(effective_fps),
        "max_frames": int(max_frames),
        "source_fps": source_fps,
        "source_frame_count": source_count,
        "source_duration": duration,
        "original_intervals": [list(item) for item in intervals_original],
        "input_intervals": intervals,
        "halo_before": float(halo_before),
        "halo_after": float(halo_after),
        "frame_indices_by_span": indices_by_span,
        "sample_timestamps_by_span": timestamps_by_span,
        "flattened_frame_indices": [index for span in indices_by_span for index in span],
        "total_frames": total_frames,
    }
