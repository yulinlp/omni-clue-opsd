#!/usr/bin/env python3
"""Repair CLUE-OPSD teacher intervals whose requested frames exceed reality.

The dataset stores a requested ``nframes`` for each annotated interval.  The
interval duration is rounded from annotation timestamps, while the decoder
uses the source video's actual timestamps.  A one-frame discrepancy is enough
for Qwen-Omni's strict ``smart_nframes`` check to abort a DDP rank.  This tool
probes every unique source with decord, clamps safe requests to an even frame
count, and merges an adjacent interval when an annotation contains fewer than
two decodable frames.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any


def _probe(path: str) -> tuple[str, int, float, str | None]:
    try:
        import decord

        reader = decord.VideoReader(path)
        return path, len(reader), float(reader.get_avg_fps()), None
    except Exception as exc:  # pragma: no cover - exercised by data preflight
        return path, 0, 0.0, f"{type(exc).__name__}: {exc}"


def _available(path_meta: tuple[int, float], start: float, end: float) -> int:
    total, fps = path_meta
    start_frame = math.ceil(max(0.0, start) * fps)
    end_frame = min(math.floor(max(0.0, end) * fps), total - 1)
    return end_frame - start_frame + 1


def _even_at_most(value: int) -> int:
    return value - (value % 2)


def _refresh_budget(row: dict[str, Any], intervals: list[list[float]], videos: list[dict[str, Any]]) -> None:
    budget = row.get("dynamic_teacher_budget") or {}
    total_frames = sum(int(video.get("nframes", 0)) for video in videos)
    budget["nframes"] = total_frames
    budget["interval_count"] = len(intervals)
    budget["clue_intervals"] = [[float(a), float(b)] for a, b in intervals]
    grid = budget.get("spatial_grid") or [1, 1]
    spatial_tokens = int(grid[0]) * int(grid[1])
    visual_tokens = (total_frames // 2) * spatial_tokens
    budget["visual_budget_tokens"] = visual_tokens
    budget["visual_tokens_per_sampled_frame"] = visual_tokens / total_frames if total_frames else 0.0
    reserved = int(budget.get("text_reserve_tokens", 0)) + int(budget.get("audio_budget_tokens", 0)) + visual_tokens
    budget["reserved_tokens"] = reserved
    budget["max_checked_tokens"] = reserved + 512
    row["dynamic_teacher_budget"] = budget
    contract = row.get("sampling_contract") or {}
    contract["teacher_frame_cap_total"] = total_frames
    contract["teacher_frame_caps"] = [int(video.get("nframes", 0)) for video in videos]
    contract["teacher_dynamic_video_budget"] = visual_tokens
    contract["teacher_dynamic_reserved_tokens"] = reserved
    row["sampling_contract"] = contract


def _refresh_prompt(row: dict[str, Any], count: int) -> None:
    prompt = str(row.get("teacher_prompt", ""))
    question_start = prompt.find("Question:")
    if question_start < 0:
        raise ValueError(f"teacher prompt has no Question marker: {row.get('case_id')}")
    row["teacher_prompt"] = "\n".join(["<video>"] * count + [prompt[question_start:]])


def repair_rows(rows: list[dict[str, Any]], metadata: dict[str, tuple[int, float, str | None]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repaired: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        videos = [dict(video) for video in row.get("teacher_videos", [])]
        intervals = [[float(a), float(b)] for a, b in row.get("clue_intervals", [])]
        if len(videos) != len(intervals):
            raise ValueError(f"teacher interval/media mismatch at row {row_index}: {row.get('case_id')}")
        output_videos: list[dict[str, Any]] = []
        output_intervals: list[list[float]] = []
        row_repairs: list[dict[str, Any]] = []
        for interval, video in zip(intervals, videos):
            path = str(video["video"]).removeprefix("file://")
            total, fps, error = metadata[path]
            if error or total <= 0 or fps <= 0:
                raise ValueError(f"decoder metadata failed for {path}: {error}")
            start, end = interval
            available = _available((total, fps), start, end)
            if available < 2:
                if output_videos and abs(output_intervals[-1][1] - start) <= 1e-3:
                    # Keep the evidence while avoiding a one-frame structured
                    # video.  The merged descriptor receives the combined
                    # requested budget and is clamped against the merged range.
                    previous = output_videos[-1]
                    previous_interval = output_intervals[-1]
                    old_end = previous_interval[1]
                    previous_interval[1] = end
                    merged_available = _available((total, fps), previous_interval[0], end)
                    old_requested = int(previous.get("nframes", 2))
                    requested = old_requested + int(video.get("nframes", 2))
                    safe = max(2, _even_at_most(min(requested, merged_available)))
                    previous["video_end"] = float(end)
                    previous["nframes"] = safe
                    row_repairs.append({
                        "kind": "merge_sub_two_frame_interval",
                        "from": [start, end],
                        "into": [previous_interval[0], end],
                        "available_frames_before_merge": available,
                        "nframes_after_merge": safe,
                    })
                    continue
                raise ValueError(
                    f"interval has fewer than two decodable frames and is not adjacent to a previous interval: "
                    f"row={row_index} case={row.get('case_id')} interval={interval} available={available}"
                )
            requested = int(video.get("nframes", 2))
            safe_limit = _even_at_most(available)
            safe = max(2, min(requested, safe_limit))
            video["nframes"] = safe
            if safe != requested:
                row_repairs.append({
                    "kind": "clamp_nframes_to_decoder_interval",
                    "interval": [start, end],
                    "requested_nframes": requested,
                    "available_frames": available,
                    "nframes_after": safe,
                })
            output_videos.append(video)
            output_intervals.append(interval)
        if row_repairs:
            _refresh_prompt(row, len(output_videos))
            row["teacher_videos"] = output_videos
            row["clue_intervals"] = output_intervals
            _refresh_budget(row, output_intervals, output_videos)
            row["decoder_preflight"] = {
                "version": "clue_interval_frame_preflight_v1",
                "repaired": True,
                "repairs": row_repairs,
            }
            report.append({
                "row_index": row_index,
                "case_id": row.get("case_id"),
                "repairs": row_repairs,
            })
        repaired.append(row)
    return repaired, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    paths = sorted({str(video["video"]).removeprefix("file://") for row in rows for video in row.get("teacher_videos", [])})
    workers = max(1, min(int(args.workers), len(paths)))
    with mp.Pool(workers) as pool:
        probes = list(pool.imap(_probe, paths, chunksize=8))
    metadata = {path: (total, fps, error) for path, total, fps, error in probes}
    repaired, report = repair_rows(rows, metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in repaired:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": "clue_interval_frame_preflight_v1",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "rows": len(rows),
        "unique_videos": len(paths),
        "repaired_rows": len(report),
        "repairs": report,
    }
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("rows", "unique_videos", "repaired_rows")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
