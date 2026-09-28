#!/usr/bin/env python3
"""Cap the dynamic reasoning OPSD media budget for full-parameter CUDA runs.

The formal dynamic-budget rows materialize ``nframes`` and an explicit resized
resolution.  Those values are ideal for the 32k-token LoRA protocol, but a
full-parameter student needs a smaller activation footprint on an 80-GiB
A100.  This utility keeps the rows, prompts, teacher contract, full time
interval and audio flag, and replaces the materialized media budget with the
same bounded ``fps``/``max_frames``/``max_pixels`` contract used by the
validated CUDA-safe OPSD run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cap_media(value: Any, *, fps: float, max_frames: int, min_pixels: int, max_pixels: int) -> Any:
    if not isinstance(value, dict):
        return value
    media = dict(value)
    # Remove the high-resolution dynamic materialization.  qwen-omni-utils
    # then samples the full [video_start, video_end] interval from this
    # explicit fps/frame/pixel cap.
    for key in ("nframes", "resized_height", "resized_width"):
        media.pop(key, None)
    media["fps"] = float(fps)
    media["max_frames"] = int(max_frames)
    media["min_pixels"] = int(min_pixels)
    media["max_pixels"] = int(max_pixels)
    return media


def cap_row(row: dict[str, Any], *, fps: float, max_frames: int, min_pixels: int, max_pixels: int) -> dict[str, Any]:
    out = deepcopy(row)
    out["videos"] = [
        cap_media(media, fps=fps, max_frames=max_frames, min_pixels=min_pixels, max_pixels=max_pixels)
        for media in out.get("videos", [])
    ]
    if "teacher_videos" in out:
        out["teacher_videos"] = [
            cap_media(media, fps=fps, max_frames=max_frames, min_pixels=min_pixels, max_pixels=max_pixels)
            for media in out.get("teacher_videos", [])
        ]
    sampling = dict(out.get("sampling_contract") or {})
    sampling.update({
        "fps": float(fps),
        "target_fps": float(fps),
        "max_frames_per_view": int(max_frames),
        "frames_per_video_input": int(max_frames),
        "teacher_frame_cap_total": int(max_frames),
        "min_pixels": int(min_pixels),
        "max_pixels": int(max_pixels),
        "cuda_memory_safe_variant": True,
        "original_dynamic_budget_version": sampling.get("dynamic_budget_version", "dynamic_video_budget_v1"),
        "original_dynamic_visual_budget": sampling.get("dynamic_video_budget"),
    })
    out["sampling_contract"] = sampling
    out["cuda_memory_safe_variant"] = {
        "max_frames_per_view": int(max_frames),
        "max_pixels": int(max_pixels),
        "fps": float(fps),
        "sampling": "uniform over the original full video interval",
        "reason": "full-parameter Qwen2.5-Omni student and teacher forwards on an 80-GiB A100",
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=7840)
    args = parser.parse_args()
    if args.fps <= 0 or args.max_frames < 64 or args.max_pixels < args.min_pixels:
        raise SystemExit("invalid safe media budget")
    rows = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(cap_row(json.loads(line), fps=args.fps, max_frames=args.max_frames,
                                    min_pixels=args.min_pixels, max_pixels=args.max_pixels))
    if not rows:
        raise SystemExit("input dataset is empty")
    ids = [str(row.get("case_id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("input dataset contains duplicate case_id values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({
        "input": str(args.input), "input_sha256": sha256(args.input),
        "output": str(args.output), "output_sha256": sha256(args.output),
        "rows": len(rows), "unique_case_ids": len(ids),
        "fps": args.fps, "max_frames": args.max_frames,
        "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
