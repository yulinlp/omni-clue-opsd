#!/usr/bin/env python3
"""Create a memory-safe OPSD matrix while retaining full-video coverage.

OPSD runs two multimodal forwards (student and answer-privileged teacher).
The paper matrix uses 768 frames and 28,672 pixels per frame, which can fit a
single SFT forward but exhausts an 80-GiB A100 during GKD.  This utility keeps
the same rows, prompts, answers, intervals, and audio contract, and only caps
the uniform temporal and spatial sampling budgets recorded in each media
mapping.  The original matrix is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cap_media(value: Any, max_frames: int, max_pixels: int) -> Any:
    if not isinstance(value, dict):
        return value
    result = dict(value)
    if "max_frames" in result:
        result["max_frames"] = min(int(result["max_frames"]), max_frames)
    else:
        result["max_frames"] = max_frames
    if "max_pixels" in result:
        result["max_pixels"] = min(int(result["max_pixels"]), max_pixels)
    else:
        result["max_pixels"] = max_pixels
    return result


def _cap_row(row: dict[str, Any], max_frames: int, max_pixels: int) -> dict[str, Any]:
    result = dict(row)
    result["videos"] = [
        _cap_media(value, max_frames, max_pixels) for value in row.get("videos", [])
    ]
    if "teacher_videos" in row:
        result["teacher_videos"] = [
            _cap_media(value, max_frames, max_pixels)
            for value in row.get("teacher_videos", [])
        ]
    sampling = dict(row.get("sampling_contract") or {})
    sampling["max_frames_per_view"] = min(
        int(sampling.get("max_frames_per_view", max_frames)), max_frames
    )
    sampling["teacher_frame_cap_total"] = min(
        int(sampling.get("teacher_frame_cap_total", max_frames)), max_frames
    )
    sampling["max_pixels"] = min(int(sampling.get("max_pixels", max_pixels)), max_pixels)
    sampling["cuda_memory_safe_variant"] = True
    sampling["original_max_frames_per_view"] = int(
        sampling.get("original_max_frames_per_view", 768)
    )
    sampling["original_max_pixels"] = int(sampling.get("original_max_pixels", 28672))
    result["sampling_contract"] = sampling
    result["cuda_memory_safe_variant"] = {
        "max_frames_per_view": max_frames,
        "max_pixels": max_pixels,
        "sampling": "uniform over the original full video interval",
        "reason": "OPSD student and teacher forwards on an 80-GiB A100",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--max-pixels", type=int, default=7840)
    args = parser.parse_args()
    if args.max_frames < 64:
        raise SystemExit("--max-frames must be >= 64 for a multi-step training run")
    if args.max_pixels < 3136:
        raise SystemExit("--max-pixels must be >= the 3136-pixel dataset minimum")
    rows = []
    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                rows.append(_cap_row(row, args.max_frames, args.max_pixels))
    if not rows:
        raise SystemExit("input dataset is empty")
    case_ids = [str(row.get("case_id")) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise SystemExit("input dataset contains duplicate case_id values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "rows": len(rows),
        "unique_case_ids": len(set(case_ids)),
        "max_frames_per_view": args.max_frames,
        "max_pixels": args.max_pixels,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
