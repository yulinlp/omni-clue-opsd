#!/usr/bin/env python3
"""Split WorldSense questions into duration-balanced shards for parallel workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TASK_KEYS = ("task0", "task1", "task2", "task3", "task4")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--media-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    args = parser.parse_args()

    payload = json.loads(args.qa.read_text(encoding="utf-8"))
    media = json.loads(args.media_index.read_text(encoding="utf-8")) if args.media_index.is_file() else {}

    def duration_for(video_id: str) -> float:
        for path, info in media.items():
            if Path(path).stem == video_id:
                return float(info.get("duration_s") or 0.0)
        return 0.0

    questions: list[tuple[float, str]] = []
    for video_id in sorted(payload):
        entry = payload[video_id] or {}
        duration = duration_for(video_id)
        for task_key in TASK_KEYS:
            if entry.get(task_key):
                questions.append((duration, f"{video_id}::{task_key}"))

    questions.sort(key=lambda item: (-item[0], item[1]))
    shards: list[list[tuple[float, str]]] = [[] for _ in range(args.num_shards)]
    for position, item in enumerate(questions):
        shards[position % args.num_shards].append(item)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"num_shards": args.num_shards, "total": len(questions), "shards": []}
    for index, shard in enumerate(shards):
        path = args.output_dir / f"shard{index}.ids"
        with path.open("w", encoding="utf-8") as handle:
            for _, question_id in shard:
                handle.write(question_id + "\n")
        manifest["shards"].append(
            {
                "index": index,
                "count": len(shard),
                "total_duration_s": round(sum(duration for duration, _ in shard), 1),
                "ids_file": str(path),
            }
        )
        print(
            f"shard{index}: {len(shard)} questions, "
            f"{round(sum(duration for duration, _ in shard) / 60.0, 1)} min of video"
        )
    (args.output_dir / "shards_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"manifest -> {args.output_dir / 'shards_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
