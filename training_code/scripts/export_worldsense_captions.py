#!/usr/bin/env python3
"""Export the caption store to a readable text file (and print simple stats)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("output/worldsense_captions/captions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("output/worldsense_captions/captions.txt"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = []
    if args.store.is_file():
        for line in args.store.open(encoding="utf-8", errors="replace"):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    rows.sort(key=lambda row: str(row.get("video_id") or ""))
    if args.limit:
        rows = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"===== {row.get('video_id')} =====\n")
            handle.write(
                f"[model={row.get('model')} fps={row.get('fps')} max_pixels={row.get('max_pixels')} "
                f"chars={len(str(row.get('caption') or ''))}]\n\n"
            )
            handle.write(str(row.get("caption") or "") + "\n\n")
    chars = [len(str(row.get("caption") or "")) for row in rows]
    print(
        json.dumps(
            {
                "captions": len(rows),
                "unique_videos": len({str(row.get("video_id")) for row in rows}),
                "mean_chars": round(sum(chars) / len(chars), 1) if chars else None,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
