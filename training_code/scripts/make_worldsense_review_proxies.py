#!/usr/bin/env python3
"""Remux review videos with -movflags +faststart for instant web playback.

The WorldSense mp4 files keep the moov atom at the end, so a browser must fetch
the tail before it can play.  Remuxing (stream copy, no re-encode) moves moov
to the front; the review server serves these files when present.

Usage:
    python scripts/make_worldsense_review_proxies.py --limit 20 --workers 4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from worldsense_review_server import REVIEW_DIR, VIDEO_DIR, build_items  # noqa: E402

FFMPEG = "/share/home/ylhu/.conda/envs/omniagent_gyh/bin/ffmpeg"
OUT_DIR = REVIEW_DIR / "faststart"


def remux(video_id: str) -> tuple[str, str]:
    src = VIDEO_DIR / f"{video_id}.mp4"
    dst = OUT_DIR / f"{video_id}.mp4"
    if dst.is_file() and dst.stat().st_size > 0:
        return video_id, "cached"
    tmp = dst.with_suffix(".tmp.mp4")
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-i", str(src),
        "-c", "copy", "-movflags", "+faststart",
        str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return video_id, f"FAILED: {result.stderr.strip()[:120]}"
    tmp.replace(dst)
    return video_id, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    video_ids = [item["video_id"] for item in build_items(limit=args.limit)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for video_id, status in pool.map(remux, video_ids):
            print(f"  {video_id}: {status}")
    total = sum(f.stat().st_size for f in OUT_DIR.glob("*.mp4"))
    print(f"faststart 目录: {OUT_DIR} | {len(list(OUT_DIR.glob('*.mp4')))} 个文件 | {total / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
