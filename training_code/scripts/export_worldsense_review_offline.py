#!/usr/bin/env python3
"""Build a self-contained offline copy of the review site.

The VS Code port forward has been flaky, so this exports everything the page
needs to run from the local disk: the UI with all data embedded, the small
evidence clips, and (optionally) the full faststart videos.

    python scripts/export_worldsense_review_offline.py --limit 20 --with-full

Outputs:
    output/worldsense_api_review/offline/            (open index.html locally)
    output/worldsense_api_review/worldsense_review_offline_clips.tar.gz
    output/worldsense_api_review/worldsense_review_offline_full.tar.gz   (--with-full)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from worldsense_review_server import REVIEW_DIR, build_items  # noqa: E402

UI_PATH = PROJECT_ROOT / "scripts" / "worldsense_review_ui.html"
OUT_DIR = REVIEW_DIR / "offline"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--with-full", action="store_true", help="include the full videos")
    args = parser.parse_args()

    items = build_items(limit=args.limit)
    stats = json.loads((PROJECT_ROOT / "output" / "worldsense_api_200" / "stats.json").read_text(encoding="utf-8"))
    html = UI_PATH.read_text(encoding="utf-8")
    inject = (
        "<script>\n"
        "window.REVIEW_OFFLINE = true;\n"
        f"window.REVIEW_ITEMS = {json.dumps(items, ensure_ascii=False)};\n"
        f"window.REVIEW_STATS = {json.dumps(stats, ensure_ascii=False)};\n"
        "</script>\n"
    )
    html = html.replace("<script>", inject + "<script>", 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")

    clips_src = REVIEW_DIR / "clips"
    clips_dst = OUT_DIR / "clips"
    clips_dst.mkdir(exist_ok=True)
    clip_count = 0
    for path in clips_src.glob("*.mp4"):
        shutil.copy2(path, clips_dst / path.name)
        clip_count += 1

    def size_of(path: Path) -> float:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6

    # 先打"仅片段"包（体积小、下载快）
    clips_tar = REVIEW_DIR / "worldsense_review_offline_clips.tar.gz"
    subprocess.run(["tar", "-czf", str(clips_tar), "-C", str(OUT_DIR.parent), "offline"], check=True)
    print(f"offline dir: {OUT_DIR} | clips={clip_count} | {size_of(OUT_DIR):.0f} MB")
    print(f"clips bundle: {clips_tar} ({clips_tar.stat().st_size / 1e6:.0f} MB)")

    if args.with_full:
        videos_dst = OUT_DIR / "videos"
        videos_dst.mkdir(exist_ok=True)
        video_count = 0
        for item in items:
            src = REVIEW_DIR / "faststart" / f"{item['video_id']}.mp4"
            if src.is_file():
                shutil.copy2(src, videos_dst / src.name)
                video_count += 1
        full_tar = REVIEW_DIR / "worldsense_review_offline_full.tar.gz"
        subprocess.run(["tar", "-czf", str(full_tar), "-C", str(OUT_DIR.parent), "offline"], check=True)
        print(f"full bundle: videos={video_count} dir={size_of(OUT_DIR):.0f} MB -> {full_tar} ({full_tar.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
