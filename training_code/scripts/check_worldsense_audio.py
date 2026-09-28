#!/usr/bin/env python3
"""Standalone audio sanity check for the WorldSense corpus.

Verifies that every video has a decodable audio stream (codec/sample rate) and
reports anything suspicious.  Run this before an API annotation run: a silent
video would silently degrade captions for speech/music questions.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from omni_opsd.worldsense.clips import audio_stream_info  # noqa: E402


def _check(path: str) -> tuple[str, dict[str, object]]:
    try:
        return path, audio_stream_info(path)
    except Exception as exc:  # noqa: BLE001
        return path, {"present": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--expected-min-rate", type=int, default=8000)
    args = parser.parse_args()

    videos = sorted(args.video_root.glob("*.mp4"))
    print(f"checking {len(videos)} videos", flush=True)
    problems: list[dict[str, object]] = []
    ok = 0
    with mp.Pool(max(1, min(args.workers, len(videos)))) as pool:
        for path, info in pool.imap_unordered(_check, [str(v) for v in videos], chunksize=8):
            if not info.get("present"):
                problems.append({"video": Path(path).name, "issue": "no_audio", **info})
            elif int(info.get("sample_rate") or 0) < args.expected_min_rate:
                problems.append({"video": Path(path).name, "issue": "low_sample_rate", **info})
            else:
                ok += 1
    summary = {"videos": len(videos), "ok": ok, "problems": len(problems), "details": problems[:50]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("videos", "ok", "problems")}, ensure_ascii=False))
    for item in problems[:10]:
        print("  PROBLEM", item)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
