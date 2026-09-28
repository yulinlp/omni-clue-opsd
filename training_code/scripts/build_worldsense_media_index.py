#!/usr/bin/env python3
"""Pre-build the WorldSense media index in parallel.

All annotation workers load this file read-only, which avoids racing probes and
repeated decord opens across the eight GPU workers.
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

from omni_opsd.worldsense.media import MediaIndex, MediaInfo, probe_video  # noqa: E402


def _probe_one(task: tuple[str, str]) -> tuple[str, dict[str, object]]:
    video_id, path = task
    try:
        info = probe_video(path)
        return video_id, {"ok": True, "info": info.as_dict()}
    except Exception as exc:  # noqa: BLE001 - reported per video
        return video_id, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    payload = json.loads(args.qa.read_text(encoding="utf-8"))
    tasks = [(video_id, str((args.video_root / f"{video_id}.mp4").resolve())) for video_id in sorted(payload)]
    print(f"probing {len(tasks)} videos with {args.workers} workers", flush=True)

    index = MediaIndex()
    failures: list[str] = []
    with mp.Pool(max(1, min(args.workers, len(tasks)))) as pool:
        for position, (video_id, result) in enumerate(pool.imap_unordered(_probe_one, tasks, chunksize=8), 1):
            key = str((args.video_root / f"{video_id}.mp4").resolve())
            if result["ok"]:
                index.entries[key] = MediaInfo(**result["info"])  # type: ignore[arg-type]
            else:
                failures.append(f"{video_id}: {result['error']}")
            if position % 200 == 0:
                print(f"  probed {position}/{len(tasks)}", flush=True)

    serializable = {key: value.as_dict() for key, value in index.entries.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output} | entries={len(serializable)} failures={len(failures)}")
    for failure in failures[:10]:
        print("  FAIL", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
