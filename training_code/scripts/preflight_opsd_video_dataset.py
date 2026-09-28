#!/usr/bin/env python3
"""Decode preflight for the full-video OE-5k OPSD dataset."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any


def probe(path: str) -> dict[str, Any]:
    try:
        import decord

        reader = decord.VideoReader(path, num_threads=1)
        total = len(reader)
        if total <= 0:
            raise ValueError("empty video")
        # Exercise the first, middle, and final random access used by the
        # uniform full-video sampler, rather than only opening the container.
        indices = sorted({0, total // 2, total - 1})
        reader.get_batch(indices)
        return {"path": path, "frames": total, "fps": float(reader.get_avg_fps()), "error": None}
    except Exception as exc:  # pragma: no cover - data-dependent
        return {"path": path, "frames": 0, "fps": 0.0, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.dataset.open(encoding="utf-8") if line.strip()]
    paths: set[str] = set()
    missing: list[str] = []
    for row in rows:
        for key in ("videos", "teacher_videos"):
            for video in row.get(key, []) or []:
                path = str(video.get("video", "")).removeprefix("file://")
                if not path:
                    missing.append(f"{row.get('case_id')}: empty video path")
                elif not Path(path).is_file() or Path(path).stat().st_size == 0:
                    missing.append(f"{row.get('case_id')}: {path}")
                else:
                    paths.add(path)
    workers = max(1, min(int(args.workers), max(len(paths), 1)))
    with mp.Pool(workers) as pool:
        probes = list(pool.imap(probe, sorted(paths), chunksize=8))
    failures = [p for p in probes if p["error"]]
    summary = {
        "version": "opsd_video_decode_preflight_v1",
        "dataset": str(args.dataset.resolve()),
        "rows": len(rows),
        "unique_videos": len(paths),
        "missing": missing,
        "probed": len(probes),
        "decode_failures": failures,
        "passed": not missing and not failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("rows", "unique_videos", "missing", "probed", "decode_failures", "passed")}, ensure_ascii=False))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
