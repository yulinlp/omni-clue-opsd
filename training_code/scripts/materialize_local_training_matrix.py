#!/usr/bin/env python3
"""Rewrite a frozen training matrix to use worker-local OmniVideo media.

Only structured media fields are changed.  Messages, answers, supervision,
evidence intervals, and sampling parameters remain byte-for-byte equivalent
after JSON decoding.  The persistent matrix stays authoritative; this local
copy is an ephemeral I/O optimization that can be regenerated at any time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _localize(value: Any, cache_root: Path, counters: dict[str, int]) -> Any:
    if isinstance(value, list):
        return [_localize(item, cache_root, counters) for item in value]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key == "video" and isinstance(item, str):
            target = cache_root / Path(item).name
            if not target.is_file():
                raise FileNotFoundError(
                    f"training media is absent from required local cache: {target} "
                    f"(persistent source: {item})"
                )
            result[key] = str(target)
            counters["media_references"] += 1
            counters["unique_sources"].add(item)
            counters["unique_targets"].add(str(target))
        else:
            result[key] = _localize(item, cache_root, counters)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--video-cache-root", type=Path, required=True)
    parser.add_argument(
        "--arms", default="sft,grpo,opsd,clue_opsd", help="comma-separated matrix arms"
    )
    args = parser.parse_args()
    arms = [item.strip() for item in args.arms.split(",") if item.strip()]
    if not arms or len(set(arms)) != len(arms):
        raise ValueError("--arms must be non-empty and unique")
    if not args.video_cache_root.is_dir():
        raise FileNotFoundError(args.video_cache_root)

    args.output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "status": "complete",
        "persistent_source_root": str(args.source_root.resolve()),
        "local_output_root": str(args.output_root.resolve()),
        "video_cache_root": str(args.video_cache_root.resolve()),
        "scratch_is_ephemeral": True,
        "arms": {},
    }
    for arm in arms:
        source = args.source_root / f"omnivideo_100k_train.{arm}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(source)
        counters: dict[str, Any] = {
            "media_references": 0,
            "unique_sources": set(),
            "unique_targets": set(),
        }
        rows: list[dict[str, Any]] = []
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(_localize(json.loads(line), args.video_cache_root, counters))
        if not rows or counters["media_references"] == 0:
            raise ValueError(f"matrix arm has no rows or media references: {source}")
        target = args.output_root / source.name
        temporary = target.with_name(f".{target.name}.partial.{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, target)
        report["arms"][arm] = {
            "rows": len(rows),
            "media_references": counters["media_references"],
            "unique_source_media": len(counters["unique_sources"]),
            "unique_local_media": len(counters["unique_targets"]),
            "source_sha256": _sha256(source),
            "local_sha256": _sha256(target),
        }

    source_summary = args.source_root / "training_matrix_summary.json"
    if source_summary.is_file():
        target_summary = args.output_root / source_summary.name
        target_summary.write_bytes(source_summary.read_bytes())
        report["persistent_summary_sha256"] = _sha256(source_summary)
    marker = args.output_root / "LOCAL_MATRIX_SUCCESS.json"
    marker.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
