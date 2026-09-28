#!/usr/bin/env python3
"""Prepare a deterministic video-disjoint VideoOdyssey OPSD staging split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from omni_opsd.data.swift_opsd import read_jsonl, swift_opsd_row, video_split, write_jsonl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-videos", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--max-pixels", type=int, default=180_000)
    parser.add_argument("--use-audio-in-video", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    if not rows:
        raise SystemExit("input manifest is empty")
    train, val = video_split(rows, val_video_count=args.val_videos, seed=args.seed)
    train_video_ids = {str(row["video_id"]) for row in train}
    val_video_ids = {str(row["video_id"]) for row in val}
    if train_video_ids & val_video_ids:
        raise RuntimeError("video-level leakage detected")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_canonical": output_dir / "video_odyssey_train.canonical.jsonl",
        "val_canonical": output_dir / "video_odyssey_val.canonical.jsonl",
        "train_swift_opsd": output_dir / "video_odyssey_train.swift_opsd.jsonl",
        "val_swift_opsd": output_dir / "video_odyssey_val.swift_opsd.jsonl",
        "val_labels": output_dir / "video_odyssey_val.labels.jsonl",
    }
    write_jsonl(paths["train_canonical"], train)
    write_jsonl(paths["val_canonical"], val)
    write_jsonl(
        paths["train_swift_opsd"],
        (
            swift_opsd_row(
                row,
                fps=args.fps,
                max_frames=args.max_frames,
                max_pixels=args.max_pixels,
                use_audio_in_video=args.use_audio_in_video,
            )
            for row in train
        ),
    )
    write_jsonl(
        paths["val_swift_opsd"],
        (
            swift_opsd_row(
                row,
                fps=args.fps,
                max_frames=args.max_frames,
                max_pixels=args.max_pixels,
                use_audio_in_video=args.use_audio_in_video,
            )
            for row in val
        ),
    )
    write_jsonl(
        paths["val_labels"],
        (
            {
                "sample_id": str(row["sample_id"]),
                "video_id": str(row["video_id"]),
                "answer": row.get("answer"),
            }
            for row in val
        ),
    )
    summary = {
        "status": "temporary_protocol_pending_dataset_research",
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": _sha256(args.manifest),
        "seed": args.seed,
        "split_unit": "video_id",
        "train_records": len(train),
        "train_videos": len(train_video_ids),
        "val_records": len(val),
        "val_videos": len(val_video_ids),
        "video_overlap": sorted(train_video_ids & val_video_ids),
        "answer_labels_excluded_from_swift_rows": True,
        "fps": args.fps,
        "max_frames": args.max_frames,
        "max_pixels": args.max_pixels,
        "use_audio_in_video": args.use_audio_in_video,
        "paths": {name: str(path) for name, path in paths.items()},
    }
    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
