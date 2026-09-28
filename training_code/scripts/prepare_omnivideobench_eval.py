#!/usr/bin/env python3
"""Materialize an answer-free OmniVideoBench evaluation split.

The released parquet contains the gold answer and reasoning annotations.  The
model-facing JSONL emitted here contains only the user question, choices, and
the referenced video; labels are written to a separate JSONL file for strict
post-hoc scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/share/home/ylhu/datasets/OmniVideoBench"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/OmniVideoBench"),
    )
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=768)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=28672)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    parquet_path = source_root / "data.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    records = parquet.read_table(parquet_path).to_pylist()
    if len(records) != 1000:
        raise ValueError(f"expected 1000 OmniVideoBench rows, found {len(records)}")

    answer_free: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    missing: list[str] = []
    seen_ids: set[str] = set()
    for index, source in enumerate(records):
        sample_id = f"omnivideobench_{index:04d}"
        if sample_id in seen_ids:
            raise ValueError(f"duplicate sample id: {sample_id}")
        seen_ids.add(sample_id)
        choices = [str(choice) for choice in (source.get("options") or [])]
        if len(choices) != 4:
            raise ValueError(f"{sample_id} has {len(choices)} choices, expected 4")
        answer = str(source.get("correct_option") or "").strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"{sample_id} has invalid correct_option={answer!r}")
        relative_video = str(source.get("video") or "")
        video_path = (source_root / relative_video).resolve()
        if not video_path.is_file() or video_path.stat().st_size == 0:
            missing.append(str(video_path))
            continue

        prompt = (
            "<video>\n"
            f"Question: {source['question']}\n"
            "Options:\n"
            + "\n".join(choices)
            + "\nAnswer with exactly one option letter."
        )
        answer_free.append(
            {
                "messages": [{"role": "user", "content": prompt}],
                "videos": [
                    {
                        "video": str(video_path),
                        "fps": args.fps,
                        "max_frames": args.max_frames,
                        "min_pixels": args.min_pixels,
                        "max_pixels": args.max_pixels,
                    }
                ],
                "case_id": sample_id,
                "prompt_id": sample_id,
                "video_id": relative_video,
                "benchmark": "OmniVideoBench",
                "question_type": str(source.get("question_type") or ""),
                "audio_type": str(source.get("audio_type") or ""),
                "sampling_contract": {
                    "fps": args.fps,
                    "min_pixels": args.min_pixels,
                    "max_pixels": args.max_pixels,
                    "max_frames_per_view": args.max_frames,
                    "frames_per_video_input": args.max_frames,
                    "use_audio_in_video": True,
                    "student_view": "full-video-uniform",
                    "held_out_evaluation": True,
                },
            }
        )
        labels.append(
            {
                "sample_id": sample_id,
                "video_id": relative_video,
                "answer": answer,
                "question_type": str(source.get("question_type") or ""),
            }
        )

    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(
            f"{len(missing)} referenced videos are missing; finish the dataset download first.\n{preview}"
        )
    if len(answer_free) != len(labels) or len(answer_free) != 1000:
        raise RuntimeError(
            f"materialization cardinality mismatch: inputs={len(answer_free)} labels={len(labels)}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "omnivideobench.answer_free.jsonl"
    label_path = output_dir / "omnivideobench.labels.jsonl"
    _write_jsonl(input_path, answer_free)
    _write_jsonl(label_path, labels)
    manifest = {
        "benchmark": "OmniVideoBench",
        "source_parquet": str(parquet_path),
        "source_parquet_sha256": _sha256(parquet_path),
        "rows": len(answer_free),
        "unique_sample_ids": len(seen_ids),
        "unique_videos": len({row["video_id"] for row in answer_free}),
        "answer_free_path": str(input_path),
        "labels_path": str(label_path),
        "answer_free_sha256": _sha256(input_path),
        "labels_sha256": _sha256(label_path),
        "fps": args.fps,
        "max_frames": args.max_frames,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "use_audio_in_video": True,
        "answer_fields_excluded_from_model_rows": True,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
