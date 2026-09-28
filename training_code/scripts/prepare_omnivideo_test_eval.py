#!/usr/bin/env python3
"""Materialize the answer-free 505-row OmniVideo-Test evaluation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_option(value: object) -> str:
    text = str(value).strip()
    return re.sub(r"^[A-Da-d][.)：:]\s*", "", text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/share/home/ylhu/Light-Omni/data/omni_benchmarks/OmniVideo-Test/test_505.jsonl"),
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("/share/home/ylhu/Light-Omni/data/omni_benchmarks/OmniVideo-Test"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/omnivideo_test_505_eval"),
    )
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=768)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=28672)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.source.open(encoding="utf-8") if line.strip()]
    if len(rows) != 505:
        raise ValueError(f"expected 505 rows, found {len(rows)}")

    eval_rows: list[dict] = []
    labels: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        sample_id = str(row.get("question_id") or "").strip()
        if not sample_id or sample_id in seen:
            raise ValueError(f"invalid or duplicate question_id: {sample_id!r}")
        seen.add(sample_id)
        answer = str(row.get("answer") or "").strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"invalid answer for {sample_id}: {answer!r}")
        options = [clean_option(x) for x in (row.get("options") or [])]
        if len(options) != 4:
            raise ValueError(f"{sample_id} has {len(options)} options")
        relative_video = str(row.get("video_path") or "")
        video_path = (args.video_root / relative_video).resolve()
        if not video_path.is_file() or video_path.stat().st_size == 0:
            raise FileNotFoundError(video_path)
        prompt = (
            "<video>\n"
            f"Question: {row['question']}\n"
            "Options:\n"
            + "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", options))
            + "\nAnswer with exactly one option letter."
        )
        eval_rows.append(
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
                "video_id": str(row.get("video_id") or ""),
                "benchmark": "OmniVideo-Test",
                "question_type": str(row.get("task") or row.get("subtask") or "unknown"),
                "sampling_contract": {
                    "fps": args.fps,
                    "min_pixels": args.min_pixels,
                    "max_pixels": args.max_pixels,
                    "max_frames_per_view": args.max_frames,
                    "frames_per_video_input": args.max_frames,
                    "use_audio_in_video": True,
                    "student_view": "released-question-specific-clip",
                    "held_out_evaluation": True,
                },
            }
        )
        labels.append(
            {
                "sample_id": sample_id,
                "video_id": str(row.get("video_id") or ""),
                "answer": answer,
                "question_type": str(row.get("task") or row.get("subtask") or "unknown"),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.output_dir / "omnivideo_test_505.answer_free.jsonl"
    label_path = args.output_dir / "omnivideo_test_505.labels.jsonl"
    with dataset.open("w", encoding="utf-8") as f:
        for row in eval_rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with label_path.open("w", encoding="utf-8") as f:
        for row in labels:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "benchmark": "OmniVideo-Test",
        "rows": len(eval_rows),
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "dataset": str(dataset.resolve()),
        "labels": str(label_path.resolve()),
        "dataset_sha256": sha256(dataset),
        "labels_sha256": sha256(label_path),
        "fps": args.fps,
        "max_frames": args.max_frames,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "use_audio_in_video": True,
        "answers_excluded_from_model_rows": True,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
