#!/usr/bin/env python3
"""Build an answer-free, video-disjoint WorldSense evaluation subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--sft-train", type=Path, required=True)
    parser.add_argument("--clue-train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=0, help="0 uses all video-disjoint questions")
    parser.add_argument("--sample-seed", type=int, default=20260930)
    parser.add_argument("--tier-metrics", type=Path, help="screening per_question.jsonl used to filter tiers")
    parser.add_argument("--tiers", help="comma-separated tiers to keep, for example A,B,D")
    parser.add_argument("--max-duration", type=float, help="maximum source video duration in seconds")
    args = parser.parse_args()
    if bool(args.tier_metrics) != bool(args.tiers):
        parser.error("--tier-metrics and --tiers must be specified together")
    allowed_tiers = None
    tier_by_id = {}
    if args.tiers:
        allowed_tiers = {tier.strip().upper() for tier in args.tiers.split(",")}
        if not allowed_tiers or not allowed_tiers <= {"A", "B", "C", "D"}:
            parser.error("--tiers must be a comma-separated subset of A,B,C,D")
        metrics = read_jsonl(args.tier_metrics)
        tier_by_id = {row["sample_id"]: row["tier"] for row in metrics}
        if len(tier_by_id) != len(metrics):
            raise ValueError("duplicate IDs in tier metrics")

    candidates = read_jsonl(args.candidates)
    sft = read_jsonl(args.sft_train)
    clue = read_jsonl(args.clue_train)
    sft_ids = {row["case_id"] for row in sft}
    clue_ids = {row["case_id"] for row in clue}
    sft_videos = {row["video_id"] for row in sft}
    clue_videos = {row["video_id"] for row in clue}
    if len(sft_ids) != len(sft) or len(clue_ids) != len(clue):
        raise ValueError("training case IDs are duplicated")
    if sft_ids != clue_ids or sft_videos != clue_videos:
        raise ValueError("SFT and CLUE did not use identical training cases/videos")
    candidate_ids = {row["sample_id"] for row in candidates}
    if not sft_ids <= candidate_ids:
        raise ValueError("some training cases are absent from the candidate source")

    pool = [
        row for row in candidates
        if row["video_id"] not in sft_videos
        and (args.max_duration is None or float(row["duration"]) <= args.max_duration)
        and (allowed_tiers is None or tier_by_id.get(row["sample_id"]) in allowed_tiers)
    ]
    if any(row["sample_id"] in sft_ids for row in pool):
        raise ValueError("a training case leaked into the held-out pool")
    if len({row["sample_id"] for row in pool}) != len(pool):
        raise ValueError("candidate case IDs are duplicated")
    count = args.sample_size or len(pool)
    if not 1 <= count <= len(pool):
        raise ValueError(f"sample-size must be between 1 and {len(pool)}")
    indices = sorted(random.Random(args.sample_seed).sample(range(len(pool)), count))
    chosen = [pool[index] for index in indices]

    inputs: list[dict] = []
    labels: list[dict] = []
    for row in chosen:
        sample_id = row["sample_id"]
        choices = [str(choice).strip() for choice in row["choices"]]
        if len(choices) not in (3, 4) or any(not choice for choice in choices):
            raise ValueError(f"invalid choices in {sample_id}")
        answer = str(row["answer"]).strip().upper()
        if answer not in "ABCD"[: len(choices)]:
            raise ValueError(f"invalid answer in {sample_id}: {answer}")
        video = Path(row["video_path"])
        if not video.is_file() or video.stat().st_size == 0:
            raise FileNotFoundError(f"missing video in {sample_id}: {video}")
        options = "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", choices))
        prompt = (
            f"<video>\nQuestion: {row['question']}\nOptions:\n{options}\n"
            "Answer with exactly one option letter."
        )
        inputs.append(
            {
                "messages": [{"role": "user", "content": prompt}],
                "videos": [{
                    "video": str(video), "fps": 2.0, "max_frames": 768,
                    "min_pixels": 3136, "max_pixels": 28672,
                }],
                "case_id": sample_id,
                "prompt_id": sample_id,
                "video_id": row["video_id"],
                "benchmark": "WorldSense",
                "question_type": row.get("question_type", ""),
                "sampling_contract": {
                    "held_out_evaluation": True,
                    "train_case_overlap": False,
                    "train_video_overlap": False,
                    "student_view": "full-video-uniform",
                    "use_audio_in_video": True,
                    "fps": 2.0, "max_frames_per_view": 768,
                    "min_pixels": 3136, "max_pixels": 28672,
                },
            }
        )
        labels.append({
            "sample_id": sample_id, "video_id": row["video_id"],
            "answer": answer, "question_type": row.get("question_type", ""),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "worldsense.answer_free.jsonl"
    labels_path = args.output_dir / "worldsense.labels.jsonl"
    write_jsonl(data_path, inputs)
    write_jsonl(labels_path, labels)
    manifest = {
        "benchmark": "WorldSense",
        "candidate_source": str(args.candidates.resolve()),
        "candidate_source_sha256": sha256(args.candidates),
        "sft_train_source": str(args.sft_train.resolve()),
        "sft_train_sha256": sha256(args.sft_train),
        "clue_train_source": str(args.clue_train.resolve()),
        "clue_train_sha256": sha256(args.clue_train),
        "candidate_rows": len(candidates),
        "training_rows": len(sft),
        "training_videos": len(sft_videos),
        "video_disjoint_pool_rows": len(pool),
        "video_disjoint_pool_videos": len({row["video_id"] for row in pool}),
        "tier_metrics_source": str(args.tier_metrics.resolve()) if args.tier_metrics else None,
        "tier_metrics_sha256": sha256(args.tier_metrics) if args.tier_metrics else None,
        "allowed_tiers": sorted(allowed_tiers) if allowed_tiers else None,
        "tier_counts": dict(sorted(Counter(tier_by_id.get(row["sample_id"]) for row in chosen).items())) if allowed_tiers else None,
        "max_duration_seconds": args.max_duration,
        "maximum_selected_duration_seconds": max(float(row["duration"]) for row in chosen),
        "rows": len(inputs),
        "sample_seed": args.sample_seed,
        "selected_pool_indices": indices,
        "unique_videos": len({row["video_id"] for row in inputs}),
        "answer_free_path": str(data_path.resolve()),
        "answer_free_sha256": sha256(data_path),
        "labels_path": str(labels_path.resolve()),
        "labels_sha256": sha256(labels_path),
        "case_overlap": 0,
        "video_overlap": 0,
        "use_audio_in_video": True,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "selected_pool_indices"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
