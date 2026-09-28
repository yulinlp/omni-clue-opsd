#!/usr/bin/env python3
"""Prepare the evidence-bearing OmniVideo-100K MCQ research split.

This script intentionally stops at canonical, answer-isolated manifests.  It
does not choose or materialize the full-video sampling policy; that policy must
pass the separate context/memory/coverage gate before training rows are built.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from omni_opsd.data.omnivideo_100k import iter_omnivideo_100k
from omni_opsd.data.swift_opsd import video_split, write_jsonl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _without_answer(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["answer"] = None
    return result


def _duration_bucket(duration: float) -> str:
    if duration < 90:
        return "60-89s"
    if duration < 120:
        return "90-119s"
    if duration < 150:
        return "120-149s"
    return "150-180s"


def _counts(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def _eligibility_error(row: dict[str, Any]) -> str | None:
    """Keep a comparable four-choice protocol and reject broken release rows."""

    choices = list(row.get("choices") or [])
    if len(choices) != 4:
        return f"choice_count_{len(choices)}"
    answer = str(row.get("answer") or "")
    if answer not in {"A", "B", "C", "D"}:
        return "answer_outside_four_choices"
    return None


def _audit_rows(rows: list[dict[str, Any]], *, require_videos: bool) -> dict[str, Any]:
    sample_ids: set[str] = set()
    evidence_durations: list[float] = []
    missing_videos: list[str] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        choices = list(row.get("choices") or [])
        answer = str(row.get("answer") or "")
        if len(choices) != 4:
            raise ValueError(f"sample {sample_id} has {len(choices)} choices, expected 4")
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"sample {sample_id} has invalid answer {answer!r}")
        duration = float(row["duration"])
        spans = list(row.get("evidence_spans") or [])
        if not spans:
            raise ValueError(f"sample {sample_id} has no evidence interval")
        for start, end in spans:
            if not 0 <= float(start) < float(end) <= duration:
                raise ValueError(
                    f"sample {sample_id} evidence [{start}, {end}] is outside [0, {duration}]"
                )
            evidence_durations.append(float(end) - float(start))
        if require_videos and not Path(row["video_path"]).is_file():
            missing_videos.append(str(row["video_path"]))
    if missing_videos:
        preview = "\n".join(missing_videos[:10])
        raise FileNotFoundError(f"{len(missing_videos)} videos are missing; first paths:\n{preview}")
    return {
        "records": len(rows),
        "videos": len({str(row["video_id"]) for row in rows}),
        "task_counts": _counts(rows, "question_type"),
        "duration_bucket_counts": dict(
            sorted(Counter(_duration_bucket(float(row["duration"])) for row in rows).items())
        ),
        "evidence_intervals": len(evidence_durations),
        "mean_evidence_interval_seconds": (
            sum(evidence_durations) / len(evidence_durations) if evidence_durations else 0.0
        ),
    }


def _balanced_atomic_subset(
    rows: list[dict[str, Any]], *, per_task: int, seed: int
) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["question_type"])].append(row)
    selected: list[dict[str, Any]] = []
    for task, task_rows in sorted(by_task.items()):
        ranked = sorted(
            task_rows,
            key=lambda row: hashlib.sha256(
                f"{seed}\0atomic\0{task}\0{row['sample_id']}".encode()
            ).hexdigest(),
        )
        selected.extend(ranked[:per_task])
    return sorted(selected, key=lambda row: (str(row["question_type"]), str(row["sample_id"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.10)
    parser.add_argument("--dev-videos", type=int)
    parser.add_argument("--atomic-per-task", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--require-videos", action="store_true")
    args = parser.parse_args()

    if args.atomic_per_task <= 0:
        raise ValueError("atomic-per-task must be positive")
    evidence_rows = [
        sample.to_record()
        for sample in iter_omnivideo_100k(
            args.annotation,
            video_root=args.video_root,
            evidence_only=True,
        )
    ]
    if not evidence_rows:
        raise SystemExit("no evidence-bearing MCQ rows found")
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for row in evidence_rows:
        reason = _eligibility_error(row)
        if reason is None:
            rows.append(row)
        else:
            exclusions.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "video_id": str(row["video_id"]),
                    "question_type": str(row["question_type"]),
                    "reason": reason,
                }
            )
    source_audit = _audit_rows(rows, require_videos=args.require_videos)
    video_count = source_audit["videos"]
    if not 0 < args.dev_fraction < 1:
        raise ValueError("dev-fraction must be between 0 and 1")
    dev_videos = args.dev_videos or max(1, round(video_count * args.dev_fraction))
    train, dev_labeled = video_split(rows, val_video_count=dev_videos, seed=args.seed)
    train_ids = {str(row["video_id"]) for row in train}
    dev_ids = {str(row["video_id"]) for row in dev_labeled}
    overlap = train_ids & dev_ids
    if overlap:
        raise RuntimeError(f"video leakage detected: {sorted(overlap)[:10]}")

    atomic_labeled = _balanced_atomic_subset(
        dev_labeled, per_task=args.atomic_per_task, seed=args.seed
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "all": output_dir / "omnivideo_100k_evidence_all.canonical.jsonl",
        "train": output_dir / "omnivideo_100k_evidence_train.canonical.jsonl",
        "dev": output_dir / "omnivideo_100k_evidence_dev.answer_free.jsonl",
        "dev_labels": output_dir / "omnivideo_100k_evidence_dev.labels.jsonl",
        "atomic": output_dir / "omnivideo_100k_atomic_dev.answer_free.jsonl",
        "atomic_labels": output_dir / "omnivideo_100k_atomic_dev.labels.jsonl",
        "exclusions": output_dir / "omnivideo_100k_evidence_exclusions.jsonl",
    }
    # OmniVideo-Test is the independent held-out benchmark.  Keep the
    # historical video-disjoint internal split for diagnostics, but also emit
    # the complete valid evidence-bearing corpus for formal post-training.
    write_jsonl(paths["all"], rows)
    write_jsonl(paths["train"], train)
    write_jsonl(paths["dev"], (_without_answer(row) for row in dev_labeled))
    write_jsonl(
        paths["dev_labels"],
        (
            {
                "sample_id": row["sample_id"],
                "video_id": row["video_id"],
                "answer": row["answer"],
                "question_type": row["question_type"],
            }
            for row in dev_labeled
        ),
    )
    write_jsonl(paths["atomic"], (_without_answer(row) for row in atomic_labeled))
    write_jsonl(
        paths["atomic_labels"],
        (
            {
                "sample_id": row["sample_id"],
                "video_id": row["video_id"],
                "answer": row["answer"],
                "question_type": row["question_type"],
            }
            for row in atomic_labeled
        ),
    )
    write_jsonl(paths["exclusions"], exclusions)

    summary = {
        "protocol_status": "pretraining_atomic_stage",
        "source": {
            "annotation": str(args.annotation.resolve()),
            "annotation_sha256": _sha256(args.annotation),
            "video_root": str(args.video_root.resolve()),
            "evidence_origin": "automated_dataset_generation_pipeline",
        },
        "seed": args.seed,
        "split_unit": "video_id",
        "video_overlap": sorted(overlap),
        "answer_labels_isolated_for_dev": True,
        "raw_evidence_records": len(evidence_rows),
        "excluded_records": len(exclusions),
        "exclusion_reason_counts": dict(
            sorted(Counter(item["reason"] for item in exclusions).items())
        ),
        "source_audit": source_audit,
        "all_audit": _audit_rows(rows, require_videos=args.require_videos),
        "train_audit": _audit_rows(train, require_videos=args.require_videos),
        "dev_audit": _audit_rows(dev_labeled, require_videos=args.require_videos),
        "atomic_audit": _audit_rows(atomic_labeled, require_videos=args.require_videos),
        "atomic_selection": {
            "kind": "deterministic_equal_cap_per_task",
            "per_task_cap": args.atomic_per_task,
        },
        "full_sampling_policy": "not_selected_by_this_script",
        "paths": {name: str(path) for name, path in paths.items()},
    }
    summary_path = output_dir / "omnivideo_100k_split_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
