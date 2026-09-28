#!/usr/bin/env python3
"""Build SFT/GRPO/OPSD/Clue-OPSD datasets with identical student inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from omni_opsd.data.swift_opsd import read_jsonl, swift_training_matrix_rows, write_jsonl


ARMS = ("sft", "grpo", "opsd", "clue_opsd")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--materialized-clue-opsd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    canonical = {str(row["sample_id"]): row for row in read_jsonl(args.canonical)}
    materialized = {
        str(row["case_id"]): row for row in read_jsonl(args.materialized_clue_opsd)
    }
    if canonical.keys() != materialized.keys():
        missing_media = sorted(canonical.keys() - materialized.keys())
        missing_labels = sorted(materialized.keys() - canonical.keys())
        raise SystemExit(
            f"dataset IDs differ: missing_media={missing_media[:5]} "
            f"missing_labels={missing_labels[:5]}"
        )

    rows_by_arm = {arm: [] for arm in ARMS}
    for case_id, canonical_row in canonical.items():
        matrix = swift_training_matrix_rows(canonical_row, materialized[case_id])
        for arm in ARMS:
            rows_by_arm[arm].append(matrix[arm])

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for arm in ARMS:
        path = output_dir / f"video_odyssey_train.{arm}.jsonl"
        write_jsonl(path, rows_by_arm[arm])
        paths[arm] = path

    missing_media = 0
    for row in rows_by_arm["clue_opsd"]:
        for value in row["videos"] + row["teacher_videos"]:
            path = value if isinstance(value, str) else value.get("video")
            if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
                missing_media += 1
    summary = {
        "rows_per_arm": {arm: len(rows) for arm, rows in rows_by_arm.items()},
        "unique_case_ids": len(canonical),
        "unique_video_ids": len({str(row["video_id"]) for row in canonical.values()}),
        "missing_media_references": missing_media,
        "canonical_sha256": _sha256(args.canonical),
        "materialized_clue_opsd_sha256": _sha256(args.materialized_clue_opsd),
        "outputs": {
            arm: {"path": str(path), "sha256": _sha256(path)}
            for arm, path in paths.items()
        },
        "comparison_contract": {
            "same_student_prompt_and_video": True,
            "sft": "gold answer teacher-forced target",
            "grpo": "gold answer exact-match sequence reward",
            "opsd": "full-video EMA teacher with privileged gold answer",
            "clue_opsd": "clue-video EMA teacher without gold answer",
        },
    }
    summary_path = output_dir / "training_matrix_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
