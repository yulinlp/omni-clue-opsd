#!/usr/bin/env python3
"""Prepare the WorldSense evidence MCQs for the Qwen2.5-Omni E-series scorer.

The selected-1500 file contains metrics but omits the question, choices,
evidence intervals, and video path.  This script retains the full candidate
rows, maps their old machine paths to the local WorldSense media, and records
the selected IDs as a reporting cohort.  It never changes the answer or Gold
intervals and never imports OmniVideo data.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(candidates: Path, selected: Path, video_root: Path, output_dir: Path) -> dict:
    rows = read_jsonl(candidates)
    chosen = read_jsonl(selected)
    ids = [str(row["sample_id"]) for row in rows]
    chosen_ids = [str(row["sample_id"]) for row in chosen]
    if not rows or len(ids) != len(set(ids)) or not chosen or len(chosen_ids) != len(set(chosen_ids)):
        raise ValueError("empty manifest or duplicate sample IDs")
    by_id = {str(row["sample_id"]): row for row in rows}
    if not set(chosen_ids) <= set(by_id):
        raise ValueError("selected IDs are absent from candidates")

    canonical = []
    for row in rows:
        sample_id = str(row["sample_id"])
        video_id = str(row["video_id"])
        path = (video_root / f"{video_id}.mp4").resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{sample_id}: {path}")
        duration = float(row["duration"])
        spans = row.get("evidence_spans") or []
        choice_count = len(row.get("choices") or [])
        answer = str(row.get("answer") or "")
        if (not row.get("question") or choice_count not in (3, 4)
                or answer not in "ABCD"[:choice_count] or duration <= 0 or not spans):
            raise ValueError(f"incomplete MCQ or evidence: {sample_id}")
        if any(float(a) < 0 or float(b) > duration or float(b) <= float(a)
               for a, b in spans) or any(spans[i][1] > spans[i + 1][0]
                                          for i in range(len(spans) - 1)):
            raise ValueError(f"invalid evidence spans: {sample_id}")
        canonical.append({**row, "video_path": str(path)})

    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = output_dir / "worldsense_candidates_3079.atomic.jsonl"
    selected_path = output_dir / "selected_1500.ids.txt"
    with canonical_path.open("w", encoding="utf-8") as handle:
        for row in canonical:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    selected_path.write_text("\n".join(chosen_ids) + "\n", encoding="utf-8")
    report = {
        "dataset": "WorldSense",
        "source_candidates": str(candidates.resolve()),
        "source_candidates_sha256": sha256(candidates),
        "source_selected": str(selected.resolve()),
        "source_selected_sha256": sha256(selected),
        "video_root": str(video_root.resolve()),
        "canonical": str(canonical_path.resolve()),
        "canonical_sha256": sha256(canonical_path),
        "selected_ids": str(selected_path.resolve()),
        "selected_ids_sha256": sha256(selected_path),
        "rows": len(rows),
        "videos": len({row["video_id"] for row in rows}),
        "selected_rows": len(chosen),
        "selected_videos": len({row["video_id"] for row in chosen}),
        "tasks": dict(sorted(Counter(row["question_type"] for row in rows).items())),
        "evidence_source": "screening/candidates.jsonl evidence_spans",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.candidates, args.selected, args.video_root, args.output_dir)
    print(json.dumps({k: result[k] for k in ("rows", "videos", "selected_rows", "selected_videos", "canonical")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
