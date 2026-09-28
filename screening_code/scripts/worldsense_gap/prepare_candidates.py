#!/usr/bin/env python3
"""Build WorldSense gap-screening candidates in the OmniVideo canonical format.

Input:
  --annotation  output/worldsense_evidence_api_full/merged.evidence.jsonl
  --qa          WorldSense QA json (video -> taskN -> question/answer/candidates)
  --media-index output/worldsense_evidence_formal/media_index.json (duration/res)
  --video-root  directory with <video_id>.mp4

Output: canonical JSONL consumed by the existing audit/score/select scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROVENANCE = "api_annotation_qwen3.8_omni_flash"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.open(encoding="utf-8", errors="replace"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotation", type=Path, required=True)
    p.add_argument("--qa", type=Path, required=True)
    p.add_argument("--media-index", type=Path, required=True)
    p.add_argument("--video-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    qa = json.loads(a.qa.read_text(encoding="utf-8"))
    media = json.loads(a.media_index.read_text(encoding="utf-8"))
    by_name = {Path(k).stem: v for k, v in media.items()}

    rows, issues = [], []
    for row in read_jsonl(a.annotation):
        if row.get("status") != "submitted":
            continue
        question_id = row["question_id"]
        video_id, _, task = question_id.partition("::")
        meta = qa.get(video_id) or {}
        task_meta = meta.get(task) or {}
        candidates = task_meta.get("candidates") or []
        if len(candidates) < 3:
            issues.append((question_id, "fewer than three options"))
            continue
        choices = [c.split(". ", 1)[1] if ". " in c else c for c in candidates]
        if len(set(choices)) != len(choices):
            issues.append((question_id, "duplicate option text"))
            continue
        answer = str(task_meta.get("answer") or "").strip()
        if answer not in "ABCD"[: len(choices)]:
            issues.append((question_id, f"answer {answer!r} outside options"))
            continue
        spans = [list(map(float, span)) for span in (row.get("clue_intervals") or [])]
        if not spans:
            issues.append((question_id, "no evidence spans"))
            continue
        info = by_name.get(video_id)
        if not info:
            issues.append((question_id, "missing media index entry"))
            continue
        duration = float(info["duration_s"])
        clean = []
        for start, end in sorted(spans):
            start, end = max(0.0, start), min(duration, end)
            if end - start < 1.5:
                continue
            if clean and start < clean[-1][1]:  # overlap: clip to the previous end
                start = clean[-1][1]
            if end - start >= 1.5:
                clean.append([round(start, 3), round(end, 3)])
        if not clean:
            issues.append((question_id, "no usable spans after clamping"))
            continue
        video_path = a.video_root / f"{video_id}.mp4"
        if not video_path.is_file():
            issues.append((question_id, "video file missing"))
            continue
        rows.append(
            {
                "sample_id": question_id,
                "video_id": video_id,
                "video_path": str(video_path),
                "duration": duration,
                "source_width": int(info.get("width") or 0),
                "source_height": int(info.get("height") or 0),
                "source_fps": float(info.get("fps") or 0.0),
                "question": task_meta.get("question"),
                "choices": choices,
                "answer": answer,
                "question_type": task_meta.get("task_type"),
                "task_domain": task_meta.get("task_domain"),
                "evidence_spans": clean,
                "evidence_provenance": PROVENANCE,
                "preparation_issues": [],
                "ordering_representation": None,
            }
        )
    if a.limit:
        rows = rows[: a.limit]
    rows.sort(key=lambda r: r["sample_id"])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    report = {
        "candidates": len(rows),
        "excluded": len(issues),
        "excluded_reasons": {},
        "videos": len({r["video_id"] for r in rows}),
        "span_counts": {},
        "provenance": PROVENANCE,
    }
    from collections import Counter

    report["excluded_reasons"] = dict(Counter(reason for _, reason in issues))
    report["span_counts"] = dict(Counter(len(r["evidence_spans"]) for r in rows))
    (a.output.parent / "prepare_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
