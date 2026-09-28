#!/usr/bin/env python3
"""Aggregate V1 sufficiency verification results into a summary."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _audio_class_map(evidence_dir: Path, qa_path: Path | None = None) -> dict[str, list[str]]:
    """Map question_id -> audio_class using the QA release (preferred)."""

    if qa_path is not None and qa_path.is_file():
        payload = json.loads(qa_path.read_text(encoding="utf-8"))
        task_keys = ("task0", "task1", "task2", "task3", "task4")
        mapping: dict[str, list[str]] = {}
        for video_id, entry in payload.items():
            classes = [str(item) for item in ((entry or {}).get("audio_class") or [])]
            for task_key in task_keys:
                if (entry or {}).get(task_key):
                    mapping[f"{video_id}::{task_key}"] = classes
        return mapping

    fallback: dict[str, list[str]] = {}
    for trace in sorted(evidence_dir.glob("shard*/evidence.trace.jsonl")):
        for row in _load_jsonl(trace):
            question_id = str(row.get("question_id") or "")
            if not question_id:
                continue
            blob = json.dumps(row.get("messages") or [], ensure_ascii=False)
            marker = "audio_class"
            if marker in blob:
                start = blob.index(marker)
                end = blob.find("]", start)
                values = [
                    item.strip().strip('"\\')
                    for item in blob[start:end].split(",")[1:]
                    if item.strip().strip('"\\')
                ]
                fallback[question_id] = values
    return fallback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qa", type=Path, default=Path("/share/home/ylhu/Light-Omni/data/omni_benchmarks/WorldSense/worldsense_qa.json"))
    args = parser.parse_args()

    rows: list[dict] = []
    for shard in sorted(args.verify_dir.glob("shard*/verify.jsonl")):
        rows.extend(_load_jsonl(shard))

    evidence_rows: list[dict] = []
    for shard in sorted(args.evidence_dir.glob("shard*/evidence.jsonl")):
        evidence_rows.extend(_load_jsonl(shard))
    status_counts = Counter(str(row.get("status")) for row in evidence_rows)
    localization_intervals = [
        len(row.get("clue_intervals") or []) for row in evidence_rows if row.get("status") == "submitted"
    ]

    audio_map = _audio_class_map(args.evidence_dir, args.qa)

    checked = [row for row in rows if int(row.get("n_intervals") or 0) > 0]
    sufficient = [row for row in checked if row.get("any_correct") is True]
    insufficient = [row for row in checked if row.get("any_correct") is False]

    def rate(part: int, total: int) -> float | None:
        return round(part / total, 4) if total else None

    by_task: dict[str, dict[str, int]] = defaultdict(lambda: {"checked": 0, "sufficient": 0})
    by_audio: dict[str, dict[str, int]] = defaultdict(lambda: {"checked": 0, "sufficient": 0})
    for row in checked:
        task = str(row.get("task_type") or "unknown")
        by_task[task]["checked"] += 1
        if row.get("any_correct"):
            by_task[task]["sufficient"] += 1
        classes = audio_map.get(str(row.get("question_id") or "")) or ["unknown"]
        for audio_class in classes:
            by_audio[audio_class]["checked"] += 1
            if row.get("any_correct"):
                by_audio[audio_class]["sufficient"] += 1

    best_p_true = [row["best_p_true"] for row in sufficient if row.get("best_p_true") is not None]
    best_margin = [row["best_margin"] for row in sufficient if row.get("best_margin") is not None]
    interval_correct = sum(int(row.get("n_correct") or 0) for row in checked)
    interval_total = sum(int(row.get("n_intervals") or 0) for row in checked)

    summary = {
        "questions_verified": len(rows),
        "questions_with_intervals": len(checked),
        "questions_sufficient": len(sufficient),
        "questions_insufficient": len(insufficient),
        "sufficiency_rate": rate(len(sufficient), len(checked)),
        "interval_accuracy": rate(interval_correct, interval_total),
        "intervals_checked": interval_total,
        "intervals_correct": interval_correct,
        "localization_status": dict(status_counts),
        "avg_localized_intervals": round(statistics.mean(localization_intervals), 2) if localization_intervals else None,
        "best_p_true_mean": round(statistics.mean(best_p_true), 4) if best_p_true else None,
        "best_p_true_median": round(statistics.median(best_p_true), 4) if best_p_true else None,
        "best_margin_mean": round(statistics.mean(best_margin), 4) if best_margin else None,
        "by_task_type": {
            key: {
                **value,
                "sufficiency_rate": rate(value["sufficient"], value["checked"]),
            }
            for key, value in sorted(by_task.items())
        },
        "by_audio_class": {
            key: {
                **value,
                "sufficiency_rate": rate(value["sufficient"], value["checked"]),
            }
            for key, value in sorted(by_audio.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
