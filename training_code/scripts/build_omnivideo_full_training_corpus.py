#!/usr/bin/env python3
"""Build the formal all-train OmniVideo-100K corpus and score supplement.

OmniVideo-Test is the independent held-out benchmark, so the formal training
candidate pool contains every valid evidence-bearing OmniVideo-100K MCQ.  The
historical video-disjoint internal split remains useful for diagnostics, but
its rows are not withheld from final post-training.

This utility also reapplies previously audited full-A/V media replacements and
writes the rows missing from an already-scored historical training manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_ids(rows: list[dict[str, Any]], label: str) -> set[str]:
    ids = [str(row.get("sample_id") or "") for row in rows]
    if "" in ids or len(ids) != len(set(ids)):
        raise ValueError(f"{label} has missing or duplicate sample IDs")
    return set(ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-canonical", type=Path, required=True)
    parser.add_argument("--historical-train", type=Path, required=True)
    parser.add_argument("--repair-audit", type=Path, required=True)
    parser.add_argument("--output-canonical", type=Path, required=True)
    parser.add_argument("--output-supplement", type=Path, required=True)
    args = parser.parse_args()

    all_rows = _read_jsonl(args.all_canonical)
    historical_rows = _read_jsonl(args.historical_train)
    all_ids = _unique_ids(all_rows, "all canonical")
    historical_ids = _unique_ids(historical_rows, "historical train")
    if not historical_ids < all_ids:
        raise ValueError("historical train must be a strict subset of all canonical rows")

    audit = json.loads(args.repair_audit.read_text(encoding="utf-8"))
    replacements: dict[str, str] = {}
    for repair in audit.get("repaired_rows") or []:
        video_id = str(repair["video_id"])
        path = str(Path(repair["new_path"]).resolve())
        previous = replacements.setdefault(video_id, path)
        if previous != path:
            raise ValueError(f"conflicting media replacement for {video_id}")
    for video_id, path in replacements.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"replacement media is missing for {video_id}: {path}")

    repaired: list[dict[str, Any]] = []
    repaired_rows = 0
    for source in all_rows:
        row = dict(source)
        replacement = replacements.get(str(row["video_id"]))
        if replacement and str(Path(row["video_path"]).resolve()) != replacement:
            row["video_path"] = replacement
            repaired_rows += 1
        if not Path(row["video_path"]).is_file():
            raise FileNotFoundError(row["video_path"])
        repaired.append(row)

    supplement = [row for row in repaired if str(row["sample_id"]) not in historical_ids]
    if len(repaired) != 20_831 or len(supplement) != 2_084:
        raise ValueError(
            f"unexpected formal corpus sizes: all={len(repaired)} supplement={len(supplement)}"
        )
    if {str(row["sample_id"]) for row in supplement} | historical_ids != all_ids:
        raise ValueError("historical and supplemental rows do not cover the formal corpus")

    _write_jsonl(args.output_canonical, repaired)
    _write_jsonl(args.output_supplement, supplement)
    summary = {
        "status": "formal_all_train_corpus",
        "independent_test": "OmniVideo-Test",
        "rows": len(repaired),
        "videos": len({str(row["video_id"]) for row in repaired}),
        "supplement_rows": len(supplement),
        "supplement_videos": len({str(row["video_id"]) for row in supplement}),
        "repaired_rows": repaired_rows,
        "repaired_video_ids": sorted(replacements),
        "task_counts": dict(sorted(Counter(str(row["question_type"]) for row in repaired).items())),
        "supplement_task_counts": dict(
            sorted(Counter(str(row["question_type"]) for row in supplement).items())
        ),
        "artifacts": {
            "canonical": str(args.output_canonical.resolve()),
            "canonical_sha256": _sha256(args.output_canonical),
            "supplement": str(args.output_supplement.resolve()),
            "supplement_sha256": _sha256(args.output_supplement),
        },
    }
    summary_path = args.output_canonical.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
