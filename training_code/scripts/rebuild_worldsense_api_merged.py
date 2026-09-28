#!/usr/bin/env python3
"""Rebuild the pilot merged evidence deterministically.

The 200-question pilot shards overlapped (454 rows for 200 questions), so every
question can have 2-3 runs with different results.  The previous merge kept an
arbitrary one (filesystem order).  This rebuild keeps the run from the lowest
shard index and writes a manifest describing the duplication, so downstream
results (stats, V1 comparison, review site) are reproducible.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def shard_index(path: Path) -> int:
    match = re.match(r"s(\d+)\.evidence\.jsonl$", path.name)
    return int(match.group(1)) if match else 999


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="output/worldsense_api_200")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    per_question: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for path in sorted((run_dir / "out").glob("s*.evidence.jsonl"), key=shard_index):
        for line in path.open(encoding="utf-8", errors="replace"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            per_question[row["question_id"]].append((shard_index(path), row))

    merged: list[dict] = []
    duplicated = 0
    differing = 0
    for question_id, runs in sorted(per_question.items()):
        runs.sort(key=lambda item: item[0])
        if len(runs) > 1:
            duplicated += 1
            intervals = {json.dumps(row.get("clue_intervals")) for _, row in runs}
            if len(intervals) > 1:
                differing += 1
        merged.append(runs[0][1])

    out = run_dir / "merged.evidence.jsonl"
    out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged),
        encoding="utf-8",
    )
    manifest = {
        "questions": len(merged),
        "duplicated_questions": duplicated,
        "duplicated_with_different_intervals": differing,
        "rule": "lowest shard index wins",
        "status": dict(Counter(row.get("status") for row in merged)),
    }
    (run_dir / "merged.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
