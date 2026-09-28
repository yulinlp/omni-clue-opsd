#!/usr/bin/env python3
"""Fail loudly if shard .ids files overlap or miss questions.

Launching overlapping shards silently annotates some questions 2-3 times (the
200-question pilot hit exactly this bug).  Run this before any launch:

    python scripts/check_shards_disjoint.py output/worldsense_evidence_formal/shard*.ids
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    owner: dict[str, list[str]] = {}
    for name in args.files:
        path = Path(name)
        for line in path.open(encoding="utf-8", errors="replace"):
            question_id = line.strip()
            if question_id:
                owner.setdefault(question_id, []).append(path.name)

    duplicated = {q: v for q, v in owner.items() if len(v) > 1}
    total_rows = sum(len(v) for v in owner.values())
    print(f"files={len(args.files)} unique_questions={len(owner)} total_rows={total_rows}")
    if duplicated:
        print(f"FAIL: {len(duplicated)} questions appear in more than one shard")
        for question_id, files in list(duplicated.items())[:5]:
            print(f"  {question_id}: {files}")
        return 1
    print("OK: shards are disjoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
