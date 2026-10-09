#!/usr/bin/env python3
"""Extract the actual 1,453 WorldSense SFT case IDs as a reporting cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.canonical.open(encoding="utf-8") as handle:
        candidates = {json.loads(line)["sample_id"] for line in handle if line.strip()}
    with args.sft.open(encoding="utf-8") as handle:
        ids = [json.loads(line)["case_id"] for line in handle if line.strip()]
    if len(candidates) != 3079 or len(ids) != 1453 or len(set(ids)) != 1453 or not set(ids) <= candidates:
        raise ValueError("SFT cohort must contain 1,453 distinct IDs from the 3,079 candidate questions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(json.dumps({"sft_rows": len(ids), "candidate_rows": len(candidates), "output": str(args.output)}))


if __name__ == "__main__":
    main()
