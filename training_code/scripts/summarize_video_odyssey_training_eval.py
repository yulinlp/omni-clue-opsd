#!/usr/bin/env python3
"""Score an ms-swift inference JSONL against isolated VideoOdyssey labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from omni_opsd.evaluation import summarize_mcq_results


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--adapter", default="")
    args = parser.parse_args()

    summary = summarize_mcq_results(
        _read_jsonl(args.results), _read_jsonl(args.labels), _read_jsonl(args.dataset)
    )
    summary.update(
        {
            "arm": args.arm,
            "adapter": args.adapter or None,
            "results": str(args.results.resolve()),
            "results_sha256": _sha256(args.results),
            "labels": str(args.labels.resolve()),
            "labels_sha256": _sha256(args.labels),
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": _sha256(args.dataset),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
