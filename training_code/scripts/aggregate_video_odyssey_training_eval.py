#!/usr/bin/env python3
"""Aggregate held-out base/SFT/GRPO/OPSD/CLUE-OPSD results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from omni_opsd.evaluation import compare_mcq_runs


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_runs(values: list[str]) -> dict[str, Path]:
    runs = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--run must be ARM=RESULTS_JSONL, got {value!r}")
        arm, raw_path = value.split("=", 1)
        if not arm or arm in runs:
            raise SystemExit(f"invalid or duplicate arm: {arm!r}")
        path = Path(raw_path)
        if not path.is_file():
            raise SystemExit(f"result file is missing: {path}")
        runs[arm] = path
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="ARM=RESULTS_JSONL")
    parser.add_argument("--reference", default="base")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()

    run_paths = parse_runs(args.run)
    labels = read_jsonl(args.labels)
    report = compare_mcq_runs(
        {arm: read_jsonl(path) for arm, path in run_paths.items()},
        labels,
        read_jsonl(args.dataset),
        reference=args.reference,
        seed=args.seed,
        resamples=args.resamples,
    )
    report["labels"] = str(args.labels.resolve())
    report["labels_sha256"] = sha256(args.labels)
    report["dataset"] = str(args.dataset.resolve())
    report["dataset_sha256"] = sha256(args.dataset)
    report["result_files"] = {
        arm: {"path": str(path.resolve()), "sha256": sha256(path)}
        for arm, path in run_paths.items()
    }

    rows = []
    for arm, summary in report["arms"].items():
        paired = report["paired_vs_reference"].get(arm, {})
        rows.append(
            {
                "arm": arm,
                "total": summary["total"],
                "correct": summary["correct"],
                "accuracy": summary["accuracy"],
                "parse_rate": summary["parse_rate"],
                "delta_vs_reference": paired.get("delta_accuracy", 0.0 if arm == args.reference else None),
                "ci_low": paired.get("ci_low"),
                "ci_high": paired.get("ci_high"),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_eval_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "training_eval_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def pct(value):
        return "--" if value is None else f"{100 * value:.2f}%"

    lines = [
        "# Qwen2.5-Omni VideoOdyssey held-out comparison",
        "",
        f"Reference: `{args.reference}`; samples per arm: {len(labels)}; paired bootstrap resamples: {args.resamples}.",
        "",
        f"| Arm | Correct | Accuracy | Parse rate | Delta vs {args.reference} | 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ci = "--" if row["ci_low"] is None else f"[{pct(row['ci_low'])}, {pct(row['ci_high'])}]"
        lines.append(
            f"| {row['arm']} | {row['correct']}/{row['total']} | {pct(row['accuracy'])} | "
            f"{pct(row['parse_rate'])} | {pct(row['delta_vs_reference'])} | {ci} |"
        )
    lines.extend(
        [
            "",
            "Unparsed responses count as wrong. All arms must contain exactly the same held-out IDs.",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "arms": list(run_paths)}))


if __name__ == "__main__":
    main()
