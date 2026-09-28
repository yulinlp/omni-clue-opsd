"""Paired bootstrap comparisons for identical sample IDs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from typing import Any, Iterable

from .evidence import evidence_duration_bucket, evidence_ratio_bucket, video_duration_bucket


def read_prediction_rows(path: str | Path) -> list[dict[str, Any]]:
    directory = Path(path)
    if directory.is_file():
        files = [directory]
    else:
        merged = directory / "predictions.jsonl"
        files = [merged] if merged.is_file() else sorted(directory.glob("predictions.shard*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no predictions JSONL under {directory}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for file in files:
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id"))
            if sample_id in seen:
                continue
            seen.add(sample_id)
            rows.append(row)
    return rows


def _evidence_duration(row: dict[str, Any]) -> float:
    return sum(
        float(end) - float(start)
        for start, end in row.get("evidence", [])
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start
    )


def row_group(row: dict[str, Any], group_by: str) -> str:
    if group_by in {"question_category", "question_type", "pipeline_family"}:
        return str(row.get(group_by) or "unknown")
    if group_by == "evidence_duration":
        return evidence_duration_bucket(_evidence_duration(row))
    if group_by == "full_video_duration":
        return video_duration_bucket(float(row.get("video_duration") or 0.0))
    if group_by == "evidence_ratio":
        duration = float(row.get("video_duration") or 0.0)
        ratio = _evidence_duration(row) / duration if duration > 0 else 0.0
        return evidence_ratio_bucket(ratio)
    if group_by == "num_evidence_spans":
        count = len(row.get("evidence", []))
        return "1" if count == 1 else ("2" if count == 2 else ">=3")
    if group_by == "all":
        return "all"
    return str(row.get(group_by) or "unknown")


def _paired_values(
    reference_rows: Iterable[dict[str, Any]],
    candidate_rows: Iterable[dict[str, Any]],
    *,
    group_by: str = "all",
    group: str | None = None,
) -> tuple[list[float], list[float], int]:
    ref = {str(row.get("sample_id")): row for row in reference_rows}
    cand = {str(row.get("sample_id")): row for row in candidate_rows}
    common = sorted(set(ref) & set(cand))
    ref_values: list[float] = []
    cand_values: list[float] = []
    excluded = 0
    for sample_id in common:
        ref_row, cand_row = ref[sample_id], cand[sample_id]
        if group is not None and row_group(ref_row, group_by) != group:
            continue
        if ref_row.get("correct") is None or cand_row.get("correct") is None:
            excluded += 1
            continue
        ref_values.append(float(bool(ref_row.get("correct"))))
        cand_values.append(float(bool(cand_row.get("correct"))))
    return ref_values, cand_values, excluded


def paired_bootstrap(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    seed: int = 0,
    resamples: int = 10_000,
    group_by: str = "all",
    group: str | None = None,
) -> dict[str, Any]:
    """Return candidate-reference accuracy delta and a paired 95% CI."""

    if resamples <= 0:
        raise ValueError("resamples must be positive")
    ref_values, cand_values, excluded = _paired_values(
        reference_rows, candidate_rows, group_by=group_by, group=group
    )
    n = len(ref_values)
    result: dict[str, Any] = {
        "group_by": group_by,
        "group": group or "all",
        "seed": seed,
        "resamples": resamples,
        "n_pairs": n,
        "excluded_unpaired_or_failed": excluded,
        "reference_accuracy": sum(ref_values) / n if n else None,
        "candidate_accuracy": sum(cand_values) / n if n else None,
        "delta_accuracy": (sum(cand_values) - sum(ref_values)) / n if n else None,
        "ci_low": None,
        "ci_high": None,
    }
    if not n:
        return result

    # NumPy is already a dependency of the media/processor path.  The small
    # pure-Python fallback keeps the comparison usable on a login node.
    try:
        import numpy as np

        ref_array = np.asarray(ref_values, dtype=np.float64)
        cand_array = np.asarray(cand_values, dtype=np.float64)
        rng = np.random.default_rng(seed)
        deltas: list[Any] = []
        remaining = resamples
        while remaining:
            batch = min(512, remaining)
            indices = rng.integers(0, n, size=(batch, n))
            deltas.append((cand_array[indices].mean(axis=1) - ref_array[indices].mean(axis=1)))
            remaining -= batch
        bootstrap_values = np.concatenate(deltas)
        result["ci_low"] = float(np.percentile(bootstrap_values, 2.5))
        result["ci_high"] = float(np.percentile(bootstrap_values, 97.5))
    except ImportError:
        rng = random.Random(seed)
        values = []
        pairs = list(zip(ref_values, cand_values))
        for _ in range(resamples):
            draw = [pairs[rng.randrange(n)] for _ in range(n)]
            values.append(sum(candidate - reference for reference, candidate in draw) / n)
        values.sort()
        result["ci_low"] = float(values[int(0.025 * (len(values) - 1))])
        result["ci_high"] = float(values[int(0.975 * (len(values) - 1))])
    return result


def paired_bootstrap_table(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    seed: int = 0,
    resamples: int = 10_000,
    groupings: Iterable[str] = (
        "all", "question_category", "evidence_duration", "full_video_duration",
        "evidence_ratio", "num_evidence_spans",
    ),
) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for group_by in groupings:
        groups = {row_group(row, group_by) for row in reference_rows}
        for group in sorted(groups):
            table.append(paired_bootstrap(
                reference_rows,
                candidate_rows,
                seed=seed,
                resamples=resamples,
                group_by=group_by,
                group=group,
            ))
    return table


def resolve_run(root: str | Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_dir() or candidate.is_file():
        return candidate
    resolved = Path(root) / value
    if resolved.exists():
        return resolved
    raise FileNotFoundError(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paired bootstrap for two golden-temporal runs")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args(argv)
    result = paired_bootstrap(
        read_prediction_rows(args.reference),
        read_prediction_rows(args.candidate),
        seed=args.seed,
        resamples=args.resamples,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
