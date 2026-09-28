"""Merge independent sample-shard outputs without caching model answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evidence import evidence_duration_bucket, evidence_ratio_bucket, video_duration_bucket


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        row.get("resource", {}).get(key)
        for row in rows
        if isinstance(row.get("resource", {}).get(key), (int, float))
    ]
    return sum(values) / len(values) if values else None


def _evidence_duration(row: dict[str, Any]) -> float:
    return sum(
        float(end) - float(start)
        for start, end in row.get("evidence", [])
        if isinstance(start, (int, float)) and isinstance(end, (int, float))
    )


def _category_value(row: dict[str, Any], key: str) -> str:
    if key == "question_category":
        return str(row.get("question_category") or "other")
    if key == "question_type":
        return str(row.get("question_type") or "unknown")
    if key == "evidence_duration":
        return evidence_duration_bucket(_evidence_duration(row))
    if key == "full_video_duration":
        return video_duration_bucket(float(row.get("video_duration") or 0.0))
    if key == "evidence_ratio":
        duration = float(row.get("video_duration") or 0.0)
        ratio = _evidence_duration(row) / duration if duration > 0 else 0.0
        return evidence_ratio_bucket(ratio)
    if key == "num_evidence_spans":
        count = len(row.get("evidence", []))
        return "1" if count == 1 else ("2" if count == 2 else ">=3")
    return "unknown"


def _metrics(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute merged metrics from merged prediction rows.

    Recomputing is preferable to averaging shard metrics: failed rows can have
    no resource record, and category denominators must remain exact after
    sample sharding.
    """
    evaluated = [row for row in rows if row.get("correct") is not None]
    correct = sum(bool(row.get("correct")) for row in evaluated)
    overall: dict[str, Any] = {
        "samples": len(rows),
        "evaluated": len(evaluated),
        "correct": correct,
        "accuracy": correct / len(evaluated) if evaluated else None,
        "failures": sum(bool(row.get("failure_reason")) for row in rows),
    }
    for key in (
        "num_video_frames", "input_tokens", "output_tokens",
        "visual_tokens_exact_before_compression", "visual_tokens_after_compression",
        "visual_budget_proxy_pixel_volume", "latency_sec", "peak_gpu_memory_mb",
        "visual_tokens_per_frame", "visual_tokens_per_second",
        "video_reader_frame_cache_hits", "video_reader_frame_cache_misses",
        "video_reader_reader_cache_hits", "video_reader_reader_cache_misses",
        "video_reader_processed_cache_hits", "video_reader_processed_cache_misses",
    ):
        value = _mean(rows, key)
        if value is not None:
            overall[f"mean_{key}"] = value

    by_category: dict[str, dict[str, Any]] = {}
    for key in (
        "question_category", "question_type", "evidence_duration", "full_video_duration",
        "evidence_ratio", "num_evidence_spans",
    ):
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(_category_value(row, key), []).append(row)
        by_category[key] = {}
        for group, group_rows in sorted(groups.items()):
            valid = [row for row in group_rows if row.get("correct") is not None]
            group_correct = sum(bool(row.get("correct")) for row in valid)
            item: dict[str, Any] = {
                "samples": len(group_rows),
                "evaluated": len(valid),
                "correct": group_correct,
                "accuracy": group_correct / len(valid) if valid else None,
            }
            visual = _mean(group_rows, "visual_tokens_exact_before_compression")
            latency = _mean(group_rows, "latency_sec")
            if visual is not None:
                item["mean_visual_tokens"] = visual
            if latency is not None:
                item["mean_latency_sec"] = latency
            by_category[key][group] = item
    return overall, by_category


def merge(directory: str | Path) -> Path:
    directory = Path(directory).resolve()
    files = sorted(directory.glob("predictions.shard*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no prediction shards in {directory}")
    rows = []
    seen = set()
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id"))
            if sample_id in seen:
                raise ValueError(f"duplicate sample id while merging: {sample_id}")
            seen.add(sample_id)
            rows.append(row)
    rows.sort(key=lambda row: str(row.get("sample_id")))
    output = directory / "predictions.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # A sharded run is a single experiment from the user's point of view.
    # Materialize the same top-level artifacts as a non-sharded run so that
    # downstream summarization never has to guess which shard to read.
    metrics, by_category = _metrics(rows)
    _write_json(directory / "metrics.json", metrics)
    _write_json(directory / "metrics_by_category.json", by_category)
    resource_rows = [row.get("resource", {}) for row in rows]
    resource_stats: dict[str, Any] = {
        "samples": len(rows),
        "pipeline_family": rows[0].get("pipeline_family", "native_qwen") if rows else "native_qwen",
        "rows": resource_rows,
        "frame_cache_hits": sum(int(row.get("video_reader_frame_cache_hits", 0)) for row in resource_rows),
        "frame_cache_misses": sum(int(row.get("video_reader_frame_cache_misses", 0)) for row in resource_rows),
        "reader_cache_hits": sum(int(row.get("video_reader_reader_cache_hits", 0)) for row in resource_rows),
        "reader_cache_misses": sum(int(row.get("video_reader_reader_cache_misses", 0)) for row in resource_rows),
        "processed_cache_hits": sum(int(row.get("video_reader_processed_cache_hits", 0)) for row in resource_rows),
        "processed_cache_misses": sum(int(row.get("video_reader_processed_cache_misses", 0)) for row in resource_rows),
        "cache_bytes_reused": sum(int(row.get("video_reader_cache_bytes_reused", 0)) for row in resource_rows),
    }
    for key, output_key in (
        ("input_tokens", "mean_input_tokens"),
        ("visual_tokens_exact_before_compression", "mean_visual_tokens"),
        ("latency_sec", "mean_latency_sec"),
    ):
        value = _mean(rows, key)
        if value is not None:
            resource_stats[output_key] = value
    _write_json(directory / "resource_stats.json", resource_stats)

    metadata_files = sorted(directory.glob("run_metadata.shard*.json"))
    if metadata_files:
        metadata = _read_json(metadata_files[0])
        metadata["merged_from_shards"] = [path.name for path in metadata_files]
        _write_json(directory / "run_metadata.json", metadata)
    config_files = sorted(directory.glob("config.shard*.yaml"))
    if config_files:
        (directory / "config.yaml").write_text(config_files[0].read_text(encoding="utf-8"), encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args(argv)
    print(f"merged predictions: {merge(args.directory)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
