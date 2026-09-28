"""Deterministic, answer-free subset and question-category preparation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from .dataset import Sample, load_samples
from .evidence import (
    evidence_duration_bucket,
    evidence_ratio_bucket,
    parse_sample_evidence,
    video_duration_bucket,
)
from .media import probe_video
from .question_categories import classify_sample


def _stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def _split_value(sample: Sample) -> str:
    return str(sample.metadata.get("split", "unknown"))


def _feature(sample: Sample, duration: float, category: dict[str, Any]) -> dict[str, Any]:
    evidence = parse_sample_evidence(sample.metadata, video_duration=duration)
    evidence_duration = evidence.duration
    ratio = evidence_duration / duration if duration > 0 else 0.0
    return {
        "sample_id": sample.sample_id,
        "video_path": sample.video_path,
        "video_duration": float(duration),
        "golden_duration": float(evidence_duration),
        "evidence_duration_bucket": evidence_duration_bucket(evidence_duration),
        "evidence_ratio": float(ratio),
        "evidence_ratio_bucket": evidence_ratio_bucket(ratio),
        "full_video_duration_bucket": video_duration_bucket(duration),
        "num_evidence_spans": evidence.num_spans,
        "span_bucket": "1" if evidence.num_spans == 1 else ("2" if evidence.num_spans == 2 else ">=3"),
        "question_category": category["category"],
        "question_category_source": category["source"],
        "question_category_confidence": category["confidence"],
    }


def _allocate_quotas(groups: dict[tuple[Any, ...], list[dict[str, Any]]], size: int, seed: int) -> dict[tuple[Any, ...], int]:
    """Allocate a proportional quota with deterministic stratum coverage."""

    if size <= 0 or not groups:
        return {}
    keys = sorted(groups, key=lambda key: _stable_rank(seed, repr(key)))
    if len(keys) > size:
        selected = keys[:size]
        return {key: 1 for key in selected}

    total = sum(len(groups[key]) for key in keys)
    quotas = {key: max(1, int(len(groups[key]) * size // total)) for key in keys}
    for key in keys:
        quotas[key] = min(quotas[key], len(groups[key]))
    while sum(quotas.values()) < size:
        candidates = [key for key in keys if quotas[key] < len(groups[key])]
        if not candidates:
            break
        candidates.sort(
            key=lambda key: (
                -(len(groups[key]) * size / total - quotas[key]),
                _stable_rank(seed, repr(key)),
            )
        )
        quotas[candidates[0]] += 1
    while sum(quotas.values()) > size:
        candidates = [key for key in keys if quotas[key] > 1]
        if not candidates:
            break
        candidates.sort(
            key=lambda key: (
                len(groups[key]) * size / total - quotas[key],
                _stable_rank(seed, repr(key)),
            )
        )
        quotas[candidates[0]] -= 1
    return quotas


def write_question_category_cache(
    samples: list[Sample],
    output: str | Path,
) -> Path:
    """Write only question/choice-derived labels; never serialize answers."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            category = classify_sample(sample.question, sample.choices)
            row = {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "choices": sample.choices,
                **category,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_question_category_cache(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    result: dict[str, dict[str, Any]] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["sample_id"])] = row
    return result


def prepare_subset(
    dataset: str | Path,
    output: str | Path,
    *,
    video_root: str | Path | None = None,
    category_cache: str | Path | None = None,
    split: str | None = "validation",
    size: int = 500,
    seed: int = 0,
) -> Path:
    if size <= 0:
        raise ValueError("subset size must be positive")
    all_samples = load_samples(dataset, video_root=video_root)
    if split is not None:
        all_samples = [sample for sample in all_samples if _split_value(sample) == split]
    if not all_samples:
        raise ValueError(f"no samples available for split={split!r}")

    category_path = Path(category_cache) if category_cache else Path(output).with_name("question_categories.jsonl")
    write_question_category_cache(all_samples, category_path)
    category_cache_rows = load_question_category_cache(category_path)

    metadata_by_video: dict[str, Any] = {}
    features: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for sample in all_samples:
        if sample.video_path not in metadata_by_video:
            try:
                metadata_by_video[sample.video_path] = probe_video(sample.video_path)
            except Exception as exc:
                failures.append({"sample_id": sample.sample_id, "reason": f"{type(exc).__name__}: {exc}"})
                continue
        metadata = metadata_by_video[sample.video_path]
        category = category_cache_rows[sample.sample_id]
        features.append(_feature(sample, metadata.duration, category))
    if failures:
        raise RuntimeError(f"failed to probe {len(failures)} videos; first={failures[0]}")
    if not features:
        raise RuntimeError("no probeable samples remain")

    target_size = min(size, len(features))
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in features:
        key = (
            item["question_category"],
            item["evidence_duration_bucket"],
            item["full_video_duration_bucket"],
            item["span_bucket"],
            item["evidence_ratio_bucket"],
        )
        groups[key].append(item)
    for items in groups.values():
        items.sort(key=lambda item: _stable_rank(seed, item["sample_id"]))
    quotas = _allocate_quotas(groups, target_size, seed)
    selected: list[dict[str, Any]] = []
    for key in sorted(quotas, key=lambda value: _stable_rank(seed, repr(value))):
        selected.extend(groups[key][: quotas[key]])
    selected.sort(key=lambda item: _stable_rank(seed, item["sample_id"]))
    selected = selected[:target_size]

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "dataset": str(Path(dataset).resolve()),
        "video_root": str(Path(video_root).resolve()) if video_root else None,
        "split": split,
        "seed": seed,
        "requested_size": size,
        "selected_size": len(selected),
        "available_size": len(features),
        "category_cache": str(category_path.resolve()),
        "selection_key": [
            "question_category",
            "evidence_duration_bucket",
            "full_video_duration_bucket",
            "span_bucket",
            "evidence_ratio_bucket",
        ],
        "selected": selected,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def load_subset(path: str | Path) -> dict[str, Any]:
    subset_path = Path(path)
    payload = json.loads(subset_path.read_text(encoding="utf-8"))
    selected = payload.get("selected")
    if not isinstance(selected, list):
        raise ValueError(f"subset missing selected list: {subset_path}")
    payload["selected_sample_ids"] = [str(item["sample_id"]) for item in selected]
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a fixed answer-free golden-temporal subset")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--category-cache", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    path = prepare_subset(
        args.dataset,
        args.output,
        video_root=args.video_root,
        category_cache=args.category_cache,
        split=args.split,
        size=args.size,
        seed=args.seed,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
