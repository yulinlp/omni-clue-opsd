"""Evaluation helpers for answer-only VideoOdyssey post-training runs."""

from __future__ import annotations

from collections import Counter
import json
from typing import Any, Iterable, Mapping

from .rewards import extract_mcq_answer


def _input_signature(row: dict[str, Any]) -> str:
    messages = [
        {"role": message.get("role"), "content": message.get("content")}
        for message in row.get("messages", [])
        if message.get("role") != "assistant"
    ]
    return json.dumps(
        {
            "messages": messages,
            "videos": row.get("videos", []),
            "audios": row.get("audios", []),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def scored_mcq_rows(
    results: Iterable[dict[str, Any]],
    labels: Iterable[dict[str, Any]],
    sources: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Strictly join held-out predictions and labels into paired score rows.

    Missing, duplicate, extra, or unparsable rows are surfaced explicitly. An
    unparsable response is counted as wrong instead of disappearing from the
    denominator.
    """

    label_map: dict[str, str] = {}
    for row in labels:
        sample_id = str(row["sample_id"])
        answer = str(row["answer"]).strip().upper()
        if sample_id in label_map:
            raise ValueError(f"duplicate label id: {sample_id}")
        if len(answer) != 1 or not answer.isalpha():
            raise ValueError(f"invalid label for {sample_id}: {answer!r}")
        label_map[sample_id] = answer

    source_map: dict[str, str] = {}
    source_ids: set[str] = set()
    if sources is not None:
        for row in sources:
            sample_id = str(row.get("case_id") or row.get("prompt_id") or "")
            if not sample_id:
                raise ValueError("source row lacks case_id and prompt_id")
            signature = _input_signature(row)
            if signature in source_map:
                raise ValueError(f"duplicate source input signature: {sample_id}")
            source_map[signature] = sample_id
            source_ids.add(sample_id)
        if source_ids != set(label_map):
            raise ValueError("source/label id mismatch")

    result_map: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for row in results:
        sample_id = str(row.get("case_id") or row.get("prompt_id") or "")
        if not sample_id and source_map:
            sample_id = source_map.get(_input_signature(row), "")
        if not sample_id:
            raise ValueError("result row lacks an ID and cannot be mapped to a source input")
        if sample_id in result_map:
            duplicate_ids.append(sample_id)
        result_map[sample_id] = row
    if duplicate_ids:
        raise ValueError(f"duplicate result ids: {sorted(set(duplicate_ids))[:5]}")

    expected_ids = set(label_map)
    actual_ids = set(result_map)
    missing_ids = sorted(expected_ids - actual_ids)
    extra_ids = sorted(actual_ids - expected_ids)
    if missing_ids or extra_ids:
        raise ValueError(
            f"result/label id mismatch: missing={missing_ids[:5]} extra={extra_ids[:5]}"
        )

    scored = []
    for sample_id in sorted(expected_ids):
        prediction = extract_mcq_answer(result_map[sample_id].get("response", ""))
        scored.append(
            {
                "sample_id": sample_id,
                "prediction": prediction,
                "correct": prediction == label_map[sample_id] if prediction is not None else False,
            }
        )
    return scored


def summarize_mcq_results(
    results: Iterable[dict[str, Any]],
    labels: Iterable[dict[str, Any]],
    sources: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute strict MCQ metrics after an exact held-out ID join."""

    scored = scored_mcq_rows(results, labels, sources)
    prediction_counts: Counter[str] = Counter(
        row["prediction"] if row["prediction"] is not None else "UNPARSED" for row in scored
    )
    parse_failures = [row["sample_id"] for row in scored if row["prediction"] is None]
    correct = sum(bool(row["correct"]) for row in scored)

    total = len(scored)
    parsed = total - len(parse_failures)
    return {
        "total": total,
        "parsed": parsed,
        "parse_failures": len(parse_failures),
        "parse_failure_ids": parse_failures,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "parse_rate": parsed / total if total else 0.0,
        "prediction_counts": dict(sorted(prediction_counts.items())),
        "strict_unparsed_counted_wrong": True,
        "identical_complete_sample_ids": True,
    }


def compare_mcq_runs(
    runs: Mapping[str, Iterable[dict[str, Any]]],
    labels: Iterable[dict[str, Any]],
    sources: Iterable[dict[str, Any]] | None = None,
    *,
    reference: str = "base",
    seed: int = 20260904,
    resamples: int = 10_000,
) -> dict[str, Any]:
    """Summarize identical-ID runs and compare each arm to a reference."""

    if reference not in runs:
        raise ValueError(f"reference run is missing: {reference}")
    if resamples <= 0:
        raise ValueError("resamples must be positive")

    from .temporal.bootstrap import paired_bootstrap

    label_rows = list(labels)
    source_rows = list(sources) if sources is not None else None
    scored: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for arm, result_rows in runs.items():
        rows = list(result_rows)
        scored[arm] = scored_mcq_rows(rows, label_rows, source_rows)
        summaries[arm] = summarize_mcq_results(rows, label_rows, source_rows)

    comparisons = {}
    for arm, rows in scored.items():
        if arm == reference:
            continue
        comparisons[arm] = paired_bootstrap(
            scored[reference], rows, seed=seed, resamples=resamples
        )
    return {
        "reference": reference,
        "seed": seed,
        "resamples": resamples,
        "arms": summaries,
        "paired_vs_reference": comparisons,
    }


def summarize_gap_recovery(
    runs: Mapping[str, Iterable[dict[str, Any]]],
    labels: Iterable[dict[str, Any]],
    sources: Iterable[dict[str, Any]] | None = None,
    *,
    initial_student: str = "initial_student",
    current_student: str = "current_student",
    teacher: str = "teacher",
    min_gap: float = 1e-8,
) -> dict[str, Any]:
    """Report the paper-style student/teacher gap recovery rate.

    The three runs are scored by :func:`scored_mcq_rows` against the same
    labels and optional source inputs, so IDs, prompt/media views, and the
    answer parsing protocol are shared.  If the teacher does not beat the
    initial student by ``min_gap``, the rate is unavailable rather than being
    fabricated or clamped.
    """

    required = (initial_student, current_student, teacher)
    missing = [name for name in required if name not in runs]
    if missing:
        raise ValueError(f"gap recovery runs are missing: {missing}")
    if min_gap < 0:
        raise ValueError("min_gap must be non-negative")

    label_rows = list(labels)
    source_rows = list(sources) if sources is not None else None
    scored = {
        name: scored_mcq_rows(runs[name], label_rows, source_rows)
        for name in required
    }
    accuracies = {
        name: (sum(bool(row["correct"]) for row in rows) / len(rows) if rows else 0.0)
        for name, rows in scored.items()
    }
    denominator = accuracies[teacher] - accuracies[initial_student]
    available = denominator > min_gap
    return {
        "initial_student": initial_student,
        "current_student": current_student,
        "teacher": teacher,
        "n_samples": len(scored[initial_student]),
        "initial_student_accuracy": accuracies[initial_student],
        "current_student_accuracy": accuracies[current_student],
        "teacher_accuracy": accuracies[teacher],
        "teacher_initial_gap": denominator,
        "gap_recovery_rate": (
            (accuracies[current_student] - accuracies[initial_student]) / denominator
            if available else None
        ),
        "available": available,
        "unavailable_reason": None if available else "teacher_initial_gap_non_positive_or_too_small",
        "same_scoring_protocol": True,
        "identical_complete_sample_ids": True,
    }
