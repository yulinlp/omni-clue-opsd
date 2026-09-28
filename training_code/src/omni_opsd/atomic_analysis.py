"""Paired analyses for the six OmniVideo atomic-view inference arms."""

from __future__ import annotations

from collections import Counter
import random
from typing import Any, Iterable, Mapping

from .evaluation import scored_mcq_rows
from .temporal.bootstrap import paired_bootstrap


ATOMIC_ARMS = (
    "A0_full_av",
    "A1_evidence_v_full_a",
    "A2_full_v_evidence_a",
    "A3_evidence_av",
    "A4_uniform_matched_av",
    "A5_full_av_timestamp",
)

PAIRWISE_CONTRASTS = {
    "joint_evidence_vs_full": ("A0_full_av", "A3_evidence_av"),
    "joint_evidence_vs_matched_budget": ("A4_uniform_matched_av", "A3_evidence_av"),
    "evidence_visual_effect": ("A0_full_av", "A1_evidence_v_full_a"),
    "evidence_audio_effect": ("A0_full_av", "A2_full_v_evidence_a"),
    "timestamp_text_effect": ("A0_full_av", "A5_full_av_timestamp"),
}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prediction_counts = Counter(
        row["prediction"] if row["prediction"] is not None else "UNPARSED" for row in rows
    )
    total = len(rows)
    parsed = sum(row["prediction"] is not None for row in rows)
    correct = sum(bool(row["correct"]) for row in rows)
    ratios = [float(row["evidence_to_full_duration_ratio"]) for row in rows]
    return {
        "total": total,
        "parsed": parsed,
        "parse_failures": total - parsed,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "parse_rate": parsed / total if total else None,
        "prediction_counts": dict(sorted(prediction_counts.items())),
        "evidence_to_full_duration_ratio": {
            "min": min(ratios) if ratios else None,
            "mean": sum(ratios) / len(ratios) if ratios else None,
            "max": max(ratios) if ratios else None,
        },
    }


def _interaction_bootstrap(
    scored: Mapping[str, list[dict[str, Any]]],
    *,
    sample_ids: list[str],
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """Bootstrap A3-A2-A1+A0 on identical examples."""

    if resamples <= 0:
        raise ValueError("resamples must be positive")
    maps = {
        arm: {str(row["sample_id"]): float(bool(row["correct"])) for row in rows}
        for arm, rows in scored.items()
    }
    required = (
        "A0_full_av",
        "A1_evidence_v_full_a",
        "A2_full_v_evidence_a",
        "A3_evidence_av",
    )
    values = [
        maps[required[3]][sample_id]
        - maps[required[2]][sample_id]
        - maps[required[1]][sample_id]
        + maps[required[0]][sample_id]
        for sample_id in sample_ids
    ]
    result: dict[str, Any] = {
        "definition": "accuracy(A3)-accuracy(A2)-accuracy(A1)+accuracy(A0)",
        "n_pairs": len(values),
        "effect": sum(values) / len(values) if values else None,
        "ci_low": None,
        "ci_high": None,
        "seed": seed,
        "resamples": resamples,
    }
    if not values:
        return result
    try:
        import numpy as np

        array = np.asarray(values, dtype=np.float64)
        rng = np.random.default_rng(seed)
        draws = []
        remaining = resamples
        while remaining:
            batch = min(512, remaining)
            indices = rng.integers(0, len(array), size=(batch, len(array)))
            draws.append(array[indices].mean(axis=1))
            remaining -= batch
        samples = np.concatenate(draws)
        result["ci_low"] = float(np.percentile(samples, 2.5))
        result["ci_high"] = float(np.percentile(samples, 97.5))
    except ImportError:
        rng = random.Random(seed)
        samples = sorted(
            sum(values[rng.randrange(len(values))] for _ in values) / len(values)
            for _ in range(resamples)
        )
        result["ci_low"] = float(samples[int(0.025 * (len(samples) - 1))])
        result["ci_high"] = float(samples[int(0.975 * (len(samples) - 1))])
    return result


def summarize_atomic_runs(
    runs: Mapping[str, Iterable[dict[str, Any]]],
    labels: Iterable[dict[str, Any]],
    sources: Mapping[str, Iterable[dict[str, Any]]],
    *,
    seed: int = 20260904,
    resamples: int = 10_000,
) -> dict[str, Any]:
    """Score all arms overall and by task with paired uncertainty estimates."""

    if set(runs) != set(ATOMIC_ARMS) or set(sources) != set(ATOMIC_ARMS):
        raise ValueError("runs and sources must contain exactly the six frozen atomic arms")
    label_rows = list(labels)
    task_by_id = {str(row["sample_id"]): str(row.get("question_type") or "unknown") for row in label_rows}
    if len(task_by_id) != len(label_rows):
        raise ValueError("duplicate label ID")
    tasks = sorted(set(task_by_id.values()))

    scored: dict[str, list[dict[str, Any]]] = {}
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm in ATOMIC_ARMS:
        source_rows = list(sources[arm])
        source_by_id = {
            str(row.get("case_id") or row.get("prompt_id") or ""): row for row in source_rows
        }
        rows = scored_mcq_rows(list(runs[arm]), label_rows, source_rows)
        for row in rows:
            row["question_type"] = task_by_id[row["sample_id"]]
            row["evidence_to_full_duration_ratio"] = float(
                source_by_id[row["sample_id"]]["view_contract"][
                    "evidence_to_full_duration_ratio"
                ]
            )
        scored[arm] = rows
        arm_summaries[arm] = {
            "overall": _summary(rows),
            "by_task": {
                task: _summary([row for row in rows if row["question_type"] == task])
                for task in tasks
            },
        }

    contrasts: dict[str, Any] = {}
    for name, (reference, candidate) in PAIRWISE_CONTRASTS.items():
        overall = paired_bootstrap(
            scored[reference], scored[candidate], seed=seed, resamples=resamples
        )
        by_task = {
            task: paired_bootstrap(
                scored[reference],
                scored[candidate],
                seed=seed,
                resamples=resamples,
                group_by="question_type",
                group=task,
            )
            for task in tasks
        }
        contrasts[name] = {
            "reference": reference,
            "candidate": candidate,
            "overall": overall,
            "by_task": by_task,
        }

    ids_by_task = {
        task: sorted(sample_id for sample_id, value in task_by_id.items() if value == task)
        for task in tasks
    }
    all_ids = sorted(task_by_id)
    interaction = {
        "overall": _interaction_bootstrap(
            scored, sample_ids=all_ids, seed=seed, resamples=resamples
        ),
        "by_task": {
            task: _interaction_bootstrap(
                scored, sample_ids=ids_by_task[task], seed=seed, resamples=resamples
            )
            for task in tasks
        },
    }
    return {
        "protocol": "OmniVideo-100K_atomic_A0-A5",
        "seed": seed,
        "resamples": resamples,
        "tasks": tasks,
        "arms": arm_summaries,
        "contrasts": contrasts,
        "visual_audio_interaction": interaction,
        "strict_unparsed_counted_wrong": True,
        "identical_complete_sample_ids": True,
    }
