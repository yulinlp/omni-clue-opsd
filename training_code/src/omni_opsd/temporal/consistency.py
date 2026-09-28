"""Answer-label-free consistency screening for atomic privileged views.

This module is deliberately an offline diagnostic.  It asks whether final
answers from independently constructed privileged views contain a useful
selection signal before any M-OPSD adapter is trained.  Ground-truth answers
are read only after selection to score the resulting prediction.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

from .bootstrap import paired_bootstrap
from .question_categories import classify_sample


def _normalise_prediction(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip().upper()
    return value if len(value) == 1 and "A" <= value <= "Z" else None


def read_view(path: str | Path, *, mode: str | None = None) -> dict[str, dict[str, Any]]:
    """Read one prediction view, optionally selecting a mode from a mixed file."""

    rows: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if mode is not None and row.get("mode") != mode:
            continue
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"missing sample_id in {path}")
        if sample_id in rows:
            raise ValueError(f"duplicate sample_id {sample_id!r} in {path} for mode={mode!r}")
        rows[sample_id] = row
    if not rows:
        raise ValueError(f"no rows selected from {path} for mode={mode!r}")
    return rows


def strict_majority_select(
    predictions: dict[str, str | None], *, anchor: str
) -> tuple[str | None, list[str], str]:
    """Return a strict-majority answer or conservatively fall back to anchor.

    Each named atomic view contributes at most one vote. Missing/invalid
    predictions abstain. A tied plurality is not considered a majority.
    """

    valid = {name: value for name, value in predictions.items() if value is not None}
    if valid:
        counts = Counter(valid.values())
        answer, count = counts.most_common(1)[0]
        unique_winner = sum(value == count for value in counts.values()) == 1
        if unique_winner and count > len(valid) / 2:
            selected = sorted(name for name, value in valid.items() if value == answer)
            return answer, selected, "strict_majority"
    anchor_answer = predictions.get(anchor)
    if anchor_answer is not None:
        return anchor_answer, [anchor], "anchor_fallback"
    return None, [], "failure_no_anchor_or_majority"


def _accuracy(rows: Iterable[dict[str, Any]]) -> float | None:
    values = [bool(row["correct"]) for row in rows if row.get("correct") is not None]
    return sum(values) / len(values) if values else None


def evaluate_consistency(
    views: dict[str, dict[str, dict[str, Any]]],
    *,
    anchor: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Evaluate strict-majority consistency without using labels for selection."""

    if anchor not in views:
        raise ValueError(f"anchor {anchor!r} is not one of {sorted(views)}")
    common_ids = sorted(set.intersection(*(set(rows) for rows in views.values())))
    if not common_ids:
        raise ValueError("the views have no common sample IDs")

    output_rows: list[dict[str, Any]] = []
    view_correct: dict[str, int] = {name: 0 for name in views}
    view_valid: dict[str, int] = {name: 0 for name in views}
    for sample_id in common_ids:
        anchor_row = views[anchor][sample_id]
        gt_answer = _normalise_prediction(anchor_row.get("gt_answer"))
        predictions = {
            name: _normalise_prediction(rows[sample_id].get("prediction"))
            for name, rows in views.items()
        }
        for name, prediction in predictions.items():
            if prediction is not None and gt_answer is not None:
                view_valid[name] += 1
                view_correct[name] += int(prediction == gt_answer)

        prediction, selected, decision = strict_majority_select(predictions, anchor=anchor)
        category = anchor_row.get("question_category")
        category_source = anchor_row.get("question_category_source")
        if not category:
            inferred = classify_sample(anchor_row.get("question", ""), anchor_row.get("choices", ()))
            category = inferred["category"]
            category_source = inferred["source"]
        correct = prediction == gt_answer if prediction is not None and gt_answer is not None else None
        anchor_prediction = predictions.get(anchor)
        anchor_correct = (
            anchor_prediction == gt_answer
            if anchor_prediction is not None and gt_answer is not None
            else None
        )
        output_rows.append({
            "sample_id": sample_id,
            "question": anchor_row.get("question"),
            "choices": anchor_row.get("choices"),
            "gt_answer": gt_answer,
            "prediction": prediction,
            "correct": correct,
            "question_category": category,
            "question_category_source": category_source,
            "atomic_predictions": predictions,
            "selected_views": selected,
            "decision": decision,
            "anchor_view": anchor,
            "anchor_prediction": anchor_prediction,
            "anchor_correct": anchor_correct,
            "changed_from_anchor": prediction is not None and prediction != anchor_prediction,
            "selection_uses_gt_answer": False,
        })

    changed = [row for row in output_rows if row["changed_from_anchor"]]
    anchor_rows = views[anchor]
    paired_reference = []
    for sample_id in common_ids:
        row = dict(anchor_rows[sample_id])
        pred = _normalise_prediction(row.get("prediction"))
        gt = _normalise_prediction(row.get("gt_answer"))
        row["correct"] = pred == gt if pred is not None and gt is not None else None
        paired_reference.append(row)
    bootstrap = paired_bootstrap(paired_reference, output_rows, seed=0, resamples=10_000)

    metrics = {
        "experiment": "E19_atomic_privileged_consistency",
        "selection_policy": "strict_majority_else_anchor",
        "selection_uses_gt_answer": False,
        "anchor": anchor,
        "views": list(views),
        "samples": len(output_rows),
        "accuracy": _accuracy(output_rows),
        "strict_majority_samples": sum(row["decision"] == "strict_majority" for row in output_rows),
        "anchor_fallback_samples": sum(row["decision"] == "anchor_fallback" for row in output_rows),
        "failures": sum(row["correct"] is None for row in output_rows),
        "changed_from_anchor": len(changed),
        "changed_and_improved": sum(
            row.get("anchor_correct") is False and row.get("correct") is True for row in changed
        ),
        "changed_and_harmed": sum(
            row.get("anchor_correct") is True and row.get("correct") is False for row in changed
        ),
        "changed_but_both_wrong": sum(
            row.get("anchor_correct") is False and row.get("correct") is False for row in changed
        ),
        "view_accuracy": {
            name: view_correct[name] / view_valid[name] if view_valid[name] else None
            for name in views
        },
        "paired_bootstrap_vs_anchor": bootstrap,
    }

    by_category: list[dict[str, Any]] = []
    categories = sorted({str(row["question_category"]) for row in output_rows})
    for category in categories:
        rows = [row for row in output_rows if row["question_category"] == category]
        by_category.append({
            "question_category": category,
            "question_category_source": rows[0]["question_category_source"],
            "samples": len(rows),
            "accuracy": _accuracy(rows),
            "strict_majority_rate": sum(row["decision"] == "strict_majority" for row in rows) / len(rows),
            "changed_from_anchor": sum(row["changed_from_anchor"] for row in rows),
            "changed_and_improved": sum(
                row["changed_from_anchor"]
                and row.get("anchor_correct") is False
                and row.get("correct") is True
                for row in rows
            ),
            "changed_and_harmed": sum(
                row["changed_from_anchor"]
                and row.get("anchor_correct") is True
                and row.get("correct") is False
                for row in rows
            ),
            "changed_but_both_wrong": sum(
                row["changed_from_anchor"]
                and row.get("anchor_correct") is False
                and row.get("correct") is False
                for row in rows
            ),
        })
    return output_rows, metrics, by_category


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", required=True, type=_parse_named_path, help="NAME=JSONL")
    parser.add_argument("--anchor-mode", default=None)
    parser.add_argument("--view", action="append", default=[], type=_parse_named_path, help="NAME=JSONL")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    anchor_name, anchor_path = args.anchor
    specs = [(anchor_name, anchor_path), *args.view]
    names = [name for name, _ in specs]
    if len(set(names)) != len(names):
        parser.error(f"view names must be unique: {names}")
    views = {
        name: read_view(path, mode=args.anchor_mode if name == anchor_name else None)
        for name, path in specs
    }
    rows, metrics, by_category = evaluate_consistency(views, anchor=anchor_name)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output / "metrics_by_category.json").write_text(
        json.dumps(by_category, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    protocol = {
        "experiment": "E19_atomic_privileged_consistency",
        "anchor": {"name": anchor_name, "path": str(anchor_path), "mode": args.anchor_mode},
        "views": [{"name": name, "path": str(path)} for name, path in args.view],
        "selection_policy": "strict_majority_else_anchor",
        "selection_uses_gt_answer": False,
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
