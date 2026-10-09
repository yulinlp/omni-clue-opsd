#!/usr/bin/env python3
"""Summarize paired WorldSense G/H/T/S/V/A scores for all and selected rows."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean

ARMS = (
    "E2_G_exact", "E4_H_halo3", "E7_T_8fps", "E8_S_highres",
    "E12_gold_v", "E13_A_audio_exact",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(canonical: Path, selected_ids: Path, results_root: Path) -> tuple[dict, list[dict]]:
    source = read_jsonl(canonical)
    by_id = {str(row["sample_id"]): row for row in source}
    selected = {line.strip() for line in selected_ids.open(encoding="utf-8") if line.strip()}
    if len(by_id) != len(source) or not selected or not selected <= set(by_id):
        raise ValueError("invalid canonical or selected IDs")
    paths = sorted(results_root.glob("shard*/results.jsonl"))
    if not paths:
        raise ValueError("no scored shards")
    scored = [row for path in paths for row in read_jsonl(path)]
    scores = {str(row["sample_id"]): row for row in scored}
    if len(scores) != len(scored) or set(scores) != set(by_id):
        raise ValueError(f"score coverage mismatch: {len(scores)} of {len(by_id)}")
    failures = [row for path in paths
                for row in (read_jsonl(path.with_name("unscored.jsonl"))
                            if path.with_name("unscored.jsonl").is_file() else [])]
    if failures:
        raise ValueError(f"{len(failures)} unscored rows remain")
    identities = {json.dumps(row["model_identity"], sort_keys=True) for row in scored}
    contracts = {json.dumps(row["score_sampling_contract"], sort_keys=True) for row in scored}
    if len(identities) != 1 or len(contracts) != 1:
        raise ValueError("model or sampling contract differs across shards")
    for sample_id, row in scores.items():
        expected_letters = "ABCD"[:len(by_id[sample_id]["choices"])]
        if row.get("choice_letters") != expected_letters or set(row.get("scores") or {}) != set(ARMS):
            raise ValueError(f"invalid choice or view contract: {sample_id}")
        for arm in ARMS:
            item = row["scores"][arm]
            probs = item.get("probabilities") or {}
            if set(probs) != set(expected_letters) or abs(sum(probs.values()) - 1) > 1e-5:
                raise ValueError(f"invalid option probabilities: {sample_id}/{arm}")
            if item.get("predicted") != max(probs, key=probs.get):
                raise ValueError(f"invalid predicted option: {sample_id}/{arm}")

    cohorts = {f"all_{len(by_id)}": list(by_id),
               f"selected_{len(selected)}": [sid for sid in by_id if sid in selected]}
    table: list[dict] = []
    for cohort, cohort_ids in cohorts.items():
        tasks = defaultdict(list)
        for sid in cohort_ids:
            tasks[str(by_id[sid]["question_type"])].append(sid)
        for task, ids in {"overall": cohort_ids, **dict(sorted(tasks.items()))}.items():
            for arm in ARMS:
                correct = []
                answer_probs = []
                tokens = []
                delta_correct = []
                delta_probability = []
                for sid in ids:
                    answer = str(by_id[sid]["answer"])
                    item = scores[sid]["scores"][arm]
                    anchor = scores[sid]["scores"]["E2_G_exact"]
                    is_correct = float(item["predicted"] == answer)
                    p_answer = float(item["probabilities"][answer])
                    correct.append(is_correct)
                    answer_probs.append(p_answer)
                    tokens.append(float(item["input_tokens"]))
                    delta_correct.append(is_correct - float(anchor["predicted"] == answer))
                    delta_probability.append(p_answer - float(anchor["probabilities"][answer]))
                table.append({
                    "cohort": cohort, "task": task, "view": arm,
                    "rows": len(ids), "videos": len({by_id[sid]["video_id"] for sid in ids}),
                    "accuracy": fmean(correct),
                    "mean_answer_probability": fmean(answer_probs),
                    "mean_input_tokens": fmean(tokens),
                    "delta_accuracy_vs_G": fmean(delta_correct),
                    "delta_answer_probability_vs_G": fmean(delta_probability),
                })
    result = {
        "dataset": "WorldSense", "status": "complete",
        "canonical": str(canonical.resolve()), "canonical_sha256": sha256(canonical),
        "selected_ids_sha256": sha256(selected_ids),
        "rows": len(source), "selected_rows": len(selected),
        "model_identity": json.loads(next(iter(identities))),
        "score_sampling_contract": json.loads(next(iter(contracts))),
        "shards": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in paths],
        "scope": "WorldSense evidence-bearing screened candidates; the selected cohort is a training cohort, not a held-out test",
    }
    return result, table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--selected-ids", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary, rows = summarize(args.canonical, args.selected_ids, args.results_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "accuracy_by_view_and_task.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "complete", "rows": summary["rows"],
                      "selected_rows": summary["selected_rows"], "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
