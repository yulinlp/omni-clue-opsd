#!/usr/bin/env python3
"""Aggregate the training-matched rerun without turning unresolved grades into errors.

All rates use the complete frozen question set. An unresolved semantic grade is
an interval [0, 1], not a false value. Confidence intervals resample videos and
condition on the generated answers from this single decoding seed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training_code/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_worldsense_openqa import extract  # noqa: E402
from score_worldsense_mcq_v2 import SCORING as MCQ_SCORING, scored_mcq_v2_rows, summarize as summarize_mcq_v2  # noqa: E402

BOOTSTRAP_SEED = 20261003
BOOTSTRAP_RESAMPLES = 5000
CI_NOTE = (
    "Video-cluster bootstrap, conditional on these generated answers. One decoding seed is evaluated; "
    "the interval does not include stochastic decoding variance, judge error, or uncertainty in the reference labels."
)
PROTOCOL_NOTE = (
    "The old/new comparison changes decoding (greedy to temperature-1 sampling), visual budget, "
    "open-QA prompt wording, open-QA completion limit (768 to 512), and semantic grading (v1 to v2). "
    "A difference cannot be attributed to any single change. Old semantic scores have known grading errors. "
    "MCQ old/new primary scores are both reparsed with the same conservative v2 option parser; the original old strict MCQ score is preserved separately."
)
OPENQA_PRIMARY_SCORING = (
    "openqa-v2-uniform-automatic: conservative rules and blinded dual judge; "
    "reference-insufficient=null; model-specific assistant/human adjudications excluded"
)


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_metrics(values: list[bool | None]) -> dict:
    """Keep null distinct from false; validate rather than coerce score values."""
    if not values or any(value is not None and type(value) is not bool for value in values):
        raise ValueError("Scores must be a nonempty list of strict boolean/null values")
    total = len(values)
    correct = sum(value is True for value in values)
    incorrect = sum(value is False for value in values)
    undecided = sum(value is None for value in values)
    return {"total": total, "confirmed_correct": correct, "confirmed_incorrect": incorrect,
            "undecided": undecided, "resolved": total - undecided,
            "accuracy": correct / total if undecided == 0 else None,
            "accuracy_lower_bound": correct / total,
            "accuracy_upper_bound": (correct + undecided) / total,
            "all_scores_resolved": undecided == 0}


def score_bounds(values: list[bool | None]) -> tuple[np.ndarray, np.ndarray]:
    score_metrics(values)
    return (np.array([1.0 if value is True else 0.0 for value in values]),
            np.array([0.0 if value is False else 1.0 for value in values]))


class VideoBootstrap:
    def __init__(self, ids: list[str], labelmap: dict[str, dict],
                 seed: int = BOOTSTRAP_SEED, resamples: int = BOOTSTRAP_RESAMPLES):
        self.videos = sorted({labelmap[key]["video_id"] for key in ids})
        video_index = {video: i for i, video in enumerate(self.videos)}
        self.membership = np.array([video_index[labelmap[key]["video_id"]] for key in ids])
        counts = np.bincount(self.membership, minlength=len(self.videos))
        self.draws = np.random.default_rng(seed).integers(0, len(self.videos), size=(resamples, len(self.videos)))
        self.denominators = counts[self.draws].sum(axis=1)

    def distribution(self, question_values: np.ndarray) -> np.ndarray:
        sums = np.bincount(self.membership, weights=question_values, minlength=len(self.videos))
        return sums[self.draws].sum(axis=1) / self.denominators

    def interval(self, lower: np.ndarray, upper: np.ndarray, all_resolved: bool) -> dict:
        low_boot, high_boot = self.distribution(lower), self.distribution(upper)
        low_ci = [float(x) for x in np.quantile(low_boot, [0.025, 0.975])]
        high_ci = [float(x) for x in np.quantile(high_boot, [0.025, 0.975])]
        return {"ci95_video_bootstrap": low_ci if all_resolved else None,
                "lower_bound_ci95_video_bootstrap": low_ci,
                "upper_bound_ci95_video_bootstrap": high_ci,
                "ci95_video_bootstrap_conservative_envelope": [low_ci[0], high_ci[1]],
                "interval_note": CI_NOTE}


def paired_comparison(model: list[bool | None], base: list[bool | None],
                      bootstrap: VideoBootstrap, identical_model: bool = False) -> dict:
    if len(model) != len(base):
        raise ValueError("Paired score sets have different lengths")
    low_model, high_model = score_bounds(model)
    low_base, high_base = score_bounds(base)
    if identical_model:
        # The same unresolved variable cancels against itself.
        low = high = np.zeros(len(model), dtype=float)
        exact = True
    else:
        low, high = low_model - high_base, high_model - low_base
        exact = not any(value is None for value in model + base)
    resolved_pairs = [(m, b) for m, b in zip(model, base) if m is not None and b is not None]
    contribution = sum(int(m) - int(b) for m, b in resolved_pairs) / len(model)
    return {"total_paired_questions": len(model), "resolved_pairs": len(resolved_pairs),
            "unresolved_pairs": len(model) - len(resolved_pairs),
            "delta_accuracy": float(low.mean()) if exact else None,
            "delta_accuracy_lower_bound": float(low.mean()), "delta_accuracy_upper_bound": float(high.mean()),
            "known_wrong_to_correct": sum(m is True and b is False for m, b in resolved_pairs),
            "known_correct_to_wrong": sum(m is False and b is True for m, b in resolved_pairs),
            "known_resolved_pair_difference_contribution_full_denominator": contribution,
            "resolved_pair_note": "Known transitions are incomplete if any pair is unresolved; the contribution still uses all questions as denominator.",
            **bootstrap.interval(low, high, exact)}


def generation_diagnostics(rows: list[dict], tokenizer, cap: int) -> dict:
    diagnostics = []
    for row in rows:
        response = str(row.get("response", ""))
        answer, analysis, valid, format_ok = extract(response)
        if row.get("final_answer", answer) != answer or row.get("valid_answer", valid) != valid:
            raise ValueError("Stored final-answer extraction differs: " + row["sample_id"])
        words = re.findall(r"\w+", analysis.lower())
        windows = [tuple(words[i:i + 20]) for i in range(max(0, len(words) - 19))]
        repeat_fraction = 1 - len(set(windows)) / len(windows) if windows else 0.0
        decoded_tokens = len(tokenizer.encode(response, add_special_tokens=False).ids) if tokenizer else None
        diagnostics.append({"sample_id": row["sample_id"], "analysis_words": len(analysis.split()),
                            "valid_final_answer": valid, "format_ok": format_ok,
                            "analysis_closed": bool(re.search(r"</analysis>", response, re.I)),
                            "answer_tag_closed": bool(re.search(r"<answer>.*?</answer>", response, re.S | re.I)),
                            "repeated_20gram_fraction": repeat_fraction,
                            "repetitive_analysis": len(words) >= 60 and repeat_fraction >= 0.4,
                            "decoded_response_tokens": decoded_tokens,
                            "near_generation_cap": decoded_tokens >= cap - 2 if decoded_tokens is not None else None})
    n = len(rows)
    return {"total": n, "valid_final_answers": sum(r["valid_final_answer"] for r in diagnostics),
            "invalid_final_answers": sum(not r["valid_final_answer"] for r in diagnostics),
            "answer_complete_rate": sum(r["valid_final_answer"] for r in diagnostics) / n,
            "closed_answer_tags": sum(r["answer_tag_closed"] for r in diagnostics),
            "strict_format_count": sum(r["format_ok"] for r in diagnostics),
            "strict_format_rate": sum(r["format_ok"] for r in diagnostics) / n,
            "unfinished_analysis_count": sum(not r["analysis_closed"] for r in diagnostics),
            "repetitive_analysis_count": sum(r["repetitive_analysis"] for r in diagnostics),
            "repetitive_analysis_rate": sum(r["repetitive_analysis"] for r in diagnostics) / n,
            "analysis_words_mean_including_unclosed": sum(r["analysis_words"] for r in diagnostics) / n,
            "analysis_over_120_count": sum(r["analysis_words"] > 120 for r in diagnostics),
            "analysis_over_120_fraction": sum(r["analysis_words"] > 120 for r in diagnostics) / n,
            "decoded_response_near_cap_count": sum(r["near_generation_cap"] is True for r in diagnostics) if tokenizer else None,
            "max_new_tokens": cap,
            "cap_detection_note": f"Decoded-text retokenization >= {cap - 2}, approximate; no original finish_reason is available.",
            "answer_complete_definition": "An unambiguous nonletter final answer exists; a missing closing answer tag remains allowed, as in the scorer.",
            "repetition_definition": "At least 60 analysis words and repeated 20-word ngram fraction >= 0.4; excludes some semantic paraphrase repetition.",
            "analysis_factuality_evaluated": False, "per_question": diagnostics}


def check_summary(summary: dict, metrics: dict, mode: str) -> None:
    expected = {"total": metrics["total"], "correct": metrics["confirmed_correct"]}
    if mode == "openqa":
        expected.update(uncertain=metrics["undecided"], incorrect=metrics["confirmed_incorrect"])
    for key, value in expected.items():
        if key in summary and summary[key] != value:
            raise ValueError(f"Summary {key} disagrees with scored rows: {summary[key]} != {value}")
    if "accuracy" in summary:
        actual = summary["accuracy"]
        required = metrics["accuracy"]
        if (actual is None) != (required is None) or (actual is not None and abs(actual - required) > 1e-12):
            raise ValueError("Summary accuracy differs from strict boolean/null row aggregation")
    for key in ("accuracy_lower_bound", "accuracy_upper_bound"):
        if key in summary and abs(summary[key] - metrics[key]) > 1e-12:
            raise ValueError("Summary bound differs from scored rows: " + key)


def reference_is_insufficient(row: dict) -> bool:
    return ("reference_insufficiency" in row or
            row.get("scoring_method") == "insufficient-reference")


def automatic_openqa_grade(row: dict) -> bool | None:
    """Recover the automatic grade using recorded evidence, never the reviewed value.

    A reference audit applies to every model and takes precedence over a later
    adjudication. Other reviewed rows require two explicit raw judge verdicts
    or an explicitly recorded pre-adjudication boolean/null value.
    """
    if reference_is_insufficient(row):
        return None
    reviewed = (row.get("scoring_method") in ("assistant-review", "human-review") or
                "assistant_adjudication" in row or "human_adjudication" in row)
    if not reviewed:
        return row["semantic_correct"]
    originals = []
    for field in ("semantic_correct_before_adjudication", "semantic_correct_before_review"):
        if field in row:
            value = row[field]
            if value is not None and type(value) is not bool:
                raise ValueError("Invalid recorded original grade: " + row["sample_id"])
            originals.append(value)
    if "judge_reviews" in row:
        reviews = row["judge_reviews"]
        if (not isinstance(reviews, list) or len(reviews) != 2 or
                any(not isinstance(review, dict) or review.get("verdict") not in
                    ("YES", "NO", "UNCERTAIN") for review in reviews)):
            raise ValueError("Invalid original dual judge evidence: " + row["sample_id"])
        verdicts = [review["verdict"] for review in reviews]
        originals.append(True if verdicts == ["YES", "YES"] else
                         False if verdicts == ["NO", "NO"] else None)
    if not originals:
        raise ValueError("Reviewed row lacks an explicit original grade or dual judge evidence: " + row["sample_id"])
    if any(value is not originals[0] for value in originals[1:]):
        raise ValueError("Recorded original grade contradicts dual judge evidence: " + row["sample_id"])
    return originals[0]


def load_model(root: Path, mode: str, name: str, ids: list[str], labelmap: dict[str, dict],
               mc_labels: list[dict], mc_sources: list[dict], tokenizer,
               require_v2: bool = True) -> tuple[list[bool | None], dict, dict, list[dict]] | None:
    directory = root / mode / name
    summary_path = directory / "summary.json"
    row_path = directory / ("results.jsonl" if mode == "mcq" else "scored.jsonl")
    if not summary_path.exists() or not row_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    if mode == "mcq":
        if require_v2 and not str(summary.get("scoring", "")).startswith("mcq-v2:"):
            # Generation may already be complete while the parent is replacing
            # a legacy summary with the uniformly applied MCQ v2 grade.
            return None
        rows = scored_mcq_v2_rows(read_rows(row_path), mc_labels, mc_sources)
    else:
        if require_v2 and not str(summary.get("scoring", "")).startswith("v2:"):
            raise ValueError("New open-QA summary is not scored with v2: " + str(summary_path))
        if require_v2 and (summary.get("judge_calibration", {}).get("evaluated") is not True or
                           summary.get("scoring_pipeline_reliable_on_calibration") is not True):
            # prepare-only summaries contain hundreds of not-yet-reviewed
            # nulls; they are pending work, not completed semantic intervals.
            return None
        rows = read_rows(row_path)
    rowmap = {row["sample_id"]: row for row in rows}
    if len(rowmap) != len(rows) or set(rowmap) != set(ids):
        raise ValueError("Missing/extra/duplicate scored IDs in " + str(row_path))
    ordered = [rowmap[key] for key in ids]
    for row in ordered:
        if mode == "openqa" and (row.get("video_id") != labelmap[row["sample_id"]]["video_id"]):
            raise ValueError("Scored video ID differs from label: " + row["sample_id"])
    stored_values = [row["correct" if mode == "mcq" else "semantic_correct"] for row in ordered]
    stored_metrics = score_metrics(stored_values)
    if mode == "mcq" and not require_v2:
        # Old files keep their original strict score. The rerun comparison
        # reparses their original responses in memory under the new parser.
        legacy_metrics = score_metrics([row["legacy_correct"] for row in ordered])
        check_summary(summary, legacy_metrics, mode)
    else:
        # Validate the on-disk reviewed scores before deriving the primary
        # comparison; primary recovery must not hide a stale/broken summary.
        check_summary(summary, stored_metrics, mode)
    values = ([automatic_openqa_grade(row) for row in ordered]
              if mode == "openqa" and require_v2 else stored_values)
    metrics = score_metrics(values)
    if mode == "openqa" and require_v2:
        reviewed_values = [None if reference_is_insufficient(row) else row["semantic_correct"]
                           for row in ordered]
        reviewer_counts = {
            reviewer: sum(row.get("scoring_method") == reviewer + "-review" or
                          reviewer + "_adjudication" in row for row in ordered)
            for reviewer in ("assistant", "human")}
        metrics.update(primary_scoring=OPENQA_PRIMARY_SCORING,
                       assistant_review_count=reviewer_counts["assistant"],
                       human_review_count=reviewer_counts["human"],
                       reviewed_sensitivity={
                           **score_metrics(reviewed_values),
                           "scoring": "Stored model-specific reviews, with the common reference-sufficiency audit preserved; secondary sensitivity result only",
                           "changed_grade_count": sum(primary is not reviewed for primary, reviewed in zip(values, reviewed_values)),
                           "reference_insufficient_review_overrides_ignored": sum(reference_is_insufficient(row) and row["semantic_correct"] is not None for row in ordered)},
                       adjudication_recovery_audit=[
                           {"sample_id": row["sample_id"], "stored_semantic_correct": row["semantic_correct"],
                            "primary_semantic_correct": value, "stored_scoring_method": row.get("scoring_method"),
                            "reference_insufficient": reference_is_insufficient(row)}
                           for row, value in zip(ordered, values) if value is not row["semantic_correct"]])
    diagnostics = {}
    if mode == "openqa":
        manifest_path = root / "data/openqa_manifest.json"
        cap = json.loads(manifest_path.read_text()).get("generation_max_new_tokens", 768) if manifest_path.exists() else 768
        diagnostics = generation_diagnostics(ordered, tokenizer, cap)
    else:
        metrics["parse_rate"] = sum(row["prediction"] is not None for row in ordered) / len(ordered)
        metrics["parse_failures"] = sum(row["prediction"] is None for row in ordered)
        v2_summary = summarize_mcq_v2(ordered)
        for key in ("strict_format_rate", "legacy_strict_accuracy", "legacy_strict_parse_rate",
                    "v2_recovered_from_legacy_unparsed", "v2_rejected_legacy_parsed"):
            metrics[key] = v2_summary[key]
        metrics["primary_scoring"] = MCQ_SCORING
    return values, summary, dict(metrics, generation_diagnostics=diagnostics), ordered


def experiment(name: str) -> str:
    return name.rsplit("_epoch", 1)[0] if "_epoch" in name else name


TABLE_FIELDS = [
    "mode", "model", "experiment", "epoch", "status", "total", "confirmed_correct", "confirmed_incorrect",
    "undecided", "accuracy", "accuracy_lower_bound", "accuracy_upper_bound", "delta_vs_base",
    "delta_vs_base_lower_bound", "delta_vs_base_upper_bound", "ci95_low", "ci95_high",
    "conservative_ci95_low", "conservative_ci95_high", "parse_or_format_rate", "valid_final_answers",
    "invalid_final_answers", "answer_complete_rate", "repetitive_analysis_count", "repetitive_analysis_rate",
    "unfinished_analysis_count", "analysis_words_mean", "analysis_over_120_fraction", "near_cap_count",
    "legacy_strict_mcq_accuracy", "legacy_strict_mcq_parse_rate", "strict_format_compliance_rate",
    "pipeline_calibration_accuracy", "raw_judge_calibration_accuracy", "tier_A_accuracy", "tier_A_lower_bound", "tier_A_upper_bound", "tier_A_undecided",
    "tier_B_accuracy", "tier_B_lower_bound", "tier_B_upper_bound", "tier_B_undecided",
    "tier_D_accuracy", "tier_D_lower_bound", "tier_D_upper_bound", "tier_D_undecided",
    "primary_scoring", "assistant_review_count", "human_review_count",
    "reviewed_sensitivity_confirmed_correct", "reviewed_sensitivity_confirmed_incorrect",
    "reviewed_sensitivity_undecided", "reviewed_sensitivity_accuracy",
    "reviewed_sensitivity_accuracy_lower_bound", "reviewed_sensitivity_accuracy_upper_bound",
    "reviewed_sensitivity_changed_grade_count",
]


def build_report(root: Path, old_root: Path | None, partial: bool, tokenizer,
                 expected_rows: int = 518) -> tuple[dict, list[dict], list[dict]]:
    tasks = json.loads((root / "tasks.json").read_text())
    if len({task["label"] for task in tasks}) != len(tasks):
        raise ValueError("Duplicate task labels")
    labels = read_rows(root / "data/worldsense.openqa.labels.jsonl")
    labelmap = {row["sample_id"]: row for row in labels}
    ids = sorted(labelmap)
    if len(ids) != len(labels) or len(ids) != expected_rows:
        raise ValueError(f"Expected {expected_rows} unique label IDs, got {len(ids)}")
    mc_labels = read_rows(root / "data/worldsense.labels.jsonl")
    mc_sources = read_rows(root / "data/worldsense.answer_free.jsonl")
    if set(ids) != {row["sample_id"] for row in mc_labels} or set(ids) != {row["case_id"] for row in mc_sources}:
        raise ValueError("MCQ/open-QA IDs differ")
    bootstrap = VideoBootstrap(ids, labelmap)
    report = {"root": str(root.resolve()), "n_questions": len(ids), "n_videos": len(bootstrap.videos),
              "expected_tasks": len(tasks) * 2, "bootstrap_unit": "video", "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
              "bootstrap_seed": BOOTSTRAP_SEED, "confidence_interval_note": CI_NOTE,
              "bootstrap_denominator_note": "Point estimates use all frozen questions. Each video-cluster resample uses all questions belonging to its sampled videos, without dropping unresolved rows.",
              "unresolved_score_policy": "Keep null. Main accuracy is unavailable until every grade is resolved; bounds use all frozen questions.",
              "openqa_primary_scoring": OPENQA_PRIMARY_SCORING,
              "openqa_review_sensitivity_policy": "Main comparisons, tiers, paired deltas and bootstrap use uniform automatic scores. Existing assistant/human reviews remain a separate sensitivity result; original scored rows and summaries are not changed.",
              "protocol_comparison_note": PROTOCOL_NOTE, "modes": {}, "old_protocol_comparison": {}}
    contract_path = root / "data/training_matched_contract.json"
    if contract_path.exists():
        report["evaluation_contract"] = json.loads(contract_path.read_text())
    table, old_table = [], []
    for mode in ("mcq", "openqa"):
        loaded, pending = {}, []
        for task in tasks:
            result = load_model(root, mode, task["label"], ids, labelmap, mc_labels, mc_sources, tokenizer)
            if result is None:
                pending.append(task["label"])
                if not partial:
                    raise ValueError("Incomplete task: " + str(root / mode / task["label"]))
            else:
                loaded[task["label"]] = result
        paired = {}
        if "base" in loaded:
            for name, (values, _, _, _) in loaded.items():
                paired[name] = paired_comparison(values, loaded["base"][0], bootstrap, name == "base")
        model_reports = {}
        old_comparisons = {}
        for task in tasks:
            name = task["label"]
            row = {field: None for field in TABLE_FIELDS}
            row.update(mode=mode, model=name, experiment=experiment(name), epoch=task.get("epoch"),
                       status="pending", total=len(ids))
            old_row = dict(mode=mode, model=name, epoch=task.get("epoch"), total=len(ids),
                           old_accuracy=None, old_original_strict_mcq_accuracy=None, old_mcq_rescored_v2_parse_rate=None,
                           old_mcq_reparsed_v2=(mode == "mcq"),
                           new_accuracy=None, new_accuracy_lower_bound=None,
                           new_accuracy_upper_bound=None, new_undecided=None,
                           observed_delta=None, observed_delta_lower_bound=None, observed_delta_upper_bound=None,
                           protocol_changed=True, single_factor_attribution_allowed=False)
            if name in loaded:
                values, summary, metrics, score_rows = loaded[name]
                low, high = score_bounds(values)
                absolute_interval = bootstrap.interval(low, high, metrics["all_scores_resolved"])
                tiers = {tier: score_metrics([values[i] for i, key in enumerate(ids) if labelmap[key]["tier"] == tier])
                         for tier in ("A", "B", "D") if any(labelmap[key]["tier"] == tier for key in ids)}
                types = {kind: score_metrics([values[i] for i, key in enumerate(ids) if labelmap[key]["question_type"] == kind])
                         for kind in sorted({labelmap[key]["question_type"] for key in ids})}
                model_reports[name] = {"epoch": task.get("epoch"), "experiment": experiment(name),
                                       "metrics": metrics, "accuracy_interval": absolute_interval,
                                       "by_tier": tiers, "by_question_type": types, "source_summary": summary,
                                       "scored_rows_sha256": sha256(root / mode / name / ("results.jsonl" if mode == "mcq" else "scored.jsonl"))}
                row.update(status="complete" if metrics["all_scores_resolved"] else "provisional-unresolved",
                           **{key: metrics[key] for key in ("confirmed_correct", "confirmed_incorrect", "undecided", "accuracy",
                                                          "accuracy_lower_bound", "accuracy_upper_bound")})
                comparison = paired.get(name, {})
                exact_ci = comparison.get("ci95_video_bootstrap") or [None, None]
                envelope = comparison.get("ci95_video_bootstrap_conservative_envelope") or [None, None]
                row.update(delta_vs_base=comparison.get("delta_accuracy"),
                           delta_vs_base_lower_bound=comparison.get("delta_accuracy_lower_bound"),
                           delta_vs_base_upper_bound=comparison.get("delta_accuracy_upper_bound"),
                           ci95_low=exact_ci[0], ci95_high=exact_ci[1],
                           conservative_ci95_low=envelope[0], conservative_ci95_high=envelope[1],
                           parse_or_format_rate=metrics.get("parse_rate", summary.get("format_rate")),
                           legacy_strict_mcq_accuracy=metrics.get("legacy_strict_accuracy"),
                           legacy_strict_mcq_parse_rate=metrics.get("legacy_strict_parse_rate"),
                           strict_format_compliance_rate=metrics.get("strict_format_rate", summary.get("format_rate")),
                           pipeline_calibration_accuracy=summary.get("judge_calibration", {}).get("accuracy"),
                           raw_judge_calibration_accuracy=summary.get("judge_calibration", {}).get("raw_judge_accuracy"))
                row.update(primary_scoring=metrics.get("primary_scoring"),
                           assistant_review_count=metrics.get("assistant_review_count"),
                           human_review_count=metrics.get("human_review_count"))
                sensitivity = metrics.get("reviewed_sensitivity", {})
                for key in ("confirmed_correct", "confirmed_incorrect", "undecided", "accuracy",
                            "accuracy_lower_bound", "accuracy_upper_bound", "changed_grade_count"):
                    row["reviewed_sensitivity_" + key] = sensitivity.get(key)
                diagnostics = metrics["generation_diagnostics"]
                if diagnostics:
                    for key in ("valid_final_answers", "invalid_final_answers", "answer_complete_rate",
                                "repetitive_analysis_count", "repetitive_analysis_rate", "unfinished_analysis_count",
                                "analysis_over_120_fraction"):
                        row[key] = diagnostics[key]
                    row.update(analysis_words_mean=diagnostics["analysis_words_mean_including_unclosed"],
                               near_cap_count=diagnostics["decoded_response_near_cap_count"])
                for tier, tier_metrics in tiers.items():
                    row[f"tier_{tier}_accuracy"] = tier_metrics["accuracy"]
                    row[f"tier_{tier}_lower_bound"] = tier_metrics["accuracy_lower_bound"]
                    row[f"tier_{tier}_upper_bound"] = tier_metrics["accuracy_upper_bound"]
                    row[f"tier_{tier}_undecided"] = tier_metrics["undecided"]
                old_row.update(new_accuracy=metrics["accuracy"], new_accuracy_lower_bound=metrics["accuracy_lower_bound"],
                               new_accuracy_upper_bound=metrics["accuracy_upper_bound"], new_undecided=metrics["undecided"])
            if old_root is not None and old_root.exists():
                old_data = old_root / "data"
                old_labels = read_rows(old_data / "worldsense.labels.jsonl")
                old_sources = read_rows(old_data / "worldsense.answer_free.jsonl")
                old_openlabels = read_rows(old_data / "worldsense.openqa.labels.jsonl")
                if {item["sample_id"] for item in old_openlabels} != set(ids):
                    raise ValueError("Old/new evaluation question IDs differ")
                old = (load_model(old_root, mode, name, ids, labelmap, old_labels, old_sources, tokenizer, require_v2=False)
                       if name in loaded else None)
                if old is not None:
                    old_row["old_accuracy"] = old[2]["accuracy"]
                    if mode == "mcq":
                        old_row["old_original_strict_mcq_accuracy"] = old[1]["accuracy"]
                        old_row["old_mcq_rescored_v2_parse_rate"] = old[2]["parse_rate"]
                    if name in loaded:
                        comparison = paired_comparison(loaded[name][0], old[0], bootstrap)
                        comparison.update(protocol_changed=True, interpretation_note=PROTOCOL_NOTE,
                                          old_generation_diagnostics=old[2]["generation_diagnostics"],
                                          new_generation_diagnostics=loaded[name][2]["generation_diagnostics"])
                        if mode == "mcq":
                            comparison.update(old_mcq_reparsed_v2=True,
                                              old_original_strict_accuracy=old[1]["accuracy"],
                                              old_primary_v2_accuracy=old[2]["accuracy"],
                                              old_primary_v2_parse_rate=old[2]["parse_rate"])
                        old_comparisons[name] = comparison
                        old_row.update(observed_delta=comparison["delta_accuracy"],
                                       observed_delta_lower_bound=comparison["delta_accuracy_lower_bound"],
                                       observed_delta_upper_bound=comparison["delta_accuracy_upper_bound"])
                elif name not in loaded:
                    old_summary_path = old_root / mode / name / "summary.json"
                    if old_summary_path.exists():
                        old_summary = json.loads(old_summary_path.read_text())
                        if old_summary.get("total") != len(ids):
                            raise ValueError("Old summary question total differs")
                        if mode == "mcq":
                            old_row["old_original_strict_mcq_accuracy"] = old_summary.get("accuracy")
                            old_row["old_mcq_reparsed_v2"] = False
                        else:
                            old_row["old_accuracy"] = old_summary.get("accuracy")
            table.append(row)
            old_table.append(old_row)
        report["modes"][mode] = {"completed_generation_and_scoring_models": list(loaded), "pending_models": pending,
                                 "fully_resolved_models": [name for name, item in loaded.items() if item[2]["all_scores_resolved"]],
                                 "models": model_reports, "paired_vs_base": paired}
        report["old_protocol_comparison"][mode] = old_comparisons
    complete_count = sum(len(mode["completed_generation_and_scoring_models"]) for mode in report["modes"].values())
    unresolved_count = sum(model["metrics"]["undecided"] for mode in report["modes"].values() for model in mode["models"].values())
    report.update(completed_tasks=complete_count, pending_tasks=len(tasks) * 2 - complete_count,
                  unresolved_semantic_grades=unresolved_count,
                  all_generation_and_scoring_tasks_complete=complete_count == len(tasks) * 2,
                  all_semantic_grades_resolved=complete_count == len(tasks) * 2 and unresolved_count == 0)
    return report, table, old_table


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def csv_text(rows: list[dict], fields: list[str]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, default=REPO / "training_runs/worldsense_observation_eval_20261001")
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--no-old-comparison", action="store_true")
    parser.add_argument("--tokenizer", type=Path,
                        default=Path("/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-assets/models/Qwen2.5-Omni-7B/tokenizer.json"))
    args = parser.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    report, table, old_table = build_report(args.root, None if args.no_old_comparison else args.old_root,
                                          args.partial, tokenizer)
    out = args.root / ("comparison_partial" if args.partial else "comparison")
    out.mkdir(parents=True, exist_ok=True)
    atomic_write(out / "summary.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    atomic_write(out / "table.csv", csv_text(table, TABLE_FIELDS))
    atomic_write(out / "old_protocol_comparison.csv", csv_text(old_table, list(old_table[0])))
    epoch_rows = sorted(table, key=lambda row: (row["mode"], row["experiment"], row["epoch"] or 0))
    atomic_write(out / "epoch_table.csv", csv_text(epoch_rows, TABLE_FIELDS))
    print(json.dumps({"completed_tasks": report["completed_tasks"], "pending_tasks": report["pending_tasks"],
                      "unresolved_semantic_grades": report["unresolved_semantic_grades"], "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
