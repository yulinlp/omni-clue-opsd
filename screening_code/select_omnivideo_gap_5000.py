#!/usr/bin/env python3
"""Freeze a 5K OmniVideo training set with a large aggregate Gold advantage.

Exact Full and designated-segment Gold scores are required for every row.
Default aggregate selection ranks paired correctness differences first, then
correct-answer probability differences, under a per-video cap. No per-row
Gold-correct or positive-gap condition is required. The former stricter policy
is retained as an explicit comparison, not a user-imposed constraint.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from typing import Any

try:
    from scripts.audit_omnivideo_score_media import validate_observed_media
    from scripts.prepare_omnivideo_exact_gold_candidates import (
        _by_id,
        _model_fingerprint,
        _read,
        _sha256,
        _spans,
        _validate_exact_contract,
        _write,
    )
except ModuleNotFoundError:  # direct execution places scripts/ on sys.path
    from audit_omnivideo_score_media import validate_observed_media
    from prepare_omnivideo_exact_gold_candidates import (  # type: ignore[no-redef]
        _by_id,
        _model_fingerprint,
        _read,
        _sha256,
        _spans,
        _validate_exact_contract,
        _write,
    )


def _mean_ci95(values: list[float]) -> list[float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return [mean, mean]
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - half, mean + half]


def _wilson95(successes: int, total: int) -> list[float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive total")
    z = 1.96
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def _score_identity_set(rows: list[dict[str, Any]]) -> set[str]:
    return {json.dumps(_model_fingerprint(row), sort_keys=True) for row in rows}


def _validate_probabilities(score: dict[str, Any], variant: str) -> None:
    """Reject corrupt or internally inconsistent scores before ranking."""
    sample_id = str(score.get("sample_id"))
    view = score["scores"][variant]
    probabilities = view.get("probabilities") or {}
    if set(probabilities) != set("ABCD"):
        raise ValueError(f"{sample_id}: {variant} requires ABCD probabilities")
    values = {key: float(value) for key, value in probabilities.items()}
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values.values()):
        raise ValueError(f"{sample_id}: {variant} invalid probability")
    if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-6):
        raise ValueError(f"{sample_id}: {variant} probabilities do not sum to one")
    predicted = str(view.get("predicted", "")).upper()
    if predicted not in values or values[predicted] != max(values.values()):
        raise ValueError(f"{sample_id}: {variant} prediction is not an argmax")
    answer = str(score["answer"]).upper()
    p_answer = float(score["answer_probability"][variant])
    if answer not in values or not math.isfinite(p_answer) or not math.isclose(
        p_answer, values[answer], rel_tol=1e-6, abs_tol=1e-9
    ):
        raise ValueError(f"{sample_id}: {variant} answer probability mismatch")


def _validate_text_score(score: dict[str, Any], source: dict[str, Any]) -> None:
    sample_id = str(source["sample_id"])
    if str(score.get("sample_id") or "") != sample_id:
        raise ValueError(f"{sample_id}: text diagnostic ID mismatch")
    if str(score.get("answer") or "").upper() != str(source.get("answer") or "").upper():
        raise ValueError(f"{sample_id}: text diagnostic answer mismatch")
    if "text" not in (score.get("scores") or {}):
        raise ValueError(f"{sample_id}: text diagnostic is missing text score")
    if "text" not in (score.get("answer_probability") or {}):
        raise ValueError(f"{sample_id}: text diagnostic is missing answer probability")


def _source_quality_issues(source: dict[str, Any]) -> list[str]:
    """Keep malformed MCQs out without changing the scored candidate universe."""
    issues = list(source.get("preparation_issues", []))
    if not _spans(source):
        issues.append("missing_valid_gold_intervals")
    question = source.get("question")
    if not isinstance(question, str) or not question.strip():
        issues.append("empty_question")
    choices = source.get("choices")
    if not isinstance(choices, list) or len(choices) != 4 or not all(
        isinstance(choice, str) and choice.strip() for choice in choices
    ):
        issues.append("invalid_four_choices")
    else:
        # Preserve semantic punctuation such as arrows, signs and decimals.
        normalized = [" ".join(choice.casefold().split()) for choice in choices]
        if len(set(normalized)) != 4:
            issues.append("duplicate_choice_text")
    spans = [tuple(span) for span in _spans(source)]
    ordered = sorted(set(spans))
    if len(ordered) != len(spans) or any(b > c for (_, b), (c, _) in zip(ordered, ordered[1:])):
        issues.append('duplicate_or_overlapping_gold_intervals_require_union_rescore')
    return issues


def media_timing_exclusions(report, canonical, canonical_sha256):
    """Conservatively quarantine mismatched source media, independent of scores."""
    if (report.get('status') != 'source_stream_timing_audit_not_semantic_gt_validation'
        or report.get('source_sha256') != canonical_sha256 or report.get('unreadable') != []):
        raise ValueError('media timing audit provenance/integrity mismatch')
    groups = {}
    for row in canonical:
        groups.setdefault(row['video_id'], []).append(row)
    items = report.get('items') or []
    if (len(items) != len(groups) or len({r['video_id'] for r in items}) != len(items)
        or {r['video_id'] for r in items} != set(groups)
        or report.get('candidate_rows') != len(canonical)
        or report.get('videos') != len(groups) or report.get('probed') != len(groups)):
        raise ValueError('media timing audit does not cover the complete source pool')
    excluded = set()
    for item in items:
        sources = groups[item['video_id']]
        duration = float(sources[0]['duration'])
        if (not math.isfinite(duration) or duration <= 0
            or any(float(s['duration']) != duration for s in sources)
            or item.get('declared_duration') != duration or item.get('qa_count') != len(sources)):
            raise ValueError('source duration/count differs from media timing audit')
        streams = item.get('streams') or []
        if len(streams) != 2 or {s['type'] for s in streams} != {'video', 'audio'}:
            raise ValueError('media timing audit needs exactly one audio and video stream')
        ends = [float(s['end']) for s in streams]
        if any(not math.isfinite(end) or end <= 0 for end in ends):
            raise ValueError('invalid measured stream endpoint')
        mismatch = any(abs(end - duration) > max(1., .02 * duration) for end in ends)
        if item.get('duration_mismatch_gt_2pct_or_1sec') is not mismatch:
            raise ValueError('media timing mismatch flag differs from actual measurements')
        if mismatch:
            excluded.update(s['sample_id'] for s in sources)
    if set(report.get('affected_sample_ids') or []) != excluded or report.get('duration_mismatch_qa_rows') != len(excluded):
        raise ValueError('media timing exclusion IDs differ from measured streams')
    return excluded


def source_pts_exclusions(report, canonical, canonical_sha256):
    if (report.get('status') != 'complete_source_pts_audit_not_semantic_gt_validation'
        or report.get('source_sha256') != canonical_sha256
        or report.get('candidate_rows') != len(canonical)):
        raise ValueError('source PTS audit provenance mismatch')
    groups = {}
    for row in canonical:
        groups.setdefault(row['video_id'], []).append(row)
    items = report.get('items') or []
    if (len(items) != len(groups) or {r['video_id'] for r in items} != set(groups)
        or report.get('videos') != len(groups) or report.get('audited_videos') != len(groups)):
        raise ValueError('source PTS audit coverage mismatch')
    excluded = set()
    for item in items:
        sources = groups[item['video_id']]
        duration = float(sources[0]['duration'])
        ids = [r['sample_id'] for r in sources]
        if (item.get('duration') != duration or set(item.get('sample_ids') or []) != set(ids)
            or len(item.get('sample_ids') or []) != len(ids)):
            raise ValueError('source PTS video-to-QA binding mismatch')
        if 'error' in item:
            if item.get('attempts') != 2 or item.get('quarantine_reasons') != ['source_pts_decode_failed_twice']:
                raise ValueError('unexpected source PTS decode failure record')
            expected = ['source_pts_decode_failed_twice']
        else:
            first, last, fps = (float(item[k]) for k in ('first_full_timestamp', 'last_full_timestamp', 'source_average_fps'))
            count = item.get('full_unique_timestamps')
            if (any(not math.isfinite(x) for x in (first, last, fps)) or fps <= 0
                or not 0 <= first < last <= duration or type(count) is not int or count < 2
                or not math.isclose(item['count_based_duration'], count / fps, rel_tol=1e-9)):
                raise ValueError('invalid source PTS observations')
            gaps = item.get('large_gaps')
            if not isinstance(gaps, list): raise ValueError('source PTS gap list missing')
            for a, b, gap in gaps:
                if not first <= a < b <= last or gap <= 1 or not math.isclose(b - a, gap, abs_tol=1e-8):
                    raise ValueError('invalid measured source PTS gap')
            expected = []
            if gaps: expected.append('source_pts_gap_gt_1s')
            if first > 1.: expected.append('source_first_frame_after_1s')
            if duration - last > 1.: expected.append('source_last_frame_more_than_1s_before_declared_end')
            if abs(count / fps - duration) > max(1., .05 * duration):
                expected.append('count_based_video_clock_disagrees_gt_5pct_or_1s')
            if item.get('quarantine_reasons') != expected:
                raise ValueError('source PTS quarantine reasons do not match observations')
        if expected: excluded.update(ids)
    if set(report.get('quarantined_sample_ids') or []) != excluded:
        raise ValueError('source PTS quarantine ID set mismatch')
    return excluded


def summarize_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError('cannot summarize an empty paired set')
    full = statistics.fmean(int(r['full_correct']) for r in rows)
    gold = statistics.fmean(int(r['gold_correct']) for r in rows)
    return {
        'rows': len(rows), 'full_accuracy': full, 'gold_accuracy': gold,
        'accuracy_gap': statistics.fmean(int(r['gold_correct']) - int(r['full_correct']) for r in rows),
        'mean_probability_gap': statistics.fmean(r['gold_minus_full_probability'] for r in rows),
        'mean_log_gap': statistics.fmean(r['privileged_learnability_log_gap'] for r in rows),
        'positive_flips': sum(not r['full_correct'] and r['gold_correct'] for r in rows),
        'negative_flips': sum(r['full_correct'] and not r['gold_correct'] for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--av-scores", type=Path, required=True)
    parser.add_argument("--gold-scores", type=Path, required=True)
    parser.add_argument("--coarse-five-view-scores", type=Path,
                        help="Optional diagnostic only; never fabricate missing text scores")
    parser.add_argument("--text-tiebreak", action="store_true",
                        help="Explicit opt-in; disabled to match historical aggregate protocol")
    parser.add_argument("--reconstructed-audit", type=Path,
                        help="Source audit from reconstructed PyAV pipeline")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=5000)
    parser.add_argument("--max-per-video", type=int, default=5)
    parser.add_argument('--selection-policy', choices=('aggregate', 'gold-correct-positive'),
                        default='aggregate')
    parser.add_argument('--media-timing-audit', type=Path,
                        help='complete source-media timing audit; quarantine mismatched videos')
    parser.add_argument('--source-pts-audit', type=Path,
                        help='complete CPU source timestamp audit; quarantine count-clock-incompatible media')
    args = parser.parse_args()
    if args.target_size <= 0 or args.max_per_video <= 0:
        raise ValueError("target size and per-video cap must be positive")

    canonical = _read(args.canonical)
    prefiltered = []
    reconstructed_media = {}
    if args.reconstructed_audit:
        from scripts.audit_omnivideo_score_media import audited_candidates
        canonical, prefiltered = audited_candidates(args.canonical, args.reconstructed_audit)
        reconstructed_media = {r["video_id"]: r for r in
            json.loads(args.reconstructed_audit.read_text())["items"]}
    timing_excluded = (media_timing_exclusions(json.loads(args.media_timing_audit.read_text()),
        canonical, _sha256(args.canonical)) if args.media_timing_audit else set())
    pts_report = json.loads(args.source_pts_audit.read_text()) if args.source_pts_audit else None
    pts_excluded = source_pts_exclusions(pts_report, canonical, _sha256(args.canonical)) if pts_report else set()
    pts_by_video = {r['video_id']: r for r in pts_report['items']} if pts_report else {}
    av_scores = _read(args.av_scores)
    gold_scores = _read(args.gold_scores)
    coarse_scores = _read(args.coarse_five_view_scores) if args.coarse_five_view_scores else []
    canonical_by_id = _by_id(canonical, "canonical")
    av_by_id = _by_id(av_scores, "Full scores")
    gold_by_id = _by_id(gold_scores, "Gold scores")
    coarse_by_id = _by_id(coarse_scores, "coarse five-view scores")
    expected_ids = set(canonical_by_id)
    score_sets = [("Full", av_by_id), ("Gold", gold_by_id)]
    if coarse_scores:
        score_sets.append(("coarse", coarse_by_id))
    for label, values in score_sets:
        if set(values) != expected_ids:
            raise ValueError(
                f"{label} score IDs do not exactly cover canonical: "
                f"actual={len(values)} expected={len(expected_ids)}"
            )
    identities = _score_identity_set(av_scores)
    if len(identities) != 1 or _score_identity_set(gold_scores) != identities:
        raise ValueError("exact Full and Gold passes did not use one identical model fingerprint")
    fingerprinted_coarse = [row for row in coarse_scores if row.get("model_identity")]
    if fingerprinted_coarse and _score_identity_set(fingerprinted_coarse) != identities:
        raise ValueError("coarse text diagnostic used a different model fingerprint")
    # Legacy scores lack model provenance. Do not invent it or allow those
    # diagnostics to influence membership or even tie-breaking.
    text_ranking_verified = bool(coarse_scores) and len(fingerprinted_coarse) == len(coarse_scores) and args.text_tiebreak
    if args.text_tiebreak and not text_ranking_verified:
        raise ValueError("text tiebreak requires fully fingerprinted diagnostic scores")

    decisions: list[dict[str, Any]] = []
    transitions: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    for source in canonical:
        sample_id = str(source["sample_id"])
        if not _spans(source):
            raise ValueError(f"{sample_id}: designated Gold interval is missing")
        av = av_by_id[sample_id]
        gold = gold_by_id[sample_id]
        coarse = coarse_by_id.get(sample_id)
        _validate_exact_contract(av, source, "av")
        _validate_exact_contract(gold, source, "gold")
        validate_observed_media(av, source, "av")
        validate_observed_media(gold, source, "gold")
        if reconstructed_media:
            expected_hash = reconstructed_media[source["video_id"]]["media_sha256"]
            for scored, variant in ((av, "av"), (gold, "gold")):
                if scored["scores"][variant]["observed_media"].get("media_sha256") != expected_hash:
                    raise ValueError(f"{sample_id}: scored media differs from source audit")
        if pts_by_video and sample_id not in pts_excluded:
            observed = av['scores']['av']['observed_media']['videos'][0]
            probed = pts_by_video[source['video_id']]
            if (abs(observed['decoder_total_frames'] - probed['full_unique_timestamps']) > 2
                or not math.isclose(observed['source_fps'], probed['source_average_fps'], rel_tol=1e-6)):
                raise ValueError(f'{sample_id}: actual Full decoder differs from independent source PTS audit')
        if coarse:
            _validate_text_score(coarse, source)
        pairs = [(av, "av"), (gold, "gold")]
        if coarse:
            pairs.append((coarse, "text"))
        for record, variant in pairs:
            _validate_probabilities(record, variant)
        answer = str(source["answer"]).upper()
        full_predicted = str(av["scores"]["av"]["predicted"]).upper()
        gold_predicted = str(gold["scores"]["gold"]["predicted"]).upper()
        text_predicted = str(coarse["scores"]["text"]["predicted"]).upper() if coarse else None
        full_correct = full_predicted == answer
        gold_correct = gold_predicted == answer
        text_correct = text_predicted == answer if coarse else None
        transitions[
            f"full_{'correct' if full_correct else 'wrong'}__gold_{'correct' if gold_correct else 'wrong'}"
        ] += 1
        p_full = float(av["answer_probability"]["av"])
        p_gold = float(gold["answer_probability"]["gold"])
        p_text = float(coarse["answer_probability"]["text"]) if coarse else None
        probability_gap = p_gold - p_full
        log_gap = math.log(max(p_gold, 1e-12)) - math.log(max(p_full, 1e-12))
        quality_issues = _source_quality_issues(source)
        if sample_id in timing_excluded:
            quality_issues.append('source_stream_duration_disagrees_with_annotation')
        if sample_id in pts_excluded:
            quality_issues.append('source_timestamps_incompatible_with_count_based_av_sampling')
        if quality_issues:
            tier, tier_order = "ineligible", 99
            eligible_reason = "invalid_source_mcq"
        elif args.selection_policy == 'aggregate':
            correctness_gap = int(gold_correct) - int(full_correct)
            tier_order = 1 - correctness_gap
            tier = {1: 'A_strict_flip', 0: 'B_neutral_accuracy', -1: 'C_negative_flip'}[correctness_gap]
            eligible_reason = 'aggregate_accuracy_then_probability_gap'
        elif gold_correct and not full_correct:
            tier, tier_order = "A_strict_flip", 0
            eligible_reason = "maximizes_paired_accuracy_gap"
        elif gold_correct and probability_gap > 0 and not text_correct and text_ranking_verified:
            tier, tier_order = "B_positive_gap_text_wrong", 1
            eligible_reason = "positive_privilege_gap_without_text_shortcut"
        elif gold_correct and probability_gap > 0:
            tier, tier_order = "C_positive_gap_text_correct", 2
            eligible_reason = "positive_privilege_gap_fill"
        else:
            tier, tier_order = "ineligible", 99
            eligible_reason = "Gold_wrong_or_nonpositive_nonflip_gap"
        decision = {
            "sample_id": sample_id,
            "video_id": str(source["video_id"]),
            "question_type": str(source.get("question_type") or "unknown"),
            "answer": answer,
            "full_predicted": full_predicted,
            "gold_predicted": gold_predicted,
            "text_predicted": text_predicted,
            "full_correct": full_correct,
            "gold_correct": gold_correct,
            "text_correct": text_correct,
            "strict_privileged_flip": bool(not full_correct and gold_correct),
            "paired_correctness_gap": int(gold_correct) - int(full_correct),
            "p_full_answer": p_full,
            "p_gold_answer": p_gold,
            "p_text_answer": p_text,
            "gold_minus_full_probability": probability_gap,
            "privileged_learnability_log_gap": log_gap,
            "selection_tier": tier,
            "eligible_reason": eligible_reason,
            "source_quality_issues": quality_issues,
            "selected": False,
            "selected_rank": None,
            "evidence_spans": _spans(source),
        }
        decisions.append(decision)
        if tier_order < 99:
            decision["_tier_order"] = tier_order
            eligible.append(decision)

    eligible.sort(
        key=lambda row: (
            int(row["_tier_order"]),
            -float(row["gold_minus_full_probability"]),
            -float(row["privileged_learnability_log_gap"]),
            -float(row["p_gold_answer"]),
            float(row["p_text_answer"]) if text_ranking_verified else 0.0,
            str(row["sample_id"]),
        )
    )
    video_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for decision in eligible:
        video_id = str(decision["video_id"])
        if video_counts[video_id] >= args.max_per_video:
            continue
        decision["selected"] = True
        decision["selected_rank"] = len(selected) + 1
        selected.append(decision)
        video_counts[video_id] += 1
        if len(selected) == args.target_size:
            break
    if len(selected) != args.target_size:
        raise RuntimeError(
            f"only {len(selected)} eligible rows survive max_per_video={args.max_per_video}; "
            f"cannot freeze requested {args.target_size}"
        )
    for decision in decisions:
        decision.pop("_tier_order", None)

    selected_ids = {str(row["sample_id"]) for row in selected}
    selected_canonical = [row for row in canonical if str(row["sample_id"]) in selected_ids]
    if len(selected_ids) != args.target_size or len(selected_canonical) != args.target_size:
        raise AssertionError("selected manifest lost or duplicated sample IDs")
    probability_gaps = [float(row["gold_minus_full_probability"]) for row in selected]
    log_gaps = [float(row["privileged_learnability_log_gap"]) for row in selected]
    strict_flips = sum(bool(row["strict_privileged_flip"]) for row in selected)
    paired = summarize_pairs(selected)
    full_accuracy = paired['full_accuracy']
    gold_accuracy = paired['gold_accuracy']
    correctness_gaps = [float(row['paired_correctness_gap']) for row in selected]
    tier_counts = Counter(str(row["selection_tier"]) for row in selected)
    task_counts = Counter(str(row["question_type"]) for row in selected)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    decisions_path = output / "omnivideo_100k_exact_full_gold_gap_decisions.jsonl"
    ranking_path = output / "omnivideo_100k_gap5000.ranking.jsonl"
    manifest_path = output / "omnivideo_100k_train.gap5000.canonical.jsonl"
    _write(decisions_path, decisions)
    _write(ranking_path, selected)
    _write(manifest_path, selected_canonical)
    _write(output / "prefilter_exclusions.jsonl", prefiltered)
    summary = {
        "reconstructed_audit": str(args.reconstructed_audit) if args.reconstructed_audit else None,
        "prefilter_excluded_rows": len(prefiltered),
        "historical_reproduction": "Reconstructed implementation; not proof of identical historical membership or scores",
        'source_pts_audit': ({'path': str(args.source_pts_audit.resolve()),
            'sha256': _sha256(args.source_pts_audit), 'quarantined_rows': len(pts_excluded),
            'policy': pts_report['policy'], 'caveat': pts_report['caveat']}
            if args.source_pts_audit else None),
        'media_timing_audit': ({'path': str(args.media_timing_audit.resolve()),
            'sha256': _sha256(args.media_timing_audit), 'quarantined_rows': len(timing_excluded),
            'policy': 'quarantine entire video when any AV stream endpoint differs by more than max(1 second, 2 percent); do not change official spans'}
            if args.media_timing_audit else None),
        "status": "frozen_exact_f1_gap5000",
        "target_rows": args.target_size,
        "selected_rows": len(selected_canonical),
        "definition": (
            "All structurally valid MCQs; paired correctness gap first, then "
            "P_gold-P_full; no per-row Gold-correct or positive-gap restriction; "
            "deterministic rank with per-video cap"
            if args.selection_policy == 'aggregate' else
            "Gold-correct; Tier A Full-wrong/Gold-correct first; then positive "
            "P_gold-P_full; text-wrong priority only when all text model fingerprints "
            "are verified; deterministic rank; per-video cap"
        ),
        "selection_policy": args.selection_policy,
        "text_diagnostic_provenance": {
            "fingerprinted_rows": len(fingerprinted_coarse),
            "total_rows": len(coarse_scores),
            "used_for_ranking": text_ranking_verified,
        },
        "statistical_caveat": (
            "Post-selection training-set statistics, not independent-test performance. "
            "Report measured Gold accuracy, not an assumed 100%. Reported IID intervals ignore video "
            "clustering and selection and must not be interpreted as generalization CIs."
        ),
        "max_per_video": args.max_per_video,
        "model_fingerprint": _model_fingerprint(av_scores[0]),
        "sampling_contract": {
            "fps": 2.0,
            "max_frames": 768,
            "min_pixels": 3136,
            "max_pixels": 28672,
            "full_timeline_covered": True,
            "text_diagnostic_media_independent": True,
            "observed_decoder_and_processor_media_validated": True,
        },
        "canonical_rows": len(canonical),
        "all_transitions": dict(sorted(transitions.items())),
        "eligible_rows_before_video_cap": len(eligible),
        "source_quality_excluded_rows": sum(bool(row["source_quality_issues"]) for row in decisions),
        "source_quality_issue_counts": dict(Counter(
            issue for row in decisions for issue in row["source_quality_issues"])),
        "selected_tier_counts": dict(sorted(tier_counts.items())),
        "selected_strict_flip_rows": strict_flips,
        "selected_negative_flip_rows": paired['negative_flips'],
        "selected_gold_wrong_rows": sum(not row['gold_correct'] for row in selected),
        "selected_nonpositive_probability_gap_rows": sum(g <= 0 for g in probability_gaps),
        "selected_full_accuracy": full_accuracy,
        "selected_gold_accuracy": gold_accuracy,
        "selected_accuracy_gap": paired['accuracy_gap'],
        "selected_accuracy_gap_descriptive_iid_ci95": _mean_ci95(correctness_gaps),
        "selected_mean_probability_gap": statistics.fmean(probability_gaps),
        "selected_mean_probability_gap_ci95": _mean_ci95(probability_gaps),
        "selected_mean_log_gap": statistics.fmean(log_gaps),
        "selected_mean_log_gap_ci95": _mean_ci95(log_gaps),
        "selected_text_wrong_rows": sum(row["text_correct"] is False for row in selected) if coarse_scores else None,
        "selected_unique_videos": len(video_counts),
        "selected_max_questions_per_video": max(video_counts.values()),
        "selected_task_counts": dict(sorted(task_counts.items())),
        "selected_per_task": {task: summarize_pairs([r for r in selected if r['question_type'] == task])
                              for task in sorted(task_counts)},
        "selected_per_video": {video: summarize_pairs([r for r in selected if r['video_id'] == video])
                               for video in sorted(video_counts)},
        "artifacts": {
            "canonical": str(args.canonical.resolve()),
            "canonical_sha256": _sha256(args.canonical),
            "av_scores": str(args.av_scores.resolve()),
            "av_scores_sha256": _sha256(args.av_scores),
            "gold_scores": str(args.gold_scores.resolve()),
            "gold_scores_sha256": _sha256(args.gold_scores),
            "coarse_five_view_scores": str(args.coarse_five_view_scores.resolve()) if args.coarse_five_view_scores else None,
            "coarse_five_view_scores_sha256": _sha256(args.coarse_five_view_scores) if args.coarse_five_view_scores else None,
            "decisions": str(decisions_path),
            "decisions_sha256": _sha256(decisions_path),
            "ranking": str(ranking_path),
            "ranking_sha256": _sha256(ranking_path),
            "training_manifest": str(manifest_path),
            "training_manifest_sha256": _sha256(manifest_path),
        },
    }
    (output / "gap5000_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "GAP5000_SUCCESS").touch()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
