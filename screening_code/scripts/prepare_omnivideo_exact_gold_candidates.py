#!/usr/bin/env python3
"""Prepare answer-safe canonical MCQs and shared scoring contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

CONTRACT = dict(
    fps=2.0,
    max_frames=768,
    min_pixels=3136,
    max_pixels=28672,
    sampling_rate=16000,
    scoring="first_token_ABCD_normalized",
    audio_max_uncovered_fraction=0.02,
    audio_max_edge_gap_seconds=1.0,
    audio_internal_gaps="reject",
)


def _read(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _by_id(rows, label):
    result = {}
    for row in rows:
        sid = row["sample_id"]
        if sid in result:
            raise ValueError(f"duplicate {label} ID: {sid}")
        result[sid] = row
    return result


def _spans(row):
    return row.get("evidence_spans", [])


def binding(row):
    return hashlib.sha256(
        json.dumps(
            {
                k: row[k]
                for k in (
                    "sample_id",
                    "video_id",
                    "video_path",
                    "duration",
                    "question",
                    "choices",
                    "answer",
                    "evidence_spans",
                )
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def _model_fingerprint(row):
    identity = row.get("model_identity")
    if not identity or not identity.get("asset_sha256"):
        raise ValueError("missing verified model identity")
    return identity


def _validate_exact_contract(score, source, variant):
    if (
        score.get("sample_id") != source["sample_id"]
        or score.get("answer") != source["answer"]
    ):
        raise ValueError("score ID/answer mismatch")
    if score.get("source_binding") != binding(source):
        raise ValueError(
            "score does not bind to current question/options/evidence/media"
        )
    if score.get("sampling_contract") != CONTRACT:
        raise ValueError("incompatible sampling/scoring contract")
    if variant not in score.get("scores", {}) or score.get("failure_reason"):
        raise ValueError("missing or failed score")


def timestamp(value):
    parts = value.strip().split(":")
    if len(parts) not in (2, 3):
        raise ValueError("timestamp must be MM:SS or HH:MM:SS")
    nums = [float(p) for p in parts]
    if any(not math.isfinite(x) or x < 0 for x in nums) or any(
        x >= 60 for x in nums[1:]
    ):
        raise ValueError("invalid timestamp")
    return sum(x * 60**i for i, x in enumerate(reversed(nums)))


def parse_spans(text, duration):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("missing_designated_segments")
    pattern = r"(\d+(?::\d+){1,2}(?:\.\d+)?)\s*[-–—]\s*(\d+(?::\d+){1,2}(?:\.\d+)?)"
    matches = list(re.finditer(pattern, text))
    if not matches or re.sub(pattern, "", text).strip(" \t\r\n[],;*`"):
        raise ValueError("unparsed_designated_segments")
    spans = [[timestamp(m[1]), timestamp(m[2])] for m in matches]
    if any(not 0 <= a < b <= duration for a, b in spans):
        raise ValueError("out_of_bounds_designated_segments")
    return spans


def canonicalize(row, video_root, ordering="indexed"):
    duration = float(row["duration"])
    if not 60 <= duration <= 180:
        raise ValueError("duration_outside_60_180")
    evidence_text = (row.get("analysis") or {}).get("designated_segments")
    if not isinstance(evidence_text, str) or not evidence_text.strip():
        raise ValueError("missing_designated_segments")
    q, choices = row.get("question"), row.get("options")
    if row["task"] == "event_sequence_ordering":
        q, choices = row.get(f"question_{ordering}"), row.get(f"options_{ordering}")
    if not isinstance(q, str) or not q.strip():
        raise ValueError("invalid_question")
    if (
        not isinstance(choices, list)
        or len(choices) != 4
        or any(not isinstance(c, str) or not c.strip() for c in choices)
    ):
        raise ValueError("invalid_choices")
    if row.get("answer") not in ("A", "B", "C", "D"):
        raise ValueError("invalid_answer")
    issues = []
    try:
        spans = parse_spans(evidence_text, duration)
    except ValueError as e:
        # Preserve the candidate universe, but quarantine before media scoring.
        # Do not silently mine timestamps from answer explanations or repair GT.
        spans = []
        issues.append(str(e))
    return dict(
        sample_id=row["question_id"],
        video_id=row["video_id"],
        video_path=str((Path(video_root) / (row["video_id"] + ".mp4")).resolve()),
        duration=duration,
        question=q,
        choices=choices,
        answer=row["answer"],
        question_type=row["task"],
        evidence_spans=spans,
        evidence_provenance="official_analysis.designated_segments",
        preparation_issues=issues,
        ordering_representation=ordering
        if row["task"] == "event_sequence_ordering"
        else None,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--video-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ordering", choices=["indexed", "textual"], default="indexed")
    a = p.parse_args()
    rows, excluded = [], []
    from collections import Counter

    for row in _read(a.source):
        try:
            rows.append(canonicalize(row, a.video_root, a.ordering))
        except (ValueError, KeyError, TypeError) as e:
            excluded.append(dict(sample_id=row.get("question_id"), reason=str(e)))
    _by_id(rows, "source")
    _write(a.output, rows)
    _write(a.output.with_suffix(".excluded.jsonl"), excluded)
    report = dict(
        source_sha256=_sha256(a.source),
        canonical_sha256=_sha256(a.output),
        rows=len(rows),
        videos=len({r["video_id"] for r in rows}),
        excluded=dict(Counter(r["reason"] for r in excluded)),
        ordering=a.ordering,
        evidence_parse_quarantines=dict(
            Counter(x for r in rows for x in r["preparation_issues"])
        ),
        historical_rows=20831,
        historical_count_match=len(rows) == 20831,
        note="Reconstructed parser; count agreement alone does not prove identical membership.",
    )
    a.output.with_suffix(".summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
