#!/usr/bin/env python3
"""Independent source timing audit and validation of client-observed input."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.prepare_omnivideo_exact_gold_candidates import _read, _sha256, _spans


def audited_candidates(canonical, audit_path):
    from select_omnivideo_gap_5000 import _source_quality_issues

    rows = _read(canonical)
    audit = json.loads(Path(audit_path).read_text())
    if audit.get("status") != "reconstructed_source_audit_v1" or audit[
        "source_sha256"
    ] != _sha256(canonical):
        raise ValueError("audit provenance mismatch")
    mapping = {r["video_id"]: r for r in audit["items"]}
    if len(mapping) != len(audit["items"]) or set(mapping) != {
        r["video_id"] for r in rows
    }:
        raise ValueError("audit coverage mismatch")
    good, bad = [], []
    for row in rows:
        item = mapping[row["video_id"]]
        reasons = _source_quality_issues(row)
        reasons += item["quarantine_reasons"]
        if item.get("duration_mismatch_gt_2pct_or_1sec"):
            reasons.append("source_stream_duration_disagrees_with_annotation")
        if reasons:
            bad.append(dict(sample_id=row["sample_id"], reasons=reasons))
        else:
            good.append(row)
    if not good:
        raise ValueError("no eligible candidates after audit")
    return good, bad


def probe(row):
    import av

    path = row["video_path"]
    with av.open(path) as c:
        streams = []
        for s in c.streams:
            if s.type in ("audio", "video"):
                if s.duration is None:
                    raise ValueError("stream has no measurable duration")
                start = float((s.start_time or 0) * s.time_base)
                streams.append(
                    dict(
                        type=s.type,
                        start=start,
                        end=start + float(s.duration * s.time_base),
                    )
                )
        if len(streams) != 2 or {s["type"] for s in streams} != {"video", "audio"}:
            raise ValueError("exactly one audio and video stream required")
        fps = float(c.streams.video[0].average_rate)
        ts = [
            float(f.pts * f.time_base) for f in c.decode(video=0) if f.pts is not None
        ]
    if len(ts) < 2 or any(b <= a for a, b in zip(ts, ts[1:])):
        raise ValueError("missing/nonmonotonic source PTS")
    duration = row["duration"]
    reasons = []
    gaps = [[a, b, b - a] for a, b in zip(ts, ts[1:]) if b - a > 1]
    if gaps:
        reasons.append("source_pts_gap_gt_1s")
    if ts[0] > 1:
        reasons.append("source_first_frame_after_1s")
    if duration - ts[-1] > 1:
        reasons.append("source_last_frame_more_than_1s_before_declared_end")
    if abs(len(ts) / fps - duration) > max(1, 0.05 * duration):
        reasons.append("count_based_video_clock_disagrees_gt_5pct_or_1s")
    if ts[0] < -1 or ts[-1] > duration + max(1, 0.02 * duration):
        reasons.append("source_pts_outside_declared_duration")
    timing_bad = any(
        abs(s["end"] - duration) > max(1, 0.02 * duration) for s in streams
    )
    return dict(
        video_id=row["video_id"],
        duration=duration,
        streams=streams,
        source_average_fps=fps,
        full_unique_timestamps=len(ts),
        timestamps=ts,
        first_full_timestamp=ts[0],
        last_full_timestamp=ts[-1],
        count_based_duration=len(ts) / fps,
        large_gaps=gaps,
        quarantine_reasons=reasons,
        duration_mismatch_gt_2pct_or_1sec=timing_bad,
        media_sha256=_sha256(path),
    )


def validate_observed_media(score, source, variant):
    obs = score["scores"][variant].get("observed_media")
    spans = [[0.0, source["duration"]]] if variant == "av" else _spans(source)
    if not obs or obs.get("input_spans") != spans:
        raise ValueError("observed spans differ from required input")
    if len(obs.get("videos", [])) != len(spans) or len(obs.get("audios", [])) != len(
        spans
    ):
        raise ValueError("not all evidence AV segments reached the processor")
    for video, audio, (a, b) in zip(obs["videos"], obs["audios"], spans):
        ts = video["sampled_timestamps"]
        if not ts or not all(a <= t <= b for t in ts) or len(ts) > 768:
            raise ValueError("invalid sampled frame timestamps")
        if any(y <= x for x, y in zip(ts, ts[1:])):
            raise ValueError("duplicate/nonmonotonic input frames")
        if (
            audio["sampling_rate"] != 16000
            or abs(audio["samples"] / 16000 - (b - a)) > 0.002
        ):
            raise ValueError("audio duration not matched to evidence")
        if video.get("visual_tokens", 0) <= 0 or not video.get("processor_grid_thw"):
            raise ValueError("official processor token audit missing")
    if not math.isfinite(obs.get("total_visual_tokens", float("nan"))):
        raise ValueError("invalid visual token accounting")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--canonical", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    rows = _read(a.canonical)
    by_video = {r["video_id"]: r for r in rows}
    items = []

    def one(row):
        for attempt in range(2):
            try:
                return probe(row)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
        return dict(
            video_id=row["video_id"],
            error=error,
            attempts=2,
            quarantine_reasons=["source_pts_decode_failed_twice"],
        )

    with ThreadPoolExecutor(a.workers) as pool:
        for item in pool.map(one, by_video.values()):
            items.append(item)
            print(
                f"audited {len(items)}/{len(by_video)} {item['video_id']}", flush=True
            )
    excluded = {
        x["video_id"]
        for x in items
        if x.get("error")
        or x["quarantine_reasons"]
        or x["duration_mismatch_gt_2pct_or_1sec"]
    }
    report = dict(
        status="reconstructed_source_audit_v1",
        source_sha256=_sha256(a.canonical),
        candidate_rows=len(rows),
        videos=len(by_video),
        items=items,
        quarantined_sample_ids=[
            r["sample_id"] for r in rows if r["video_id"] in excluded
        ],
        caveat="Source timing integrity, not semantic ground-truth validation",
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report) + "\n")
    print(f"quarantined {len(report['quarantined_sample_ids'])} QA")


if __name__ == "__main__":
    main()
