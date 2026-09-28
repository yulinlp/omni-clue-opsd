#!/usr/bin/env python3
"""Fast source audit for WorldSense gap candidates.

The OmniVideo audit decoded every frame to read real PTS; WorldSense videos
were already frame-decoded by the annotation pipeline and are CFR, so this
audit reads stream metadata (frame count, average rate, stream durations) and
derives the frame timeline analytically, then quarantines videos whose streams
disagree with the annotation.  Media hashes are streamed once per video.

Output keeps the OmniVideo audit schema (status / source_sha256 / items /
quarantined_sample_ids) so downstream code can stay unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STATUS = "worldsense_source_audit_v1"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).open(encoding="utf-8", errors="replace") if l.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(row: dict) -> dict:
    import av

    path = row["video_path"]
    with av.open(path) as c:
        vstreams = [s for s in c.streams if s.type == "video"]
        astreams = [s for s in c.streams if s.type == "audio"]
        reasons = []
        if len(vstreams) != 1:
            reasons.append("not_exactly_one_video_stream")
        if len(astreams) != 1:
            reasons.append("not_exactly_one_audio_stream")
        fps = float(vstreams[0].average_rate) if vstreams else 0.0
        frames = int(vstreams[0].frames or 0) if vstreams else 0
        vdur = float(vstreams[0].duration * vstreams[0].time_base) if vstreams and vstreams[0].duration else 0.0
        adur = float(astreams[0].duration * astreams[0].time_base) if astreams and astreams[0].duration else 0.0
        container = float(c.duration * av.time_base) if c.duration else 0.0
    duration = float(row["duration"])
    if fps <= 0 or frames < 2:
        reasons.append("missing_frame_count_or_fps")
    if not 0.9 * duration <= (vdur or container) <= 1.1 * duration + 1:
        reasons.append("video_stream_duration_disagrees_with_annotation")
    if adur and not 0.9 * duration <= adur <= 1.1 * duration + 1:
        reasons.append("audio_stream_duration_disagrees_with_annotation")
    timestamps = [i / fps for i in range(frames)] if fps > 0 else []
    return dict(
        video_id=row["video_id"],
        duration=duration,
        source_average_fps=fps,
        full_unique_timestamps=frames,
        timestamps=timestamps,
        video_stream_duration=vdur,
        audio_stream_duration=adur,
        container_duration=container,
        quarantine_reasons=reasons,
        duration_mismatch_gt_2pct_or_1sec=bool(
            reasons and "video_stream_duration_disagrees_with_annotation" in reasons
        ),
        media_sha256=sha256(path),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--canonical", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    rows = read_jsonl(a.canonical)
    by_video = {r["video_id"]: r for r in rows}
    items = []

    def one(row):
        for _ in range(2):
            try:
                return probe(row)
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
        return dict(video_id=row["video_id"], error=error, quarantine_reasons=["source_probe_failed"])

    with ThreadPoolExecutor(a.workers) as pool:
        for i, item in enumerate(pool.map(one, by_video.values()), 1):
            items.append(item)
            if i % 100 == 0:
                print(f"audited {i}/{len(by_video)}", flush=True)
    excluded = {
        x["video_id"]
        for x in items
        if x.get("error") or x["quarantine_reasons"] or x.get("duration_mismatch_gt_2pct_or_1sec")
    }
    report = dict(
        status=STATUS,
        source_sha256=sha256(a.canonical),
        candidate_rows=len(rows),
        videos=len(by_video),
        items=items,
        quarantined_sample_ids=[r["sample_id"] for r in rows if r["video_id"] in excluded],
        caveat="Metadata-based timing audit (CFR assumption); semantic validation is separate",
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report), encoding="utf-8")
    print(json.dumps(dict(videos=len(by_video), quarantined_videos=len(excluded),
                          quarantined_samples=len(report["quarantined_sample_ids"]))))


if __name__ == "__main__":
    main()
