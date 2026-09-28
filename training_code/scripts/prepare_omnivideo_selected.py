#!/usr/bin/env python3
"""Build an exact, ordered selected-ID canonical manifest; never resplit/filter it."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from omni_opsd.data.omnivideo_100k import iter_omnivideo_100k
from omni_opsd.data.swift_opsd import write_jsonl


def prepare(annotation: Path, sample_ids: Path, video_dir: Path, expected_count: int):
    ids = [s.strip() for s in sample_ids.read_text().splitlines() if s.strip()]
    if len(ids) != expected_count or len(set(ids)) != len(ids):
        raise ValueError(f"expected {expected_count} unique IDs, got {len(ids)} rows/{len(set(ids))} unique")
    wanted = set(ids)
    found = {}
    for sample in iter_omnivideo_100k(annotation, video_root=video_dir, sample_ids=wanted):
        row = sample.to_record()
        sid = row['sample_id']
        if sid in found:
            raise ValueError(f"duplicate annotation ID: {sid}")
        # --video-dir means the directory containing <video_id>.mp4 directly.
        media = (video_dir / f"{row['video_id']}.mp4").resolve()
        if not media.is_file() or media.stat().st_size == 0:
            raise ValueError(f"missing/empty video: {media}")
        row['video_path'] = str(media)
        if len(row['choices']) != 4 or row['answer'] not in {'A', 'B', 'C', 'D'}:
            raise ValueError(f"invalid four-choice answer: {sid}")
        if not row['duration'] or row['duration'] <= 0 or not row['evidence_spans']:
            raise ValueError(f"missing duration/evidence: {sid}")
        for start, end in row['evidence_spans']:
            if not 0 <= start < end <= row['duration']:
                raise ValueError(f"invalid evidence: {sid}")
        found[sid] = row
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"{len(missing)} IDs absent from annotation: {sorted(missing)[:10]}")
    return [found[sid] for sid in ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation', type=Path, required=True)
    parser.add_argument('--sample-ids', type=Path, required=True)
    parser.add_argument('--video-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-count', type=int, default=5000)
    args = parser.parse_args()
    rows = prepare(args.annotation, args.sample_ids, args.video_dir, args.expected_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / 'gap5000.canonical.jsonl'
    write_jsonl(output, rows)
    summary = {'rows': len(rows), 'unique_videos': len({r['video_id'] for r in rows}),
               'task_counts': dict(Counter(r['question_type'] for r in rows)),
               'id_order_preserved': True, 'all_media_present': True,
               'video_dir': str(args.video_dir.resolve()),
               'sources': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (args.annotation, args.sample_ids, output)}}
    (args.output_dir / 'selection_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
