"""OmniVideo-100K MCQ adapter with released designated evidence segments.

The release's cross-segment MCQ rows contain ``analysis.designated_segments``.
They are question-level evidence intervals produced by the dataset's automated
generation pipeline.  The adapter preserves that provenance instead of calling
them human-annotated oracle evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .common import parse_time_ranges, strip_choice_prefix
from .schema import CanonicalSample, ModalityEvidence


def iter_omnivideo_100k(
    annotation: str | Path,
    *,
    video_root: str | Path,
    evidence_only: bool = False,
    sample_ids: set[str] | None = None,
) -> Iterator[CanonicalSample]:
    with Path(annotation).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if sample_ids is not None and str(row.get("question_id")) not in sample_ids:
                continue
            video_id = str(row["video_id"])
            analysis = row.get("analysis") or {}
            spans = parse_time_ranges(analysis.get("designated_segments"))
            duration = float(row["duration"]) if row.get("duration") is not None else None
            if duration is not None:
                spans = [
                    type(span)(max(0.0, span.start), min(duration, span.end))
                    for span in spans
                    if min(duration, span.end) > max(0.0, span.start)
                ]
            if evidence_only and not spans:
                continue

            # Event-sequence-ordering examples deliberately omit the generic
            # fields.  The indexed representation keeps the events and answer
            # choices self-contained and is therefore the canonical variant.
            question = row.get("question") or row.get("question_indexed")
            options = row.get("options") or row.get("options_indexed") or []
            if not question:
                raise ValueError(f"sample {row.get('question_id')} has no question text")
            yield CanonicalSample(
                sample_id=str(row.get("question_id", f"{video_id}:unknown")),
                benchmark="OmniVideo-100K",
                video_id=video_id,
                video_path=str(Path(video_root) / str(row.get("video_path", f"{video_id}.mp4"))),
                question=str(question),
                choices=[strip_choice_prefix(value) for value in options],
                answer=str(row.get("answer")).strip().upper() if row.get("answer") is not None else None,
                duration=duration,
                question_type=str(row.get("task") or row.get("subtask") or "unknown"),
                category_provenance="dataset",
                evidence=ModalityEvidence(
                    visual=list(spans),
                    audio=list(spans),
                    provenance=(
                        "dataset_generated_designated_segments" if spans else "not_provided"
                    ),
                ),
                metadata={
                    "search_tag": row.get("search_tag"),
                    "language": row.get("language"),
                    "resolution": row.get("resolution"),
                    "task": row.get("task"),
                    "subtask": row.get("subtask"),
                    "connections": analysis.get("connections"),
                    "designated_segments_raw": analysis.get("designated_segments"),
                    "evidence_annotation_origin": "automated_dataset_generation_pipeline",
                    "event_representation": (
                        "indexed" if row.get("question") is None and row.get("question_indexed") else None
                    ),
                },
            )
