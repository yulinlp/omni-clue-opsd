"""VideoOdyssey-AV adapter with dataset-provided temporal references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .common import parse_time_ranges, strip_choice_prefix
from .schema import CanonicalSample, ModalityEvidence


def iter_video_odyssey(
    annotation: str | Path,
    *,
    video_root: str | Path,
    subtitle_root: str | Path | None = None,
) -> Iterator[CanonicalSample]:
    records = json.loads(Path(annotation).read_text(encoding="utf-8"))
    for video in records:
        video_id = str(video["video_id"])
        duration = float(video["duration_minutes"]) * 60.0
        video_path = Path(video_root) / str(video.get("video_path", f"{video_id}.mp4"))
        subtitle = Path(subtitle_root) / f"{video_id}.srt" if subtitle_root else None
        for question in video.get("questions", []):
            spans = parse_time_ranges(question.get("time_reference"))
            yield CanonicalSample(
                sample_id=str(question["question_id"]),
                benchmark="VideoOdyssey-AV",
                video_id=video_id,
                video_path=str(video_path),
                question=str(question["question"]),
                choices=[strip_choice_prefix(choice) for choice in question.get("options", [])],
                answer=str(question.get("answer")) if question.get("answer") is not None else None,
                duration=duration,
                question_type="|".join(map(str, question.get("question_type", []))) or "unknown",
                category_provenance="dataset",
                subtitle_path=str(subtitle) if subtitle is not None else None,
                evidence=ModalityEvidence(
                    visual=spans,
                    audio=spans,
                    subtitle=spans,
                    provenance="dataset_time_reference",
                ),
                metadata={
                    "video_category": video.get("video_category"),
                    "audio_type": question.get("audio_type", []),
                    "time_reference": question.get("time_reference"),
                },
            )
