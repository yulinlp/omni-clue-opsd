"""Adapter for the human-verified OmniVideo-Test release."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .common import strip_choice_prefix
from .schema import CanonicalSample, ModalityEvidence


def iter_omnivideo_test(
    annotation: str | Path, *, video_root: str | Path
) -> Iterator[CanonicalSample]:
    with Path(annotation).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            start = float(row.get("start_time", 0.0))
            end = float(row.get("end_time", row["duration"]))
            if end <= start:
                raise ValueError(f"sample {row.get('question_id')} has an invalid clip range")
            video_id = str(row["video_id"])
            yield CanonicalSample(
                sample_id=str(row["question_id"]),
                benchmark="OmniVideo-Test",
                video_id=video_id,
                video_path=str(Path(video_root) / str(row["video_path"])),
                question=str(row["question"]),
                choices=[strip_choice_prefix(value) for value in row.get("options", [])],
                answer=str(row["answer"]).strip().upper(),
                # The distributed file is already the question-specific clip.
                duration=end - start,
                question_type=str(row.get("task") or row.get("subtask") or "unknown"),
                category_provenance="dataset",
                evidence=ModalityEvidence(provenance="not_released"),
                metadata={
                    "source_video_duration": float(row["duration"]),
                    "source_clip_start": start,
                    "source_clip_end": end,
                    "search_tag": row.get("search_tag"),
                    "language": row.get("language"),
                    "resolution": row.get("resolution"),
                    "task": row.get("task"),
                    "subtask": row.get("subtask"),
                    "human_verified": True,
                },
            )
