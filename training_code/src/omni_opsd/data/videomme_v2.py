"""Video-MME-v2 parquet adapter."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterator

from .common import strip_choice_prefix
from .schema import CanonicalSample, ModalityEvidence


def _split_options(value: object) -> list[str]:
    return [strip_choice_prefix(item) for item in re.split(r"\r?\n", str(value)) if item.strip()]


def iter_videomme_v2(annotation: str | Path, *, video_root: str | Path) -> Iterator[CanonicalSample]:
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("Video-MME-v2 parquet conversion requires omni-opsd[analysis]") from error
    frame = pd.read_parquet(annotation)
    for row in frame.to_dict(orient="records"):
        video_id = str(row["video_id"])
        candidates = [Path(video_root) / f"{video_id}.mp4", Path(video_root) / video_id / f"{video_id}.mp4"]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        yield CanonicalSample(
            sample_id=str(row["question_id"]),
            benchmark="Video-MME-v2",
            video_id=video_id,
            video_path=str(path),
            question=str(row["question"]),
            choices=_split_options(row.get("options", "")),
            answer=str(row.get("answer")) if row.get("answer") is not None else None,
            question_type=str(row.get("third_head", row.get("second_head", "unknown"))),
            category_provenance="dataset",
            evidence=ModalityEvidence(provenance="not_provided"),
            metadata={key: row.get(key) for key in ("url", "group_type", "group_structure", "level", "second_head", "third_head")},
        )
