"""Dataset adapters for evidence-aware evaluation.

The adapter accepts the repository's CG-Bench POC manifest, NextGQA grounded
JSONL, and simple JSONL/CSV records without making any benchmark path a
compile-time constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
from typing import Any, Mapping

from .evidence import SampleEvidence, parse_sample_evidence


def _as_choices(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [str(value[key]) for key in sorted(value) if str(key).upper() in {"A", "B", "C", "D", "E", "F", "G"}]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return _as_choices(decoded)
    if isinstance(value, list):
        choices: list[str] = []
        for item in value:
            text = str(item)
            if len(text) >= 3 and text[0].isalpha() and text[1:3] in {". ", ") "}:
                text = text[3:]
            choices.append(text)
        return choices
    return [str(value)]


def _pick_video_path(row: Mapping[str, Any], video_root: str | Path | None) -> str:
    for key in ("video_path", "video", "path"):
        value = row.get(key)
        if value:
            path = Path(str(value))
            if path.is_file() or path.is_absolute() or video_root is None:
                return str(path)
            return str(Path(video_root) / path)
    for key in ("video_name", "video_id", "videoID", "key"):
        value = row.get(key)
        if value:
            name = str(value)
            root = Path(video_root) if video_root is not None else Path(".")
            candidates = [root / name, root / f"{name}.mp4"]
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate)
            return str(candidates[-1])
    raise ValueError(f"record has no video path/id: {row.keys()}")


@dataclass
class Sample:
    sample_id: str
    video_path: str
    question: str
    choices: list[str] = field(default_factory=list)
    answer: Any = None
    answer_text: str | None = None
    evidence: SampleEvidence = field(default_factory=SampleEvidence)
    question_type: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_multiple_choice(self) -> bool:
        return bool(self.choices)


def normalize_record(row: Mapping[str, Any], index: int, video_root: str | Path | None = None) -> Sample:
    sample_id = str(row.get("id", row.get("sample_id", row.get("qa_id", index))))
    question = str(row.get("question", row.get("question_text", "")))
    if not question:
        raise ValueError(f"sample {sample_id} has an empty question")
    choices = _as_choices(row.get("choices", row.get("options")))
    answer = row.get("answer", row.get("correct_answer", row.get("label")))
    answer_text = row.get("answer_text")
    duration = row.get("duration")
    if not isinstance(duration, (int, float)):
        duration = row.get("video_duration")
    evidence = parse_sample_evidence(row, video_duration=float(duration) if isinstance(duration, (int, float)) else None)
    metadata = dict(row)
    return Sample(
        sample_id=sample_id,
        video_path=_pick_video_path(row, video_root),
        question=question,
        choices=choices,
        answer=answer,
        answer_text=str(answer_text) if answer_text is not None else None,
        evidence=evidence,
        question_type=str(row.get("question_type", row.get("category", "unknown"))),
        metadata=metadata,
    )


def _read_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            if isinstance(value.get("data"), list):
                return value["data"]
            return [value]
        raise ValueError(f"unsupported JSON root in {path}")
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"unsupported dataset format: {path}")


def load_samples(
    path: str | Path,
    *,
    video_root: str | Path | None = None,
    max_samples: int | None = None,
    sample_ids: set[str] | None = None,
) -> list[Sample]:
    records = _read_records(Path(path).resolve())
    result: list[Sample] = []
    for index, row in enumerate(records):
        sample = normalize_record(row, index, video_root)
        if sample_ids is not None and sample.sample_id not in sample_ids:
            continue
        result.append(sample)
        if max_samples is not None and len(result) >= max_samples:
            break
    return result
