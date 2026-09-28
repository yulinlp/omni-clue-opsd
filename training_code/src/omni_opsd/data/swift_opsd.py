"""Build leak-resistant ms-swift OPSD rows from canonical video samples."""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from omni_opsd.data.common import parse_timestamp


def video_split(
    rows: Sequence[dict[str, Any]], *, val_video_count: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split deterministically by ``video_id`` rather than by QA sample."""

    video_ids = sorted({str(row["video_id"]) for row in rows})
    if not 0 < val_video_count < len(video_ids):
        raise ValueError("val_video_count must be between 1 and the number of videos minus 1")
    ranked = sorted(
        video_ids,
        key=lambda video_id: hashlib.sha256(f"{seed}\0{video_id}".encode()).hexdigest(),
    )
    val_ids = set(ranked[:val_video_count])
    train = [row for row in rows if str(row["video_id"]) not in val_ids]
    val = [row for row in rows if str(row["video_id"]) in val_ids]
    return train, val


def prompt_for(row: dict[str, Any]) -> str:
    choices = list(row.get("choices") or [])
    if not choices:
        raise ValueError(f"sample {row.get('sample_id')} has no choices")
    if len(choices) > 26:
        raise ValueError("only A-Z multiple-choice labels are supported")
    options = "\n".join(f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices))
    return (
        f"Question: {row['question']}\n"
        f"Options:\n{options}\n"
        "Answer with exactly one option letter."
    )


def _metadata_time_spans(row: dict[str, Any], *, point_radius: float) -> list[list[float]]:
    value = (row.get("metadata") or {}).get("time_reference")
    if not value:
        return []
    result: list[list[float]] = []
    for segment in re.split(r"\s*[,;|]\s*", str(value).strip()):
        match = re.match(r"^\s*(.+?)\s*[-–—]\s*(.+?)\s*$", segment)
        if not match:
            continue
        start = parse_timestamp(match.group(1))
        end = parse_timestamp(match.group(2))
        if end == start:
            start -= point_radius
            end += point_radius
        result.append([start, end])
    return result


def evidence_spans(row: dict[str, Any], *, point_radius: float = 1.0) -> list[list[float]]:
    raw_spans = _metadata_time_spans(row, point_radius=point_radius)
    if not raw_spans:
        raw_spans = row.get("evidence_spans")
    if not raw_spans:
        raw_spans = (row.get("evidence") or {}).get("visual")
    duration = float(row["duration"])
    spans: list[list[float]] = []
    for raw in raw_spans or []:
        start, end = float(raw[0]), float(raw[1])
        start = max(0.0, min(duration, start))
        end = max(0.0, min(duration, end))
        if end > start:
            spans.append([start, end])
    if not spans:
        raise ValueError(f"sample {row.get('sample_id')} has no non-empty evidence spans")
    return spans


def _allocate_frame_caps(spans: Sequence[Sequence[float]], max_frames: int) -> list[int]:
    """Allocate a fixed aggregate teacher budget across temporal spans."""

    if max_frames < len(spans):
        raise ValueError("max_frames must be at least the number of evidence spans")
    durations = [float(end) - float(start) for start, end in spans]
    total = sum(durations)
    raw = [max_frames * duration / total for duration in durations]
    caps = [max(1, math.floor(value)) for value in raw]
    while sum(caps) > max_frames:
        candidates = [index for index, value in enumerate(caps) if value > 1]
        index = min(candidates, key=lambda item: raw[item] - caps[item])
        caps[index] -= 1
    for index in sorted(range(len(caps)), key=lambda item: raw[item] - caps[item], reverse=True):
        if sum(caps) == max_frames:
            break
        caps[index] += 1
    return caps


def _video_spec(
    path: str,
    start: float,
    end: float,
    *,
    fps: float,
    max_frames: int,
    min_pixels: int,
    max_pixels: int,
) -> dict[str, Any]:
    return {
        "video": str(path),
        "video_start": float(start),
        "video_end": float(end),
        "fps": float(fps),
        "max_frames": int(max_frames),
        "min_pixels": int(min_pixels),
        "max_pixels": int(max_pixels),
    }


def swift_opsd_row(
    row: dict[str, Any],
    *,
    fps: float = 2.0,
    max_frames: int = 256,
    min_pixels: int = 3136,
    max_pixels: int = 180_000,
    use_audio_in_video: bool = False,
) -> dict[str, Any]:
    """Return one student-full/teacher-clue row without the answer label."""

    duration = float(row["duration"])
    if duration <= 0:
        raise ValueError(f"sample {row.get('sample_id')} has invalid duration")
    spans = evidence_spans(row)
    frame_caps = _allocate_frame_caps(spans, max_frames)
    prompt = prompt_for(row)
    teacher_videos = [
        _video_spec(
            row["video_path"],
            start,
            end,
            fps=fps,
            max_frames=cap,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        for (start, end), cap in zip(spans, frame_caps)
    ]
    return {
        "messages": [{"role": "user", "content": "<video>\n" + prompt}],
        "videos": [
            _video_spec(
                row["video_path"],
                0.0,
                duration,
                fps=fps,
                max_frames=max_frames,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        ],
        "teacher_prompt": "\n".join(["<video>"] * len(spans)) + "\n" + prompt,
        "teacher_videos": teacher_videos,
        "prompt_id": str(row["sample_id"]),
        "case_id": str(row["sample_id"]),
        "video_id": str(row["video_id"]),
        "clue_intervals": spans,
        "sampling_contract": {
            "fps": float(fps),
            "min_pixels": int(min_pixels),
            "max_frames_per_view": int(max_frames),
            "max_pixels": int(max_pixels),
            "student_view": "full-video-uniform",
            "teacher_view": "dataset-time-reference",
            "teacher_frame_cap_total": sum(frame_caps),
            # Full raw audio for VideoOdyssey can span 1-4 hours and is not
            # bounded by the visual frame cap.  Callers must opt into it and
            # perform a context-length preflight before a full training run.
            "use_audio_in_video": bool(use_audio_in_video),
            "answer_label_in_model_row": False,
        },
    }


def _gold_answer(row: dict[str, Any]) -> str:
    """Return a validated multiple-choice answer letter."""

    answer = str(row.get("answer", "")).strip().upper()
    choices = list(row.get("choices") or [])
    if not re.fullmatch(r"[A-Z]", answer):
        raise ValueError(f"sample {row.get('sample_id')} has invalid answer {answer!r}")
    if not choices or ord(answer) - ord("A") >= len(choices):
        raise ValueError(
            f"sample {row.get('sample_id')} answer {answer!r} is outside its choices"
        )
    return answer


def _matrix_base(
    canonical_row: dict[str, Any], materialized_opsd_row: dict[str, Any]
) -> dict[str, Any]:
    """Copy the common full-video student input for post-training controls."""

    expected_id = str(canonical_row["sample_id"])
    actual_id = str(materialized_opsd_row["case_id"])
    if expected_id != actual_id:
        raise ValueError(f"case mismatch: canonical={expected_id}, materialized={actual_id}")
    videos = deepcopy(materialized_opsd_row.get("videos") or [])
    messages = deepcopy(materialized_opsd_row.get("messages") or [])
    if len(videos) != 1 or len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError(f"sample {expected_id} lacks one full-video user input")
    return {
        "messages": messages,
        "videos": videos,
        "prompt_id": expected_id,
        "case_id": expected_id,
        "video_id": str(canonical_row["video_id"]),
        "sampling_contract": deepcopy(materialized_opsd_row["sampling_contract"]),
    }


def swift_training_matrix_rows(
    canonical_row: dict[str, Any], materialized_opsd_row: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Build comparable SFT, GRPO, OPSD, and Clue-OPSD rows.

    All four arms share the exact same materialized full-video student input.
    Only the supervision channel differs:

    * SFT: the gold option is the assistant target;
    * GRPO: the gold option is kept as ``solution`` for the reward plugin;
    * OPSD: an EMA teacher sees the full video plus the gold option;
    * Clue-OPSD: an EMA teacher sees clue video only and never sees the answer.
    """

    answer = _gold_answer(canonical_row)
    base = _matrix_base(canonical_row, materialized_opsd_row)

    sft = deepcopy(base)
    sft["messages"].append({"role": "assistant", "content": answer})
    sft["sampling_contract"]["answer_label_in_model_row"] = True
    sft["experiment_arm"] = "sft"
    sft["supervision_contract"] = {
        "kind": "gold-answer-teacher-forcing",
        "gold_answer_in_student_target": True,
        "gold_answer_in_reward": False,
        "gold_answer_in_teacher_prompt": False,
        "teacher_view": None,
    }

    grpo = deepcopy(base)
    grpo["solution"] = answer
    grpo["experiment_arm"] = "grpo"
    grpo["supervision_contract"] = {
        "kind": "gold-answer-sequence-reward",
        "gold_answer_in_student_target": False,
        "gold_answer_in_reward": True,
        "gold_answer_in_teacher_prompt": False,
        "teacher_view": None,
    }

    opsd = deepcopy(base)
    student_prompt = str(opsd["messages"][0]["content"])
    opsd["teacher_prompt"] = (
        f"{student_prompt}\n\n"
        f"Privileged information: the correct option is {answer}. "
        "Return exactly that option letter for the original question."
    )
    opsd["teacher_videos"] = deepcopy(opsd["videos"])
    opsd["sampling_contract"]["teacher_view"] = "full-video-uniform"
    opsd["experiment_arm"] = "opsd"
    opsd["supervision_contract"] = {
        "kind": "answer-privileged-on-policy-self-distillation",
        "gold_answer_in_student_target": False,
        "gold_answer_in_reward": False,
        "gold_answer_in_teacher_prompt": True,
        "teacher_view": "full-video-uniform",
    }

    clue_opsd = deepcopy(materialized_opsd_row)
    clue_opsd.pop("answer", None)
    clue_opsd.pop("solution", None)
    clue_opsd["experiment_arm"] = "clue_opsd"
    clue_opsd["supervision_contract"] = {
        "kind": "clue-privileged-on-policy-self-distillation",
        "gold_answer_in_student_target": False,
        "gold_answer_in_reward": False,
        "gold_answer_in_teacher_prompt": False,
        "teacher_view": "dataset-clue-interval",
    }
    return {"sft": sft, "grpo": grpo, "opsd": opsd, "clue_opsd": clue_opsd}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
