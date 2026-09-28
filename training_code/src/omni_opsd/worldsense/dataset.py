"""Flatten the WorldSense QA release into agent question records."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .media import MediaIndex
from .schema import QuestionRecord

TASK_KEYS = ("task0", "task1", "task2", "task3", "task4")


def _duration_from_string(value: object) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)", str(value or ""))
    return float(match.group(1)) if match else 0.0


def _parse_question_ids(question_ids: set[str]) -> dict[str, set[str]]:
    wanted: dict[str, set[str]] = {}
    for question_id in question_ids:
        if "::" not in question_id:
            raise ValueError(f"question_id must look like '<video_id>::taskN': {question_id!r}")
        video_id, task_key = question_id.split("::", 1)
        wanted.setdefault(video_id, set()).add(task_key)
    return wanted


def load_questions(
    qa_path: str | Path,
    video_root: str | Path,
    media_index: MediaIndex,
    *,
    question_ids: set[str] | None = None,
    video_ids: set[str] | None = None,
    task_types: set[str] | None = None,
    limit: int | None = None,
) -> list[QuestionRecord]:
    """Load and flatten every requested task.

    Filtering happens before media probing so that selecting a handful of
    questions never decodes the whole corpus.  Duration comes from the probed
    media index (authoritative), falling back to the ``video_duration`` string.
    """

    payload = json.loads(Path(qa_path).read_text(encoding="utf-8"))
    root = Path(video_root)
    wanted_by_video = _parse_question_ids(question_ids) if question_ids else None
    records: list[QuestionRecord] = []
    for video_id in sorted(payload):
        if video_ids is not None and video_id not in video_ids:
            continue
        if wanted_by_video is not None and video_id not in wanted_by_video:
            continue
        entry = payload[video_id] or {}
        wanted_tasks = wanted_by_video.get(video_id) if wanted_by_video is not None else None
        pending: list[tuple[int, str, dict[str, object]]] = []
        for task_index, task_key in enumerate(TASK_KEYS):
            if wanted_tasks is not None and task_key not in wanted_tasks:
                continue
            task = entry.get(task_key)
            if not task:
                continue
            if task_types is not None and str(task.get("task_type")) not in task_types:
                continue
            pending.append((task_index, task_key, task))
        if not pending:
            continue
        video_path = root / f"{video_id}.mp4"
        media = media_index.get(video_path)
        duration = media.duration_s or _duration_from_string(entry.get("video_duration"))
        for task_index, task_key, task in pending:
            records.append(
                QuestionRecord(
                    question_id=f"{video_id}::{task_key}",
                    video_id=video_id,
                    task_index=task_index,
                    task_domain=str(task.get("task_domain") or ""),
                    task_type=str(task.get("task_type") or ""),
                    question=str(task.get("question") or ""),
                    candidates=[str(item) for item in (task.get("candidates") or [])],
                    video_caption=str(entry.get("video_caption") or ""),
                    domain=str(entry.get("domain") or ""),
                    sub_category=str(entry.get("sub_category") or ""),
                    audio_class=[str(item) for item in (entry.get("audio_class") or [])],
                    duration_s=float(duration),
                    video_path=str(video_path),
                    answer_letter=str(task.get("answer") or "") or None,
                )
            )
            if limit is not None and len(records) >= limit:
                return records
    return records
