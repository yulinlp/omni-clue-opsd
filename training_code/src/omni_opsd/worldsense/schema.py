"""Typed records for the WorldSense evidence-localization agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class QuestionRecord:
    """One flattened WorldSense question (video_id + taskN)."""

    question_id: str
    video_id: str
    task_index: int
    task_domain: str
    task_type: str
    question: str
    candidates: list[str]
    video_caption: str
    domain: str
    sub_category: str
    audio_class: list[str]
    duration_s: float
    video_path: str
    answer_letter: str | None = None

    @property
    def needs_audio(self) -> bool:
        return bool(self.audio_class)

    def prompt_metadata(self) -> dict[str, Any]:
        """The metadata block shown to the agent (never includes the answer)."""

        return {
            "question_id": self.question_id,
            "video_id": self.video_id,
            "task_index": self.task_index,
            "video_duration_seconds": round(self.duration_s, 3),
            "domain": self.domain,
            "sub_category": self.sub_category,
            "audio_class": list(self.audio_class),
            "task_domain": self.task_domain,
            "task_type": self.task_type,
            "video_synopsis": self.video_caption,
            "question": self.question,
            "options": list(self.candidates),
        }


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    observation: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": float(self.start),
            "end": float(self.end),
            "observation": self.observation,
        }

    @property
    def duration(self) -> float:
        return float(self.end) - float(self.start)


@dataclass
class BootstrapResult:
    """Extra context a backend may prepare before the tool loop starts.

    The API backend uses it to run the two-stage "timestamped caption ->
    inspect plan" flow: the caption is injected as a user message and the
    full-video survey is skipped because the caption already summarises it.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    skip_survey: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTraceEvent:
    turn: int
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"turn": int(self.turn), "kind": self.kind, **self.payload}
