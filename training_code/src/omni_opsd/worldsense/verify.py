"""V1 sufficiency verification for candidate evidence intervals.

The teacher receives ONLY the candidate interval (video + synchronized audio)
and the question, and must answer correctly.  A passing interval is by
construction sufficient privileged evidence for OPSD distillation; the recorded
``p_true``/``margin``/``entropy`` support confidence-based routing later.

The gold answer is used here only to measure the interval, never in the prompt.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import AgentConfig
from .prompts import build_system_prompt
from .schema import QuestionRecord
from .views import ViewRequest, make_media_message

DEFAULT_LETTERS = ("A", "B", "C", "D")
ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-Ha-h])\s*</answer>")
LETTER_RE = re.compile(r"(?<![A-Za-z])([A-Ha-h])(?![A-Za-z])")


@dataclass(frozen=True)
class VerifierConfig:
    fps: float = 2.0
    max_pixels: int = 156_800
    modality: str = "av"
    max_new_tokens: int = 4
    letters: tuple[str, ...] = DEFAULT_LETTERS

    def as_dict(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "max_pixels": self.max_pixels,
            "modality": self.modality,
            "max_new_tokens": self.max_new_tokens,
            "letters": list(self.letters),
        }


@dataclass
class SufficiencyResult:
    question_id: str
    video_id: str
    interval: dict[str, Any]
    answer_text: str = ""
    answer_letter: str | None = None
    predicted_letter: str | None = None
    gold_letter: str | None = None
    correct: bool | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    p_true: float | None = None
    p_max: float | None = None
    margin: float | None = None
    entropy: float | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "video_id": self.video_id,
            "interval": dict(self.interval),
            "answer_text": self.answer_text,
            "answer_letter": self.answer_letter,
            "predicted_letter": self.predicted_letter,
            "gold_letter": self.gold_letter,
            "correct": self.correct,
            "probabilities": dict(self.probabilities),
            "p_true": self.p_true,
            "p_max": self.p_max,
            "margin": self.margin,
            "entropy": self.entropy,
            "error": self.error,
        }


def parse_option_letter(text: str, letters: Sequence[str] = DEFAULT_LETTERS) -> str | None:
    """Extract the chosen option letter from the teacher reply."""

    if not text:
        return None
    tagged = ANSWER_TAG_RE.search(text)
    if tagged:
        return tagged.group(1).upper()
    for match in LETTER_RE.finditer(text):
        candidate = match.group(1).upper()
        if candidate in letters:
            return candidate
    return None


def letter_probabilities(
    first_scores: Sequence[float] | None,
    letter_ids: dict[str, list[int]],
    letters: Sequence[str] = DEFAULT_LETTERS,
) -> dict[str, float]:
    """Renormalised probability over the option letters from next-token logits."""

    if first_scores is None:
        return {}
    log_sums: dict[str, float] = {}
    for letter in letters:
        ids = letter_ids.get(letter) or []
        values = [float(first_scores[index]) for index in ids if index < len(first_scores)]
        if not values:
            continue
        peak = max(values)
        log_sums[letter] = peak + math.log(sum(math.exp(value - peak) for value in values))
    if not log_sums:
        return {}
    peak = max(log_sums.values())
    weights = {letter: math.exp(value - peak) for letter, value in log_sums.items()}
    total = sum(weights.values())
    return {letter: weight / total for letter, weight in weights.items()}


def _entropy(probabilities: dict[str, float]) -> float | None:
    values = [value for value in probabilities.values() if value > 0]
    if not values:
        return None
    return float(-sum(value * math.log(value) for value in values))


def build_sufficiency_messages(
    record: QuestionRecord,
    interval: dict[str, Any],
    verifier: VerifierConfig,
    agent_config: AgentConfig | None = None,
) -> list[dict[str, Any]]:
    """Interval-only teacher prompt; the gold answer never appears."""

    options = "\n".join(record.candidates)
    system_prompt = (
        "You are answering a multiple-choice question using ONLY the short clip "
        "(video with synchronized audio) provided below. The clip is a segment cut "
        "from a longer video and may or may not contain the decisive evidence.\n"
        "Rules:\n"
        "- Use only what you can see and hear in the clip.\n"
        "- Never assume the answer from outside knowledge.\n"
        "- Reply with exactly one option letter and no explanation."
    )
    view = ViewRequest(
        start=float(interval["start"]),
        end=float(interval["end"]),
        fps=verifier.fps,
        max_pixels=verifier.max_pixels,
        modality=verifier.modality,
        focus="decide the correct option from this clip alone",
    )
    question_text = (
        f"Question: {record.question}\n"
        f"Options:\n{options}\n\n"
        f"Task type: {record.task_type}\n"
        "The correct option letter is:"
    )
    return [
        {"role": "system", "content": system_prompt},
        make_media_message(view, record.video_path, record.duration_s),
        {"role": "user", "content": question_text},
    ]


def verify_interval(
    client: Any,
    record: QuestionRecord,
    interval: dict[str, Any],
    verifier: VerifierConfig,
    agent_config: AgentConfig | None = None,
) -> SufficiencyResult:
    """Run the interval-only sufficiency check for one candidate interval."""

    result = SufficiencyResult(
        question_id=record.question_id,
        video_id=record.video_id,
        interval={"start": float(interval["start"]), "end": float(interval["end"])},
        gold_letter=(record.answer_letter or "").upper() or None,
    )
    messages = build_sufficiency_messages(record, interval, verifier, agent_config)
    try:
        text, scores = client.generate_with_scores(messages, max_new_tokens=verifier.max_new_tokens)
    except Exception as exc:  # noqa: BLE001 - keep the batch running
        result.error = f"{type(exc).__name__}: {exc}"
        return result
    result.answer_text = text
    result.answer_letter = parse_option_letter(text, verifier.letters)
    result.predicted_letter = result.answer_letter
    if result.gold_letter and result.answer_letter:
        result.correct = result.answer_letter == result.gold_letter
    if scores is not None:
        if isinstance(scores, dict):
            probabilities = {
                str(letter).upper(): float(value)
                for letter, value in scores.items()
                if str(letter).upper() in verifier.letters
            }
        else:
            try:
                letter_ids = client.option_token_ids(list(verifier.letters))
            except Exception:
                letter_ids = {}
            probabilities = letter_probabilities(scores, letter_ids, verifier.letters)
        if probabilities:
            result.probabilities = probabilities
            result.p_max = max(probabilities.values())
            if not result.predicted_letter:
                result.predicted_letter = max(probabilities, key=probabilities.get)
            if result.gold_letter and result.gold_letter in probabilities:
                result.p_true = probabilities[result.gold_letter]
                others = [value for letter, value in probabilities.items() if letter != result.gold_letter]
                result.margin = result.p_true - (max(others) if others else 0.0)
            result.entropy = _entropy(probabilities)
    return result


def verify_question(
    client: Any,
    record: QuestionRecord,
    intervals: list[dict[str, Any]],
    verifier: VerifierConfig,
    agent_config: AgentConfig | None = None,
) -> dict[str, Any]:
    """Verify every candidate interval and summarise the question."""

    results = [
        verify_interval(client, record, interval, verifier, agent_config)
        for interval in intervals
    ]
    correct_flags = [bool(item.correct) for item in results if item.correct is not None]
    summary: dict[str, Any] = {
        "question_id": record.question_id,
        "video_id": record.video_id,
        "task_type": record.task_type,
        "gold_letter": (record.answer_letter or "").upper() or None,
        "n_intervals": len(results),
        "any_correct": any(correct_flags) if correct_flags else None,
        "n_correct": sum(1 for flag in correct_flags if flag),
        "results": [item.as_dict() for item in results],
        "verifier": verifier.as_dict(),
    }
    passing = [item for item in results if item.correct]
    if passing:
        best = max(
            passing,
            key=lambda item: (item.p_true if item.p_true is not None else -1.0, item.p_max or -1.0),
        )
        summary["best_interval"] = best.interval
        summary["best_p_true"] = best.p_true
        summary["best_margin"] = best.margin
    return summary
