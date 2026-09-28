"""Answer-free question taxonomy used for formal compute ablations.

The classifier is deliberately small and deterministic.  It only consumes the
question and the option text, never the answer, evidence, prediction, or any
model output.  Its labels are therefore *inferred* categories, not benchmark
ground truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Iterable


QUESTION_CATEGORIES = (
    "temporal_order",
    "before_after",
    "action",
    "counting",
    "state_change",
    "causal",
    "object",
    "attribute",
    "OCR",
    "dialogue",
    "other",
)


@dataclass(frozen=True)
class QuestionCategory:
    category: str
    source: str = "inferred_rule"
    confidence: float = 0.0
    matched_rules: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9? ]", " ", text.lower())).strip()


def _contains(text: str, phrases: Iterable[str]) -> list[str]:
    return [phrase for phrase in phrases if phrase in text]


def classify_question(question: str, choices: Iterable[str] | None = None) -> QuestionCategory:
    """Classify a question using only text visible before inference.

    The ordering is intentional: e.g. ``why ... after ...`` is causal, while
    an explicit ``what happened after ...`` is a before/after question.  The
    result is conservative; ambiguous cases fall back to ``other`` rather
    than pretending to be a ground-truth label.
    """

    question_text = _normalise(question)
    option_text = " ".join(_normalise(str(item)) for item in (choices or ()))
    text = f"{question_text} {option_text}".strip()

    rules: list[tuple[str, tuple[str, ...], str, float]] = [
        (
            "ocr",
            (
                "read", "written", "text", "letter", "word", "sign", "logo",
                "label", "title", "caption", "screen", "display", "number on",
                "what does it say", "what is written",
            ),
            "OCR",
            0.95,
        ),
        (
            "counting",
            (
                "how many", "how much", "number of", "count", "counted",
                "how often", "how many times", "once", "twice",
            ),
            "counting",
            0.95,
        ),
        (
            "causal",
            ("why ", "why?", "reason", "because", "purpose", "so that", "in order to"),
            "causal",
            0.93,
        ),
        (
            "before_after",
            (
                " before ", " after ", "before?", "after?", "what happens next",
                "what did .* do next", "following", "prior to", "subsequent",
            ),
            "before_after",
            0.92,
        ),
        (
            "temporal_order",
            (
                "which happened first", "what happened first", "first or",
                "in what order", "sequence", "order of events", "earlier or later",
                "at the beginning", "in the middle", "towards the end",
            ),
            "temporal_order",
            0.88,
        ),
        (
            "dialogue",
            (
                "say", "said", "tell", "told", "ask", "asked", "talk", "talking",
                "speak", "speaking", "conversation", "dialogue", "sound", "hear",
                "voice", "speaker", "music", "sing", "shout", "noise",
            ),
            "dialogue",
            0.88,
        ),
        (
            "state_change",
            (
                "change", "become", "turn into", "start", "stop", "no longer",
                "end up", "before and after", "put on", "take off", "open", "close",
                "appear", "disappear",
            ),
            "state_change",
            0.82,
        ),
        (
            "attribute",
            (
                "color", "colour", "wear", "wearing", "shirt", "dress", "size",
                "shape", "material", "appearance", "look like", "which one is",
                "bigger", "smaller", "longer", "shorter", "same color",
            ),
            "attribute",
            0.84,
        ),
        (
            "object",
            (
                "what is", "which object", "which item", "which thing", "what object",
                "what item", "who is", "where is", "where are", "which person",
                "which animal", "what kind of",
            ),
            "object",
            0.70,
        ),
        (
            "action",
            (
                "what did", "what does", "what do", "what is .* doing", "how did",
                "how does", "how are", "what happened", "what are .* doing",
            ),
            "action",
            0.62,
        ),
    ]

    for rule_name, phrases, category, confidence in rules:
        matched = _contains(text, phrases)
        # A few entries above use a regex-like ``.*`` marker.  Keep matching
        # answer-free and lightweight without making the complete classifier a
        # regex engine.
        if not matched:
            for phrase in phrases:
                if ".*" in phrase:
                    prefix, suffix = phrase.split(".*", 1)
                    if prefix in text and suffix in text:
                        matched.append(phrase)
        if matched:
            return QuestionCategory(category, "inferred_rule", confidence, tuple(matched))
    return QuestionCategory("other", "inferred_rule", 0.25, ())


def classify_sample(question: str, choices: Iterable[str] | None = None) -> dict[str, Any]:
    """JSON-friendly answer-free classifier record."""

    return classify_question(question, choices).as_dict()
