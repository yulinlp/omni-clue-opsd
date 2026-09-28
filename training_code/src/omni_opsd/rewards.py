"""Small, dependency-free reward helpers for post-training experiments."""

from __future__ import annotations

import re
from typing import Any


def completion_text(value: Any) -> str:
    """Normalize the completion representations used by ms-swift."""

    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("content", ""))
    if isinstance(value, list):
        for item in reversed(value):
            if isinstance(item, dict) and item.get("role") == "assistant":
                return str(item.get("content", ""))
        if value:
            return completion_text(value[-1])
    return str(value)


def extract_mcq_answer(value: Any) -> str | None:
    """Extract one A-Z option letter without matching letters inside words."""

    text = completion_text(value).strip().upper()
    patterns = (
        r"<ANSWER>\s*[\(\[]?([A-Z])[\)\]]?\s*</ANSWER>",
        r"(?:FINAL\s+ANSWER|ANSWER|OPTION|CHOICE)\s*(?:IS|:|=)?\s*[\(\[]?([A-Z])[\)\]]?",
        r"^\s*[\(\[]?([A-Z])[\)\].:]?\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return match.group(1)
    return None


def mcq_exact_rewards(completions: list[Any], solutions: list[Any]) -> list[float]:
    if len(completions) != len(solutions):
        raise ValueError("completion and solution counts differ")
    return [
        float(extract_mcq_answer(completion) == str(solution).strip().upper())
        for completion, solution in zip(completions, solutions)
    ]
