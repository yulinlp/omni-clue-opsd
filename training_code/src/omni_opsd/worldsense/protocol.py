"""Strict parsing of the single-JSON controller protocol."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterator

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
ALLOWED_ACTIONS = ("inspect", "get_media_info", "submit")


@dataclass(frozen=True)
class Action:
    action: str
    observation: str
    payload: dict[str, Any]
    raw: str = ""


def _iter_json_objects(text: str) -> Iterator[str]:
    depth = 0
    start: int | None = None
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start : index + 1]
                    start = None


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    candidates = [match.group(1).strip() for match in FENCE_RE.finditer(text)]
    candidates.append(text.strip())
    for candidate in candidates:
        for blob in _iter_json_objects(candidate):
            try:
                parsed = json.loads(blob)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def parse_action(text: str) -> tuple[Action | None, str | None]:
    parsed = extract_json_object(text)
    if parsed is None:
        return None, (
            'your reply is not a JSON object. Reply with exactly one JSON object, '
            'e.g. {"observation": "...", "action": "inspect", ...}'
        )
    action = str(parsed.get("action") or "").strip()
    if action not in ALLOWED_ACTIONS:
        return None, f'unknown action {action!r}; allowed actions: {", ".join(ALLOWED_ACTIONS)}'
    observation = str(parsed.get("observation") or "").strip()
    if not observation:
        return None, 'every reply must include a non-empty "observation" field'
    return Action(action=action, observation=observation, payload=parsed, raw=text), None
