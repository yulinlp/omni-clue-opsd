"""Dependency-light parsing helpers for Omni benchmark metadata."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Iterable

from omni_opsd.temporal.evidence import EvidenceSpan


def parse_timestamp(value: str | int | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    parts = [float(part) for part in str(value).strip().split(":")]
    if not parts or len(parts) > 3:
        raise ValueError(f"invalid timestamp: {value!r}")
    result = 0.0
    for part in parts:
        result = result * 60.0 + part
    return result


def parse_time_ranges(value: Any) -> list[EvidenceSpan]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        result: list[EvidenceSpan] = []
        for item in value:
            result.extend(parse_time_ranges(item))
        return sorted(result)
    if isinstance(value, dict):
        start = value.get("start", value.get("start_time"))
        end = value.get("end", value.get("end_time"))
        return [EvidenceSpan(parse_timestamp(start), parse_timestamp(end))]
    # Real releases use several equivalent spellings, including one bracketed
    # interval per line (``[00:10 - 00:14]``), unbracketed newline-separated
    # intervals, and semicolon-separated intervals.  Extract timestamp pairs
    # rather than relying on one particular delimiter.
    text = str(value).strip()
    timestamp = r"\d+(?::\d+(?:\.\d+)?){0,2}"
    result = []
    for match in re.finditer(
        rf"(?<![\d:])({timestamp})\s*[-–—]\s*({timestamp})(?![\d:])",
        text,
    ):
        start, end = parse_timestamp(match.group(1)), parse_timestamp(match.group(2))
        if end > start:
            result.append(EvidenceSpan(start, end))
    return sorted(result)


def strip_choice_prefix(value: Any) -> str:
    return re.sub(r"^\s*[A-Ha-h][\.)]\s*", "", str(value)).strip()


def write_jsonl(rows: Iterable[dict[str, Any]], output: str | Path) -> int:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count
