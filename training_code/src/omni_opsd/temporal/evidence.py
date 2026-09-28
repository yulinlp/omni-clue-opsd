"""Canonical temporal evidence data structures and pure planning helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True, order=True)
class EvidenceSpan:
    """A half-open source-video interval in seconds.

    The annotation is represented as ``[start, end]`` in JSON for readability,
    but all sampling helpers treat it as a continuous interval and place frame
    timestamps strictly inside it whenever possible.
    """

    start: float
    end: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.start) and math.isfinite(self.end)):
            raise ValueError(f"evidence timestamps must be finite: {self!r}")
        if self.end <= self.start:
            raise ValueError(f"evidence span must have positive length: {self!r}")

    @property
    def duration(self) -> float:
        return self.end - self.start

    def as_list(self) -> list[float]:
        return [float(self.start), float(self.end)]


@dataclass
class SampleEvidence:
    spans: list[EvidenceSpan] = field(default_factory=list)
    video_duration: float | None = None

    def __post_init__(self) -> None:
        if self.video_duration is not None and self.video_duration < 0:
            raise ValueError("video duration cannot be negative")

    @property
    def duration(self) -> float:
        return sum(span.duration for span in self.spans)

    @property
    def num_spans(self) -> int:
        return len(self.spans)

    def as_lists(self) -> list[list[float]]:
        return [span.as_list() for span in self.spans]


@dataclass(frozen=True)
class FrameInfo:
    """Frame timestamp metadata retained across crop/materialisation steps."""

    original_timestamp: float
    local_timestamp: float
    source: str = "global"

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_timestamp": float(self.original_timestamp),
            "local_timestamp": float(self.local_timestamp),
            "source": self.source,
        }


@dataclass(frozen=True)
class ROICrop:
    """A precomputed or manually supplied normalized pixel ROI."""

    frame_timestamp: float
    x1: float
    y1: float
    x2: float
    y2: float
    source: str = "precomputed"

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"ROI must have positive area: {self!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_timestamp": float(self.frame_timestamp),
            "x1": float(self.x1),
            "y1": float(self.y1),
            "x2": float(self.x2),
            "y2": float(self.y2),
            "source": self.source,
        }


def _as_float_pair(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        start = value.get("start", value.get("start_time", value.get("begin")))
        end = value.get("end", value.get("end_time", value.get("stop")))
        if start is None or end is None:
            return None
        return float(start), float(end)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def _candidate_evidence_values(record: Mapping[str, Any]) -> list[Any]:
    # Dataset records often carry both a broad question window
    # (evidence_start/evidence_end) and the actual sparse spans
    # (evidence_spans/clue_intervals).  These are aliases/metadata, not two
    # independent pieces of evidence.  Prefer the most specific list and only
    # fall back to a scalar interval when no list field is populated.
    # Prefer the flat visual alias emitted by canonical manifests.  The
    # structured ``evidence`` mapping may also contain independent audio and
    # subtitle windows and is therefore not itself a temporal span.
    for key in ("evidence_spans", "clue_intervals", "clues", "temporal_evidence", "evidence"):
        value = record.get(key)
        if value is not None:
            if key == "evidence" and isinstance(value, Mapping) and "visual" in value:
                value = value.get("visual")
            if isinstance(value, Mapping):
                return [value]
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                if value:
                    return list(value)
            else:
                return [value]
    if record.get("evidence_start") is not None and record.get("evidence_end") is not None:
        return [(record["evidence_start"], record["evidence_end"])]
    if record.get("start_time") is not None and record.get("end_time") is not None:
        return [(record["start_time"], record["end_time"])]
    return []


def merge_overlapping_spans(
    spans: Iterable[EvidenceSpan | Sequence[float]],
    *,
    touching_epsilon: float = 1e-9,
) -> list[EvidenceSpan]:
    """Sort and union overlapping (or numerically touching) spans."""

    normalized = [
        span if isinstance(span, EvidenceSpan) else EvidenceSpan(float(span[0]), float(span[1]))
        for span in spans
    ]
    if not normalized:
        return []
    ordered = sorted(normalized, key=lambda item: (item.start, item.end))
    merged: list[EvidenceSpan] = [ordered[0]]
    for current in ordered[1:]:
        previous = merged[-1]
        if current.start <= previous.end + touching_epsilon:
            merged[-1] = EvidenceSpan(previous.start, max(previous.end, current.end))
        else:
            merged.append(current)
    return merged


def clamp_span(span: EvidenceSpan | Sequence[float], video_duration: float) -> EvidenceSpan | None:
    """Clamp one span to ``[0, video_duration]`` and drop empty results."""

    if video_duration < 0:
        raise ValueError("video duration cannot be negative")
    if isinstance(span, EvidenceSpan):
        start_value, end_value = span.start, span.end
    else:
        start_value, end_value = float(span[0]), float(span[1])
    start = min(max(float(start_value), 0.0), float(video_duration))
    end = min(max(float(end_value), 0.0), float(video_duration))
    if end <= start:
        return None
    return EvidenceSpan(start, end)


def parse_sample_evidence(
    record: Mapping[str, Any],
    *,
    video_duration: float | None = None,
    merge: bool = False,
) -> SampleEvidence:
    """Read all supported annotation spellings into the canonical structure."""

    duration = video_duration
    if duration is None:
        for key in ("duration_seconds", "video_duration", "duration"):
            value = record.get(key)
            if isinstance(value, (int, float)):
                duration = float(value)
                break
    spans: list[EvidenceSpan] = []
    for value in _candidate_evidence_values(record):
        pair = _as_float_pair(value)
        if pair is None:
            continue
        try:
            span = EvidenceSpan(*pair)
        except ValueError:
            # A malformed/zero-width annotation must not silently become a
            # valid clip.  It is ignored here so non-evidence datasets can
            # still be loaded; strict golden mode checks for an empty result.
            continue
        if duration is not None:
            span = clamp_span(span, duration)
        if span is not None:
            spans.append(span)
    if merge:
        spans = merge_overlapping_spans(spans)
    else:
        spans = sorted(spans, key=lambda item: (item.start, item.end))
    return SampleEvidence(spans=spans, video_duration=duration)


def expand_evidence(
    spans: SampleEvidence | Iterable[EvidenceSpan | Sequence[float]],
    halo_before: float,
    halo_after: float | None = None,
    *,
    video_duration: float | None = None,
    merge: bool = True,
) -> SampleEvidence:
    """Expand each interval, clamp it, and optionally merge halo collisions."""

    if halo_before < 0 or (halo_after is not None and halo_after < 0):
        raise ValueError("evidence halo must be non-negative")
    after = halo_before if halo_after is None else halo_after
    if isinstance(spans, SampleEvidence):
        source = spans.spans
        duration = video_duration if video_duration is not None else spans.video_duration
    else:
        source = [s if isinstance(s, EvidenceSpan) else EvidenceSpan(float(s[0]), float(s[1])) for s in spans]
        duration = video_duration
    expanded: list[EvidenceSpan] = []
    for span in source:
        candidate = EvidenceSpan(span.start - halo_before, span.end + after)
        if duration is not None:
            candidate = clamp_span(candidate, duration)
        if candidate is not None:
            expanded.append(candidate)
    if merge:
        expanded = merge_overlapping_spans(expanded)
    return SampleEvidence(expanded, duration)


def crop_video_by_spans(
    video_path: str,
    spans: SampleEvidence | Iterable[EvidenceSpan | Sequence[float]],
    *,
    video_duration: float | None = None,
    source: str = "golden",
) -> list[dict[str, Any]]:
    """Create native-reader clip descriptors without re-encoding the video.

    Each non-contiguous span remains a separate descriptor.  The original
    boundaries are retained in metadata so a caller cannot accidentally treat
    the clips as one continuous timeline.
    """

    if isinstance(spans, SampleEvidence):
        evidence = spans
        duration = video_duration if video_duration is not None else evidence.video_duration
    else:
        evidence = SampleEvidence(
            [s if isinstance(s, EvidenceSpan) else EvidenceSpan(float(s[0]), float(s[1])) for s in spans],
            video_duration,
        )
        duration = video_duration
    result: list[dict[str, Any]] = []
    for index, span in enumerate(evidence.spans):
        clamped = clamp_span(span, duration) if duration is not None else span
        if clamped is None:
            continue
        result.append({
            "video": video_path,
            "video_start": float(clamped.start),
            "video_end": float(clamped.end),
            "span_index": index,
            "source": source,
            "original_span": clamped.as_list(),
        })
    return result


def _even_frame_count(duration: float, fps: float, *, min_frames: int = 2, max_frames: int | None = None) -> int:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if min_frames < 1:
        raise ValueError("min_frames must be positive")
    count = max(min_frames, int(math.ceil(duration * fps)))
    if count % 2:
        count += 1
    if max_frames is not None:
        if max_frames < min_frames:
            raise ValueError("max_frames must be >= min_frames")
        max_even = max_frames if max_frames % 2 == 0 else max_frames - 1
        count = min(count, max_even)
        count = max(count, min_frames)
    return count


def sample_frames_from_spans(
    spans: SampleEvidence | Iterable[EvidenceSpan | Sequence[float]],
    fps: float,
    *,
    native_fps: float | None = None,
    video_duration: float | None = None,
    max_frames: int | None = None,
    source: str = "golden",
) -> list[FrameInfo]:
    """Return deterministic frame timestamps while preserving absolute time.

    Requested FPS is capped at native FPS.  No timestamp duplication or frame
    interpolation is performed when a request exceeds the source rate.
    """

    if isinstance(spans, SampleEvidence):
        evidence = spans
        duration = video_duration if video_duration is not None else evidence.video_duration
    else:
        evidence = SampleEvidence(
            [s if isinstance(s, EvidenceSpan) else EvidenceSpan(float(s[0]), float(s[1])) for s in spans],
            video_duration,
        )
        duration = video_duration
    effective_fps = min(float(fps), float(native_fps)) if native_fps is not None else float(fps)
    if effective_fps <= 0:
        raise ValueError("effective FPS must be positive")
    frames: list[FrameInfo] = []
    for span in evidence.spans:
        clamped = clamp_span(span, duration) if duration is not None else span
        if clamped is None:
            continue
        count = _even_frame_count(clamped.duration, effective_fps, max_frames=max_frames)
        step = clamped.duration / count
        for index in range(count):
            timestamp = clamped.start + (index + 0.5) * step
            local = timestamp - clamped.start
            frames.append(FrameInfo(timestamp, local, source))
    return frames


def jitter_evidence(
    evidence: SampleEvidence,
    amount: float,
    mode: str,
    *,
    video_duration: float | None = None,
    seed: int = 0,
) -> SampleEvidence:
    """Apply E6 boundary perturbations without creating negative spans."""

    if amount < 0:
        raise ValueError("jitter amount must be non-negative")
    duration = video_duration if video_duration is not None else evidence.video_duration
    rng = random.Random(seed)
    result: list[EvidenceSpan] = []
    for span in evidence.spans:
        if mode == "expand":
            start, end = span.start - amount, span.end + amount
        elif mode == "shrink":
            shrink = min(amount, max(0.0, (span.duration - 1e-6) / 2.0))
            start, end = span.start + shrink, span.end - shrink
        elif mode == "shift-left":
            start, end = span.start - amount, span.end - amount
        elif mode == "shift-right":
            start, end = span.start + amount, span.end + amount
        elif mode == "random":
            left = rng.uniform(-amount, amount)
            right = rng.uniform(-amount, amount)
            start, end = span.start + left, span.end + right
            if end <= start:
                center = (start + end) / 2.0
                half = max(1e-6, span.duration / 4.0)
                start, end = center - half, center + half
        else:
            raise ValueError(f"unknown evidence jitter mode: {mode}")
        candidate = clamp_span((start, end), duration) if duration is not None else EvidenceSpan(start, end)
        if candidate is not None:
            result.append(candidate)
    return SampleEvidence(result, duration)


def evidence_duration_bucket(duration: float) -> str:
    if duration < 2:
        return "<2s"
    if duration < 5:
        return "2-5s"
    if duration < 15:
        return "5-15s"
    if duration < 30:
        return "15-30s"
    if duration < 60:
        return "30-60s"
    return ">60s"


def video_duration_bucket(duration: float) -> str:
    """Stable bins for full-video duration, separate from evidence bins."""

    if duration < 10:
        return "<10s"
    if duration < 30:
        return "10-30s"
    if duration < 60:
        return "30-60s"
    if duration < 90:
        return "60-90s"
    return ">=90s"


def evidence_ratio_bucket(ratio: float) -> str:
    """Bins used by the formal protocol for evidence/full-video ratio."""

    if ratio < 0.01:
        return "<1%"
    if ratio < 0.05:
        return "1-5%"
    if ratio < 0.10:
        return "5-10%"
    if ratio < 0.25:
        return "10-25%"
    return ">25%"
