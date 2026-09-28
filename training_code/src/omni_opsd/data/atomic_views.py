"""Paired OmniVideo atomic-view manifests with explicit modality contracts."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Iterable, Sequence

from omni_opsd.data.swift_opsd import prompt_for


ATOMIC_ARMS = (
    "A0_full_av",
    "A1_evidence_v_full_a",
    "A2_full_v_evidence_a",
    "A3_evidence_av",
    "A4_uniform_matched_av",
    "A5_full_av_timestamp",
)


def merge_spans(
    spans: Iterable[Sequence[float]], *, duration: float
) -> list[list[float]]:
    """Clamp, sort and merge overlapping intervals."""

    cleaned: list[list[float]] = []
    for raw in spans:
        start = max(0.0, min(float(duration), float(raw[0])))
        end = max(0.0, min(float(duration), float(raw[1])))
        if end > start:
            cleaned.append([start, end])
    cleaned.sort()
    merged: list[list[float]] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    if not merged:
        raise ValueError("no non-empty evidence interval remains")
    return merged


def uniform_matched_spans(
    evidence_spans: Sequence[Sequence[float]], *, duration: float
) -> list[list[float]]:
    """Spread the same ordered window lengths uniformly over the timeline."""

    lengths = [float(end) - float(start) for start, end in evidence_spans]
    occupied = sum(lengths)
    if occupied > duration + 1e-6:
        raise ValueError("evidence duration exceeds the source duration")
    gap = max(0.0, duration - occupied) / (len(lengths) + 1)
    cursor = gap
    result: list[list[float]] = []
    for length in lengths:
        result.append([cursor, cursor + length])
        cursor += length + gap
    return result


def _video_spec(
    path: str,
    start: float,
    end: float,
    *,
    fps: float,
    min_pixels: int,
    max_pixels: int,
    max_frames: int,
) -> dict[str, Any]:
    return {
        "video": path,
        "video_start": float(start),
        "video_end": float(end),
        "fps": float(fps),
        "min_pixels": int(min_pixels),
        "max_pixels": int(max_pixels),
        "max_frames": int(max_frames),
    }


def _audio_spec(path: str, start: float, end: float) -> dict[str, Any]:
    return {
        "audio": path,
        "audio_start": float(start),
        "audio_end": float(end),
    }


def _timestamp_text(spans: Sequence[Sequence[float]]) -> str:
    rendered = "; ".join(f"{start:.3f}-{end:.3f} seconds" for start, end in spans)
    return f"Dataset-provided relevant intervals: {rendered}."


def _user_message(*, video_count: int, audio_count: int, prompt: str) -> dict[str, str]:
    tags = ["<video>"] * video_count + ["<audio>"] * audio_count
    return {"role": "user", "content": "\n".join([*tags, prompt])}


def _spans_equal(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        math.isclose(float(a), float(c), abs_tol=1e-6)
        and math.isclose(float(b), float(d), abs_tol=1e-6)
        for (a, b), (c, d) in zip(left, right)
    )


def build_atomic_views(
    row: dict[str, Any],
    full_proxy: dict[str, Any],
    *,
    fps: float,
    min_pixels: int,
    max_pixels: int,
    max_frames: int,
) -> dict[str, dict[str, Any]]:
    """Build A0--A5 with identical question/checkpoint-facing metadata."""

    if row.get("answer") not in (None, ""):
        raise ValueError(f"answer leaked into atomic row {row.get('sample_id')}")
    duration = float(row["duration"])
    if duration <= 0:
        raise ValueError("duration must be positive")
    source = str(row["video_path"])
    if str(full_proxy["source_path"]) != source:
        raise ValueError("full proxy audit does not match canonical video path")
    if not full_proxy.get("endpoint_coverage_pass"):
        raise ValueError("full proxy failed endpoint coverage")
    visual_spans = merge_spans((row.get("evidence") or {}).get("visual") or [], duration=duration)
    audio_spans = merge_spans((row.get("evidence") or {}).get("audio") or [], duration=duration)
    if not _spans_equal(visual_spans, audio_spans):
        raise ValueError("A3/A4 require aligned dataset visual and audio evidence spans")
    matched_spans = uniform_matched_spans(visual_spans, duration=duration)
    evidence_duration = sum(end - start for start, end in visual_spans)
    proxy_path = str(full_proxy["proxy_path"])
    full_video = _video_spec(
        proxy_path,
        0.0,
        duration,
        fps=fps,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_frames=max_frames,
    )
    evidence_videos = [
        _video_spec(
            source,
            start,
            end,
            fps=fps,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_frames=max_frames,
        )
        for start, end in visual_spans
    ]
    matched_videos = [
        _video_spec(
            source,
            start,
            end,
            fps=fps,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_frames=max_frames,
        )
        for start, end in matched_spans
    ]
    full_audio = [_audio_spec(source, 0.0, duration)]
    evidence_audios = [_audio_spec(source, start, end) for start, end in audio_spans]
    prompt = prompt_for(row)

    common: dict[str, Any] = {
        "prompt_id": str(row["sample_id"]),
        "case_id": str(row["sample_id"]),
        "video_id": str(row["video_id"]),
        "question_type": str(row["question_type"]),
        "benchmark": str(row.get("benchmark") or "OmniVideo-100K"),
    }

    def make(
        arm: str,
        *,
        videos: list[dict[str, Any]],
        audios: list[dict[str, Any]] | None,
        use_audio_in_video: bool,
        text: str = prompt,
        selected_spans: Sequence[Sequence[float]] | None = None,
    ) -> dict[str, Any]:
        item = deepcopy(common)
        item["messages"] = [
            _user_message(
                video_count=len(videos),
                audio_count=len(audios or []),
                prompt=text,
            )
        ]
        item["videos"] = deepcopy(videos)
        if audios:
            item["audios"] = deepcopy(audios)
        item["view_contract"] = {
            "arm": arm,
            "full_policy": "F1_2fps_3136_28672px_continuous_audio",
            "use_audio_in_video": use_audio_in_video,
            "source_duration": duration,
            "evidence_spans": deepcopy(visual_spans),
            "selected_spans": deepcopy(list(selected_spans or [])),
            "evidence_duration": evidence_duration,
            "evidence_to_full_duration_ratio": evidence_duration / duration,
            "answer_available_to_model": False,
        }
        item["sampling_contract"] = {
            "fps": float(fps),
            "min_pixels": int(min_pixels),
            "max_pixels": int(max_pixels),
            "max_frames": int(max_frames),
            "use_audio_in_video": use_audio_in_video,
            "held_out_evaluation": True,
        }
        return item

    return {
        "A0_full_av": make(
            "A0_full_av", videos=[full_video], audios=None, use_audio_in_video=True
        ),
        "A1_evidence_v_full_a": make(
            "A1_evidence_v_full_a",
            videos=evidence_videos,
            audios=full_audio,
            use_audio_in_video=False,
            selected_spans=visual_spans,
        ),
        "A2_full_v_evidence_a": make(
            "A2_full_v_evidence_a",
            videos=[full_video],
            audios=evidence_audios,
            use_audio_in_video=False,
            selected_spans=audio_spans,
        ),
        "A3_evidence_av": make(
            "A3_evidence_av",
            videos=evidence_videos,
            audios=None,
            use_audio_in_video=True,
            selected_spans=visual_spans,
        ),
        "A4_uniform_matched_av": make(
            "A4_uniform_matched_av",
            videos=matched_videos,
            audios=None,
            use_audio_in_video=True,
            selected_spans=matched_spans,
        ),
        "A5_full_av_timestamp": make(
            "A5_full_av_timestamp",
            videos=[full_video],
            audios=None,
            use_audio_in_video=True,
            text=_timestamp_text(visual_spans) + "\n" + prompt,
            selected_spans=visual_spans,
        ),
    }
