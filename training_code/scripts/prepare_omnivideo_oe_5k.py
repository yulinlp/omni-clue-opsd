#!/usr/bin/env python3
"""Materialize the ID-frozen 5k OmniVideo-100K direct-answer split.

The ID manifest is a gzipped JSON object rather than one ID per line.  This
utility verifies its source checksum, preserves the manifest order, resolves
all videos from the extracted ``videos/<video_id>.mp4`` tree, and writes
answer-isolated and SFT-ready views.  The SFT/rollout prompt is deliberately
non-reasoning: the model is asked to emit only a concise final answer.  It
does not resample or reselect the 5,000 questions.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from omni_opsd.data.common import parse_time_ranges
from omni_opsd.data.dynamic_budget import (
    dynamic_budget_for,
    dynamic_clue_budget_for,
    dynamic_sampling_contract,
    dynamic_video_spec,
)
from omni_opsd.data.swift_opsd import write_jsonl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_ids(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    ids = [str(value) for value in payload.get("question_ids", [])]
    expected = int(payload.get("count", len(ids)))
    if len(ids) != expected or len(set(ids)) != expected:
        raise ValueError(f"ID manifest count/uniqueness mismatch: expected={expected}, rows={len(ids)}")
    if payload.get("id_field") != "question_id":
        raise ValueError(f"unsupported ID field: {payload.get('id_field')!r}")
    return payload


def _compact_metadata(source: dict[str, Any]) -> dict[str, Any]:
    analysis = source.get("analysis") or {}
    return {
        "search_tag": source.get("search_tag"),
        "language": source.get("language"),
        "resolution": source.get("resolution"),
        "task": source.get("task"),
        "subtask": source.get("subtask"),
        "connections": analysis.get("connections"),
        "designated_segments": analysis.get("designated_segments"),
        "source_video_path": source.get("video_path"),
        "source_dataset": "OmniVideo-100K/train_oe_70k.jsonl",
    }


def _canonical(source: dict[str, Any], video_path: Path) -> dict[str, Any]:
    question_id = str(source["question_id"])
    answer = str(source.get("answer") or "").strip()
    if not answer:
        raise ValueError(f"{question_id}: empty open-ended answer")
    duration = float(source.get("duration") or 0.0)
    if duration <= 0:
        raise ValueError(f"{question_id}: invalid duration={duration}")
    question = str(source.get("question") or "").strip()
    if not question:
        raise ValueError(f"{question_id}: empty question")
    return {
        "sample_id": question_id,
        "question_id": question_id,
        "benchmark": "OmniVideo-100K",
        "video_id": str(source["video_id"]),
        "video_path": str(video_path.resolve()),
        "question": question,
        "answer": answer,
        "duration": duration,
        "resolution": str(source.get("resolution") or ""),
        "question_type": str(source.get("task") or source.get("subtask") or "unknown"),
        "task": str(source.get("task") or ""),
        "subtask": str(source.get("subtask") or ""),
        "analysis": dict(source.get("analysis") or {}),
        "metadata": _compact_metadata(source),
    }


def _user_prompt(row: dict[str, Any]) -> str:
    return (
        "<video>\n"
        f"Question: {row['question']}\n"
        "Answer the question directly using the relevant video and audio evidence. "
        "Do not provide a step-by-step reasoning trace or meta-commentary. "
        "Give a concise, self-contained answer with the key supporting evidence."
    )


def _training_row(row: dict[str, Any], budget: dict[str, Any], *, include_answer: bool) -> dict[str, Any]:
    prompt = _user_prompt(row)
    result: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "videos": [dynamic_video_spec(row, budget)],
        "case_id": row["sample_id"],
        "prompt_id": row["sample_id"],
        "video_id": row["video_id"],
        "benchmark": "OmniVideo-100K",
        "question_type": row["question_type"],
        "response_format": "direct_answer",
        "dynamic_student_budget": budget,
        "sampling_contract": dynamic_sampling_contract(
            budget, use_audio_in_video=True, teacher_view="full-video-uniform"
        ),
        "metadata": row["metadata"],
    }
    if include_answer:
        result["messages"].append({"role": "assistant", "content": row["answer"]})
        result["gold_answer"] = row["answer"]
        # The shared training-arm validator uses this explicit tag to prevent
        # accidentally feeding an arm-specific row to another launcher.
        result["experiment_arm"] = "sft"
        result["supervision_contract"] = {
            "kind": "direct-answer-gold-answer-teacher-forcing",
            "gold_answer_in_student_target": True,
            "reasoning_disabled": True,
            "use_audio_in_video": True,
        }
    return result


def _opsd_training_row(row: dict[str, Any], budget: dict[str, Any]) -> dict[str, Any]:
    """Build a full-video ordinary OPSD row for an open-ended answer.

    The student/rollout view is the non-reasoning direct-answer prompt.  The
    teacher gets the identical full-video descriptor plus the reference answer
    in a privileged prompt, while the raw answer is omitted from the row so it
    cannot become a student target through dataset collation.
    """

    result = _training_row(row, budget, include_answer=False)
    student_prompt = str(result["messages"][0]["content"])
    result.update({
        "teacher_prompt": (
            f"{student_prompt}\n\n"
            "Privileged information: The reference answer is:\n"
            f"{row['answer']}\n\n"
            "Use the reference answer and the video/audio evidence to produce a concise, "
            "self-contained answer to the original question. Do not mention the privileged "
            "information or this instruction."
        ),
        "teacher_videos": [dict(media) for media in result["videos"]],
        "dynamic_teacher_budget": dict(budget),
        "experiment_arm": "opsd",
        "supervision_contract": {
            "kind": "direct-answer-privileged-full-video-on-policy-self-distillation",
            "gold_answer_in_student_target": False,
            "gold_answer_in_reward": False,
            "gold_answer_in_teacher_prompt": True,
            "reasoning_disabled": True,
            "teacher_view": "full-video-uniform",
            "teacher_completion": "student_on_policy_completion_token_ids",
        },
    })
    # Keep the teacher-side budget explicit and identical to the student side.
    result["sampling_contract"].update({
        "teacher_view": "full-video-uniform",
        "teacher_frame_cap_total": int(budget["nframes"]),
        "teacher_dynamic_video_budget": int(budget["visual_budget_tokens"]),
        "teacher_dynamic_audio_budget": int(budget["audio_budget_tokens"]),
        "teacher_dynamic_reserved_tokens": int(budget["reserved_tokens"]),
    })
    return result


def _designated_spans(row: dict[str, Any]) -> list[list[float]]:
    """Parse the release's optional golden clue intervals.

    The OE release has 297 rows without timestamp annotations.  Those rows are
    retained in the 5k training set and use an explicit full-video teacher
    fallback; they are never silently dropped.
    """

    raw = (row.get("metadata") or {}).get("designated_segments")
    spans = []
    duration = float(row["duration"])
    for span in parse_time_ranges(raw):
        start = max(0.0, min(duration, float(span.start)))
        end = max(0.0, min(duration, float(span.end)))
        if end > start:
            spans.append([start, end])
    return spans


def _allocate_clue_frame_caps(spans: list[list[float]], total_frames: int) -> list[int]:
    """Allocate an even shared frame budget proportionally across clue spans."""

    if not spans or total_frames < 2 * len(spans) or total_frames % 2:
        raise ValueError(f"invalid clue frame allocation: intervals={len(spans)}, frames={total_frames}")
    durations = [max(0.0, float(end) - float(start)) for start, end in spans]
    if any(duration <= 0 for duration in durations):
        raise ValueError("clue intervals must have positive duration")
    pair_budget = total_frames // 2
    raw_pairs = [pair_budget * duration / sum(durations) for duration in durations]
    pair_caps = [max(1, math.floor(value)) for value in raw_pairs]
    while sum(pair_caps) > pair_budget:
        candidates = [index for index, value in enumerate(pair_caps) if value > 1]
        if not candidates:
            raise ValueError(f"cannot reduce clue frame allocation: {pair_caps}")
        index = min(candidates, key=lambda item: raw_pairs[item] - pair_caps[item])
        pair_caps[index] -= 1
    for index in sorted(range(len(pair_caps)), key=lambda item: raw_pairs[item] - pair_caps[item], reverse=True):
        if sum(pair_caps) >= pair_budget:
            break
        pair_caps[index] += 1
    if sum(pair_caps) != pair_budget:
        raise ValueError(f"failed to allocate clue frames: {pair_caps} vs {pair_budget}")
    return [2 * value for value in pair_caps]


def _clue_video_specs(row: dict[str, Any], spans: list[list[float]], budget: dict[str, Any]) -> list[dict[str, Any]]:
    caps = _allocate_clue_frame_caps(spans, int(budget["nframes"]))
    max_pixels = int(budget["resized_height"] * budget["resized_width"])
    return [
        {
            "video": str(row["video_path"]),
            "video_start": float(start),
            "video_end": float(end),
            "nframes": int(frame_cap),
            "resized_height": int(budget["resized_height"]),
            "resized_width": int(budget["resized_width"]),
            "min_pixels": 3_136,
            "max_pixels": max_pixels,
        }
        for (start, end), frame_cap in zip(spans, caps)
    ]


def _clue_training_row(row: dict[str, Any], budget: dict[str, Any], *, visual_budget_cap: int) -> dict[str, Any]:
    """Build an OE Clue-OPSD row with clue media and no answer leakage.

    The student sees the full video and a direct-answer prompt.  The teacher
    sees only the designated clue intervals and the same prompt, without the
    gold answer.  A small annotated-segment-free subset uses a recorded
    full-video teacher fallback so the 5,000-row corpus stays intact and
    auditable.
    """

    result = _training_row(row, budget, include_answer=False)
    student_prompt = str(result["messages"][0]["content"])
    prompt_body = student_prompt.split("\n", 1)[1]
    spans = _designated_spans(row)
    if spans:
        teacher_budget = dynamic_clue_budget_for(row, spans, visual_budget_cap=visual_budget_cap)
        teacher_videos = _clue_video_specs(row, spans, teacher_budget)
        teacher_view = "dataset-clue-interval"
        teacher_placeholders = "\n".join(["<video>"] * len(teacher_videos))
    else:
        teacher_budget = budget
        teacher_videos = [dynamic_video_spec(row, budget)]
        teacher_view = "full-video-fallback-no-annotated-clue"
        teacher_placeholders = "<video>"
    result.update({
        "teacher_prompt": f"{teacher_placeholders}\n{prompt_body}",
        "teacher_videos": teacher_videos,
        "clue_intervals": spans,
        "dynamic_teacher_budget": teacher_budget,
        "experiment_arm": "clue_opsd",
        "supervision_contract": {
            "kind": "direct-answer-clue-on-policy-self-distillation",
            "gold_answer_in_student_target": False,
            "gold_answer_in_reward": False,
            "gold_answer_in_teacher_prompt": False,
            "gold_answer_available_for_auxiliary_ce": False,
            "reasoning_disabled": True,
            "teacher_view": teacher_view,
            "teacher_completion": "student_on_policy_completion_token_ids",
        },
    })
    result["sampling_contract"].update({
        "teacher_view": teacher_view,
        "teacher_frame_cap_total": int(sum(int(media["nframes"]) for media in teacher_videos)),
        "teacher_frame_caps": [int(media["nframes"]) for media in teacher_videos],
        "teacher_resized_height": int(teacher_budget["resized_height"]),
        "teacher_resized_width": int(teacher_budget["resized_width"]),
        "teacher_max_pixels": int(teacher_budget["resized_height"] * teacher_budget["resized_width"]),
        "teacher_dynamic_video_budget": int(teacher_budget["visual_budget_tokens"]),
        "teacher_dynamic_audio_budget": int(teacher_budget["audio_budget_tokens"]),
        "teacher_dynamic_reserved_tokens": int(teacher_budget["reserved_tokens"]),
    })
    return result


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    write_jsonl(path, rows)
    return sum(1 for _ in path.open(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True,
                        help="OmniVideo-100K root containing videos/<video_id>.mp4")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-budget-cap", type=int, default=24_000,
                        help="dynamic visual budget cap for the 3B full-EMA run (24k is the documented section-6/7 budget)")
    args = parser.parse_args()

    annotation = args.annotation.resolve()
    ids_path = args.ids.resolve()
    video_root = args.video_root.resolve()
    output_dir = args.output_dir.resolve()
    id_manifest = _read_ids(ids_path)
    wanted = list(id_manifest["question_ids"])
    wanted_set = set(wanted)
    source_sha = _sha256(annotation)
    if source_sha != id_manifest.get("source_sha256"):
        raise ValueError(
            f"source SHA256 mismatch: manifest={id_manifest.get('source_sha256')} actual={source_sha}"
        )

    found: dict[str, dict[str, Any]] = {}
    source_rows: dict[str, dict[str, Any]] = {}
    with annotation.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            source = json.loads(line)
            question_id = str(source.get("question_id") or "")
            if question_id not in wanted_set:
                continue
            if question_id in source_rows:
                raise ValueError(f"duplicate source question_id={question_id} at line {line_number}")
            source_rows[question_id] = source
            video_id = str(source.get("video_id") or "")
            relative_video = str(source.get("video_path") or f"videos/{video_id}.mp4")
            # The release uses videos/<video_id>.mp4; resolve by video_id so a
            # stale relative path cannot silently point outside the extracted tree.
            video_path = (video_root / "videos" / f"{video_id}.mp4").resolve()
            if relative_video != f"videos/{video_id}.mp4":
                alternate = (video_root / relative_video).resolve()
                if alternate.is_file() and alternate.stat().st_size > 0:
                    video_path = alternate
            if not video_path.is_file() or video_path.stat().st_size == 0:
                raise FileNotFoundError(f"{question_id}: missing video {video_path}")
            found[question_id] = _canonical(source, video_path)

    missing_ids = [question_id for question_id in wanted if question_id not in found]
    if missing_ids:
        raise ValueError(f"{len(missing_ids)} IDs absent from annotation; first={missing_ids[:5]}")
    if len(found) != 5_000:
        raise ValueError(f"expected 5000 selected rows, found {len(found)}")
    rows = [found[question_id] for question_id in wanted]
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("selected rows contain duplicate sample IDs")

    # Dynamic budgets are materialized once and reused by all downstream arms.
    budgets: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []
    answer_free_rows: list[dict[str, Any]] = []
    opsd_rows: list[dict[str, Any]] = []
    clue_opsd_rows: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    clue_fallback_count = 0
    for row in rows:
        budget = dynamic_budget_for(row, visual_budget_cap=args.visual_budget_cap)
        budgets.append(budget)
        sft_rows.append(_training_row(row, budget, include_answer=True))
        answer_free_rows.append(_training_row(row, budget, include_answer=False))
        opsd_rows.append(_opsd_training_row(row, budget))
        clue_row = _clue_training_row(row, budget, visual_budget_cap=args.visual_budget_cap)
        clue_opsd_rows.append(clue_row)
        clue_fallback_count += int(not clue_row["clue_intervals"])
        labels.append({
            "sample_id": row["sample_id"],
            "question_id": row["question_id"],
            "video_id": row["video_id"],
            "answer": row["answer"],
            "question_type": row["question_type"],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "selected_raw": output_dir / "selected_raw.jsonl",
        "canonical": output_dir / "omnivideo_oe_5k.canonical.jsonl",
        "sft": output_dir / "omnivideo_oe_5k.sft.jsonl",
        "answer_free": output_dir / "omnivideo_oe_5k.answer_free.jsonl",
        "opsd": output_dir / "omnivideo_oe_5k.opsd.jsonl",
        "clue_opsd": output_dir / "omnivideo_oe_5k.clue_opsd.jsonl",
        "labels": output_dir / "omnivideo_oe_5k.labels.jsonl",
    }
    _write_rows(paths["selected_raw"], (source_rows[question_id] for question_id in wanted))
    _write_rows(paths["canonical"], rows)
    _write_rows(paths["sft"], sft_rows)
    _write_rows(paths["answer_free"], answer_free_rows)
    _write_rows(paths["opsd"], opsd_rows)
    _write_rows(paths["clue_opsd"], clue_opsd_rows)
    _write_rows(paths["labels"], labels)

    visual_tokens = [int(b["visual_budget_tokens"]) for b in budgets]
    frames = [int(b["nframes"]) for b in budgets]
    tasks = Counter(row["question_type"] for row in rows)
    videos = sorted({row["video_id"] for row in rows})
    manifest = {
        "status": "id_frozen_materialized",
        "benchmark": "OmniVideo-100K",
        "format": "direct_answer",
        "prompt_contract": {
            "mode": "non_reasoning",
            "answer_only": True,
            "reasoning_disabled": True,
        },
        "selection": id_manifest.get("selection"),
        "count": len(rows),
        "unique_videos": len(videos),
        "clue_opsd": {
            "rows": len(clue_opsd_rows),
            "annotated_clue_rows": len(clue_opsd_rows) - clue_fallback_count,
            "full_video_fallback_rows": clue_fallback_count,
            "teacher_view": "dataset-clue-interval-with-explicit-fallback",
            "gold_answer_in_teacher_prompt": False,
            "gold_answer_in_student_target": False,
            "auxiliary_answer_ce": False,
        },
        "opsd": {
            "rows": len(opsd_rows),
            "teacher_view": "full-video-uniform",
            "gold_answer_in_teacher_prompt": True,
            "gold_answer_in_student_target": False,
            "reasoning_disabled": True,
        },
        "id_order_preserved": True,
        "all_media_present": True,
        "source": {
            "annotation": str(annotation),
            "annotation_sha256": source_sha,
            "ids_manifest": str(ids_path),
            "ids_manifest_sha256": _sha256(ids_path),
            "video_root": str(video_root),
        },
        "sampling": {
            "dynamic_budget_version": "dynamic_video_budget_v1",
            "visual_budget_cap": args.visual_budget_cap,
            "visual_tokens_min": min(visual_tokens),
            "visual_tokens_max": max(visual_tokens),
            "frames_min": min(frames),
            "frames_max": max(frames),
            "fps": 2.0,
            "use_audio_in_video": True,
            "context_tokens": 32768,
        },
        "task_counts": dict(sorted(tasks.items())),
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": {name: _sha256(path) for name, path in paths.items()},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
