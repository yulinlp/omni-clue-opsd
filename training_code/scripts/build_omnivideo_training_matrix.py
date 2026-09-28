#!/usr/bin/env python3
"""Build the four comparable OmniVideo-100K post-training arms.

The canonical training manifest contains the offline answer labels.  This
builder turns each row into four model-facing contracts while keeping the
student input byte-for-byte equivalent across arms:

* SFT: answer as the assistant target;
* GRPO: answer only in the external reward solution;
* OPSD: full A/V EMA-teacher prompt with privileged answer;
* CLUE-OPSD: evidence-interval A/V EMA-teacher prompt without the answer.

The full student view follows the frozen F1 policy.  Videos are referenced by
their original paths (no lossy training-time proxy is needed); qwen-omni's
processor performs the same 2 FPS, pixel-bounded decode used by the gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from omni_opsd.data.swift_opsd import (
    read_jsonl,
    swift_opsd_row,
    swift_training_matrix_rows,
    write_jsonl,
)


ARMS = ("sft", "grpo", "opsd", "clue_opsd")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _proxy_map(path: Path | None) -> dict[str, str]:
    """Load a verified source->full-timeline proxy inventory."""

    if path is None:
        return {}
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            source = str(Path(item["source_path"]).resolve())
            proxy = str(Path(item["proxy_path"]).resolve())
            if not item.get("endpoint_coverage_pass") or not os.path.isfile(proxy):
                raise ValueError(f"unverified or missing full-timeline proxy: {proxy}")
            previous = mapping.setdefault(source, proxy)
            if previous != proxy:
                raise ValueError(f"conflicting proxy inventory for {source}")
    if not mapping:
        raise ValueError(f"proxy audit is empty: {path}")
    return mapping


def _rewrite_proxy_paths(row: dict[str, Any], mapping: dict[str, str]) -> None:
    """Point structured video descriptors at verified full-timeline proxies."""

    if not mapping:
        return
    for field in ("videos", "teacher_videos"):
        for descriptor in row.get(field) or []:
            if not isinstance(descriptor, dict):
                raise ValueError(f"proxy rewrite requires structured {field} descriptors")
            source = str(Path(descriptor.get("video", "")).resolve())
            if source not in mapping:
                raise KeyError(f"no full-timeline proxy for {source}")
            descriptor["video"] = mapping[source]
            # qwen_omni_utils defaults to 128 video tokens/frame.  F1 uses a
            # deliberately smaller lower bound so its 28,672-pixel frame cap
            # remains valid at training time as well as inference time.
            descriptor.setdefault("min_pixels", 3136)


def _media_path(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("video") or value.get("audio") or "")
    return ""


def _validate_matrix(
    canonical: list[dict[str, Any]],
    rows_by_arm: dict[str, list[dict[str, Any]]],
    *,
    fps: float,
    max_frames: int,
    min_pixels: int,
    max_pixels: int,
    use_audio_in_video: bool,
    proxy_map: dict[str, str],
) -> dict[str, Any]:
    expected_ids = [str(row["sample_id"]) for row in canonical]
    durations = {str(row["sample_id"]): float(row["duration"]) for row in canonical}
    canonical_answers = {
        str(row["sample_id"]): str(row.get("answer", "")).upper() for row in canonical
    }
    for arm in ARMS:
        rows = rows_by_arm[arm]
        actual_ids = [str(row.get("case_id", "")) for row in rows]
        if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
            raise ValueError(f"{arm}: paired case IDs are not identical to canonical input")
        for row in rows:
            case_id = str(row["case_id"])
            messages = row.get("messages") or []
            if not messages or messages[0].get("role") != "user":
                raise ValueError(f"{arm}/{case_id}: missing user prompt")
            sampling = row.get("sampling_contract") or {}
            if sampling.get("fps") != fps:
                raise ValueError(f"{arm}/{case_id}: FPS contract mismatch")
            if sampling.get("max_frames_per_view") != max_frames:
                raise ValueError(f"{arm}/{case_id}: frame cap mismatch")
            if sampling.get("max_pixels") != max_pixels:
                raise ValueError(f"{arm}/{case_id}: pixel budget mismatch")
            if sampling.get("min_pixels", min_pixels) != min_pixels:
                raise ValueError(f"{arm}/{case_id}: minimum pixel budget mismatch")
            if bool(sampling.get("use_audio_in_video")) != use_audio_in_video:
                raise ValueError(f"{arm}/{case_id}: audio contract mismatch")
            videos = row.get("videos") or []
            if len(videos) != 1:
                raise ValueError(f"{arm}/{case_id}: student must have one full video")
            spec = videos[0]
            if spec.get("min_pixels", min_pixels) != min_pixels:
                raise ValueError(f"{arm}/{case_id}: student minimum pixel budget mismatch")
            if float(spec.get("video_start", -1.0)) != 0.0:
                raise ValueError(f"{arm}/{case_id}: student video does not start at zero")
            duration = durations[case_id]
            if abs(float(spec.get("video_end", -1.0)) - duration) > 1e-6:
                raise ValueError(f"{arm}/{case_id}: student video does not cover full duration")
            path = _media_path(spec)
            if not path or not os.path.isfile(path):
                raise FileNotFoundError(f"{arm}/{case_id}: missing student video {path}")

            if arm == "sft":
                if len(messages) != 2 or messages[-1].get("role") != "assistant":
                    raise ValueError(f"{arm}/{case_id}: missing gold assistant target")
                if messages[-1].get("content") != canonical_answers[case_id]:
                    raise ValueError(f"{arm}/{case_id}: SFT target mismatch")
            elif arm == "grpo":
                if row.get("solution") != canonical_answers[case_id] or len(messages) != 1:
                    raise ValueError(f"{arm}/{case_id}: invalid reward-only contract")
            elif arm == "opsd":
                contract = row.get("supervision_contract") or {}
                if not contract.get("gold_answer_in_teacher_prompt"):
                    raise ValueError(f"{arm}/{case_id}: privileged teacher contract missing")
                if row.get("teacher_videos") != row.get("videos"):
                    raise ValueError(f"{arm}/{case_id}: OPSD teacher must see same full video")
            else:
                if row.get("answer") is not None or row.get("solution") is not None:
                    raise ValueError(f"{arm}/{case_id}: answer leaked into CLUE row")
                if not row.get("teacher_videos"):
                    raise ValueError(f"{arm}/{case_id}: CLUE teacher has no evidence video")
                for teacher in row["teacher_videos"]:
                    teacher_path = _media_path(teacher)
                    if not teacher_path or not os.path.isfile(teacher_path):
                        raise FileNotFoundError(
                            f"{arm}/{case_id}: missing teacher video {teacher_path}"
                        )

    # Compare only the student user prompt and video specs.  Supervision is
    # expected to differ, but the model-facing full student input must not.
    reference = rows_by_arm["grpo"]
    for arm in ARMS:
        for index, row in enumerate(rows_by_arm[arm]):
            ref = reference[index]
            if row["messages"][0] != ref["messages"][0] or row["videos"] != ref["videos"]:
                raise ValueError(f"student input mismatch at {arm}/{row['case_id']}")

    return {
        "rows": len(canonical),
        "unique_case_ids": len(expected_ids),
        "unique_video_ids": len({str(row["video_id"]) for row in canonical}),
        "task_counts": dict(sorted(Counter(str(row["question_type"]) for row in canonical).items())),
        "student_inputs_identical": True,
        "full_video_coverage": True,
        "answer_isolated_by_contract": True,
        "full_timeline_proxy_inventory": bool(proxy_map),
        "proxy_video_count": len(proxy_map),
        "f1_policy": {
            "fps": fps,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "max_frames_safety_ceiling": max_frames,
            "use_audio_in_video": use_audio_in_video,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=768)
    parser.add_argument("--min-pixels", type=int, default=3_136)
    parser.add_argument("--max-pixels", type=int, default=28_672)
    parser.add_argument(
        "--proxy-audit",
        type=Path,
        help="verified full-timeline proxy audit; rewrite all model-facing video paths",
    )
    parser.add_argument(
        "--use-audio-in-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include the continuous audio stream in full-video inputs (default: true)",
    )
    args = parser.parse_args()
    if args.fps <= 0 or args.max_frames < 2 or args.min_pixels <= 0 or args.max_pixels < args.min_pixels:
        raise ValueError("invalid F1 sampling parameters")

    canonical = read_jsonl(args.canonical)
    if not canonical:
        raise SystemExit("canonical manifest is empty")
    if len({str(row["sample_id"]) for row in canonical}) != len(canonical):
        raise ValueError("canonical manifest contains duplicate sample IDs")

    proxy_map = _proxy_map(args.proxy_audit)
    rows_by_arm = {arm: [] for arm in ARMS}
    for row in canonical:
        # This row is the answer-free, evidence-view teacher contract.  The
        # original source media is intentionally retained so the processor can
        # apply the frozen F1 decode policy during training.
        clue_row = swift_opsd_row(
            row,
            fps=args.fps,
            max_frames=args.max_frames,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            use_audio_in_video=args.use_audio_in_video,
        )
        _rewrite_proxy_paths(clue_row, proxy_map)
        matrix = swift_training_matrix_rows(row, clue_row)
        for arm in ARMS:
            rows_by_arm[arm].append(matrix[arm])

    summary = _validate_matrix(
        canonical,
        rows_by_arm,
        fps=args.fps,
        max_frames=args.max_frames,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_audio_in_video=args.use_audio_in_video,
        proxy_map=proxy_map,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for arm in ARMS:
        path = output_dir / f"omnivideo_100k_train.{arm}.jsonl"
        write_jsonl(path, rows_by_arm[arm])
        paths[arm] = path
    summary.update(
        {
            "canonical": str(args.canonical.resolve()),
            "canonical_sha256": _sha256(args.canonical),
            "proxy_audit": str(args.proxy_audit.resolve()) if args.proxy_audit else None,
            "outputs": {
                arm: {"path": str(path), "sha256": _sha256(path)}
                for arm, path in paths.items()
            },
            "comparison_contract": {
                "sft": "gold answer teacher forcing",
                "grpo": "gold answer exact-match sequence reward",
                "opsd": "full A/V EMA teacher with privileged gold answer",
                "clue_opsd": "evidence-interval A/V EMA teacher without gold answer",
            },
        }
    )
    summary_path = output_dir / "training_matrix_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
