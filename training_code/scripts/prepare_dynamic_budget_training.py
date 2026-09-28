#!/usr/bin/env python3
"""Prepare the current dynamic-budget SFT, OPSD and CLUE-OPSD JSONL inputs.

The legacy four-arm matrix remains untouched.  This command writes a new
versioned dynamic-budget stream for each current arm, including a CLUE teacher
whose evidence intervals share one context/visual budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from omni_opsd.data.dynamic_budget import (
    dynamic_budget_for,
    dynamic_clue_budget_for,
    dynamic_sampling_contract,
    dynamic_video_spec,
)
from omni_opsd.data.swift_opsd import evidence_spans, read_jsonl, write_jsonl


REASONING_PROMPT = (
    "Question: {question}\n"
    "Options:\n{options}\n"
    "Briefly analyze the video and audio evidence relevant to the question and compare the answer options. "
    "Write the analysis first, then give exactly one option letter inside <answer>...</answer>. "
    "Keep the analysis concise (at most 120 words)."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _answer(row: dict[str, Any]) -> str:
    answer = str(row.get("answer", "")).strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        raise ValueError(f"invalid four-choice answer for {row.get('sample_id')}: {answer!r}")
    choices = list(row.get("choices") or [])
    if len(choices) != 4:
        raise ValueError(f"expected four choices for {row.get('sample_id')}, got {len(choices)}")
    return answer


def _student_prompt(row: dict[str, Any]) -> str:
    choices = list(row.get("choices") or [])
    options = "\n".join(f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices))
    return "<video>\n" + REASONING_PROMPT.format(question=row["question"], options=options)


def _sft_target(row: dict[str, Any], answer: str) -> str:
    connections = str((row.get("metadata") or {}).get("connections") or "").strip()
    if not connections:
        raise ValueError(f"missing metadata.connections for SFT sample {row.get('sample_id')}")
    return f"{connections}\n<answer>{answer}</answer>"


def _allocate_clue_frame_caps(spans: list[list[float]], total_frames: int) -> list[int]:
    """Split one even frame budget across evidence intervals.

    Each structured interval receives an even number of frames so Qwen's
    temporal merge remains well-defined.  The allocation is proportional to
    interval duration and uses the largest remainders for the final pairs.
    """

    if not spans or total_frames < 2 * len(spans) or total_frames % 2:
        raise ValueError(f"invalid clue frame allocation: intervals={len(spans)}, frames={total_frames}")
    durations = [max(0.0, float(end) - float(start)) for start, end in spans]
    if any(duration <= 0 for duration in durations):
        raise ValueError("clue intervals must have positive duration")
    pair_budget = total_frames // 2
    raw_pairs = [pair_budget * duration / sum(durations) for duration in durations]
    pair_caps = [max(1, math.floor(value)) for value in raw_pairs]
    if sum(pair_caps) > pair_budget:
        for index in sorted(range(len(pair_caps)), key=lambda i: raw_pairs[i] - pair_caps[i]):
            if sum(pair_caps) <= pair_budget:
                break
            if pair_caps[index] > 1:
                pair_caps[index] -= 1
    for index in sorted(range(len(pair_caps)), key=lambda i: raw_pairs[i] - pair_caps[i], reverse=True):
        if sum(pair_caps) >= pair_budget:
            break
        pair_caps[index] += 1
    if sum(pair_caps) != pair_budget:
        raise ValueError(f"failed to allocate clue frames: {pair_caps} vs {pair_budget}")
    return [2 * value for value in pair_caps]


def _clue_video_specs(
    row: dict[str, Any], spans: list[list[float]], budget: dict[str, Any]
) -> list[dict[str, Any]]:
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


def _base_fields(row: dict[str, Any], budget: dict[str, Any]) -> dict[str, Any]:
    spec = dynamic_video_spec(row, budget)
    return {
        "messages": [{"role": "user", "content": _student_prompt(row)}],
        "videos": [spec],
        "prompt_id": str(row["sample_id"]),
        "case_id": str(row["sample_id"]),
        "video_id": str(row["video_id"]),
        "response_format": "reasoning",
        "dynamic_student_budget": deepcopy(budget),
        "sampling_contract": dynamic_sampling_contract(budget, use_audio_in_video=True),
    }


def _build_rows(
    row: dict[str, Any],
    *,
    visual_budget_cap: int = 24_000,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    answer = _answer(row)
    if visual_budget_cap <= 0:
        raise ValueError("visual_budget_cap must be positive")
    budget = dynamic_budget_for(row, visual_budget_cap=visual_budget_cap)
    base = _base_fields(row, budget)

    sft = deepcopy(base)
    sft["messages"].append({"role": "assistant", "content": _sft_target(row, answer)})
    sft["experiment_arm"] = "sft"
    sft["sft_target_source"] = "metadata.connections + gold_answer"
    sft["sampling_contract"]["answer_label_in_model_row"] = True
    sft["supervision_contract"] = {
        "kind": "gold-answer-reasoning-teacher-forcing",
        "gold_answer_in_student_target": True,
        "gold_answer_in_reward": False,
        "gold_answer_in_teacher_prompt": False,
        "teacher_view": None,
    }

    opsd = deepcopy(base)
    student_prompt = str(opsd["messages"][0]["content"])
    opsd["teacher_prompt"] = (
        f"{student_prompt}\n\n"
        f"Privileged information: the correct option is {answer}. "
        "Use the video and audio evidence to explain this answer briefly, then give the option "
        "inside <answer>...</answer>."
    )
    opsd["teacher_videos"] = deepcopy(opsd["videos"])
    opsd["experiment_arm"] = "opsd"
    opsd["sampling_contract"]["answer_label_in_model_row"] = False
    opsd["sampling_contract"]["teacher_view"] = "full-video-uniform"
    opsd["supervision_contract"] = {
        "kind": "answer-privileged-on-policy-self-distillation",
        "gold_answer_in_student_target": False,
        "gold_answer_in_reward": False,
        "gold_answer_in_teacher_prompt": True,
        "teacher_view": "full-video-uniform",
        "teacher_completion": "student_on_policy_completion_token_ids",
        "distillation": {
            "support": "teacher_top_20_vocabulary",
            "jsd_beta": 0.5,
            "temperature": 1.0,
            "lmbda": 1.0,
            "sft_alpha": 0.0,
            "lora_shadow_ema_alpha": 0.05,
        },
    }

    clue_opsd = deepcopy(base)
    spans = evidence_spans(row)
    clue_budget = dynamic_clue_budget_for(row, spans, visual_budget_cap=visual_budget_cap)
    # CLUE's teacher receives both the annotated evidence clips and the gold
    # option.  The student row remains answer-free; ``gold_answer`` is kept as
    # a side-channel for the auxiliary answer-token CE implemented by the GKD
    # trainer, while the teacher prompt carries the same answer explicitly.
    clue_opsd["teacher_prompt"] = (
        "\n".join(["<video>"] * len(spans)) + "\n" + student_prompt.split("\n", 1)[1] + "\n\n"
        f"Privileged information: the correct option is {answer}. "
        "Use the evidence clips and audio to explain why, then give exactly that option "
        "inside <answer>...</answer>."
    )
    clue_opsd["teacher_videos"] = _clue_video_specs(row, spans, clue_budget)
    clue_opsd["gold_answer"] = answer
    clue_opsd["clue_intervals"] = deepcopy(spans)
    clue_opsd["dynamic_teacher_budget"] = deepcopy(clue_budget)
    clue_opsd["sampling_contract"].update({
        "teacher_view": "dataset-clue-interval",
        "teacher_frame_cap_total": int(clue_budget["nframes"]),
        "teacher_frame_caps": [int(media["nframes"]) for media in clue_opsd["teacher_videos"]],
        "teacher_resized_height": int(clue_budget["resized_height"]),
        "teacher_resized_width": int(clue_budget["resized_width"]),
        "teacher_max_pixels": int(clue_budget["resized_height"] * clue_budget["resized_width"]),
        "teacher_dynamic_video_budget": int(clue_budget["visual_budget_tokens"]),
        "teacher_dynamic_audio_budget": int(clue_budget["audio_budget_tokens"]),
        "teacher_dynamic_reserved_tokens": int(clue_budget["reserved_tokens"]),
    })
    clue_opsd["experiment_arm"] = "clue_opsd"
    clue_opsd["supervision_contract"] = {
        "kind": "clue-privileged-on-policy-self-distillation",
        "gold_answer_in_student_target": False,
        "gold_answer_in_reward": False,
        "gold_answer_in_teacher_prompt": True,
        "gold_answer_available_for_auxiliary_ce": True,
        "teacher_view": "dataset-clue-interval",
        "teacher_completion": "student_on_policy_completion_token_ids",
        "distillation": {
            "support": "teacher_top_20_vocabulary",
            "jsd_beta": 0.5,
            "temperature": 1.0,
            "lmbda": 1.0,
            "sft_alpha": 0.0,
            "gold_answer_ce_alpha": 0.25,
            "full_parameter_ema_alpha": 0.05,
        },
    }
    return sft, opsd, clue_opsd, budget


def _validate_rows(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    if len(rows) != 5000:
        raise ValueError(f"{arm}: expected 5000 rows, got {len(rows)}")
    case_ids = [str(row.get("case_id")) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"{arm}: duplicate case_id")
    frame_counts = [int(row["sampling_contract"]["frames_per_video_input"]) for row in rows]
    visual_tokens = [int(row["dynamic_student_budget"]["visual_budget_tokens"])
                     for row in rows]
    visual_tokens_per_frame = [
        float(row["dynamic_student_budget"]["visual_tokens_per_sampled_frame"])
        for row in rows
    ]
    if max(frame_counts) > 300:
        raise ValueError(f"{arm}: frame count exceeds 300: {max(frame_counts)}")
    if max(visual_tokens) > 24_000:
        raise ValueError(f"{arm}: visual budget exceeds 24000: {max(visual_tokens)}")
    if min(visual_tokens_per_frame) < 100 or max(visual_tokens_per_frame) > 128:
        raise ValueError(
            f"{arm}: visual tokens/frame outside [100, 128]: "
            f"{min(visual_tokens_per_frame)}..{max(visual_tokens_per_frame)}"
        )
    for row in rows:
        sampling = row["sampling_contract"]
        budget = row["dynamic_student_budget"]
        if sampling["use_audio_in_video"] is not True:
            raise ValueError(f"{arm}/{row['case_id']}: audio contract is not enabled")
        if budget["max_checked_tokens"] > 32_768:
            raise ValueError(f"{arm}/{row['case_id']}: context headroom exceeded")
        for field in ("nframes", "resized_height", "resized_width"):
            if field not in row["videos"][0]:
                raise ValueError(f"{arm}/{row['case_id']}: missing explicit {field}")
    return {
        "rows": len(rows),
        "unique_case_ids": len(case_ids),
        "unique_video_ids": len({str(row["video_id"]) for row in rows}),
        "frame_count_min": min(frame_counts),
        "frame_count_max": max(frame_counts),
        "visual_tokens_min": min(visual_tokens),
        "visual_tokens_max": max(visual_tokens),
        "visual_tokens_mean": sum(visual_tokens) / len(visual_tokens),
        "visual_tokens_per_sampled_frame_min": min(visual_tokens_per_frame),
        "visual_tokens_per_sampled_frame_max": max(visual_tokens_per_frame),
        "audio_enabled": True,
        "student_inputs_dynamic": True,
    }


def _validate_clue_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _validate_rows(rows, "clue_opsd")
    teacher_frames: list[int] = []
    teacher_visual_tokens: list[int] = []
    teacher_tokens_per_frame: list[float] = []
    for row in rows:
        contract = row.get("supervision_contract") or {}
        if row.get("experiment_arm") != "clue_opsd":
            raise ValueError(f"clue_opsd/{row.get('case_id')}: arm mismatch")
        if not contract.get("gold_answer_in_teacher_prompt"):
            raise ValueError(f"clue_opsd/{row.get('case_id')}: teacher prompt lacks gold answer contract")
        answer = str(row.get("gold_answer", "")).strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"clue_opsd/{row.get('case_id')}: missing/invalid gold_answer")
        if f"correct option is {answer}" not in str(row.get("teacher_prompt", "")):
            raise ValueError(f"clue_opsd/{row.get('case_id')}: teacher prompt does not contain gold answer")
        if "answer" in row or "solution" in row:
            raise ValueError(f"clue_opsd/{row.get('case_id')}: raw answer/solution column leaked")
        spans = row.get("clue_intervals") or []
        teacher_videos = row.get("teacher_videos") or []
        budget = row.get("dynamic_teacher_budget") or {}
        if len(spans) != len(teacher_videos) or not teacher_videos:
            raise ValueError(f"clue_opsd/{row.get('case_id')}: interval/media mismatch")
        frame_count = sum(int(media.get("nframes", 0)) for media in teacher_videos)
        if frame_count != int(budget.get("nframes", -1)):
            raise ValueError(f"clue_opsd/{row.get('case_id')}: teacher frame budget mismatch")
        if int(budget.get("visual_budget_tokens", 0)) > 24_000:
            raise ValueError(f"clue_opsd/{row.get('case_id')}: teacher visual budget exceeds 24000")
        if int(budget.get("max_checked_tokens", 99_999)) > 32_768:
            raise ValueError(f"clue_opsd/{row.get('case_id')}: teacher context headroom exceeded")
        expected_placeholders = "\n".join(["<video>"] * len(spans))
        if not str(row.get("teacher_prompt", "")).startswith(expected_placeholders + "\n"):
            raise ValueError(f"clue_opsd/{row.get('case_id')}: teacher placeholders mismatch")
        for media in teacher_videos:
            for field in ("nframes", "resized_height", "resized_width"):
                if field not in media:
                    raise ValueError(f"clue_opsd/{row.get('case_id')}: missing teacher {field}")
        teacher_frames.append(frame_count)
        teacher_visual_tokens.append(int(budget["visual_budget_tokens"]))
        teacher_tokens_per_frame.append(float(budget["visual_tokens_per_sampled_frame"]))
    summary.update({
        "teacher_frame_count_min": min(teacher_frames),
        "teacher_frame_count_max": max(teacher_frames),
        "teacher_visual_tokens_min": min(teacher_visual_tokens),
        "teacher_visual_tokens_max": max(teacher_visual_tokens),
        "teacher_visual_tokens_mean": sum(teacher_visual_tokens) / len(teacher_visual_tokens),
        "teacher_visual_tokens_per_sampled_frame_min": min(teacher_tokens_per_frame),
        "teacher_visual_tokens_per_sampled_frame_max": max(teacher_tokens_per_frame),
        "teacher_inputs_dynamic": True,
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gate-size", type=int, default=32)
    parser.add_argument(
        "--visual-budget-cap",
        type=int,
        default=24_000,
        help=(
            "maximum visual tokens per materialized view; the documented value is "
            "24000, while smaller hardware-safe caps may be used for an explicitly "
            "labelled engineering run"
        ),
    )
    args = parser.parse_args()

    canonical = read_jsonl(args.canonical)
    if len(canonical) != 5000:
        raise ValueError(f"expected frozen gap5000 canonical input, got {len(canonical)} rows")
    if len({str(row["sample_id"]) for row in canonical}) != len(canonical):
        raise ValueError("canonical input contains duplicate sample IDs")

    sft_rows: list[dict[str, Any]] = []
    opsd_rows: list[dict[str, Any]] = []
    clue_opsd_rows: list[dict[str, Any]] = []
    for row in canonical:
        sft, opsd, clue_opsd, _ = _build_rows(
            row, visual_budget_cap=args.visual_budget_cap
        )
        sft_rows.append(sft)
        opsd_rows.append(opsd)
        clue_opsd_rows.append(clue_opsd)

    root = args.output_root.resolve()
    paths = {
        "sft": root / "sft" / "formal" / "data" / "sft.jsonl",
        "opsd": root / "opsd" / "formal" / "data" / "reasoning.jsonl",
        "sft_gate": root / "sft" / "gate" / "data" / "sft.jsonl",
        "opsd_gate": root / "opsd" / "gate" / "data" / "reasoning.jsonl",
        "clue_opsd": root / "clue_opsd" / "formal" / "data" / "reasoning.jsonl",
        "clue_opsd_gate": root / "clue_opsd" / "gate" / "data" / "reasoning.jsonl",
    }
    write_jsonl(paths["sft"], sft_rows)
    write_jsonl(paths["opsd"], opsd_rows)
    write_jsonl(paths["clue_opsd"], clue_opsd_rows)
    if args.gate_size <= 0 or args.gate_size > len(canonical):
        raise ValueError("--gate-size must be between 1 and 5000")
    write_jsonl(paths["sft_gate"], sft_rows[: args.gate_size])
    write_jsonl(paths["opsd_gate"], opsd_rows[: args.gate_size])
    write_jsonl(paths["clue_opsd_gate"], clue_opsd_rows[: args.gate_size])

    summary = {
        "version": "dynamic_video_budget_v1",
        "canonical": str(args.canonical.resolve()),
        "canonical_sha256": _sha256(args.canonical),
        "modified_arms": ["sft", "opsd", "clue_opsd"],
        "unchanged_arms": ["grpo"],
        "sft": _validate_rows(sft_rows, "sft"),
        "opsd": _validate_rows(opsd_rows, "opsd"),
        "clue_opsd": _validate_clue_rows(clue_opsd_rows),
        "gate_size": args.gate_size,
        "outputs": {key: {"path": str(path), "sha256": _sha256(path)} for key, path in paths.items()},
        "training_parameters": {
            "model": "Qwen2.5-Omni-7B",
            "lora_rank": 64,
            "lora_alpha": 128,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "global_batch_size": 32,
            "max_steps": 157,
            "sft_learning_rate": 1e-5,
            "opsd_learning_rate": 2e-6,
            "max_grad_norm": 0.0,
            "rollout": {
                "use_vllm": True,
                "vllm_mode": "colocate",
                "vllm_tensor_parallel_size": 2,
                "temperature": 1.0,
                "top_k": 20,
                "top_p": 0.95,
                "max_completion_length": 512,
            },
            "distillation": {
                "support": "top_20_vocabulary",
                "jsd_beta": 0.5,
                "temperature": 1.0,
                "lmbda": 1.0,
                "sft_alpha": 0.0,
                "lora_shadow_ema_alpha": 0.05,
            },
            "clue_opsd": {
                "teacher_view": "dataset-clue-interval",
                "shared_visual_budget": True,
                "lora_shadow_ema_alpha": 0.05,
            },
            "visual_budget_cap_override": args.visual_budget_cap,
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "dynamic_budget_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
