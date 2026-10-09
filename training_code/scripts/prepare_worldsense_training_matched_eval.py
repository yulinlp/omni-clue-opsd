#!/usr/bin/env python3
"""Freeze the existing 518 held-out questions with the training student AV budget.

This creates a separate evaluation dataset. The prior evaluation, model weights,
training examples, and labels remain untouched. Only answer-free user messages
and full-video student media are passed to model inference.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training_code/src"))
from omni_opsd.data.dynamic_budget import (  # noqa: E402
    dynamic_budget_for,
    dynamic_sampling_contract,
    dynamic_video_spec,
)

OPEN_INSTRUCTION = (
    "Briefly analyze the video and audio evidence in English using at most 120 words. "
    "Write your analysis inside <analysis>...</analysis>, then give the answer "
    "in natural language inside <answer>...</answer>. Do not output an option letter."
)
TRAIN_FILES = (
    "worldsense_observation_sft_lora_20260930/data/sft_observation_openqa.jsonl",
    "worldsense_observation_sft_full_npu96_w6w7_20260930/data/sft_observation_openqa.jsonl",
    "worldsense_openqa_thinking_full_observation_20260930/data/clue_openqa_thinking.jsonl",
    "worldsense_clue_full_no_observation_npu96_w10w11_20260930/data/clue_openqa_thinking.jsonl",
)
CLUE_ARGS = (
    REPO / "training_runs/worldsense_openqa_thinking_full_observation_20260930"
    "/outputs/formal/v0-20260930-165152/args.json"
)
MODEL = Path("/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-assets/models/Qwen2.5-Omni-7B")


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_write(path: Path, content: bytes) -> None:
    """Allow idempotent reruns, but never silently replace an existing dataset."""
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite a different frozen file: {path}")
        return
    path.write_bytes(content)


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def summary(numbers: list[int | float]) -> dict:
    return {"min": min(numbers), "mean": statistics.mean(numbers),
            "median": statistics.median(numbers), "max": max(numbers)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path,
                        default=REPO / "training_runs/worldsense_observation_eval_20261001")
    parser.add_argument("--candidate-source", type=Path,
                        default=REPO / "data/atomic_eval_20260929/worldsense_candidates_3079.atomic.jsonl")
    args = parser.parse_args()
    output = args.root.resolve() / "data"
    output.mkdir(parents=True, exist_ok=True)
    source = args.source_root.resolve() / "data"
    old_mcq = read_rows(source / "worldsense.answer_free.jsonl")
    old_openqa = read_rows(source / "worldsense.openqa.jsonl")
    mcq_labels = read_rows(source / "worldsense.labels.jsonl")
    open_labels = read_rows(source / "worldsense.openqa.labels.jsonl")
    candidates = {row["sample_id"]: row for row in read_rows(args.candidate_source)}
    ids = [row["case_id"] for row in old_mcq]
    if len(ids) != 518 or len(set(ids)) != 518:
        raise ValueError("Expected exactly 518 unique frozen held-out questions")
    if ids != [row["case_id"] for row in old_openqa]:
        raise ValueError("Existing MCQ and open-QA question order differs")
    if any(set(ids) != {row["sample_id"] for row in labels}
           or len(labels) != 518 for labels in (mcq_labels, open_labels)):
        raise ValueError("Label sets differ from the frozen 518 input IDs")
    open_by_id = {row["sample_id"]: row for row in open_labels}
    eval_videos = {candidates[case_id]["video_id"] for case_id in ids}
    train_audits = []
    for name in TRAIN_FILES:
        path = REPO / "training_runs" / name
        rows = read_rows(path)
        train_ids = {row["case_id"] for row in rows}
        train_videos = {row["video_id"] for row in rows}
        case_overlap = sorted(set(ids) & train_ids)
        video_overlap = sorted(eval_videos & train_videos)
        if case_overlap or video_overlap:
            raise ValueError(f"Training leakage in {path}: cases={case_overlap}, videos={video_overlap}")
        # The same function must reproduce the frozen training student budget.
        budget_mismatches = []
        for row in rows:
            old_budget = row["dynamic_student_budget"]
            audit_source = {"duration": old_budget["audio_seconds"],
                            "metadata": {"resolution": old_budget["source_resolution"]}}
            if dynamic_budget_for(audit_source) != old_budget:
                budget_mismatches.append(row["case_id"])
        if budget_mismatches:
            raise ValueError(f"Dynamic budget implementation differs from {path}: {budget_mismatches[:10]}")
        train_audits.append({"source": str(path), "sha256": sha256(path), "rows": len(rows),
                             "case_overlap": 0, "video_overlap": 0,
                             "student_budget_reproduced_rows": len(rows)})

    train_args = json.loads(CLUE_ARGS.read_text())
    generation = {key: train_args[key] for key in
                  ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "seed")}
    if generation != {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
                      "repetition_penalty": 1.0, "seed": 20260904}:
        raise ValueError(f"Unexpected current CLUE generation settings: {generation}")
    if train_args["max_completion_length"] != 512 or train_args["max_length"] != 32768:
        raise ValueError("Unexpected training completion/context length")
    training_prompt = read_rows(REPO / "training_runs" / TRAIN_FILES[2])[0]["messages"][0]["content"]
    if not training_prompt.endswith("\n" + OPEN_INSTRUCTION):
        raise ValueError("Open-QA prompt no longer matches the current CLUE student training prompt")

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    mcq, openqa, budgets = [], [], []
    text_counts = {"mcq": [], "openqa": []}
    forbidden = {"answer", "solution", "correct_option", "teacher_prompt", "teacher_videos",
                 "gold_answer", "gold_answer_text", "observation", "clue_intervals"}
    for old in old_mcq:
        case_id = old["case_id"]
        candidate = candidates[case_id]
        label = open_by_id[case_id]
        if not 0 < candidate["duration"] <= 300 or label["tier"] not in {"A", "B", "D"}:
            raise ValueError(f"Invalid held-out duration or tier: {case_id}")
        if candidate["question"] != label["question"]:
            raise ValueError(f"Question differs from existing label: {case_id}")
        if candidate["choices"][ord(candidate["answer"]) - ord("A")] != label["gold_answer_text"]:
            raise ValueError(f"Answer label differs from source: {case_id}")
        width, height = candidate.get("source_width"), candidate.get("source_height")
        if not width or not height:
            raise ValueError(f"Missing measured source resolution for {case_id}")
        dynamic_source = dict(candidate, metadata={"resolution": f"{width}x{height}"})
        budget = dynamic_budget_for(dynamic_source)
        spec = dynamic_video_spec(candidate, budget)
        video = Path(spec["video"])
        if video != Path(old["videos"][0]["video"]) or not video.is_file() or not video.stat().st_size:
            raise ValueError(f"Changed or missing full video: {case_id}")
        contract = dynamic_sampling_contract(budget, teacher_view="not_used_in_evaluation")
        # Remove teacher-related metadata: evaluation only uses the student view.
        contract.pop("teacher_view")
        contract.pop("teacher_frame_cap_total")
        contract.update(held_out_evaluation=True, train_case_overlap=False,
                        train_video_overlap=False, answer_label_in_model_row=False,
                        video_reader="pyav_seek", video_interval="full_video",
                        audio_sample_rate=16000, audio_max_seconds=300)
        base = {"case_id": case_id, "prompt_id": case_id, "video_id": candidate["video_id"],
                "benchmark": "WorldSense", "question_type": candidate["question_type"],
                "videos": [spec], "dynamic_student_budget": budget, "sampling_contract": contract}
        row_mcq = copy.deepcopy(base)
        row_mcq["messages"] = copy.deepcopy(old["messages"])
        row_open = copy.deepcopy(base)
        row_open["messages"] = [{"role": "user", "content":
                                 "<video>\nQuestion: " + candidate["question"] + "\n" + OPEN_INSTRUCTION}]
        if row_mcq["videos"] != row_open["videos"]:
            raise ValueError(f"MCQ/open-QA media differs: {case_id}")
        for mode, row, cap in (("mcq", row_mcq, 8), ("openqa", row_open, 512)):
            if forbidden & row.keys() or len(row["messages"]) != 1 or row["messages"][0]["role"] != "user":
                raise ValueError(f"Answer or teacher information in input: {case_id}")
            # Reserve 128 extra text tokens for the chat template/system message.
            text_tokens = len(tokenizer.encode(row["messages"][0]["content"], add_special_tokens=False).ids)
            if text_tokens + cap + 128 > budget["text_reserve_tokens"]:
                raise ValueError(f"Question/completion exceeds training text reserve: {mode}/{case_id}")
            text_counts[mode].append(text_tokens)
        mcq.append(row_mcq)
        openqa.append(row_open)
        budgets.append({"case_id": case_id, "video_id": candidate["video_id"],
                        "duration_seconds": candidate["duration"], "source_width": width,
                        "source_height": height, "tier": label["tier"],
                        "budget": budget, "video": spec})

    for name, rows in (("worldsense.answer_free.jsonl", mcq), ("worldsense.openqa.jsonl", openqa),
                       ("media_budget_audit.jsonl", budgets)):
        frozen_write(output / name, jsonl_bytes(rows))
    for name in ("worldsense.labels.jsonl", "worldsense.openqa.labels.jsonl"):
        frozen_write(output / name, (source / name).read_bytes())
    contract = {
        "benchmark": "WorldSense", "rows": len(ids), "unique_videos": len(eval_videos),
        "source_root": str(args.source_root.resolve()), "same_prior_ids_and_order": True,
        "same_prior_label_bytes": True, "same_mcq_openqa_media": True,
        "same_training_student_dynamic_budget_function": True,
        "candidate_source": str(args.candidate_source.resolve()), "candidate_source_sha256": sha256(args.candidate_source),
        "dynamic_budget_source": str(REPO / "training_code/src/omni_opsd/data/dynamic_budget.py"),
        "dynamic_budget_source_sha256": sha256(REPO / "training_code/src/omni_opsd/data/dynamic_budget.py"),
        "training_generation_args": str(CLUE_ARGS), "training_generation_args_sha256": sha256(CLUE_ARGS),
        "training_sources_audit": train_audits,
        "tier_counts": dict(sorted(Counter(row["tier"] for row in open_labels).items())),
        "maximum_selected_duration_seconds": max(row["duration_seconds"] for row in budgets),
        "case_overlap": 0, "video_overlap": 0,
        "use_audio_in_video": True, "video_reader": "pyav_seek", "audio_sample_rate": 16000,
        "audio_max_seconds": 300, "context_tokens": 32768, "text_reserve_tokens": 2048,
        "visual_budget_cap": 24000, "target_fps": 2.0, "max_total_frames": 300,
        "generation": dict(generation, do_sample=True),
        "nframes": summary([row["budget"]["nframes"] for row in budgets]),
        "visual_tokens": summary([row["budget"]["visual_budget_tokens"] for row in budgets]),
        "estimated_audio_budget_tokens": summary([row["budget"]["audio_budget_tokens"] for row in budgets]),
        "max_checked_tokens": summary([row["budget"]["max_checked_tokens"] for row in budgets]),
        "frame_dimensions_height_width": dict(sorted(Counter(
            f"{row['budget']['resized_height']}x{row['budget']['resized_width']}" for row in budgets).items())),
        "media_budget_audit_sha256": sha256(output / "media_budget_audit.jsonl"),
        "text_token_counts_without_chat_template": {mode: summary(counts) for mode, counts in text_counts.items()},
        "text_reserve_audit": "native tokenizer user-message tokens + completion cap + 128 template/system tokens <= 2048",
    }
    for mode, input_name, label_name, cap in (
        ("mcq", "worldsense.answer_free.jsonl", "worldsense.labels.jsonl", 8),
        ("openqa", "worldsense.openqa.jsonl", "worldsense.openqa.labels.jsonl", 512),
    ):
        manifest = dict(contract, evaluation_mode=mode, choices_visible_to_respondent=(mode == "mcq"),
                        answers_visible_to_respondent=False,
                        answer_free_path=str(output / input_name), answer_free_sha256=sha256(output / input_name),
                        labels_path=str(output / label_name), labels_sha256=sha256(output / label_name),
                        generation_max_new_tokens=cap)
        if mode == "openqa":
            manifest.update(instruction=OPEN_INSTRUCTION, same_training_clue_student_instruction=True,
                            options_dependent_wording_count=sum(row["options_dependent_wording"] for row in open_labels))
        name = "manifest.json" if mode == "mcq" else "openqa_manifest.json"
        frozen_write(output / name, (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode())
    frozen_write(output / "training_matched_contract.json", (json.dumps(contract, ensure_ascii=False, indent=2) + "\n").encode())
    print(json.dumps({"root": str(args.root.resolve()), "rows": len(ids), "unique_videos": len(eval_videos),
                      "generation": contract["generation"], "nframes": contract["nframes"],
                      "visual_tokens": contract["visual_tokens"], "text_tokens": contract["text_token_counts_without_chat_template"],
                      "frame_dimensions": contract["frame_dimensions_height_width"], "training_leakage": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
