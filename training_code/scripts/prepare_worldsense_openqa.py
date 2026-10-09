#!/usr/bin/env python3
"""Convert the screened WorldSense MCQ rows into direct-answer SFT and CLUE rows.

The video sampling and clue intervals are copied unchanged. The option list is
removed from both model prompts; the correct option *text* becomes the target.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path


ANSWER_TAG = re.compile(r"<answer>\s*([A-D])\s*</answer>")
OPTION_LINE = re.compile(r"^([A-D])\.\s*(.+)$", re.MULTILINE)
INSTRUCTION = "Answer the question directly in natural language. Give only the answer, without analysis or an option letter."

# This option is relative to the discarded list. The source reasoning identifies
# the destination explicitly, so use that meaning as the open-ended target.
OPEN_ANSWER_OVERRIDES = {"xmHjHCiU::task0": "Her old filming location."}


def convert(sft: dict, clue: dict) -> tuple[dict, dict]:
    case_id = sft["case_id"]
    if clue["case_id"] != case_id:
        raise ValueError(f"SFT/CLUE row order differs: {case_id} != {clue['case_id']}")
    original_user = sft["messages"][0]["content"]
    if original_user.count("\nOptions:\n") != 1:
        raise ValueError(f"unexpected question/options layout: {case_id}")
    question, option_block = original_user.split("\nOptions:\n", 1)
    if not question.startswith("<video>\nQuestion: "):
        raise ValueError(f"missing video question prefix: {case_id}")
    options = dict(OPTION_LINE.findall(option_block))
    letter_match = ANSWER_TAG.search(sft["messages"][-1]["content"])
    if letter_match is None or letter_match.group(1) not in options:
        raise ValueError(f"missing correct option: {case_id}")
    letter = letter_match.group(1)
    option_text = options[letter].strip()
    answer = OPEN_ANSWER_OVERRIDES.get(case_id, option_text)
    if not answer or re.search(r"\b(?:all|none) of the above\b", answer, re.I):
        raise ValueError(f"answer still depends on the option list: {case_id}: {answer}")

    student_prompt = question + "\n" + INSTRUCTION
    teacher_video_prefix = "\n".join(["<video>"] * len(clue["teacher_videos"]))
    teacher_question = question.split("\n", 1)[1]
    teacher_prompt = (teacher_video_prefix + "\n" + teacher_question
                      + "\nVerified correct answer: " + answer + "\n" + INSTRUCTION)

    sft_out = copy.deepcopy(sft)
    sft_out["messages"] = [
        {"role": "user", "content": student_prompt},
        {"role": "assistant", "content": answer},
    ]
    sft_out["experiment_arm"] = "worldsense_openqa_sft"
    sft_out["gold_answer_text"] = answer
    sft_out["source_option_letter"] = letter

    clue_out = copy.deepcopy(clue)
    clue_out["messages"] = copy.deepcopy(sft_out["messages"])
    clue_out["teacher_prompt"] = teacher_prompt
    clue_out["experiment_arm"] = "worldsense_openqa_clue_opsd"
    clue_out["gold_answer_text"] = answer
    clue_out["source_option_letter"] = letter
    clue_out["solution"] = answer
    clue_out["response_format"] = "direct_answer"
    clue_out["supervision_contract"] = {
        "kind": "openqa-gold-answer-conditioned-clue-gkd",
        "student_prompt_has_gold": False,
        "teacher_prompt_has_gold": True,
        "dataset_completion_is_gold_answer_text": True,
        "teacher_view": "annotated-clue-intervals",
        "student_view": "full-video-uniform",
        "rollout_or_dataset_completion": True,
    }
    return sft_out, clue_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-source", type=Path, required=True)
    parser.add_argument("--clue-source", type=Path, required=True)
    parser.add_argument("--sft-output", type=Path, required=True)
    parser.add_argument("--clue-output", type=Path, required=True)
    args = parser.parse_args()
    sft_rows = [json.loads(line) for line in args.sft_source.open(encoding="utf-8")]
    clue_rows = [json.loads(line) for line in args.clue_source.open(encoding="utf-8")]
    if len(sft_rows) != len(clue_rows):
        raise ValueError(f"SFT/CLUE row counts differ: {len(sft_rows)} != {len(clue_rows)}")
    sft_ids = [row["case_id"] for row in sft_rows]
    if len(set(sft_ids)) != len(sft_ids):
        raise ValueError("duplicate case_id in source SFT data")
    args.sft_output.parent.mkdir(parents=True, exist_ok=True)
    args.clue_output.parent.mkdir(parents=True, exist_ok=True)
    with args.sft_output.open("w", encoding="utf-8") as sft_file, args.clue_output.open("w", encoding="utf-8") as clue_file:
        for sft, clue in zip(sft_rows, clue_rows):
            sft_out, clue_out = convert(sft, clue)
            sft_file.write(json.dumps(sft_out, ensure_ascii=False) + "\n")
            clue_file.write(json.dumps(clue_out, ensure_ascii=False) + "\n")
    print(f"wrote {len(sft_rows)} open-QA SFT and CLUE rows")


if __name__ == "__main__":
    main()
