#!/usr/bin/env python3
"""Build the WorldSense SFT JSONL (dynamic budget, reasoning format).

Input:
  --selected    outputs/worldsense_gap/metrics/selected_1500_300s.jsonl
  --annotation  output/worldsense_evidence_api_full/merged.evidence.jsonl
                (teacher observation used as the analysis supervision)
  --qa          WorldSense QA json
Output:
  swift SFT jsonl with messages / videos / dynamic_student_budget, matching the
  GAP5000 dynamic-budget schema so the same training entry points apply.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omni_opsd.data.dynamic_budget import (
    dynamic_budget_for,
    dynamic_sampling_contract,
    dynamic_video_spec,
)

REASONING_PROMPT = (
    "Question: {question}\n"
    "Options:\n{options}\n"
    "Briefly analyze the video and audio evidence relevant to the question and compare the answer options. "
    "Write the analysis first, then give exactly one option letter inside <answer>...</answer>. "
    "Keep the analysis concise (at most 120 words)."
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).open(encoding="utf-8", errors="replace") if l.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selected", type=Path, required=True)
    p.add_argument("--annotation", type=Path, required=True)
    p.add_argument("--qa", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--answer-only", action="store_true", help="drop the analysis supervision")
    a = p.parse_args()

    selected = read_jsonl(a.selected)
    annotation = {r["question_id"]: r for r in read_jsonl(a.annotation)}
    qa = json.loads(a.qa.read_text(encoding="utf-8"))

    rows, skipped = [], []
    for row in selected:
        question_id = row["sample_id"]
        video_id, _, task = question_id.partition("::")
        meta = (qa.get(video_id) or {}).get(task) or {}
        choices = [c.split(". ", 1)[1] if ". " in c else c for c in (meta.get("candidates") or [])]
        answer = str(meta.get("answer") or "").strip().upper()
        if len(choices) not in (3, 4) or answer not in "ABCD"[: len(choices)]:
            skipped.append((question_id, "invalid options/answer"))
            continue
        source = annotation.get(question_id)
        if not source or source.get("status") != "submitted":
            skipped.append((question_id, "no submitted annotation"))
            continue
        duration = float(row["duration"])
        video_path = f"/share/home/ylhu/Light-Omni/data/omni_benchmarks/WorldSense/videos/{video_id}.mp4"
        media_row = {
            "duration": duration,
            "video_path": video_path,
            "metadata": {"resolution": f"{row.get('source_width', 640)}x{row.get('source_height', 360)}"},
        }
        budget = dynamic_budget_for(media_row)
        observation = str(source.get("observation") or "").strip()
        if a.answer_only or not observation:
            target = f"<answer>{answer}</answer>"
        else:
            target = f"{observation}\n<answer>{answer}</answer>"
        options = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": "<video>\n" + REASONING_PROMPT.format(question=meta.get("question"), options=options)},
                    {"role": "assistant", "content": target},
                ],
                "videos": [dynamic_video_spec(media_row, budget)],
                "prompt_id": question_id,
                "case_id": question_id,
                "video_id": video_id,
                "response_format": "reasoning",
                "experiment_arm": "sft",
                "sft_target_source": "gold_answer_only" if (a.answer_only or not observation) else "api_annotation_observation + gold_answer",
                "supervision_contract": {
                    "kind": "gold-answer-reasoning-teacher-forcing",
                    "gold_answer_in_student_target": True,
                    "gold_answer_in_reward": False,
                    "gold_answer_in_teacher_prompt": False,
                    "teacher_view": None,
                },
                "sampling_contract": dynamic_sampling_contract(budget, use_audio_in_video=True),
                "dynamic_student_budget": budget,
            }
        )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    report = {
        "rows": len(rows),
        "skipped": len(skipped),
        "skipped_reasons": {},
        "videos": len({r["video_id"] for r in rows}),
        "answer_only": a.answer_only,
        "with_analysis": sum(1 for r in rows if "analysis" in r["messages"][1]["content"] or len(r["messages"][1]["content"]) > 40),
    }
    from collections import Counter

    report["skipped_reasons"] = dict(Counter(reason for _, reason in skipped))
    (a.output.parent / "prepare_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
