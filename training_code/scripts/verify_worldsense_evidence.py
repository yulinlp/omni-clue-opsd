#!/usr/bin/env python3
"""Run V1 sufficiency verification over localized WorldSense evidence intervals.

For every candidate interval the teacher sees ONLY the interval media plus the
question and must pick the correct option.  Output rows record correctness and
the letter distribution (p_true / margin / entropy) per interval.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from omni_opsd.worldsense.clients import TransformersOmniClient
from omni_opsd.worldsense.dataset import load_questions
from omni_opsd.worldsense.media import MediaIndex
from omni_opsd.worldsense.verify import VerifierConfig, verify_question

DEFAULT_MODEL = "/share/home/ylhu/models/Qwen3-Omni-30B-A3B-Instruct"
HIGH_RES_TASKS = {
    "Text and Diagram Understanding",
    "Attribute Recognition",
    "Attribute Reasoning",
    "Object Counting",
    "Action Counting",
    "Fine-grained Perception",
}
HIGH_RES_PIXELS = 313_600


def verifier_for_record(record, base: VerifierConfig) -> VerifierConfig:
    """Task-type-aware spatial budget: small text/detail tasks get more pixels."""

    if record.task_type in HIGH_RES_TASKS and base.max_pixels < HIGH_RES_PIXELS:
        return replace(base, max_pixels=HIGH_RES_PIXELS)
    return base


class MockVerifierClient:
    """CPU stand-in: picks a letter deterministically and exposes a fake score vector."""

    def __init__(self, letters: tuple[str, ...] = ("A", "B", "C", "D")):
        self.letters = letters

    def generate_with_scores(self, messages, *, max_new_tokens: int = 4):
        text_blob = json.dumps(messages, ensure_ascii=False)
        index = sum(ord(char) for char in text_blob) % len(self.letters)
        letter = self.letters[index]
        scores = [0.1] * len(self.letters)
        scores[index] = 2.0
        return f"<answer>{letter}</answer>", scores

    def option_token_ids(self, letters):
        return {letter: [index] for index, letter in enumerate(letters)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True, help="localization JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--media-index-cache", type=Path, default=None)
    parser.add_argument("--backend", choices=("local", "api"), default="local",
                        help="local = on-GPU Transformers; api = Qwen3.8-Omni-Flash cloud API")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-model", default="qwen3.8-omni-flash")
    parser.add_argument("--api-base-url", default=None, help="or env DASHSCOPE_BASE_URL")
    parser.add_argument("--api-key", default=None, help="or env DASHSCOPE_API_KEY")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--api-enable-thinking", action="store_true")
    parser.add_argument("--clip-cache-dir", default="output/worldsense_api_clips")
    parser.add_argument("--max-clip-mb", type=float, default=7.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--max-pixels", type=int, default=156_800)
    parser.add_argument("--modality", default="av", choices=("av", "video", "audio"))
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question-ids", default=None)
    parser.add_argument("--question-id-file", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--mock", action="store_true", help="CPU scripted verifier")
    return parser.parse_args()


def _split(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _read_id_file(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def _load_evidence(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            question_id = str(row.get("question_id") or "")
            if question_id:
                rows[question_id] = row
    return rows


def _done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("question_id"):
                done.add(str(row["question_id"]))
    return done


def main() -> int:
    args = parse_args()
    evidence = _load_evidence(args.evidence)
    if not evidence:
        print(f"no evidence rows in {args.evidence}")
        return 1

    question_ids = _read_id_file(args.question_id_file) or _split(args.question_ids)
    if question_ids is None:
        question_ids = set(evidence)
    else:
        question_ids &= set(evidence)

    media_index = MediaIndex.load(args.media_index_cache) if args.media_index_cache else MediaIndex()
    records = load_questions(
        args.qa,
        args.video_root,
        media_index,
        question_ids=question_ids,
        limit=args.limit,
    )
    print(f"verifying {len(records)} questions from {args.evidence}")

    verifier = VerifierConfig(
        fps=args.fps,
        max_pixels=args.max_pixels,
        modality=args.modality,
        max_new_tokens=args.max_new_tokens,
    )
    if args.mock:
        client = MockVerifierClient(verifier.letters)
    elif args.backend == "api":
        from omni_opsd.worldsense.api_client import QwenAPIOmniClient

        print(f"using cloud API verifier: {args.api_model}")
        client = QwenAPIOmniClient(
            model=args.api_model,
            base_url=args.api_base_url,
            api_key=args.api_key,
            api_key_env=args.api_key_env,
            clip_cache_dir=args.clip_cache_dir,
            max_clip_mb=args.max_clip_mb,
            media_index=media_index,
            enable_thinking=args.api_enable_thinking,
        )
    else:
        print(f"loading verifier teacher {args.model} on {args.device}")
        client = TransformersOmniClient(
            args.model,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn,
        )

    done = set() if args.no_resume else _done_ids(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    by_status: dict[str, int] = {}
    with args.output.open("a", encoding="utf-8") as handle:
        for record in records:
            if record.question_id in done:
                continue
            row = evidence.get(record.question_id) or {}
            intervals = [
                {"start": float(item["start"]), "end": float(item["end"])}
                for item in (row.get("intervals") or [])
                if isinstance(item, dict) and "start" in item and "end" in item
            ]
            if str(row.get("status")) != "submitted" or not intervals:
                summary = {
                    "question_id": record.question_id,
                    "video_id": record.video_id,
                    "task_type": record.task_type,
                    "gold_letter": (record.answer_letter or "").upper() or None,
                    "n_intervals": 0,
                    "any_correct": None,
                    "n_correct": 0,
                    "results": [],
                    "verifier": verifier.as_dict(),
                    "skipped": str(row.get("status") or "missing"),
                }
            else:
                summary = verify_question(
                    client, record, intervals, verifier_for_record(record, verifier)
                )
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
            handle.flush()
            processed += 1
            if summary.get("any_correct") is True:
                key = "sufficient"
            elif summary.get("n_intervals"):
                key = "insufficient"
            else:
                key = "skipped"
            by_status[key] = by_status.get(key, 0) + 1
            print(
                f"[{processed}] {record.question_id} n={summary['n_intervals']} "
                f"correct={summary.get('n_correct')} any={summary.get('any_correct')} "
                f"best_p_true={summary.get('best_p_true')}",
                flush=True,
            )

    sufficient = by_status.get("sufficient", 0)
    total_checked = sufficient + by_status.get("insufficient", 0)
    rate = (sufficient / total_checked) if total_checked else None
    print(
        json.dumps(
            {
                "processed": processed,
                "by_status": by_status,
                "sufficiency_rate": round(rate, 4) if rate is not None else None,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
