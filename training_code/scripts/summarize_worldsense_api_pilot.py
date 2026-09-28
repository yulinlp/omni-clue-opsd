#!/usr/bin/env python3
"""Summarize the API annotation pilot into a single stats.json.

Reads the merged pilot evidence, per-shard traces, the caption store and the
same-question local 30B results, then writes a JSON document that the review
website (and any report) can display directly.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not Path(path).is_file():
        return rows
    for line in Path(path).open(encoding="utf-8", errors="replace"):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def dist(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    p90 = ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "p90": round(p90, 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "histogram": {str(k): v for k, v in sorted(Counter(int(v) for v in values).items())},
    }


def last_traces(pattern: str) -> dict[str, dict[str, Any]]:
    last: dict[str, dict[str, Any]] = {}
    for path in glob.glob(pattern):
        for row in load_jsonl(path):
            if row.get("question_id"):
                last[row["question_id"]] = row
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="output/worldsense_api_200")
    parser.add_argument(
        "--local-evidence",
        default="output/worldsense_evidence_v2_formal/shard*/evidence.jsonl",
    )
    parser.add_argument(
        "--qa",
        default="/share/home/ylhu/Light-Omni/data/omni_benchmarks/WorldSense/worldsense_qa.json",
    )
    parser.add_argument("--output", default="output/worldsense_api_200/stats.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    evidence = load_jsonl(run_dir / "merged.evidence.jsonl")
    traces = last_traces(str(run_dir / "out" / "s*.trace.jsonl"))
    captions = load_jsonl(run_dir / "captions.jsonl")

    submitted = [r for r in evidence if r.get("status") == "submitted"]
    status_counts = Counter(r.get("status") for r in evidence)

    # ---- inspect / turns / elapsed -------------------------------------------------
    inspect_calls = [int(r.get("inspect_calls") or 0) for r in submitted]
    turns = [int(r.get("turns") or 0) for r in submitted]
    elapsed = [float(r.get("elapsed_s") or 0.0) for r in submitted]

    rejections: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    bootstrap: list[dict[str, Any]] = []
    for trace in traces.values():
        for event in trace.get("events") or []:
            kind = str(event.get("kind"))
            event_counts[kind] += 1
            if kind == "inspect_rejected":
                rejections[str(event.get("error"))[:60]] += 1
            if kind == "bootstrap":
                bootstrap.append(event)

    # Every bootstrap event in every trace line is a real API call, including
    # the ones from runs that were later restarted; summing them gives the true
    # caption call count for the pilot.
    caption_calls_all = 0
    for path in glob.glob(str(run_dir / "out" / "s*.trace.jsonl")):
        for trace in load_jsonl(path):
            for event in trace.get("events") or []:
                if str(event.get("kind")) == "bootstrap":
                    caption_calls_all += int(event.get("caption_attempts") or 1)

    # ---- caption stage -------------------------------------------------------------
    caption_sources = Counter(str(b.get("caption_source")) for b in bootstrap)
    caption_attempts = [int(b.get("caption_attempts") or 0) for b in bootstrap if b.get("caption_attempts")]
    caption_chars = [int(b.get("caption_chars") or 0) for b in bootstrap if b.get("caption_chars")]
    caption_seconds = [
        float(b.get("caption_seconds") or 0.0)
        for b in bootstrap
        if b.get("caption_source") == "generated"
    ]
    caption_truncated = sum(1 for b in bootstrap if str(b.get("caption_truncated")) == "True")
    caption_invalid = sum(1 for b in bootstrap if str(b.get("caption_valid")) == "False")

    # ---- intervals / observations --------------------------------------------------
    n_intervals = [len(r.get("clue_intervals") or []) for r in submitted]
    interval_lengths = [
        end - start
        for r in submitted
        for start, end in (r.get("clue_intervals") or [])
    ]
    obs_lengths = [len(r.get("observation") or "") for r in submitted]
    confidences = [float(r["confidence"]) for r in submitted if r.get("confidence") is not None]

    # ---- local 30B comparison ------------------------------------------------------
    local_rows = {
        r["question_id"]: r
        for path in glob.glob(args.local_evidence)
        for r in load_jsonl(path)
        if r.get("status") == "submitted"
    }
    both = [r for r in submitted if r["question_id"] in local_rows]

    def iou(a: list[float], b: list[float]) -> float:
        inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
        union = max(a[1], b[1]) - min(a[0], b[0])
        return inter / union if union > 0 else 0.0

    best_iou = [
        max(
            (iou(a, b) for a in r.get("clue_intervals") or [] for b in local_rows[r["question_id"]].get("clue_intervals") or []),
            default=0.0,
        )
        for r in both
    ]
    local_inspect = [int(local_rows[r["question_id"]].get("inspect_calls") or 0) for r in both]
    local_turns = [int(local_rows[r["question_id"]].get("turns") or 0) for r in both]

    # ---- V1 auto judge (partial same-question comparison) --------------------------
    v1: dict[str, Any] = {}
    for tag in ("api", "local"):
        rows = load_jsonl(run_dir / f"v1cmp_{tag}.verify.jsonl")
        judged = [r for r in rows if r.get("any_correct") is not None]
        v1[tag] = {
            "verified": len(judged),
            "sufficient": sum(1 for r in judged if r.get("any_correct") is True),
            "rate": round(
                sum(1 for r in judged if r.get("any_correct") is True) / len(judged), 3
            )
            if judged
            else None,
        }

    # ---- QA metadata (domains / task types) ---------------------------------------
    qa = json.load(open(args.qa, encoding="utf-8")) if Path(args.qa).is_file() else {}
    domains = Counter()
    task_types = Counter()
    for r in submitted:
        video_id, task = (r["question_id"].split("::") + [""])[:2]
        meta = qa.get(video_id, {})
        domains[str(meta.get("domain"))] += 1
        task_types[str(meta.get(f"{task}", {}).get("task_type"))] += 1

    manifest_path = run_dir / "merged.manifest.json"
    pilot_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )

    generated_calls = int(caption_sources.get("generated", 0))
    reused_calls = int(caption_sources.get("reused", 0))
    total_calls = generated_calls + sum(turns)

    stats = {
        "run_dir": str(run_dir),
        "questions": {
            "total": len(evidence),
            "submitted": len(submitted),
            "status": {str(k): v for k, v in status_counts.items()},
            "unique_videos": len({r["video_id"] for r in evidence}),
        },
        "tool_calls": {
            "inspect": dist([float(v) for v in inspect_calls]),
            "inspect_total": int(sum(inspect_calls)),
            "zero_inspect": sum(1 for v in inspect_calls if v == 0),
            "rejections_total": int(sum(rejections.values())),
            "rejections_by_reason": {k: v for k, v in rejections.most_common()},
            "get_media_info": int(event_counts.get("get_media_info", 0)),
        },
        "turns": {
            "per_question": dist([float(v) for v in turns]),
            "total": int(sum(turns)),
            "elapsed_seconds": dist(elapsed),
        },
        "api_calls": {
            "caption_generated_final": generated_calls,
            "caption_reused_final": reused_calls,
            "caption_calls_including_restarts": caption_calls_all,
            "caption_fallback_survey": int(event_counts.get("survey", 0)),
            "agent_turns": int(sum(turns)),
            "total_final_run": total_calls,
            "per_question_final_run": round(total_calls / max(1, len(evidence)), 2),
        },
        "caption_quality": {
            "sources": {str(k): v for k, v in caption_sources.items()},
            "attempts": dist([float(v) for v in caption_attempts]),
            "chars": dist([float(v) for v in caption_chars]),
            "generated_seconds": dist(caption_seconds),
            "generated_seconds_total": round(sum(caption_seconds), 1),
            "truncated": caption_truncated,
            "invalid": caption_invalid,
            "store_entries": len(captions),
            "store_unique_questions": len({c["question_id"] for c in captions}),
        },
        "annotations": {
            "intervals_per_question": dist([float(v) for v in n_intervals]),
            "interval_length_seconds": dist(interval_lengths),
            "observation_chars": dist([float(v) for v in obs_lengths]),
            "confidence": dist(confidences),
        },
        "events": {str(k): v for k, v in event_counts.most_common()},
        "pilot_sharding": {
            "note": (
                "200 题 pilot 的分片文件有重叠（454 行 / 200 唯一题），每题被跑了 2-3 次；"
                "合并文件按'最小分片号优先'取一次运行，两次区间不同的题占多数。"
                "全量 3172 题的 8 分片已校验无重叠。"
            ),
            **pilot_manifest,
        },
        "comparison_local_30b": {
            "same_questions": len(both),
            "api_inspect_mean": round(statistics.mean(inspect_calls), 2) if inspect_calls else None,
            "local_inspect_mean": round(statistics.mean(local_inspect), 2) if local_inspect else None,
            "api_turns_mean": round(statistics.mean(turns), 2) if turns else None,
            "local_turns_mean": round(statistics.mean(local_turns), 2) if local_turns else None,
            "interval_iou_median": round(statistics.median(best_iou), 3) if best_iou else None,
            "interval_iou_over_03": sum(1 for v in best_iou if v > 0.3),
        },
        "v1_auto_judge": v1,
        "domains": {str(k): v for k, v in domains.most_common()},
        "task_types": {str(k): v for k, v in task_types.most_common()},
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
