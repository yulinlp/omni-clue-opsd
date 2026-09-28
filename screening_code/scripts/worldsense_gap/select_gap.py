#!/usr/bin/env python3
"""Paired Full/Gold metrics, tiers and the N-sample selection for WorldSense.

Sort key (descending, mirroring the OmniVideo aggregate policy):
    Δacc -> Δp -> Δlogp -> P(GT|Gold) -> sample_id
Selection: cap 5 questions per video, take the top N.

Tiers:
    A  Δacc = +1                      core gain (Full wrong -> Gold right)
    B  Δacc = 0 and Δp >= 0.10        probability gain on unchanged outcomes
    C  Δacc = -1                      suspect annotations -> audit list
    D  the rest
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    return [json.loads(l) for l in Path(path).open(encoding="utf-8", errors="replace") if l.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--canonical", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--target-size", type=int, default=2000)
    p.add_argument("--max-duration", type=float, default=0.0, help="drop questions longer than this (0=keep all)")
    p.add_argument("--max-per-video", type=int, default=5)
    a = p.parse_args()

    canon = {r["sample_id"]: r for r in read_jsonl(a.canonical)}
    full = {r["sample_id"]: r for r in read_jsonl(a.run_dir / "full_run/full.jsonl")}
    gold = {r["sample_id"]: r for r in read_jsonl(a.run_dir / "gold_run/gold.jsonl")}
    ids = sorted(set(full) & set(gold))
    if a.max_duration > 0:
        ids = [sid for sid in ids if float(canon[sid]["duration"]) <= a.max_duration]
    if not ids:
        raise SystemExit("no paired scores found")

    rows = []
    for sid in ids:
        f, g, c = full[sid], gold[sid], canon[sid]
        p_f = f["answer_probability"]
        p_g = g["answer_probability"]
        logp_f = f["logprobs"][f["answer"]]
        logp_g = g["logprobs"][g["answer"]]
        dacc = (1 if g["correct"] else 0) - (1 if f["correct"] else 0)
        rows.append(
            {
                "sample_id": sid,
                "video_id": c["video_id"],
                "question_type": c["question_type"],
                "task_domain": c.get("task_domain"),
                "duration": c["duration"],
                "n_spans": len(c["evidence_spans"]),
                "gold_total_seconds": round(sum(b - x for x, b in c["evidence_spans"]), 2),
                "answer": c["answer"],
                "pred_full": f["predicted"],
                "pred_gold": g["predicted"],
                "correct_full": f["correct"],
                "correct_gold": g["correct"],
                "p_gt_full": round(p_f, 6),
                "p_gt_gold": round(p_g, 6),
                "logp_full": round(logp_f, 6),
                "logp_gold": round(logp_g, 6),
                "delta_acc": dacc,
                "delta_p": round(p_g - p_f, 6),
                "delta_logp": round(logp_g - logp_f, 6),
                "prompt_tokens_full": f.get("prompt_tokens"),
                "prompt_tokens_gold": g.get("prompt_tokens"),
            }
        )
        if dacc == 1:
            rows[-1]["tier"] = "A"
        elif dacc == 0 and rows[-1]["delta_p"] >= 0.10:
            rows[-1]["tier"] = "B"
        elif dacc == -1:
            rows[-1]["tier"] = "C"
        else:
            rows[-1]["tier"] = "D"

    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_question.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )

    n = len(rows)
    fa = sum(r["correct_full"] for r in rows) / n
    ga = sum(r["correct_gold"] for r in rows) / n
    dp = [r["delta_p"] for r in rows]
    b = sum(1 for r in rows if not r["correct_full"] and r["correct_gold"])
    c_ = sum(1 for r in rows if r["correct_full"] and not r["correct_gold"])
    # exact McNemar (two-sided, capped at 1)
    tail = sum(math.comb(b + c_, k) for k in range(0, min(b, c_) + 1)) / 2 ** (b + c_)
    mcnemar = min(1.0, 2 * tail)
    tiers = Counter(r["tier"] for r in rows)
    summary = {
        "questions": n,
        "full_accuracy": round(fa, 4),
        "gold_accuracy": round(ga, 4),
        "accuracy_gap_pp": round((ga - fa) * 100, 2),
        "delta_p_mean": round(statistics.mean(dp), 4),
        "delta_p_median": round(statistics.median(dp), 4),
        "delta_p_positive": sum(1 for x in dp if x > 0),
        "delta_p_ge_0.1": sum(1 for x in dp if x >= 0.1),
        "flips_full_wrong_to_gold_right": b,
        "flips_gold_wrong_to_full_right": c_,
        "mcnemar_p": round(mcnemar, 4),
        "tiers": dict(tiers),
        "videos": len({r["video_id"] for r in rows}),
    }
    # per task type
    by = defaultdict(list)
    for r in rows:
        by[r["question_type"] or "?"].append(r)
    per_task = {}
    for task, v in sorted(by.items(), key=lambda x: -len(x[1])):
        per_task[task] = {
            "n": len(v),
            "full": round(sum(x["correct_full"] for x in v) / len(v), 4),
            "gold": round(sum(x["correct_gold"] for x in v) / len(v), 4),
            "delta_acc": round(sum(x["delta_acc"] for x in v) / len(v), 3),
            "delta_p": round(statistics.mean(x["delta_p"] for x in v), 4),
        }
    (out / "summary.json").write_text(
        json.dumps({"overall": summary, "per_task": per_task}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    # ---- selection ---------------------------------------------------------
    ordered = sorted(
        rows,
        key=lambda r: (-r["delta_acc"], -r["delta_p"], -r["delta_logp"], -r["p_gt_gold"], r["sample_id"]),
    )
    selected, per_video = [], Counter()
    for r in ordered:
        if len(selected) >= a.target_size:
            break
        if per_video[r["video_id"]] >= a.max_per_video:
            continue
        per_video[r["video_id"]] += 1
        selected.append(r)
    sel_stats = {
        "selected": len(selected),
        "videos": len({r["video_id"] for r in selected}),
        "full_accuracy": round(sum(r["correct_full"] for r in selected) / len(selected), 4),
        "gold_accuracy": round(sum(r["correct_gold"] for r in selected) / len(selected), 4),
        "accuracy_gap_pp": round(
            (sum(r["correct_gold"] for r in selected) - sum(r["correct_full"] for r in selected)) / len(selected) * 100, 2
        ),
        "delta_p_mean": round(statistics.mean(r["delta_p"] for r in selected), 4),
        "tiers": dict(Counter(r["tier"] for r in selected)),
    }
    tag = f"_{int(a.max_duration)}s" if a.max_duration else ""
    (out / f"selected_{a.target_size}{tag}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected), encoding="utf-8"
    )
    (out / "audit_tier_c.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows if r["tier"] == "C"),
        encoding="utf-8",
    )
    sel_stats["max_duration"] = a.max_duration
    (out / f"selection_summary{tag}.json").write_text(
        json.dumps(sel_stats, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps({"overall": summary, "selection": sel_stats}, ensure_ascii=False, indent=1))
    print("\nper-task (top 12):")
    for task, v in list(per_task.items())[:12]:
        print(f"  {task[:32]:34s} n={v['n']:>4d} full={v['full']*100:>5.1f}% gold={v['gold']*100:>5.1f}% Δacc={v['delta_acc']:+.2f} Δp={v['delta_p']:+.4f}")


if __name__ == "__main__":
    main()
