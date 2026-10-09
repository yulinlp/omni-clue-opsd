#!/usr/bin/env python3
"""Audit and summarize paired WorldSense Full/G/H/T/S/V/A and optional E5 scores."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean


ATOMIC = (
    "E2_G_exact", "E4_H_halo3", "E7_T_8fps", "E8_S_highres",
    "E12_gold_v", "E13_A_audio_exact",
)
FULL = "E0_full_av"
E5 = "E5_global_coarse_dense"
LABELS = {FULL: "Full video", "E2_G_exact": "G", "E4_H_halo3": "H",
          "E7_T_8fps": "T", "E8_S_highres": "S", "E12_gold_v": "V",
          "E13_A_audio_exact": "A", E5: "E5 coarse+dense"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(root: Path, expected: set[str], arms: tuple[str, ...]) -> tuple[dict, list[dict], dict]:
    paths = sorted(root.glob("shard*/results.jsonl"))
    if len(paths) != 8:
        raise ValueError(f"expected eight result shards at {root}; got {len(paths)}")
    scored = [row for path in paths for row in read_jsonl(path)]
    by_id = {str(row["sample_id"]): row for row in scored}
    if len(scored) != len(by_id) or set(by_id) != expected:
        raise ValueError(f"score coverage mismatch at {root}: {len(scored)} rows, {len(by_id)} distinct IDs, expected {len(expected)}")
    unscored = [row for path in paths for row in read_jsonl(path.with_name("unscored.jsonl"))
                if path.with_name("unscored.jsonl").is_file()]
    if unscored:
        raise ValueError(f"{len(unscored)} unscored rows at {root}")
    identities = {json.dumps(row["model_identity"], sort_keys=True) for row in scored}
    contracts = {json.dumps(row["score_sampling_contract"], sort_keys=True) for row in scored}
    if len(identities) != 1 or len(contracts) != 1:
        raise ValueError(f"inconsistent model or sampling contract at {root}")
    contract = json.loads(next(iter(contracts)))
    if tuple(contract.pop("variants")) != arms:
        raise ValueError(f"wrong views at {root}")
    for sid, row in by_id.items():
        if set(row.get("scores") or {}) != set(arms):
            raise ValueError(f"missing views for {sid} at {root}")
    audit = {"root": str(root.resolve()), "model_identity": json.loads(next(iter(identities))),
             "base_sampling": contract,
             "shards": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in paths]}
    return by_id, scored, audit


def summarize(canonical: Path, sft_ids: Path, atomic_root: Path, full_root: Path,
              e5_root: Path | None = None) -> tuple[dict, list[dict]]:
    source = read_jsonl(canonical)
    by_id = {str(row["sample_id"]): row for row in source}
    if len(source) != 3079 or len(by_id) != len(source):
        raise ValueError("canonical candidates must contain 3,079 distinct rows")
    selected_list = [line.strip() for line in sft_ids.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = set(selected_list)
    if len(selected_list) != 1453 or len(selected) != len(selected_list) or not selected <= set(by_id):
        raise ValueError("SFT cohort must contain 1,453 distinct candidate IDs")
    atomic, _, atomic_audit = load_run(atomic_root, set(by_id), ATOMIC)
    full, _, full_audit = load_run(full_root, set(by_id), (FULL,))
    if atomic_audit["model_identity"] != full_audit["model_identity"] or atomic_audit["base_sampling"] != full_audit["base_sampling"]:
        raise ValueError("Full and atomic runs have different model or base sampling contracts")
    e5 = None
    e5_audit = None
    if e5_root is not None:
        e5, _, e5_audit = load_run(e5_root, set(by_id), (E5,))
        if atomic_audit["model_identity"] != e5_audit["model_identity"] or atomic_audit["base_sampling"] != e5_audit["base_sampling"]:
            raise ValueError("E5 and atomic runs have different model or base sampling contracts")

    observed_frames = []
    duration_over_cap = 0
    decoder_backends: Counter[str] = Counter()
    with_observed_audio = 0
    for sid, row in by_id.items():
        letters = "ABCD"[:len(row["choices"])]
        runs = [(atomic, ATOMIC), (full, (FULL,))]
        if e5 is not None:
            runs.append((e5, (E5,)))
        for run, arms in runs:
            record = run[sid]
            if record["choice_letters"] != letters or record["answer"] != row["answer"]:
                raise ValueError(f"choice or answer mismatch for {sid}")
            for arm in arms:
                score = record["scores"][arm]
                probs = score["probabilities"]
                if set(probs) != set(letters) or abs(sum(probs.values()) - 1) > 1e-5 or score["predicted"] != max(probs, key=probs.get):
                    raise ValueError(f"invalid option scores for {sid}/{arm}")
        item = full[sid]["scores"][FULL]
        contract = item["e_series_contract"]
        if (contract["video_intervals"] != [[0.0, float(row["duration"])]]
                or not contract["use_audio_in_video"]
                or not item["observed_media"]["videos"]):
            raise ValueError(f"Full audiovisual timeline not present for {sid}")
        frames = sum(int(video["decoded_tensor_shape"][0]) for video in item["observed_media"]["videos"])
        observed_frames.append(frames)
        decoder_backends.update(video["video_backend"] for video in item["observed_media"]["videos"])
        with_observed_audio += bool(contract.get("observed_audio_sample_counts"))
        duration_over_cap += float(row["duration"]) > 384.0
        if e5 is not None:
            e5_item = e5[sid]["scores"][E5]
            e5_contract = e5_item["e_series_contract"]
            intervals = e5_contract["video_intervals"]
            per_video = e5_contract["sampling"]["per_video"]
            if (e5_contract["arm"] != E5 or e5_contract["use_audio_in_video"]
                    or len(intervals) < 2 or intervals[-1] != [0.0, float(row["duration"])]
                    or e5_contract["audio_intervals"] != row["evidence_spans"]
                    or len(e5_item["observed_media"]["videos"]) != len(intervals)
                    or len(e5_contract.get("observed_audio_sample_counts") or []) != len(row["evidence_spans"])
                    or any(spec["fps"] != 8.0 for spec in per_video[:-1])
                    or per_video[-1]["fps"] != 0.25
                    or per_video[-1]["max_pixels"] != 102400):
                raise ValueError(f"E5 media contract differs for {sid}")

    cohorts = {"all_3079": list(by_id), "sft_1453": [sid for sid in by_id if sid in selected]}
    table = []
    for cohort, ids in cohorts.items():
        tasks: dict[str, list[str]] = defaultdict(list)
        for sid in ids:
            tasks[str(by_id[sid]["question_type"])].append(sid)
        for task, members in {"overall": ids, **dict(sorted(tasks.items()))}.items():
            for arm in (FULL, *ATOMIC, *((E5,) if e5 is not None else ())):
                correct = 0
                probabilities = []
                tokens = []
                g_correct = 0
                for sid in members:
                    answer = by_id[sid]["answer"]
                    run = full if arm == FULL else e5 if arm == E5 else atomic
                    score = run[sid]["scores"][arm]
                    anchor = atomic[sid]["scores"]["E2_G_exact"]
                    correct += score["predicted"] == answer
                    g_correct += anchor["predicted"] == answer
                    probabilities.append(float(score["probabilities"][answer]))
                    tokens.append(int(score["input_tokens"]))
                table.append({"cohort": cohort, "task": task, "view": LABELS[arm],
                              "arm": arm, "questions": len(members),
                              "videos": len({by_id[sid]["video_id"] for sid in members}),
                              "correct": correct, "accuracy": correct / len(members),
                              "mean_answer_probability": fmean(probabilities),
                              "mean_input_tokens": fmean(tokens),
                              "delta_accuracy_vs_G": (correct - g_correct) / len(members)})

    summary = {"dataset": "WorldSense", "status": "complete",
               "canonical": str(canonical.resolve()), "canonical_sha256": sha256(canonical),
               "sft_ids": str(sft_ids.resolve()), "sft_ids_sha256": sha256(sft_ids),
               "candidate_questions": len(source), "sft_questions": len(selected),
               "model_identity": atomic_audit["model_identity"],
               "base_sampling": atomic_audit["base_sampling"],
               "atomic_run": atomic_audit, "full_run": full_audit,
               "full_video_audit": {"videos_over_384_seconds": duration_over_cap,
                                    "decoder_backends": dict(sorted(decoder_backends.items())),
                                    "questions_with_observed_audio": with_observed_audio,
                                    "min_decoded_frames": min(observed_frames),
                                    "max_decoded_frames": max(observed_frames),
                                    "mean_decoded_frames": fmean(observed_frames)},
               "metric": "Top-1 of option-normalized first-token probability; no chain-of-thought generation",
               "scope": "3,079 evidence-bearing screened WorldSense candidates and the 1,453 actual SFT case IDs; SFT is a selected training cohort, not a held-out test set"}
    if e5_audit is not None:
        summary["e5_run"] = e5_audit
        summary["e5_view"] = "Gold video at 8 FPS plus full video at 0.25 FPS, with explicit Gold audio; composite diagnostic"
    return summary, table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--sft-ids", type=Path, required=True)
    parser.add_argument("--atomic-root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--e5-root", type=Path, help="optional complete eight-shard E5 run")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary, table = summarize(args.canonical, args.sft_ids, args.atomic_root, args.full_root, args.e5_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "accuracy_by_view_and_task.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    overall = [row for row in table if row["task"] == "overall"]
    title = "# WorldSense Full video、六视角与 E5 评测" if args.e5_root else "# WorldSense Full video 与六视角评测"
    lines = [title, "",
             "同一 Qwen2.5-Omni-7B 模型、选项归一化首 token 概率；正确率为最高概率选项与答案一致的比例。", "",
             "| 题集 | 视角 | 答对/题数 | 正确率 | 相对 G 变化 |", "|---|---|---:|---:|---:|"]
    for row in overall:
        lines.append(f"| {row['cohort']} | {row['view']} | {row['correct']}/{row['questions']} | {row['accuracy']:.2%} | {row['delta_accuracy_vs_G']:+.2%} |")
    lines.extend(["", "全量为 3,079 条带证据标注的候选题；1,453 条为 SFT 实际 case_id，属于筛选后的训练题集。",
                  "超过 384 秒的视频仍在完整时间范围均匀采样，最多取 768 帧；任务细分见 CSV。", ""])
    if args.e5_root:
        lines.extend(["E5 是组合诊断：Gold 片段 8 FPS，加整段视频 0.25 FPS，并单独输入 Gold 音频。", ""])
    (args.output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "candidate_questions": 3079,
                      "sft_questions": 1453, "report": str(args.output_dir)}))


if __name__ == "__main__":
    main()
