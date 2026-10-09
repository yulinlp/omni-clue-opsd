#!/usr/bin/env python3
"""Score WorldSense E-series views with the Omni-OPSD media code.

Each process owns one NPU and one disjoint shard.  WorldSense has both three-
and four-choice questions, so option probabilities are normalized only over
the letters actually shown to the model.  The frozen OmniVideo scorer and its
four-choice contract are left untouched.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


ARMS = (
    "E2_G_exact", "E4_H_halo3", "E7_T_8fps", "E8_S_highres",
    "E12_gold_v", "E13_A_audio_exact",
)
ALLOWED_ARMS = (*ARMS, "E0_full_av", "E5_global_coarse_dense")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Omni-OPSD checkout")
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unscored-output", type=Path, required=True)
    parser.add_argument("--variants", default=",".join(ARMS))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sort-duration-desc", action="store_true")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="validate all media descriptors without loading the model")
    args = parser.parse_args()
    arms = tuple(part.strip() for part in args.variants.split(",") if part.strip())
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= set(ALLOWED_ARMS):
        raise ValueError(f"invalid views: {arms}")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard assignment")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    source = args.source_root.resolve()
    sys.path.insert(0, str(source / "scripts"))
    sys.path.insert(0, str(source / "src"))
    import score_omnivideo_e_series_mcq as e_series

    rows = [e_series._canonical_e(row, Path("/")) for row in read_jsonl(args.annotation)]
    if args.sort_duration_desc:
        rows.sort(key=lambda row: (-float(row["duration"]), row["sample_id"]))
    if args.limit is not None:
        rows = rows[:args.limit]
    rows = rows[args.shard_index :: args.num_shards]
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sample IDs in shard")
    for row in rows:
        count = len(row["choices"])
        if count not in (3, 4) or row["answer"] not in "ABCD"[:count]:
            raise ValueError(f"invalid WorldSense choices: {row['sample_id']}")
        if not Path(row["video_path"]).is_file():
            raise FileNotFoundError(row["video_path"])
        if not row["evidence_spans"]:
            raise ValueError(f"missing evidence: {row['sample_id']}")
        if args.validate_only:
            for arm in arms:
                e_series._conversation_e(row, arm)
    if args.validate_only:
        print(json.dumps({"validated_rows": len(rows), "views": list(arms)}), flush=True)
        return

    done = {r["sample_id"]: r for r in read_jsonl(args.output)} if args.resume else {}
    failed = {r["sample_id"]: r for r in read_jsonl(args.unscored_output)} if args.resume else {}
    if not set(done) <= set(ids) or not set(failed) <= set(ids):
        raise ValueError("resume files contain IDs outside the shard")
    for sample_id, record in done.items():
        if set(record.get("scores") or {}) != set(arms):
            raise ValueError(f"resume views differ: {sample_id}")

    import torch
    torch.set_num_threads(int(os.environ.get("OMNI_OPSD_TORCH_THREADS", "1")))
    torch.set_num_interop_threads(1)
    import torch_npu  # noqa: F401
    from qwen_omni_utils import process_mm_info
    if {"E0_full_av", "E5_global_coarse_dense"} & set(arms):
        # torchvision.read_video materializes every Full-video frame before
        # sampling and can exhaust RAM on long/high-resolution WorldSense
        # clips. E5 also has a full-timeline video branch. Reuse the
        # Omni-OPSD streaming PyAV reader, which preserves frame indices
        # while holding only selected RGB frames.
        from omni_opsd.data.sparse_full_video import install_sparse_full_reader
        install_sparse_full_reader()
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation=os.environ.get("OMNI_OPSD_ATTN_IMPL", "sdpa"),
        device_map=args.device,
    )
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model, local_files_only=True)
    all_letter_ids = e_series.base._letter_ids(processor.tokenizer)
    identity = e_series.base._model_identity(args.model)
    started = time.monotonic()
    with e_series.installed():
        for index, row in enumerate(rows, 1):
            sample_id = row["sample_id"]
            if sample_id in done:
                continue
            letters = "ABCD"[:len(row["choices"])]
            letter_ids = {letter: all_letter_ids[letter] for letter in letters}
            record = {
                "sample_id": sample_id,
                "video_id": row["video_id"],
                "question_type": row["question_type"],
                "answer": row["answer"],
                "choice_letters": letters,
                "evidence_spans": row["evidence_spans"],
                "model_identity": identity,
                "score_sampling_contract": {**e_series.SAMPLING, "variants": list(arms)},
                "sparse_full_decode_requested": bool({"E0_full_av", "E5_global_coarse_dense"} & set(arms)),
                "scores": {},
            }
            try:
                for arm in arms:
                    record["scores"][arm] = e_series.base._score_one(
                        model, processor, process_mm_info, row, arm,
                        letter_ids=letter_ids, torch_module=torch, **e_series.SAMPLING,
                    )
            except Exception as exc:
                failed[sample_id] = {
                    "sample_id": sample_id, "video_id": row["video_id"],
                    "failed_view": arm, "error_type": type(exc).__name__, "error": str(exc),
                }
            else:
                record["answer_probability"] = {
                    arm: float(record["scores"][arm]["probabilities"][row["answer"]])
                    for arm in arms
                }
                done[sample_id] = record
                failed.pop(sample_id, None)
            if index == 1 or index % 5 == 0:
                write_jsonl(args.output, list(done.values()))
                write_jsonl(args.unscored_output, list(failed.values()))
                print(json.dumps({"progress": index, "total": len(rows), "scored": len(done),
                                  "unscored": len(failed), "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    write_jsonl(args.output, list(done.values()))
    write_jsonl(args.unscored_output, list(failed.values()))
    print(json.dumps({"rows": len(rows), "scored": len(done), "unscored": len(failed)}), flush=True)


if __name__ == "__main__":
    main()
