#!/usr/bin/env python3
"""Encode one OPSD row with the same Swift Qwen-Omni template used by GKD.

This is a diagnostic helper for the vLLM/Transformers multimodal boundary.  It
does not load model weights, so it can isolate processor/template failures
before starting a distributed training run.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--max-pixels", type=int, default=7840)
    parser.add_argument("--mode", choices=("train", "vllm"), default="vllm")
    parser.add_argument("--truncation-strategy", choices=("raise", "left", "right", "split"), default="raise")
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            if index == args.index:
                row = json.loads(line)
                break
        else:
            raise IndexError(f"no non-empty row at physical line {args.index}")

    # Import after the arguments so environment variables such as
    # USE_AUDIO_IN_VIDEO and FORCE_QWENVL_VIDEO_READER are already visible.
    from swift.model import get_model_processor
    from swift.template import get_template

    processor = get_model_processor(args.model, load_model=False)[1]
    template = get_template(
        processor,
        template_type="qwen2_5_omni",
        max_length=args.max_length,
        max_pixels=args.max_pixels,
        truncation_strategy=args.truncation_strategy,
    )
    template.set_mode(args.mode)

    print(json.dumps({
        "case_id": row.get("case_id"),
        "mode": args.mode,
        "truncation_strategy": args.truncation_strategy,
        "use_audio_in_video": os.environ.get("USE_AUDIO_IN_VIDEO"),
    }, ensure_ascii=False), flush=True)
    try:
        encoded = template.encode(row, return_length=True)
    except Exception as exc:  # noqa: BLE001 - diagnostic must preserve traceback
        print(f"ENCODE_ERROR {type(exc).__name__}: {exc!r}", flush=True)
        traceback.print_exc()
        return 1
    print(json.dumps({
        "encoded_length": encoded.get("length"),
        "input_ids": len(encoded.get("input_ids", [])),
        "keys": sorted(encoded),
        "video_grid_thw": str(encoded.get("video_grid_thw")),
        "audio_features": str(getattr(encoded.get("input_features"), "shape", None)),
        "mm_processor_kwargs": repr(encoded.get("mm_processor_kwargs")),
        "decoded_prompt": processor.tokenizer.decode(encoded.get("input_ids", []), skip_special_tokens=False),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
