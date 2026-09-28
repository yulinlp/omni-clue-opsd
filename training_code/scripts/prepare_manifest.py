#!/usr/bin/env python3
"""Convert supported Omni datasets to the canonical JSONL manifest."""

from __future__ import annotations

import argparse

from omni_opsd.data.common import write_jsonl
from omni_opsd.data.omnivideo_100k import iter_omnivideo_100k
from omni_opsd.data.omnivideo_test import iter_omnivideo_test
from omni_opsd.data.video_odyssey import iter_video_odyssey
from omni_opsd.data.videomme_v2 import iter_videomme_v2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=("video_odyssey", "omnivideo_100k", "omnivideo_test", "videomme_v2"),
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--subtitle-root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.dataset == "video_odyssey":
        samples = iter_video_odyssey(
            args.source, video_root=args.video_root, subtitle_root=args.subtitle_root
        )
    elif args.dataset == "omnivideo_100k":
        samples = iter_omnivideo_100k(args.source, video_root=args.video_root)
    elif args.dataset == "omnivideo_test":
        samples = iter_omnivideo_test(args.source, video_root=args.video_root)
    else:
        samples = iter_videomme_v2(args.source, video_root=args.video_root)
    count = write_jsonl((sample.to_record() for sample in samples), args.output)
    print(f"wrote {count} samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
