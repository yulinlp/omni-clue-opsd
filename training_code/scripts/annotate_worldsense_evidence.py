#!/usr/bin/env python3
"""Run the WorldSense adaptive evidence-localization agent.

Localization-only build: the agent explores the video through the ``inspect``
tool and submits candidate evidence intervals with per-interval observations.
Candidate fusion and sufficiency verification are intentionally not part of
this stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from omni_opsd.worldsense.clients import MockOmniClient, TransformersOmniClient
from omni_opsd.worldsense.config import AgentConfig
from omni_opsd.worldsense.dataset import load_questions
from omni_opsd.worldsense.media import MediaIndex
from omni_opsd.worldsense.prompts import build_initial_user_prompt, build_system_prompt
from omni_opsd.worldsense.runner import run_batch
from omni_opsd.worldsense.views import MIN_PIXELS, clamp_view, render_conversation

DEFAULT_MODEL = "/share/home/ylhu/models/Qwen3-Omni-30B-A3B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, required=True, help="worldsense_qa.json")
    parser.add_argument("--video-root", type=Path, required=True, help="directory with <video_id>.mp4")
    parser.add_argument("--output", type=Path, required=True, help="evidence-localization JSONL")
    parser.add_argument("--trace", type=Path, default=None, help="per-episode trace JSONL")
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
    parser.add_argument("--proxy-policy", choices=("size_aware", "fixed"), default="size_aware",
                        help="API mode: size_aware keeps the highest resolution that fits the byte budget")
    parser.add_argument("--no-caption", action="store_true",
                        help="API mode: skip the timestamped full-video caption prelude")
    parser.add_argument("--caption-max-tokens", type=int, default=8000)
    parser.add_argument("--caption-store", default="output/worldsense_captions/captions.jsonl",
                        help="API mode: JSONL store so each video is captioned once")
    parser.add_argument("--no-caption-reuse", action="store_true",
                        help="API mode: always regenerate captions instead of reusing the store")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-inspect", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--video-ids", default=None, help="comma separated video_id allow-list")
    parser.add_argument("--question-ids", default=None, help="comma separated question_id allow-list")
    parser.add_argument("--question-id-file", type=Path, default=None, help="newline separated question_id allow-list")
    parser.add_argument("--task-types", default=None, help="comma separated task_type allow-list")
    parser.add_argument("--max-view-seconds", type=float, default=120.0)
    parser.add_argument("--max-media-in-context", type=int, default=2)
    parser.add_argument("--no-full-scan", action="store_true", help="disable the mandatory full-video survey")
    parser.add_argument("--full-scan-fps", type=float, default=2.0)
    parser.add_argument("--full-scan-fallback-fps", type=float, default=1.0)
    parser.add_argument("--full-scan-max-pixels", type=int, default=156_800)
    parser.add_argument("--no-synopsis", action="store_true", help="hide the dataset caption (ablation)")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--mock", action="store_true", help="scripted controller, no model")
    parser.add_argument("--dry-run", action="store_true", help="print prompts/media specs, no model")
    return parser.parse_args()


def _split(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _read_id_file(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    ids = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    if not ids:
        raise SystemExit(f"question id file is empty: {path}")
    return ids


def mock_script(record, config: AgentConfig) -> list[str]:
    duration = record.duration_s
    start = round(max(0.0, min(duration - 2.0, duration * 0.1)), 2)
    end = round(min(duration, start + 10.0), 2)
    zoom_end = round(min(duration, start + 5.0), 2)
    return [
        json.dumps(
            {
                "observation": "Coarse scan of the opening region to find candidate evidence.",
                "action": "inspect",
                "start": start,
                "end": end,
                "fps": 2,
                "max_pixels": config.default_max_pixels,
                "modality": "av",
                "focus": "locate the evidence for the asked fact",
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {"observation": "Confirm the media parameters before narrowing.", "action": "get_media_info"},
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "observation": "The evidence looks concentrated in the first half of the scan.",
                "action": "inspect",
                "start": start,
                "end": zoom_end,
                "fps": 4,
                "max_pixels": config.max_pixels_max,
                "modality": "video",
                "focus": "read small on-screen details",
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "observation": "Decisive evidence located inside the zoomed window.",
                "action": "submit",
                "intervals": [
                    {
                        "start": start,
                        "end": zoom_end,
                        "observation": "This window contains the visual detail the question asks about.",
                    }
                ],
                "confidence": 0.6,
            },
            ensure_ascii=False,
        ),
    ]


def main() -> int:
    args = parse_args()
    config = AgentConfig(
        max_turns=args.max_turns,
        max_inspect=args.max_inspect,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_view_seconds=args.max_view_seconds,
        max_media_in_context=args.max_media_in_context,
        force_full_scan=not args.no_full_scan,
        full_scan_fps=args.full_scan_fps,
        full_scan_fallback_fps=args.full_scan_fallback_fps,
        full_scan_max_pixels=args.full_scan_max_pixels,
        include_synopsis=not args.no_synopsis,
        include_caption_in_metadata=not args.no_synopsis,
        caption_stage=not args.no_caption,
        caption_max_tokens=args.caption_max_tokens,
    )
    media_index = MediaIndex.load(args.media_index_cache) if args.media_index_cache else MediaIndex()
    question_ids = _read_id_file(args.question_id_file) or _split(args.question_ids)
    records = load_questions(
        args.qa,
        args.video_root,
        media_index,
        question_ids=question_ids,
        video_ids=_split(args.video_ids),
        task_types=_split(args.task_types),
        limit=args.limit,
    )
    print(f"loaded {len(records)} questions from {args.qa}")
    if not records:
        return 1

    if args.dry_run:
        record = records[0]
        system_prompt = build_system_prompt(record, config)
        print("=" * 100)
        print("SYSTEM PROMPT (first record)")
        print("=" * 100)
        print(system_prompt)
        print("=" * 100)
        print("INITIAL USER PROMPT")
        print("=" * 100)
        print(build_initial_user_prompt(record, config))
        print("=" * 100)
        for record in records:
            print(
                json.dumps(
                    {
                        "question_id": record.question_id,
                        "forced_survey": {
                            "start": 0.0,
                            "end": round(record.duration_s, 3),
                            "fps": config.full_scan_fps,
                            "fallback_fps": config.full_scan_fallback_fps,
                            "max_pixels": config.full_scan_max_pixels,
                            "max_frames": config.full_scan_max_frames,
                            "modality": "av",
                        } if config.force_full_scan else None,
                        "synopsis_in_prompt": "video_synopsis" in system_prompt,
                    },
                    ensure_ascii=False,
                )
            )
            view = clamp_view(
                {
                    "start": record.duration_s * 0.1,
                    "end": record.duration_s * 0.1 + 8.0,
                    "fps": 2.0,
                    "max_pixels": config.default_max_pixels,
                    "modality": "av",
                    "focus": "dry-run view",
                },
                record.duration_s,
                config,
            )
            rendered, interleaved = render_conversation(
                [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                    {
                        "role": "user",
                        "content": view.text(),
                        "_worldsense_view": {
                            **view.as_dict(),
                            "video_path": record.video_path,
                            "duration_s": record.duration_s,
                            "max_frames": config.max_frames,
                        },
                    },
                ]
            )
            video_entry = {
                key: value
                for key, value in rendered[-1]["content"][0].items()
                if key in ("video_start", "video_end", "fps", "min_pixels", "max_pixels", "max_frames")
            }
            print(
                json.dumps(
                    {
                        "question_id": record.question_id,
                        "task_type": record.task_type,
                        "audio_class": record.audio_class,
                        "duration_s": round(record.duration_s, 3),
                        "rendered_use_audio_in_video": interleaved,
                        "rendered_video_entry": video_entry,
                    },
                    ensure_ascii=False,
                )
            )
        return 0

    trace_path = args.trace or args.output.with_suffix(".trace.jsonl")
    if args.mock:
        client = None
        client_factory = lambda record: MockOmniClient(mock_script(record, config))  # noqa: E731
    elif args.backend == "api":
        from omni_opsd.worldsense.api_client import QwenAPIOmniClient

        print(f"using cloud API backend: {args.api_model}")
        client = QwenAPIOmniClient(
            model=args.api_model,
            base_url=args.api_base_url,
            api_key=args.api_key,
            api_key_env=args.api_key_env,
            clip_cache_dir=args.clip_cache_dir,
            caption_store_path=args.caption_store,
            caption_reuse=not args.no_caption_reuse,
            proxy_policy=args.proxy_policy,
            max_clip_mb=args.max_clip_mb,
            media_index=media_index,
            enable_thinking=args.api_enable_thinking,
        )
        client_factory = None
    else:
        print(f"loading controller {args.model} on {args.device}")
        client = TransformersOmniClient(
            args.model,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn,
        )
        client_factory = None

    summary = run_batch(
        records,
        client,
        config,
        media_index,
        args.output,
        trace_path,
        resume=not args.no_resume,
        client_factory=client_factory,
    )
    if args.media_index_cache:
        media_index.save(args.media_index_cache)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
