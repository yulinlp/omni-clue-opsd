#!/usr/bin/env python3
"""WorldSense Full vs Gold first-token MCQ scoring with the dynamic budget.

Reuses the OmniVideo scoring protocol (exact A/B/C/D first-token logprobs from
a vLLM OpenAI endpoint) and the media decode helpers, but replaces the fixed
sampling contract with the approved dynamic budget:

    full view: one span [0, duration]
    gold view: the annotated evidence spans, sharing ONE frame budget

Each request carries its own mm_processor_kwargs (max_pixels / fps) so the
server never has to guess, and every score row records the frame counts,
resized grids and token accounting for auditing.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from omni_opsd.selection.media import (  # noqa: E402
    decode_audio,
    decode_video,
    validate_audio_coverage,
    wav_bytes,
)
from scripts.score_omnivideo_gap import (  # noqa: E402
    complete_option_logprobs,
    option_probabilities,
    post,
)
from worldsense_gap.budget import compute_budget  # noqa: E402

PROMPT = (
    "Question: {question}\n\nOptions:\n{choices}\n\n"
    "Answer with exactly one uppercase option letter."
)
MIN_PIXELS = 3136


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not Path(path).is_file():
        return rows
    for line in Path(path).open(encoding="utf-8", errors="replace"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def span_frames(timestamps, span, budget_frames, total_duration, all_spans):
    """Distribute the shared frame budget over one span, even frame count."""

    a, b = span
    share = (b - a) / total_duration if total_duration > 0 else 1.0
    n = max(2, int(round(budget_frames * share)))
    n -= n % 2
    import numpy as np

    times = np.asarray(timestamps)
    available = np.flatnonzero((times >= a) & (times <= b))
    if not len(available):
        raise ValueError("no source frames within span")
    n = min(n, len(available))
    return available[np.linspace(0, len(available) - 1, n).round().astype(int)].tolist()


def prepare_view(row, spans, budget, audit, processor, cache_root):
    """Build content parts (video JPEG frame lists + WAV audio) for one view."""

    total_duration = sum(b - a for a, b in spans)
    plans = [span_frames(audit["timestamps"], s, budget.frames, total_duration, spans) for s in spans]
    # Frames depend on the pixel budget, so the cache key includes it; audio
    # does not and lives in a shared per-video directory.
    cache = (
        Path(cache_root) / f"frames_{audit['media_sha256'][:16]}_px{budget.max_pixels}"
        if cache_root
        else None
    )
    frames = decode_video(
        row["video_path"],
        {i for plan in plans for i in plan},
        cache,
        min_pixels=MIN_PIXELS,
        max_pixels=budget.max_pixels,
    )
    audio_cache = (
        Path(cache_root) / f"audio_{audit['media_sha256'][:16]}" / f"audio_{round(row['duration'], 3)}.npz"
        if cache_root
        else None
    )
    if audio_cache and audio_cache.is_file():
        import numpy as np

        with np.load(audio_cache) as saved:
            pcm, covered = saved["pcm"], saved["covered"]
    else:
        pcm, covered = decode_audio(row["video_path"], row["duration"])
        if audio_cache:
            import tempfile

            import numpy as np

            audio_cache.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=audio_cache.parent, delete=False) as f:
                np.savez(f, pcm=pcm, covered=covered)
                temp = f.name
            os.replace(temp, audio_cache)
    content, videos, audios = [], [], []
    for span, plan in zip(spans, plans):
        import base64

        import numpy as np
        from PIL import Image

        arrays = np.stack([frames[i][1] for i in plan])
        visual = processor.video_processor(
            videos=[arrays],
            return_tensors="pt",
            size={"shortest_edge": MIN_PIXELS, "longest_edge": budget.max_pixels},
            do_sample_frames=False,
            device="cpu",
        )
        grid = visual["video_grid_thw"][0].tolist()
        tokens = int(np.prod(grid) // processor.video_processor.merge_size**2)
        if grid[1] * grid[2] * 14**2 > budget.max_pixels + 28 * 28 * 4:
            raise ValueError("processor exceeded the requested pixel budget")
        a, b = (round(t * 16000) for t in span)
        audio = pcm[a:b]
        missing = validate_audio_coverage(covered[a:b])
        video_url = "data:video/jpeg;base64," + ",".join(
            base64.b64encode(frames[i][0]).decode() for i in plan
        )
        audio_url = "data:audio/wav;base64," + base64.b64encode(wav_bytes(audio)).decode()
        content.extend(
            [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "audio_url", "audio_url": {"url": audio_url}},
            ]
        )
        videos.append(
            dict(
                sampled_timestamps=[audit["timestamps"][i] for i in plan],
                num_frames=len(plan),
                processor_grid_thw=grid,
                visual_tokens=tokens,
                encoded_width=int(arrays.shape[2]),
                encoded_height=int(arrays.shape[1]),
            )
        )
        audios.append(dict(samples=len(audio), sampling_rate=16000, uncovered_fraction=missing))
    observed = dict(
        input_spans=[list(map(float, s)) for s in spans],
        videos=videos,
        audios=audios,
        total_visual_tokens=sum(v["visual_tokens"] for v in videos),
        budget=budget.as_dict(),
        media_sha256=audit["media_sha256"],
        frames_total=sum(v["num_frames"] for v in videos),
    )
    return content, observed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--canonical", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--model", default="Qwen2.5-Omni-7B")
    p.add_argument("--endpoint", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, default=REPO / "outputs/worldsense_gap/media_cache")
    p.add_argument("--views", default="full,gold")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260923)
    a = p.parse_args()

    # --- service identity -----------------------------------------------------
    models_url = a.endpoint.rsplit("/v1/", 1)[0] + "/v1/models"
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(models_url, timeout=15) as r:
        served = json.load(r)
    matches = [m for m in served["data"] if m["id"] == a.model]
    if len(matches) != 1:
        raise ValueError(f"model {a.model} not served exactly once")

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(a.model_dir, local_files_only=True, use_fast=False)
    tokenizer = processor.tokenizer
    token_ids = {}
    for k in "ABCD":
        ids = tokenizer.encode(k, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError("ABCD must each encode as one token")
        token_ids[k] = ids[0]

    rows = read_jsonl(a.canonical)
    audit = json.loads(a.audit.read_text())
    by_video = {x["video_id"]: x for x in audit["items"]}
    excluded = set(audit.get("quarantined_sample_ids") or [])
    rows = [r for r in rows if r["sample_id"] not in excluded]
    if a.limit:
        rows = rows[: a.limit]
    views = a.views.split(",")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (a.output_dir / "score.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    run = dict(
        canonical_sha256=sha256(a.canonical),
        audit_sha256=sha256(a.audit),
        endpoint=a.endpoint,
        model=a.model,
        model_dir=str(a.model_dir),
        prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
        budget="dynamic: audio=25/s, visual=min(24000,32768-2048-audio), 2fps<=300 frames, no upscale",
        seed=a.seed,
        views=views,
        selected_ids=[r["sample_id"] for r in rows],
    )
    run_path = a.output_dir / "run.json"
    if run_path.exists() and json.loads(run_path.read_text()) != run:
        raise ValueError("resume configuration differs; use a new output directory")
    run_path.write_text(json.dumps(run, indent=2) + "\n")

    done: dict[str, set[str]] = {}
    for view in views:
        path = a.output_dir / f"{view}.jsonl"
        done[view] = {row["sample_id"] for row in read_jsonl(path)}

    def score_one(job):
        row, view = job
        start = time.monotonic()
        try:
            spans = [[0.0, row["duration"]]] if view == "full" else row["evidence_spans"]
            budget = compute_budget(
                sum(b - a for a, b in spans), row["source_width"], row["source_height"]
            )
            content, observed = prepare_view(
                row, spans, budget, by_video[row["video_id"]], processor, a.cache_root
            )
            prompt = PROMPT.format(
                question=row["question"],
                choices="\n".join(f"{k}. {v}" for k, v in zip("ABCD", row["choices"])),
            )
            content.append({"type": "text", "text": prompt})
            payload = dict(
                model=a.model,
                messages=[{"role": "user", "content": content}],
                temperature=1.0,
                max_tokens=1,
                seed=a.seed,
                logprobs=True,
                top_logprobs=20,
                allowed_token_ids=list(token_ids.values()),
                return_tokens_as_token_ids=True,
                mm_processor_kwargs=dict(
                    min_pixels=MIN_PIXELS,
                    max_pixels=budget.max_pixels,
                    size={"shortest_edge": MIN_PIXELS, "longest_edge": budget.max_pixels},
                    fps=round(budget.fps, 4),
                    do_sample_frames=False,
                    use_audio_in_video=False,
                ),
            )
            for attempt in range(a.retries + 1):
                try:
                    response = post(a.endpoint, payload, timeout=600)
                    break
                except Exception:
                    if attempt == a.retries:
                        raise
                    time.sleep(min(2**attempt, 4))
            response, extra = complete_option_logprobs(a.endpoint, payload, response, token_ids)
            probabilities, logprobs = option_probabilities(response, token_ids)
            predicted = max(probabilities, key=probabilities.get)
            return view, dict(
                sample_id=row["sample_id"],
                answer=row["answer"],
                view=view,
                probabilities=probabilities,
                predicted=predicted,
                logprobs=logprobs,
                correct=predicted == row["answer"],
                answer_probability=probabilities[row["answer"]],
                observed_media=observed,
                prompt_tokens=response.get("usage", {}).get("prompt_tokens"),
                elapsed_sec=round(time.monotonic() - start, 1),
                extra_option_requests=extra,
            ), None
        except Exception as e:  # noqa: BLE001
            return view, None, dict(
                sample_id=row["sample_id"], view=view, error=f"{type(e).__name__}: {e}"
            )

    jobs = [(r, v) for r in rows for v in views if r["sample_id"] not in done[v]]
    failures = 0
    with ThreadPoolExecutor(a.workers) as pool:
        for i, (view, result, error) in enumerate(pool.map(score_one, jobs)):
            path = a.output_dir / ("failures.jsonl" if error else f"{view}.jsonl")
            with path.open("a") as f:
                f.write(json.dumps(error or result, ensure_ascii=False, allow_nan=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            failures += int(error is not None)
            if (i + 1) % 25 == 0 or error:
                print(
                    json.dumps(
                        dict(progress=f"{i + 1}/{len(jobs)}", view=view, error=error,
                             sample_id=(error or result)["sample_id"])
                    ),
                    flush=True,
                )
    if failures:
        raise SystemExit(f"{failures} failed requests; rerun to retry")
    (a.output_dir / "SCORES_SUCCESS").write_text("all requested ids and views complete\n")
    print(json.dumps(dict(done=len(jobs), failures=0)))


if __name__ == "__main__":
    main()
