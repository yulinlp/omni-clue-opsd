#!/usr/bin/env python3
"""Review website for WorldSense API annotations.

Serves a small SPA that plays the original video next to the question, the
video metadata and the annotated evidence intervals so a human can judge the
annotation quality.  Verdicts are appended to a JSONL file.

Usage:
    python scripts/worldsense_review_server.py --host 0.0.0.0 --port 8710
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = PROJECT_ROOT / "output" / "worldsense_evidence_api_full"
REVIEW_DIR = PROJECT_ROOT / "output" / "worldsense_api_review"
VIDEO_DIR = Path("/share/home/ylhu/Light-Omni/data/omni_benchmarks/WorldSense/videos")
QA_PATH = Path("/share/home/ylhu/Light-Omni/data/omni_benchmarks/WorldSense/worldsense_qa.json")
MEDIA_INDEX = PROJECT_ROOT / "output" / "worldsense_evidence_formal" / "media_index.json"
LOCAL_EVIDENCE_GLOB = str(
    PROJECT_ROOT / "output" / "worldsense_evidence_v2_formal" / "shard*" / "evidence.jsonl"
)
UI_PATH = Path(__file__).resolve().parent / "worldsense_review_ui.html"
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

app = FastAPI(title="WorldSense API annotation review")


@app.middleware("http")
async def log_with_timestamp(request: Request, call_next):
    """Access log with wall-clock time so client reports can be correlated."""

    started = time.time()
    response = await call_next(request)
    elapsed = (time.time() - started) * 1000
    client = request.client.host if request.client else "-"
    if request.url.path.startswith("/media"):
        print(
            f"[{time.strftime('%H:%M:%S')}] MEDIA {client} "
            f"range={request.headers.get('range', '-')} -> {response.status_code} ({elapsed:.0f}ms)",
            flush=True,
        )
    else:
        print(
            f"[{time.strftime('%H:%M:%S')}] {client} {request.method} {request.url.path} "
            f"-> {response.status_code} ({elapsed:.0f}ms)",
            flush=True,
        )
    return response
_ITEMS: list[dict[str, Any]] | None = None


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.is_file():
        return rows
    for line in p.open(encoding="utf-8", errors="replace"):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def last_traces() -> dict[str, dict[str, Any]]:
    last: dict[str, dict[str, Any]] = {}
    for path in sorted(RUN_DIR.glob("out/s*.trace.jsonl")):
        for row in load_jsonl(path):
            if row.get("question_id"):
                last[row["question_id"]] = row
    return last


def compact_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = {"survey", "inspect", "inspect_rejected", "inspect_upgraded", "bootstrap", "submitted", "intervals_adjusted", "submit_rejected", "protocol_error"}
    out: list[dict[str, Any]] = []
    for event in events or []:
        kind = str(event.get("kind"))
        if kind not in keep:
            continue
        row: dict[str, Any] = {"turn": event.get("turn"), "kind": kind}
        for key in ("view", "error", "reason", "status", "intervals", "caption_source", "caption_attempts", "caption_chars"):
            if key in event:
                value = event[key]
                if key == "view" and isinstance(value, dict):
                    value = {k: value.get(k) for k in ("start", "end", "fps", "max_pixels", "modality")}
                row[key] = value
        out.append(row)
    return out


def load_translation_cache() -> dict[str, str]:
    """hash(text) -> Chinese translation (raw cache from translate script)."""

    path = REVIEW_DIR / "translations.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def text_key(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def shard_index(path: Path) -> int:
    match = re.match(r"s(\d+)\.evidence\.jsonl$", path.name)
    return int(match.group(1)) if match else 999


def collect_runs(run_dir: Path = RUN_DIR) -> dict[str, list[dict[str, Any]]]:
    """Every (evidence row, trace) pair per question, ordered by shard index.

    The 200-question pilot shards overlapped, so many questions were annotated
    2-3 times with different results; the review UI must pair each evidence row
    with the trace of the SAME run instead of mixing them.
    """

    runs: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((run_dir / "out").glob("s*.evidence.jsonl"), key=shard_index):
        trace_path = path.with_name(path.name.replace(".evidence.jsonl", ".trace.jsonl"))
        traces = {row["question_id"]: row for row in load_jsonl(trace_path) if row.get("question_id")}
        for row in load_jsonl(path):
            question_id = row.get("question_id")
            if not question_id:
                continue
            trace = traces.get(question_id) or {}
            runs.setdefault(question_id, []).append(
                {
                    "shard": path.name.split(".")[0],
                    "shard_index": shard_index(path),
                    "annotation": {
                        "status": row.get("status"),
                        "clue_intervals": row.get("clue_intervals"),
                        "observation": row.get("observation"),
                        "confidence": row.get("confidence"),
                        "turns": row.get("turns"),
                        "inspect_calls": row.get("inspect_calls"),
                        "elapsed_s": row.get("elapsed_s"),
                        "interval_adjustments": row.get("interval_adjustments"),
                        "media_views": row.get("media_views"),
                    },
                    "events": compact_events(trace.get("events") or []),
                }
            )
    return runs


def build_items(limit: int, run_dir: Path = RUN_DIR) -> list[dict[str, Any]]:
    all_runs = collect_runs(run_dir)
    # The sample is taken from the completed questions of THIS run, spread over
    # the whole set; the primary run (lowest shard index) is what the item shows.
    candidates = sorted(
        qid
        for qid, runs in all_runs.items()
        if runs and (runs[0]["annotation"].get("status") == "submitted")
    )
    step = max(1, len(candidates) // limit)
    sample_ids = candidates[::step][:limit]
    qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
    media_index = json.loads(MEDIA_INDEX.read_text(encoding="utf-8")) if MEDIA_INDEX.is_file() else {}
    local = {
        r["question_id"]: r
        for path in sorted(Path(PROJECT_ROOT).glob("output/worldsense_evidence_v2_formal/shard*/evidence.jsonl"))
        for r in load_jsonl(path)
        if r.get("status") == "submitted"
    }
    captions: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(run_dir / "captions.jsonl"):
        captions[row["question_id"]] = row
    v1: dict[str, dict[str, dict[str, Any]]] = {"api": {}, "local": {}}
    for tag in ("api", "local"):
        for row in load_jsonl(run_dir / f"v1cmp_{tag}.verify.jsonl"):
            v1[tag][row["question_id"]] = row
    translation_cache = load_translation_cache()

    items: list[dict[str, Any]] = []
    for question_id in sample_ids:
        runs = all_runs.get(question_id) or []
        if not runs:
            continue
        row = runs[0]["annotation"]
        video_id, _, task = question_id.partition("::")
        meta = qa.get(video_id, {})
        task_meta = meta.get(task, {}) if isinstance(meta.get(task), dict) else {}
        media = media_index.get(f"{video_id}.mp4") or media_index.get(str(VIDEO_DIR / f"{video_id}.mp4")) or {}
        caption = captions.get(question_id)
        local_row = local.get(question_id)
        def zh(text: str | None) -> str | None:
            if not text:
                return None
            return translation_cache.get(text_key(text))

        runs = all_runs.get(question_id) or []
        item = {
            "question_id": question_id,
            "video_id": video_id,
            "task": task,
            "qa": {
                "question": task_meta.get("question"),
                "question_zh": zh(task_meta.get("question")),
                "candidates": task_meta.get("candidates"),
                "candidates_zh": [zh(c) for c in (task_meta.get("candidates") or [])],
                "answer": task_meta.get("answer"),
                "task_type": task_meta.get("task_type"),
                "task_domain": task_meta.get("task_domain"),
            },
            "video": {
                "duration_s": row.get("duration_s") or media.get("duration_s"),
                "width": media.get("width"),
                "height": media.get("height"),
                "fps": media.get("fps"),
                "n_frames": media.get("n_frames"),
                "has_audio": media.get("has_audio"),
                "domain": meta.get("domain"),
                "sub_category": meta.get("sub_category"),
                "video_duration": meta.get("video_duration"),
                "audio_class": meta.get("audio_class"),
                "video_caption": meta.get("video_caption"),
                "video_caption_zh": zh(meta.get("video_caption")),
            },
            "runs": [
                {
                    "shard": run["shard"],
                    "annotation": run["annotation"],
                    "annotation_zh": {"observation_zh": zh(run["annotation"].get("observation"))},
                    "events": run["events"],
                    "primary": run is runs[0],
                }
                for run in runs
            ],
            "annotation": {
                "status": row.get("status"),
                "clue_intervals": row.get("clue_intervals"),
                "observation": row.get("observation"),
                "observation_zh": zh(row.get("observation")),
                "confidence": row.get("confidence"),
                "turns": row.get("turns"),
                "inspect_calls": row.get("inspect_calls"),
                "elapsed_s": row.get("elapsed_s"),
                "interval_adjustments": row.get("interval_adjustments"),
                "media_views": row.get("media_views"),
            },
            "events": (runs[0]["events"] if runs else []),
            "caption": {
                "chars": len(caption.get("caption") or "") if caption else 0,
                "seconds": caption.get("caption_seconds") if caption else None,
                "text": (caption.get("caption") if caption else None),
                "text_zh": zh(caption.get("caption") if caption else None),
            },
            "local": {
                "clue_intervals": local_row.get("clue_intervals"),
                "observation": local_row.get("observation"),
                "observation_zh": zh(local_row.get("observation")),
                "confidence": local_row.get("confidence"),
                "inspect_calls": local_row.get("inspect_calls"),
                "turns": local_row.get("turns"),
            }
            if local_row
            else None,
            "v1": {
                "api": v1["api"].get(question_id),
                "local": v1["local"].get(question_id),
            },
        }
        items.append(item)
    return items


def items() -> list[dict[str, Any]]:
    global _ITEMS
    if _ITEMS is None:
        _ITEMS = build_items(limit=20)
    return _ITEMS


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if not UI_PATH.is_file():
        raise HTTPException(500, "UI file missing")
    return HTMLResponse(UI_PATH.read_text(encoding="utf-8"))


@app.get("/api/items")
def api_items() -> JSONResponse:
    return JSONResponse({"items": items()})


@app.get("/api/progress")
def api_progress() -> JSONResponse:
    """Live progress of the running annotation (row counts + statuses)."""

    import time as _time

    now = _time.time()
    cached = getattr(api_progress, "_cache", None)
    if cached and now - cached[0] < 20:
        return JSONResponse(cached[1])
    counts = []
    status: dict[str, int] = {}
    total = 0
    for path in sorted((RUN_DIR / "out").glob("s*.evidence.jsonl"), key=shard_index):
        expected = 0
        ids = RUN_DIR / "shards" / f"{path.name.split('.')[0]}.ids"
        if ids.is_file():
            expected = sum(1 for line in ids.open(encoding="utf-8") if line.strip())
        rows = load_jsonl(path)
        total += len(rows)
        for row in rows:
            key = str(row.get("status"))
            status[key] = status.get(key, 0) + 1
        counts.append({"shard": path.name.split(".")[0], "rows": len(rows), "expected": expected})
    payload = {
        "total": total,
        "expected": sum(c["expected"] for c in counts),
        "status": status,
        "shards": counts,
    }
    api_progress._cache = (now, payload)
    return JSONResponse(payload)


@app.get("/api/stats")
def api_stats() -> JSONResponse:
    path = RUN_DIR / "stats.json"
    if not path.is_file():
        raise HTTPException(404, "stats.json not found; run summarize_worldsense_api_pilot.py")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/verdicts")
def api_verdicts() -> JSONResponse:
    return JSONResponse({"verdicts": load_jsonl(REVIEW_DIR / "verdicts.jsonl")})


@app.post("/api/verdict")
async def api_save_verdict(request: Request) -> JSONResponse:
    payload = await request.json()
    question_id = str(payload.get("question_id") or "")
    if not question_id:
        raise HTTPException(400, "question_id required")
    row = {
        "question_id": question_id,
        "verdict": str(payload.get("verdict") or ""),
        "note": str(payload.get("note") or ""),
        "reviewer": str(payload.get("reviewer") or "anonymous"),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    with (REVIEW_DIR / "verdicts.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return JSONResponse({"ok": True, "verdict": row})


@app.get("/clip/{question_id}/{index}")
def clip(question_id: str, index: int):
    """Small pre-cut evidence clip (interval +-2s) for flaky connections."""

    if not re.match(r"^[A-Za-z0-9_:.-]+$", question_id) or not (0 <= index <= 50):
        raise HTTPException(400, "bad clip id")
    path = REVIEW_DIR / "clips" / f"{question_id.replace('::', '_')}_{index}.mp4"
    if not path.is_file():
        raise HTTPException(404, "clip not found")
    return FileResponse(
        path, media_type="video/mp4", headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"}
    )


@app.get("/media/{video_id}")
def media(video_id: str, request: Request):
    if not VIDEO_ID_RE.match(video_id):
        raise HTTPException(400, "bad video id")
    # Prefer the faststart remux (moov at the front) so playback starts without
    # downloading the tail first; fall back to the original file.
    faststart = REVIEW_DIR / "faststart" / f"{video_id}.mp4"
    path = faststart if faststart.is_file() else VIDEO_DIR / f"{video_id}.mp4"
    if not path.is_file():
        raise HTTPException(404, "video not found")
    size = path.stat().st_size
    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(path, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})
    # Only the first range is honoured; browsers use a single range for mp4.
    range_header = range_header.split(",")[0].strip()
    match = re.match(r"bytes=(\d*)-(\d*)", range_header)
    if not match:
        raise HTTPException(400, "bad range")
    first, second = match.group(1), match.group(2)
    if not first and second:
        # Suffix range "bytes=-N": the LAST N bytes.  Players request the tail
        # to read the moov atom of non-faststart mp4 files; answering with the
        # first N bytes instead made the video spin forever.
        suffix = int(second)
        start = max(0, size - suffix)
        end = size - 1
    else:
        start = int(first or 0)
        end = int(second) if second else size - 1
    end = min(end, size - 1)
    if start > end:
        raise HTTPException(416, "range not satisfiable")
    length = end - start + 1

    def iter_file() -> Iterator[bytes]:
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(1 << 20, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        iter_file(),
        status_code=206,
        media_type="video/mp4",
        headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
        },
    )


def main() -> None:
    global _ITEMS, RUN_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8710)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--run-dir", default=str(RUN_DIR))
    args = parser.parse_args()
    RUN_DIR = Path(args.run_dir)
    _ITEMS = build_items(limit=args.limit, run_dir=RUN_DIR)
    print(f"serving {len(_ITEMS)} questions on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=True)


if __name__ == "__main__":
    main()
