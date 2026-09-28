#!/usr/bin/env python3
"""30-minute monitor for the API annotation run (runs on the login node).

Checks per-shard progress, log freshness and rate-limit/error signals, restarts
stalled shards, and records a status line plus a JSON snapshot for the
concurrency test (``--workers`` records the current worker count).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path("/share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907")

ERROR_PATTERNS = {
    "rate_limit": re.compile(r"HTTP 429|status_code=429|Too Many Requests|RateLimitError|rate limit|Throttling|quota exceeded", re.I),
    "traceback": re.compile(r"Traceback"),
    "api_error": re.compile(r"Error code:|APIError|APIConnectionError|APITimeoutError|InternalServerError", re.I),
    "caption_invalid": re.compile(r"caption_valid.*False"),
}


def now_iso() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def count_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    rows = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows += 1
    return rows


def _current_run_text(text: str) -> str:
    """Only the part of the log written by the latest worker start."""

    marker = text.rfind("questions from")
    return text[marker:] if marker >= 0 else text


def scan_log(path: Path) -> dict[str, int]:
    counts = {name: 0 for name in ERROR_PATTERNS}
    if not path.is_file():
        return counts
    try:
        text = _current_run_text(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return counts
    tail = "\n".join(text.splitlines()[-400:])
    for name, pattern in ERROR_PATTERNS.items():
        counts[name] = len(pattern.findall(tail))
    return counts


def caption_stats(run_dir: Path) -> dict[str, int]:
    stats = {"captions": 0, "truncated": 0, "invalid": 0, "reused": 0}
    store = run_dir / "captions.jsonl"
    if store.is_file():
        with store.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip():
                    stats["captions"] += 1
    for trace in sorted((run_dir / "out").glob("s*.trace.jsonl")):
        with trace.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for event in row.get("events") or []:
                    if event.get("kind") != "bootstrap":
                        continue
                    if event.get("caption_source") == "reused":
                        stats["reused"] += 1
                    if event.get("caption_truncated"):
                        stats["truncated"] += 1
                    if event.get("caption_valid") is False:
                        stats["invalid"] += 1
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "output/worldsense_api_200")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--interval", type=int, default=1800)
    parser.add_argument("--stale-seconds", type=int, default=1500)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    shards = sorted((run_dir / "shards").glob("s*.ids"))
    expected = {}
    for path in shards:
        index = int(path.stem[1:])
        expected[index] = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    total_expected = sum(expected.values())

    log_path = run_dir / "api_monitor.log"
    status_path = run_dir / "api_monitor_status.json"
    while True:
        timestamp = now_iso()
        progress = []
        restarts: list[str] = []
        rate_limit_total = 0
        for index in sorted(expected):
            evidence = run_dir / "out" / f"s{index}.evidence.jsonl"
            log = run_dir / "out" / f"s{index}.log"
            rows = count_rows(evidence)
            age = time.time() - log.stat().st_mtime if log.is_file() else None
            errors = scan_log(log)
            rate_limit_total += errors["rate_limit"]
            restarted = ""
            if rows < expected[index] and age is not None and age > args.stale_seconds:
                launcher = run_dir / "launch_one.sh"
                if launcher.is_file():
                    result = subprocess.run(
                        ["bash", str(launcher), str(index)], capture_output=True, text=True, timeout=60
                    )
                    restarted = (result.stdout or result.stderr).strip().splitlines()[-1] if (result.stdout or result.stderr) else "restart"
                    restarts.append(f"s{index}:{restarted}")
            progress.append(
                {
                    "shard": index,
                    "rows": rows,
                    "expected": expected[index],
                    "log_age_s": round(age, 1) if age is not None else None,
                    "errors": errors,
                }
            )
        total_rows = sum(item["rows"] for item in progress)
        stats = caption_stats(run_dir)
        snapshot = {
            "timestamp": timestamp,
            "workers": args.workers,
            "rows": total_rows,
            "expected": total_expected,
            "rate_limit_signals": rate_limit_total,
            "caption_stats": stats,
            "restarts": restarts,
            "progress": progress,
        }
        status_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        line = (
            f"[{timestamp}] workers={args.workers} rows={total_rows}/{total_expected} "
            f"rate_limit={rate_limit_total} captions={stats['captions']} "
            f"truncated={stats['truncated']} invalid={stats['invalid']} reused={stats['reused']} "
            f"restarts={restarts or 'none'} "
            + " ".join(f"s{item['shard']}={item['rows']}" for item in progress)
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)
        if total_rows >= total_expected:
            print("[api-monitor] all shards complete", flush=True)
            return 0
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
