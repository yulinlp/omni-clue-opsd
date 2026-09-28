#!/usr/bin/env python3
"""30-minute monitor for the formal WorldSense annotation run.

Runs on the login node.  Reads per-shard progress from the shared filesystem,
checks GPU utilisation on gpu06, restarts stale shards through the wxzhao
allocation, and appends a status line to the monitor log.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path("/share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from shared_gpu_exec import run_remote  # noqa: E402

DEFAULT_SHARD_DIR = PROJECT_ROOT / "output/worldsense_evidence_formal"


def now_iso() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def shard_progress(shard_dir: Path, index: int) -> dict[str, object]:
    output = shard_dir / f"shard{index}" / "evidence.jsonl"
    status_counts: dict[str, int] = {}
    rows = 0
    if output.is_file():
        with output.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows += 1
                status = str(row.get("status") or "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
    heartbeat_paths = [
        output,
        shard_dir / f"shard{index}" / "supervisor.log",
        shard_dir / f"shard{index}" / "supervisor_out.log",
        shard_dir / f"shard{index}" / "supervisor_restart.log",
    ]
    mtimes = [p.stat().st_mtime for p in heartbeat_paths if p.exists()]
    heartbeat_age = (time.time() - max(mtimes)) if mtimes else None
    return {
        "index": index,
        "rows": rows,
        "status_counts": status_counts,
        "age_s": round(heartbeat_age, 1) if heartbeat_age is not None else None,
        "output": str(output),
    }


def _gpu_memory_used_mib(text: str, index: int) -> float | None:
    for line in text.splitlines():
        match = re.match(rf"\s*{index}\s*,\s*(\d+)\s*MiB", line)
        if match:
            return float(match.group(1))
    return None


def gpu_status() -> tuple[str, str]:
    command = (
        "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; "
        "echo '---procs---'; "
        "ps -eo pid,etime,cmd | grep -E 'annotate_worldsense|supervise_shard' | grep -v grep | wc -l"
    )
    ok, text, _ = run_remote("wxzhao", "gpu06", command, timeout=120)
    return ("ok" if ok else "fail"), text


def restart_shard(shard_dir: Path, index: int) -> str:
    command = (
        f"OMNI_OPSD_SHARD_INDEX={index} "
        f"OMNI_OPSD_SHARD_DIR={shard_dir} "
        f"OMNI_OPSD_AGENT_CUDA_VISIBLE_DEVICES={index} "
        f"setsid nohup bash {PROJECT_ROOT}/output/worldsense_agent_jobs/supervise_shard.sh "
        f"> {shard_dir}/shard{index}/supervisor_restart.log 2>&1 < /dev/null & echo restarted_pid=$!"
    )
    ok, text, _ = run_remote("wxzhao", "gpu06", command, timeout=120)
    return f"{'ok' if ok else 'fail'} {text.strip().splitlines()[-1] if text.strip() else ''}".strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--interval", type=int, default=1800)
    parser.add_argument("--stale-seconds", type=int, default=1500)
    parser.add_argument("--gpu-busy-mib", type=float, default=5_000.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-loops", type=int, default=0)
    args = parser.parse_args()

    manifest_path = args.shard_dir / "shards_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {"shards": []}
    expected = {int(item["index"]): int(item["count"]) for item in manifest.get("shards", [])}
    total_expected = sum(expected.values())

    log_path = args.shard_dir / "monitor.log"
    status_path = args.shard_dir / "monitor_status.json"
    loops = 0
    while True:
        loops += 1
        timestamp = now_iso()
        progress = [shard_progress(args.shard_dir, index) for index in sorted(expected)]
        total_rows = sum(int(item["rows"]) for item in progress)
        gpu_state, gpu_text = gpu_status()

        restarts: list[str] = []
        for item in progress:
            index = int(item["index"])
            expected_rows = expected.get(index, 0)
            if int(item["rows"]) >= expected_rows:
                continue
            age = item["age_s"]
            if age is None or float(age) <= args.stale_seconds:
                continue
            used_mib = _gpu_memory_used_mib(gpu_text, index)
            if used_mib is not None and used_mib > args.gpu_busy_mib:
                restarts.append(f"shard{index}:alive-elsewhere(gpu={used_mib:.0f}MiB)")
                continue
            restarts.append(f"shard{index}:restart({restart_shard(args.shard_dir, index)})")

        summary = {
            "timestamp": timestamp,
            "total_rows": total_rows,
            "total_expected": total_expected,
            "progress": progress,
            "gpu_state": gpu_state,
            "gpu_text": gpu_text.strip().splitlines()[:12],
            "restarts": restarts,
        }
        status_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        line = (
            f"[{timestamp}] rows={total_rows}/{total_expected} gpu={gpu_state} restarts={restarts or 'none'} "
            + " ".join(f"s{int(item['index'])}={item['rows']}/{expected.get(int(item['index']), 0)}" for item in progress)
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

        if total_expected and total_rows >= total_expected:
            print("[monitor] all shards complete", flush=True)
            return 0
        if args.once:
            return 0
        if args.max_loops and loops >= args.max_loops:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
