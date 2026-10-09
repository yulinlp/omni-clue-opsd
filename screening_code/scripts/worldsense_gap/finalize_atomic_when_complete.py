#!/usr/bin/env python3
"""Wait for eight WorldSense score shards, then validate and publish the report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--selected-ids", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        raise ValueError("poll interval is too short")
    script = Path(__file__).with_name("summarize_atomic_views.py")
    while True:
        pending = []
        active = []
        for index in range(8):
            shard = args.run_root / f"shard{index:02d}"
            pid_file = shard / "PID"
            if not pid_file.is_file():
                pending.append(index)
                continue
            pid = int(pid_file.read_text().strip())
            if alive(pid):
                active.append(index)
        print(json.dumps({"pending_launch": pending, "active": active,
                          "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z")}), flush=True)
        if not pending and not active:
            break
        time.sleep(args.poll_seconds)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(script),
               "--canonical", str(args.canonical),
               "--selected-ids", str(args.selected_ids),
               "--results-root", str(args.run_root),
               "--output-dir", str(args.report_dir)]
    try:
        subprocess.run(command, check=True)
    except Exception as exc:
        (args.run_root / "ANALYSIS_FAILED").write_text(str(exc) + "\n", encoding="utf-8")
        raise
    (args.run_root / "ANALYSIS_SUCCESS").touch()


if __name__ == "__main__":
    main()
