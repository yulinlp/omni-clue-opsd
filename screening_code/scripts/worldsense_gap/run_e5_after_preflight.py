#!/usr/bin/env python3
"""Launch eight WorldSense E5 shards after preflight and publish a paired report."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
DATA = PROJECT / "data/atomic_eval_20260929"
PREFLIGHT = DATA / "e5_preflight_20260930"
RUN = DATA / "e5_8npu_20260930"
REPORT = DATA / "report_full_atomic_e5_20260930"
CANONICAL = DATA / "worldsense_candidates_3079.atomic.jsonl"
SFT_IDS = DATA / "sft_1453.ids.txt"
ATOMIC = DATA / "full_8npu"
FULL = DATA / "full_video_8npu"
ARM = "E5_global_coarse_dense"


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    try:
        expected = {json.loads(line)["sample_id"] for line in PREFLIGHT.joinpath("manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
        scored = [json.loads(line) for line in PREFLIGHT.joinpath("results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        unscored = PREFLIGHT.joinpath("unscored.jsonl").read_text(encoding="utf-8").strip()
        if len(expected) != 3 or len(scored) != 3 or {row["sample_id"] for row in scored} != expected or unscored:
            raise ValueError("E5 preflight must score three distinct rows without failures")
        if not (ATOMIC / "ANALYSIS_SUCCESS").is_file() or not (FULL / "ANALYSIS_SUCCESS").is_file():
            raise ValueError("paired reference runs are incomplete")

        env = os.environ.copy()
        env.update({"ATOMIC_RUN_NAME": RUN.name, "ATOMIC_VARIANTS": ARM})
        print(json.dumps({"stage": "launch_e5", "run": str(RUN)}), flush=True)
        subprocess.run(["bash", str(HERE / "run_atomic_8npu.sh")], env=env, check=True)
        while True:
            missing = []
            active = []
            for index in range(8):
                pid_file = RUN / f"shard{index:02d}" / "PID"
                if not pid_file.is_file():
                    missing.append(index)
                elif alive(int(pid_file.read_text().strip())):
                    active.append(index)
            print(json.dumps({"stage": "e5_scoring", "missing_shards": missing,
                              "active_shards": active,
                              "time": time.strftime("%Y-%m-%d %H:%M:%S %Z")}), flush=True)
            if not missing and not active:
                break
            time.sleep(60)

        command = [sys.executable, str(HERE / "summarize_full_and_atomic.py"),
                   "--canonical", str(CANONICAL), "--sft-ids", str(SFT_IDS),
                   "--atomic-root", str(ATOMIC), "--full-root", str(FULL),
                   "--e5-root", str(RUN), "--output-dir", str(REPORT)]
        subprocess.run(command, check=True)
        (RUN / "ANALYSIS_FAILED").unlink(missing_ok=True)
        (RUN / "ANALYSIS_SUCCESS").touch()
        print(json.dumps({"stage": "complete", "report": str(REPORT)}), flush=True)
    except Exception as exc:
        (RUN / "ANALYSIS_FAILED").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
