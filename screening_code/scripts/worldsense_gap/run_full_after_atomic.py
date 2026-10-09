#!/usr/bin/env python3
"""After the six-view run finishes, score Full video and publish paired results."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
WORKSPACE = PROJECT.parent
DATA = PROJECT / "data/atomic_eval_20260929"
ATOMIC = DATA / "full_8npu"
FULL = DATA / "full_video_8npu"
PREFLIGHT = DATA / "full_video_preflight_longest"
REPORT = DATA / "report_full_and_atomic"
SOURCE = WORKSPACE / "Omni-OPSD"
PYTHON = Path("/home/ma-user/anaconda3/envs/PyTorch-2.9.0/bin/python")
MODEL = WORKSPACE / "omni-opsd-assets/models/Qwen2.5-Omni-7B"
CANONICAL = DATA / "worldsense_candidates_3079.atomic.jsonl"
SFT = DATA / "sft_1453.ids.txt"


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_shards() -> None:
    while True:
        active = []
        missing = []
        for index in range(8):
            shard = FULL / f"shard{index:02d}"
            pid_file = shard / "PID"
            if not pid_file.is_file():
                missing.append(index)
            elif alive(int(pid_file.read_text().strip())):
                active.append(index)
        print(json.dumps({"stage": "full_video", "active_shards": active,
                          "missing_shards": missing,
                          "time": time.strftime("%Y-%m-%d %H:%M:%S %Z")}), flush=True)
        if not active and not missing:
            return
        time.sleep(60)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    try:
        while not (ATOMIC / "ANALYSIS_SUCCESS").is_file():
            if (ATOMIC / "ANALYSIS_FAILED").is_file():
                raise RuntimeError("six-view analysis failed; Full video must wait for a valid six-view report")
            print(json.dumps({"stage": "waiting_for_six_views",
                              "time": time.strftime("%Y-%m-%d %H:%M:%S %Z")}), flush=True)
            time.sleep(60)

        PREFLIGHT.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({"ASCEND_RT_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1", "OMNI_OPSD_TORCH_THREADS": "1",
                    "PYTHONPATH": f"{SOURCE / 'src'}:{WORKSPACE / 'omni-opsd-runtime/python-deps'}"})
        preflight_cmd = ["taskset", "-c", "1-7", str(PYTHON), str(HERE / "score_atomic_views.py"),
                         "--source-root", str(SOURCE), "--annotation", str(CANONICAL),
                         "--model", str(MODEL), "--output", str(PREFLIGHT / "results.jsonl"),
                         "--unscored-output", str(PREFLIGHT / "unscored.jsonl"),
                         "--variants", "E0_full_av", "--sort-duration-desc", "--limit", "1",
                         "--device", "npu:0", "--resume"]
        print(json.dumps({"stage": "preflight_longest_full_video"}), flush=True)
        with (PREFLIGHT / "score.log").open("a", encoding="utf-8") as log:
            subprocess.run(preflight_cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        with (PREFLIGHT / "results.jsonl").open(encoding="utf-8") as handle:
            preflight = [json.loads(line) for line in handle if line.strip()]
        if len(preflight) != 1 or (PREFLIGHT / "unscored.jsonl").read_text(encoding="utf-8").strip():
            raise RuntimeError("longest Full video preflight did not produce exactly one valid score")

        launch_env = os.environ.copy()
        launch_env.update({"ATOMIC_RUN_NAME": "full_video_8npu", "ATOMIC_VARIANTS": "E0_full_av"})
        print(json.dumps({"stage": "launch_full_video_8npu"}), flush=True)
        subprocess.run(["bash", str(HERE / "run_atomic_8npu.sh")], env=launch_env, check=True)
        wait_shards()
        command = [sys.executable, str(HERE / "summarize_full_and_atomic.py"),
                   "--canonical", str(CANONICAL), "--sft-ids", str(SFT),
                   "--atomic-root", str(ATOMIC), "--full-root", str(FULL),
                   "--output-dir", str(REPORT)]
        subprocess.run(command, check=True)
        (FULL / "ANALYSIS_FAILED").unlink(missing_ok=True)
        (FULL / "ANALYSIS_SUCCESS").touch()
        print(json.dumps({"stage": "complete", "report": str(REPORT)}), flush=True)
    except Exception as exc:
        FULL.mkdir(parents=True, exist_ok=True)
        (FULL / "ANALYSIS_FAILED").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
