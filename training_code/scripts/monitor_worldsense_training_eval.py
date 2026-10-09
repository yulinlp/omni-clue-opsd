#!/usr/bin/env python3
"""Write live progress snapshots for the two training and evaluation queues."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_openqa_20260929")
OMNI = ROOT / "eval_omnivideobench200"
WORLD = ROOT / "eval_worldsense_holdout200"
RUNS = {
    "sft": ROOT / "outputs/sft_formal/v0-20260929-174408",
    "clue": ROOT / "outputs/clue_formal/v0-20260929-174734",
}
NAMES = ("base", "sft_epoch3", "sft_epoch2", "sft_epoch1", "clue_epoch3", "clue_epoch2", "clue_epoch1")


def read_status(path: Path) -> str:
    try:
        return next(line.removeprefix("status=") for line in path.read_text().splitlines() if line.startswith("status="))
    except (OSError, StopIteration):
        return "NOT_STARTED"


def last_training_record(path: Path) -> dict:
    try:
        lines = path.read_text().splitlines()
        for line in reversed(lines):
            row = json.loads(line)
            if "global_step/max_steps" in row:
                return {key: row[key] for key in ("global_step/max_steps", "epoch", "loss", "train_runtime", "remaining_time") if key in row}
    except (OSError, ValueError):
        pass
    return {}


def snapshot() -> dict:
    arms = {}
    for arm, run in RUNS.items():
        log = run / "logging.jsonl"
        arms[arm] = {
            "queue_status": read_status(OMNI / f"{arm}.STATUS.txt"),
            "training": last_training_record(log),
            "training_log_updated_at": datetime.fromtimestamp(log.stat().st_mtime, timezone.utc).isoformat() if log.exists() else None,
            "checkpoints": [epoch for epoch, step in ((1, 45), (2, 90), (3, 135)) if (run / f"checkpoint-{step}/adapter_model.safetensors").is_file()],
        }
    return {
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "arms": arms,
        "evaluations": {
            benchmark: {
                "completed": [name for name in NAMES if (root / "results" / name / "summary.json").is_file()],
                "remaining": [name for name in NAMES if not (root / "results" / name / "summary.json").is_file()],
                "comparison_ready": (root / "comparison_all/training_eval_summary.json").is_file(),
            }
            for benchmark, root in (("OmniVideoBench", OMNI), ("WorldSense", WORLD))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be positive")
    out = ROOT / "eval_monitor"
    out.mkdir(parents=True, exist_ok=True)
    previous = None
    while True:
        state = snapshot()
        temporary = out / f"status.json.tmp.{os.getpid()}"
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(out / "status.json")
        signature = json.dumps({"arms": state["arms"], "evaluations": state["evaluations"]}, sort_keys=True)
        if signature != previous:
            with (out / "timeline.jsonl").open("a") as handle:
                handle.write(json.dumps(state, ensure_ascii=False) + "\n")
            previous = signature
        statuses = {arm: value["queue_status"] for arm, value in state["arms"].items()}
        if all(status.startswith("FAILED") or status == f"COMPLETED_{arm.upper()}_ALL_EVALUATIONS"
               for arm, status in statuses.items()):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
