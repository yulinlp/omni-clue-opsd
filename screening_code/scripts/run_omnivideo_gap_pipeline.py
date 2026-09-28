#!/usr/bin/env python3
"""One entry point for prepare/extract/audit/score/select; resume-safe scoring."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", type=Path, default=Path("configs/omnivideo_gap5000.yaml")
    )
    p.add_argument(
        "--stage",
        choices=["prepare", "check", "extract", "audit", "score", "select", "all"],
        default="all",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Two archive videos; never a 5K scientific result",
    )
    p.add_argument("--print-only", action="store_true")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    c = yaml.safe_load(a.config.read_text())
    run = root / c["run_dir"]
    canonical = run / "candidates.jsonl"
    selected_input = run / "smoke.jsonl" if a.smoke else canonical
    audit = run / ("smoke_audit.json" if a.smoke else "audit.json")
    score_name = c.get("score_run_name", c["av_mode"])
    selection_name = c.get("selection_run_name", c["av_mode"])
    scores = (
        run
        / ("smoke_scores" if a.smoke else "scores")
        / (c["av_mode"] if a.smoke else score_name)
    )
    output = (
        run
        / ("smoke_selection" if a.smoke else "gap5000")
        / (c["av_mode"] if a.smoke else selection_name)
    )
    steps = {
        "prepare": [
            "-m",
            "scripts.prepare_omnivideo_exact_gold_candidates",
            "--source",
            c["source"],
            "--video-root",
            root / c["video_root"],
            "--output",
            canonical,
            "--ordering",
            c["ordering"],
        ],
        "check": [
            "-m",
            "scripts.check_omnivideo_gap_service",
            "--endpoint",
            c["endpoint"],
            "--model",
            c["model"],
            "--model-dir",
            c["model_dir"],
            "--require-mode",
            c["av_mode"],
            "--output",
            run / "service_probe.json",
        ],
        "extract": [
            "-m",
            "scripts.extract_omnivideo_media",
            "--canonical",
            canonical,
            "--archive-root",
            c["archive_root"],
        ],
        "audit": [
            "-m",
            "scripts.audit_omnivideo_score_media",
            "--canonical",
            selected_input,
            "--output",
            audit,
            "--workers",
            c["audit_workers"],
        ],
        "score": [
            "-m",
            "scripts.score_omnivideo_gap",
            "--canonical",
            selected_input,
            "--audit",
            audit,
            "--model-dir",
            c["model_dir"],
            "--model",
            c["model"],
            "--endpoint",
            c["endpoint"],
            "--output-dir",
            scores,
            "--workers",
            c["workers"],
            "--av-mode",
            c["av_mode"],
            "--seed",
            c["seed"],
        ],
        "select": [
            "select_omnivideo_gap_5000.py",
            "--canonical",
            selected_input,
            "--reconstructed-audit",
            audit,
            "--av-scores",
            scores / "av.jsonl",
            "--gold-scores",
            scores / "gold.jsonl",
            "--output-dir",
            output,
            "--target-size",
            2 if a.smoke else c["target_size"],
            "--max-per-video",
            1 if a.smoke else c["max_per_video"],
            "--selection-policy",
            c["selection_policy"],
        ],
    }
    if a.smoke:
        steps["extract"] += ["--limit-videos", 2, "--smoke-manifest", selected_input]
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(root / "src")
        + os.pathsep
        + str(root)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    for name, args in steps.items():
        if a.stage not in (name, "all"):
            continue
        cmd = [sys.executable, *map(str, args)]
        import shlex

        print(f"{name}: {shlex.join(cmd)}", flush=True)
        if not a.print_only:
            subprocess.run(cmd, check=True, cwd=root, env=env)


if __name__ == "__main__":
    main()
