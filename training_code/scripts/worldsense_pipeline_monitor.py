#!/usr/bin/env python3
"""Two-stage pipeline monitor: annotation -> V1 sufficiency verification.

Every ``--interval`` seconds this daemon

1. reports per-shard progress for the current stage,
2. restarts a shard whose heartbeat is stale while its dedicated GPU is idle,
3. advances each shard to the verification stage as soon as its annotation
   file is complete (so a finished GPU starts V1 without waiting for the rest),
4. writes the V1 aggregate summary once every shard is verified.

The daemon only ever acts on gpu06 through the wxzhao allocation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path("/share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from shared_gpu_exec import run_remote  # noqa: E402

JOB_DIR = PROJECT_ROOT / "output/worldsense_agent_jobs"


def now_iso() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    rows = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                continue
            rows += 1
    return rows


def heartbeat_age(paths: list[Path]) -> float | None:
    mtimes = [path.stat().st_mtime for path in paths if path.exists()]
    if not mtimes:
        return None
    return time.time() - max(mtimes)


def gpu_text(account: str, node: str) -> str:
    command = (
        "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; "
        "echo '---procs---'; "
        "ps -eo pid,etime,cmd | grep -E 'annotate_worldsense|verify_worldsense|supervise_' | grep -v grep | wc -l"
    )
    ok, text, _ = run_remote(account, node, command, timeout=120)
    return text if ok else ""


def gpu_used_mib(text: str, index: int) -> float | None:
    import re

    for line in text.splitlines():
        match = re.match(rf"\s*{index}\s*,\s*(\d+)\s*MiB", line)
        if match:
            return float(match.group(1))
    return None


def launch_shard(stage: str, shard: int, shard_dir: Path, verify_dir: Path, account: str, node: str) -> str:
    if stage == "annotate":
        command = (
            f"mkdir -p {shard_dir}/shard{shard} && "
            f"OMNI_OPSD_SHARD_INDEX={shard} OMNI_OPSD_SHARD_DIR={shard_dir} "
            f"OMNI_OPSD_AGENT_CUDA_VISIBLE_DEVICES={shard} "
            f"setsid nohup bash {JOB_DIR}/supervise_shard.sh "
            f"> {shard_dir}/shard{shard}/supervisor_restart.log 2>&1 < /dev/null & echo started=$!"
        )
    else:
        command = (
            f"mkdir -p {verify_dir}/shard{shard} && "
            f"OMNI_OPSD_SHARD_INDEX={shard} OMNI_OPSD_SHARD_DIR={shard_dir} "
            f"OMNI_OPSD_VERIFY_DIR={verify_dir} "
            f"OMNI_OPSD_AGENT_CUDA_VISIBLE_DEVICES={shard} "
            f"setsid nohup bash {JOB_DIR}/supervise_verify.sh "
            f"> {verify_dir}/shard{shard}/supervisor_restart.log 2>&1 < /dev/null & echo started=$!"
        )
    ok, text, _ = run_remote(account, node, command, timeout=120)
    tail = text.strip().splitlines()[-1] if text.strip() else ""
    return f"{'ok' if ok else 'fail'}:{tail}"


def summarize(verify_dir: Path, shard_dir: Path) -> dict[str, object]:
    script = PROJECT_ROOT / "scripts/summarize_worldsense_verify.py"
    output = verify_dir / "summary.json"
    command = [
        sys.executable,
        str(script),
        "--verify-dir",
        str(verify_dir),
        "--evidence-dir",
        str(shard_dir),
        "--output",
        str(output),
    ]
    try:
        subprocess.run(command, check=False, timeout=600)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    return {"error": "summary not written"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", default="wxzhao")
    parser.add_argument("--node", default="gpu06")
    parser.add_argument("--shard-dir", type=Path, default=PROJECT_ROOT / "output/worldsense_evidence_formal")
    parser.add_argument("--verify-dir", type=Path, default=PROJECT_ROOT / "output/worldsense_verify_formal")
    parser.add_argument("--interval", type=int, default=1800)
    parser.add_argument("--stale-seconds", type=int, default=1500)
    parser.add_argument("--gpu-busy-mib", type=float, default=5_000.0)
    parser.add_argument("--auto-verify", action="store_true", default=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    manifest_path = args.shard_dir / "shards_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {int(item["index"]): int(item["count"]) for item in manifest["shards"]}
    total_expected = sum(expected.values())

    args.verify_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.verify_dir / "pipeline_monitor.log"
    state_path = args.verify_dir / "pipeline_state.json"

    while True:
        timestamp = now_iso()
        gpu = gpu_text(args.account, args.node)
        progress = []
        actions: list[str] = []
        active_verify = False
        for index in sorted(expected):
            shard_a = args.shard_dir / f"shard{index}" / "evidence.jsonl"
            shard_v = args.verify_dir / f"shard{index}" / "verify.jsonl"
            rows_a = jsonl_rows(shard_a)
            rows_v = jsonl_rows(shard_v)
            complete_a = rows_a >= expected[index]
            complete_v = rows_v >= expected[index]
            stage = "done" if complete_v else ("verify" if complete_a else "annotate")
            if stage == "verify":
                active_verify = True
            age_a = heartbeat_age(
                [
                    shard_a,
                    args.shard_dir / f"shard{index}" / "supervisor.log",
                    args.shard_dir / f"shard{index}" / "supervisor_out.log",
                ]
            )
            age_v = heartbeat_age(
                [
                    shard_v,
                    args.verify_dir / f"shard{index}" / "supervisor.log",
                    args.verify_dir / f"shard{index}" / "supervisor_out.log",
                ]
            )
            used = gpu_used_mib(gpu, index)

            if stage == "annotate" and (age_a is None or age_a > args.stale_seconds):
                if used is not None and used > args.gpu_busy_mib:
                    actions.append(f"s{index}:annotate-alive(gpu={used:.0f}MiB)")
                else:
                    actions.append(f"s{index}:annotate-restart({launch_shard("annotate", index, args.shard_dir, args.verify_dir, args.account, args.node)})")
            elif stage == "verify" and (age_v is None or age_v > args.stale_seconds):
                if used is not None and used > args.gpu_busy_mib:
                    actions.append(f"s{index}:verify-alive(gpu={used:.0f}MiB)")
                else:
                    actions.append(f"s{index}:verify-start({launch_shard("verify", index, args.shard_dir, args.verify_dir, args.account, args.node)})")
            progress.append(
                {
                    "index": index,
                    "stage": stage,
                    "annotated": rows_a,
                    "verified": rows_v,
                    "expected": expected[index],
                    "age_annotate_s": round(age_a, 1) if age_a is not None else None,
                    "age_verify_s": round(age_v, 1) if age_v is not None else None,
                    "gpu_used_mib": used,
                }
            )

        total_a = sum(item["annotated"] for item in progress)
        total_v = sum(item["verified"] for item in progress)
        line = (
            f"[{timestamp}] annotate={total_a}/{total_expected} verify={total_v}/{total_expected} "
            f"actions={actions or 'none'} "
            + " ".join(
                f"s{item['index']}[{item['stage'][0]}]{item['annotated']}|{item['verified']}"
                for item in progress
            )
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)
        state_path.write_text(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "total_annotated": total_a,
                    "total_verified": total_v,
                    "expected": total_expected,
                    "progress": progress,
                    "actions": actions,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        if total_v >= total_expected:
            summary = summarize(args.verify_dir, args.shard_dir)
            print(json.dumps({"pipeline": "complete", "summary": summary}, ensure_ascii=False, indent=2), flush=True)
            return 0
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
