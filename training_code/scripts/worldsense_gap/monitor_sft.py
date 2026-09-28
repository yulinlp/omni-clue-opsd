#!/usr/bin/env python3
"""30-minute monitor for the WorldSense SFT runs (LoRA on gpu07, full on gpu02).

Parses the swift progress bar from each run's log, appends a JSON snapshot to
<out>/monitor.log and prints a compact line.  Access: gpu07 via plain ssh (own
allocation), gpu02 through the shared-account helper.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/share/home/ylhu/Omni-OPSD/OmniOPSD_training_code_20260907")
HELPER = REPO / "scripts/shared_gpu_exec.py"
PROGRESS_RE = re.compile(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([0-9:]+)<([0-9:]+),\s*([0-9.]+)s?/it\]")
LOSS_RE = re.compile(r"'loss':\s*([0-9.]+)")

RUNS = {
    "lora": {"node": "gpu07", "log": "output/worldsense_sft_lora_gpu07/sft.log", "account": None},
    "full": {"node": "gpu02", "log": "output/worldsense_sft_full_gpu02/sft.log", "account": "xysui"},
}


def remote(node: str, account: str | None, command: str, timeout: int = 90) -> str:
    if account:
        result = subprocess.run(
            [sys.executable, str(HELPER), "--account", account, "--node", node,
             "--timeout", str(timeout - 10), "--command", command],
            capture_output=True, text=True, timeout=timeout,
        )
        out = result.stdout + result.stderr
    else:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node, command],
            capture_output=True, text=True, timeout=timeout,
        )
        out = result.stdout + result.stderr
    return "\n".join(l for l in out.splitlines() if "Certificate invalid" not in l and "__SHARED" not in l)


def probe(name: str, cfg: dict) -> dict:
    command = (
        f"cd {REPO} && "
        f"grep -aoE \"[0-9]+%\\|[^|]*\\| [0-9]+/[0-9]+ \\[[^]]+\\]\" {cfg['log']} | tail -1; "
        f"grep -aoE \"'loss': [0-9.]+\" {cfg['log']} | tail -1; "
        f"ps -eo cmd | grep -c '[s]wift sft'; "
        f"nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -2"
    )
    text = remote(cfg["node"], cfg["account"], command)
    progress = PROGRESS_RE.search(text)
    loss = LOSS_RE.search(text)
    alive = bool(re.search(r"^[1-9]\d*$", text, re.M))
    gpu = re.findall(r"\d+, (\d+) MiB", text)
    entry: dict = {
        "run": name,
        "node": cfg["node"],
        "alive": alive,
        "gpu_used_mib": gpu[:2],
    }
    if progress:
        step, total, elapsed, remaining = int(progress.group(2)), int(progress.group(3)), progress.group(4), progress.group(5)
        entry.update(
            step=step, total_steps=total, s_per_it=float(progress.group(6)),
            elapsed=elapsed, eta=remaining,
            eta_hours=round((total - step) * float(progress.group(6)) / 3600, 2),
        )
    if loss:
        entry["loss"] = float(loss.group(1))
    return entry


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=REPO / "output/worldsense_sft_monitor")
    p.add_argument("--interval", type=int, default=1800)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    while True:
        snapshot = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": {}}
        for name, cfg in RUNS.items():
            try:
                snapshot["runs"][name] = probe(name, cfg)
            except Exception as exc:  # noqa: BLE001
                snapshot["runs"][name] = {"run": name, "error": f"{type(exc).__name__}: {exc}"}
        line = json.dumps(snapshot, ensure_ascii=False)
        (a.out / "monitor.log").open("a").write(line + "\n")
        (a.out / "monitor_status.json").write_text(json.dumps(snapshot, indent=1))
        print(line, flush=True)
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
