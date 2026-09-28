#!/usr/bin/env python3
"""30-minute monitor for the WorldSense Full/Gold gap screening.

Logs progress, accuracy, token usage, failures and service health for both
runs; appends one line per check to <run-dir>/monitor.log and writes a JSON
snapshot.  Safe to run repeatedly (``--once``) or as a daemon.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

REPO = Path("/share/home/ylhu/Omni-OPSD")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.is_file():
        for line in path.open(encoding="utf-8", errors="replace"):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def service_ok(endpoint: str) -> bool:
    url = endpoint.rsplit("/v1/", 1)[0] + "/v1/models"
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=8) as r:
            return any(m["id"] == "Qwen2.5-Omni-7B" for m in json.load(r)["data"])
    except Exception:
        return False


def check(run_dir: Path, endpoints: dict[str, str]) -> dict:
    snapshot: dict = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "views": {}}
    for view, endpoint in endpoints.items():
        out = run_dir / f"{view}_run"
        rows = read_jsonl(out / f"{view}.jsonl")
        failures = read_jsonl(out / "failures.jsonl")
        done = len(rows)
        correct = sum(1 for r in rows if r.get("correct"))
        tokens = sorted((r.get("prompt_tokens") or 0) for r in rows)
        snapshot["views"][view] = {
            "done": done,
            "failures": len(failures),
            "accuracy": round(correct / done, 4) if done else None,
            "median_prompt_tokens": tokens[len(tokens) // 2] if tokens else None,
            "service": service_ok(endpoint),
            "last_error": (failures[-1].get("error") if failures else None),
        }
    return snapshot


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--full-endpoint", default="http://gpu07:8091/v1/chat/completions")
    p.add_argument("--gold-endpoint", default="http://gpu02:8092/v1/chat/completions")
    p.add_argument("--interval", type=int, default=1800)
    p.add_argument("--expected", type=int, default=0)
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    endpoints = {"full": a.full_endpoint, "gold": a.gold_endpoint}
    a.run_dir.mkdir(parents=True, exist_ok=True)
    while True:
        snapshot = check(a.run_dir, endpoints)
        if a.expected:
            done = min(v["done"] for v in snapshot["views"].values())
            snapshot["progress"] = f"{done}/{a.expected}"
        line = json.dumps(snapshot, ensure_ascii=False)
        (a.run_dir / "monitor.log").open("a").write(line + "\n")
        (a.run_dir / "monitor_status.json").write_text(json.dumps(snapshot, indent=1))
        print(line, flush=True)
        if a.once:
            break
        done = min(v["done"] for v in snapshot["views"].values())
        if a.expected and done >= a.expected and not any(v["failures"] for v in snapshot["views"].values()):
            print("all requested questions scored", flush=True)
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
