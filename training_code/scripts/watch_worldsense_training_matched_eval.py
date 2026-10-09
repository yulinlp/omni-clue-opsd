#!/usr/bin/env python3
"""Persist read-only health observations for this evaluation root every 60 seconds.

Only workers registered by this root's launcher are inspected. Exit files,
queue state and result activity provide health evidence; shared lease locks
are advisory because cross-node visibility is not established. No SSH,
signals, automatic restarts, or changes to NPU jobs are performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
FATAL = re.compile(r"Traceback \(most recent call last\)|(?:RuntimeError|OutOfMemoryError|ChildFailedError|AssertionError|ValueError|ImportError|ModuleNotFoundError):")


def now() -> str:
    return datetime.now().astimezone().isoformat()


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(0.05)


def atomic_write(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def timestamp(value: str | None) -> float | None:
    if not value:
        return None
    return datetime.fromisoformat(value).timestamp()


def count_jsonl(path: Path) -> dict:
    if not path.exists():
        return {"rows": 0, "mtime": None, "incomplete_final_line": False, "malformed_complete_lines": 0}
    rows = bad = 0
    incomplete = False
    # Each prediction is small relative to available CPU memory. Reading a
    # point-in-time byte snapshot also avoids confusing concurrent appends.
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            json.loads(line)
            rows += 1
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i == len(lines) - 1 and not line.endswith(b"\n"):
                incomplete = True
            else:
                bad += 1
    return {"rows": rows, "mtime": path.stat().st_mtime, "incomplete_final_line": incomplete,
            "malformed_complete_lines": bad}


def lease_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        with path.open("r+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return "held"
            fcntl.flock(handle, fcntl.LOCK_UN)
            return "free"
    except OSError:
        return "unavailable"


def issue(kind: str, severity: str, message: str, **details) -> dict:
    key = kind + ":" + str(details.get("task", details.get("worker", details.get("path", "monitor"))))
    return {"id": key, "kind": kind, "severity": severity, "message": message, **details}


def tail_current_log(path: Path, started: float | None) -> str:
    if not path.exists() or (started is not None and path.stat().st_mtime < started):
        return ""
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 65536))
        return handle.read().decode(errors="replace")


def snapshot(root: Path, previous: dict | None = None, stall_seconds: int = 1800) -> dict:
    current_time = time.time()
    tasks = load_json(root / "tasks.json")
    state = load_json(root / "queue_state.json", {})
    labels = count_jsonl(root / "data/worldsense.labels.jsonl")
    expected = labels["rows"]
    if expected != 518:
        raise ValueError(f"This monitor requires the frozen 518 questions, got {expected}")
    result = {"time": now(), "timestamp": current_time, "root": str(root.resolve()),
              "monitor_pid": os.getpid(), "interval_seconds": 60,
              "liveness_method": "Only this root's registered worker exit files, queue state and result/log activity. Lease locks are advisory because cross-node flock visibility is not established; no process signals or remote probes.",
              "expected_questions_per_model": expected, "expected_models": len(tasks),
              "expected_generation_tasks": len(tasks) * 2, "jobs": [], "workers": {}, "issues": []}
    old_workers = (previous or {}).get("workers", {})
    registered = {}
    for path in sorted((root / "logs").glob("*.launch.json")):
        launch = load_json(path)
        worker = launch.get("worker", "")
        command = launch.get("command", [])
        command_root = command[command.index("--root") + 1] if "--root" in command else None
        owns_root = command_root is not None and Path(command_root).resolve() == root.resolve()
        owns_script = any(Path(part).name == "queue_worldsense_training_matched_eval.py" for part in command)
        if not worker or not owns_root or not owns_script:
            result["issues"].append(issue("invalid_worker_registration", "error", "Launcher registration does not identify this root's queue worker", path=str(path)))
            continue
        registered[worker] = launch
        started = timestamp(launch.get("launched_at"))
        age = current_time - started if started is not None else None
        pid_path = root / "logs" / f"{worker}.pid"
        pid = int(pid_path.read_text()) if pid_path.exists() else None
        lease = lease_status(root / "logs" / f"{worker}.lease.lock")
        exit_path = root / "logs" / f"{worker}.exit"
        exit_code = int(exit_path.read_text()) if exit_path.exists() and (started is None or exit_path.stat().st_mtime >= started) else None
        previous_worker = old_workers.get(worker, {})
        same_launch = previous_worker.get("launch_pid") == launch["pid"]
        free_streak = previous_worker.get("consecutive_free_lease_polls", 0) + 1 if lease == "free" and same_launch else int(lease == "free")
        active = [key for key, item in state.items() if item.get("worker") == worker and item.get("status") == "running"]
        detail = {"launch_pid": launch["pid"], "queue_pid_file": pid, "lease_status": lease,
                  "consecutive_free_lease_polls": free_streak, "exit_code": exit_code,
                  "launched_at": launch.get("launched_at"), "active_tasks": active,
                  "alive_evidence": lease == "held", "process_cmdline_verified": False,
                  "lease_note": "A free local lease is not proof of a stopped remote worker on this shared filesystem.",
                  "active_output_evidence": False}
        result["workers"][worker] = detail
        if pid is not None and pid != launch["pid"]:
            result["issues"].append(issue("worker_pid_mismatch", "warning", "Queue PID file differs from launcher PID; inspect this registered worker", worker=worker))
        if exit_code not in (None, 0):
            result["issues"].append(issue("worker_nonzero_exit", "error", "Registered evaluation worker recorded a nonzero exit", worker=worker, exit_code=exit_code, tasks=active))

    for key, item in sorted(state.items()):
        mode, name = key.split("/", 1)
        if mode not in ("mcq", "openqa", "score") or name not in {task["label"] for task in tasks}:
            result["issues"].append(issue("unknown_queue_task", "warning", "Queue contains a task outside the frozen task list", task=key))
            continue
        folder = root / ("openqa" if mode == "score" else mode) / name
        started = timestamp(item.get("started_at"))
        elapsed = current_time - started if started is not None else None
        job = {"task": key, "status": item.get("status"), "worker": item.get("worker"),
               "queue_pid": item.get("pid"), "devices": item.get("devices"),
               "started_at": item.get("started_at"), "elapsed_seconds": elapsed,
               "exit_code": item.get("exit_code"), "cards": []}
        if item.get("worker") not in registered:
            result["issues"].append(issue("task_worker_unregistered", "warning", "Task worker has no launcher registration in this root", task=key, worker=item.get("worker")))
        card_root = folder / ("judge_v2" if mode == "score" else "shards")
        devices = str(item.get("devices", "")).split(",")
        latest = None
        malformed = 0
        for card in sorted(card_root.glob("card_*"), key=lambda p: int(p.name.split("_")[-1])):
            count = count_jsonl(card / "results.jsonl")
            input_count = count_jsonl(card / "input.jsonl")["rows"]
            index = int(card.name.split("_")[-1])
            detail = {"shard": card.name, "device": devices[index] if mode != "score" and index < len(devices) else None,
                      "rows_written": count["rows"], "expected_rows": input_count,
                      "fraction": count["rows"] / input_count if input_count else None,
                      "seconds_since_result_write": current_time - count["mtime"] if count["mtime"] else None,
                      "incomplete_final_write": count["incomplete_final_line"],
                      "malformed_complete_lines": count["malformed_complete_lines"]}
            job["cards"].append(detail)
            malformed += count["malformed_complete_lines"]
            if count["mtime"]:
                latest = max(latest or count["mtime"], count["mtime"])
            if count["rows"] > input_count and input_count:
                result["issues"].append(issue("shard_excess_rows", "error", "Shard result contains more rows than its assigned input", task=key, path=str(card)))
            if item.get("status") == "running":
                content = tail_current_log(card / "infer.log", started)
                match = FATAL.search(content)
                if match:
                    result["issues"].append(issue("current_shard_fatal_log", "error", "Fatal traceback/error marker in the current shard log", task=key, path=str(card / "infer.log"), marker=match.group(0)))
        if malformed:
            result["issues"].append(issue("malformed_result_lines", "error", "Nonfinal completed JSONL lines are malformed", task=key, malformed_lines=malformed))
        shard_count = sum(card["rows_written"] for card in job["cards"])
        merged = count_jsonl(folder / "results.jsonl") if mode != "score" else {"rows": 0}
        job["rows_written"] = max(shard_count, merged["rows"])
        job["expected_rows"] = sum(card["expected_rows"] for card in job["cards"]) if mode == "score" else expected
        job["seconds_since_result_write"] = current_time - latest if latest else None
        job["progress"] = f"{job['rows_written']}/{job['expected_rows']}"
        job["phase"] = "prefill_or_loading" if job["rows_written"] == 0 and item.get("status") == "running" else item.get("status")
        worker_detail = result["workers"].get(item.get("worker"))
        if worker_detail is not None and item.get("status") == "running" and latest is not None and current_time - latest < stall_seconds:
            worker_detail["active_output_evidence"] = True
            worker_detail["latest_result_age_seconds"] = min(worker_detail.get("latest_result_age_seconds", current_time - latest), current_time - latest)
        if item.get("status") == "failed":
            result["issues"].append(issue("task_failed", "error", "Evaluation task failed; preserve outputs and diagnose before restarting", task=key, worker=item.get("worker"), exit_code=item.get("exit_code")))
        if item.get("status") == "running" and elapsed and elapsed > stall_seconds:
            activity = latest
            for path in card_root.glob("card_*/infer.log"):
                activity = max(activity or path.stat().st_mtime, path.stat().st_mtime)
            if activity is None or current_time - activity > stall_seconds:
                result["issues"].append(issue("task_no_recent_activity", "warning", "No new result or shard log activity for 30 minutes; slow prefill is not automatically treated as failure", task=key))
        if mode != "score" and job["rows_written"] and elapsed:
            rate = job["rows_written"] / elapsed
            remaining = max(0, expected - job["rows_written"])
            job["elapsed_average_rows_per_second"] = rate
            job["rough_remaining_seconds_range"] = [remaining / rate * 0.5, remaining / rate * 2.0]
            job["eta_note"] = "Rough elapsed-average estimate including loading; video lengths and future input difficulty differ."
        result["jobs"].append(job)

    summary_counts = {"mcq": 0, "openqa": 0}
    uncertain = 0
    for task in tasks:
        for mode in ("mcq", "openqa"):
            summary = load_json(root / mode / task["label"] / "summary.json")
            if summary is not None:
                if summary.get("total") != expected:
                    result["issues"].append(issue("summary_wrong_total", "error", "Published summary does not contain all 518 questions", task=mode + "/" + task["label"]))
                else:
                    summary_counts[mode] += 1
                    uncertain += int(summary.get("uncertain", 0))
    jobs = {job["task"]: job for job in result["jobs"]}
    observed = sum(jobs.get(mode + "/" + task["label"], {}).get("rows_written", 0)
                   for task in tasks for mode in ("mcq", "openqa"))
    progress = {"mcq_completed": sum(state.get("mcq/" + task["label"], {}).get("status") == "complete" for task in tasks),
                "mcq_total": len(tasks),
                "openqa_generated": sum(state.get("openqa/" + task["label"], {}).get("status") in ("generated", "complete") for task in tasks),
                "openqa_generation_total": len(tasks),
                "openqa_scoring_completed": sum(state.get("score/" + task["label"], {}).get("status") == "complete" for task in tasks),
                "openqa_scoring_total": len(tasks), "summary_files_complete": summary_counts,
                "generation_rows_written": observed, "generation_rows_expected": expected * len(tasks) * 2,
                "unresolved_semantic_grades": uncertain}
    result["progress"] = progress
    if previous and current_time > previous.get("timestamp", current_time):
        delta = observed - previous.get("progress", {}).get("generation_rows_written", observed)
        seconds = current_time - previous["timestamp"]
        if delta > 0:
            rate = delta / seconds
            remaining = max(0, progress["generation_rows_expected"] - observed)
            result["rough_generation_eta"] = {"recent_rows_per_second": rate, "observed_window_seconds": seconds,
                                               "remaining_seconds_range": [remaining / rate * 0.5, remaining / rate * 2],
                                               "note": "Generation only, extrapolating current leased nodes. Excludes semantic scoring and future resource/phase changes; unstable for one short window."}
    complete = all(state.get(mode + "/" + task["label"], {}).get("status") == "complete"
                   for task in tasks for mode in ("mcq", "openqa"))
    complete = complete and summary_counts == {"mcq": len(tasks), "openqa": len(tasks)}
    result["all_26_tasks_and_summaries_complete"] = complete
    result["all_semantic_grades_resolved"] = complete and uncertain == 0
    result["healthy"] = not any(item["severity"] == "error" for item in result["issues"])
    return result


def persist(root: Path, observation: dict, previous: dict | None) -> None:
    atomic_write(root / "health_status.json", observation)
    with (root / "monitor_history.jsonl").open("a") as handle:
        handle.write(json.dumps(observation, ensure_ascii=False, allow_nan=False) + "\n")
    old_issues = {item["id"]: item["severity"] for item in (previous or {}).get("issues", [])}
    new_issues = [item for item in observation["issues"] if old_issues.get(item["id"]) != item["severity"]]
    resolved_issues = [key for key in old_issues if key not in {item["id"] for item in observation["issues"]}]
    if new_issues or resolved_issues:
        with (root / "alerts.jsonl").open("a") as handle:
            handle.write(json.dumps({"time": observation["time"], "issues": new_issues, "cleared_issue_ids": resolved_issues}, ensure_ascii=False) + "\n")
    atomic_write(root / "repair_needed.json", {"time": observation["time"],
                                               "automatic_restart_enabled": False,
                                               "items": observation["issues"],
                                               "unresolved_semantic_grades": observation.get("progress", {}).get("unresolved_semantic_grades"),
                                               "operator": "Root agent diagnoses and repairs only this evaluation's owned tasks."})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--stall-seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.interval < 10:
        raise ValueError("Monitoring interval must be at least 10 seconds")
    root = args.root.resolve()
    (root / "logs").mkdir(exist_ok=True)
    monitor_lock = (root / "logs/training_matched_monitor.lock").open("a")
    if not args.once:
        fcntl.flock(monitor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root / "logs/training_matched_monitor.pid").write_text(str(os.getpid()) + "\n")
    previous = load_json(root / "health_status.json")
    while True:
        try:
            observation = snapshot(root, previous, args.stall_seconds)
            observation["interval_seconds"] = args.interval
        except Exception as error:
            observation = {"time": now(), "timestamp": time.time(), "root": str(root), "monitor_pid": os.getpid(),
                           "healthy": False, "all_26_tasks_and_summaries_complete": False,
                           "issues": [issue("monitor_observation_error", "warning", f"{type(error).__name__}: {error}")]}
        persist(root, observation, previous)
        print(json.dumps({"time": observation["time"], "progress": observation.get("progress"),
                          "healthy": observation["healthy"], "issues": observation["issues"]}, ensure_ascii=False), flush=True)
        if args.once:
            return
        if observation.get("all_26_tasks_and_summaries_complete"):
            command = [sys.executable, str(REPO / "training_code/scripts/aggregate_worldsense_training_matched_eval.py"), "--root", str(root)]
            with (root / "logs/final_aggregate.log").open("a") as log:
                completed = subprocess.run(command, stdout=log, stderr=log)
            if completed.returncode:
                observation["issues"].append(issue("final_aggregation_failed", "error", "Final aggregation failed; inspect final_aggregate.log", exit_code=completed.returncode))
                observation["healthy"] = False
                persist(root, observation, previous)
            else:
                atomic_write(root / "monitor_completion.json", {"time": now(), "exit_code": 0,
                                                               "all_26_tasks_and_summaries_complete": True,
                                                               "all_semantic_grades_resolved": observation["all_semantic_grades_resolved"],
                                                               "unresolved_semantic_grades": observation["progress"]["unresolved_semantic_grades"],
                                                               "final_aggregation": str(root / "comparison/summary.json")})
                (root / "logs/training_matched_monitor.exit").write_text("0\n")
                return
        previous = observation
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
