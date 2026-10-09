#!/usr/bin/env python3
"""Observe card-pool evaluation health without touching jobs or queue reservations."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training_code/scripts"))
from distributed_directory_lock import DirectoryLock
from watch_worldsense_training_matched_eval import (
    atomic_write, count_jsonl, load_json, now, tail_current_log, timestamp,
)

FATAL = re.compile(r"Traceback \(most recent call last\)|(?:RuntimeError|OutOfMemoryError|ChildFailedError|AssertionError|ValueError|ImportError|ModuleNotFoundError):")


def alert(kind, severity, message, subject, **details):
    return dict(id=f"{kind}:{subject}", kind=kind, severity=severity,
                message=message, subject=subject, **details)


def absolute(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def units_of(state):
    raw = state.get("units", {})
    if isinstance(raw, list):
        return [(str(unit.get("id", i)), unit) for i, unit in enumerate(raw)]
    if isinstance(raw, dict):
        return list(raw.items())
    raise ValueError("card_pool_state.units must be a list or mapping")


def output_detail(folder, result_path=None, expected=None):
    source = count_jsonl(folder / "input.jsonl")
    results = count_jsonl(result_path or folder / "results.jsonl")
    expected = source["rows"] if expected is None else int(expected)
    log = folder / "infer.log"
    return dict(folder=str(folder), rows_written=results["rows"], expected_rows=expected,
                fraction=results["rows"] / expected if expected else None,
                result_mtime=results["mtime"],
                log_mtime=log.stat().st_mtime if log.exists() else None,
                incomplete_final_write=results["incomplete_final_line"],
                malformed_complete_lines=results["malformed_complete_lines"])


def inspect_output(issues, subject, detail, status, started, current_time, stall_seconds):
    if detail["malformed_complete_lines"]:
        issues.append(alert("malformed_result_lines", "error", "Completed JSONL lines are malformed",
                            subject, malformed_lines=detail["malformed_complete_lines"]))
    expected = detail["expected_rows"]
    if expected and detail["rows_written"] > expected:
        issues.append(alert("unit_excess_rows", "error", "Result count exceeds assigned input",
                            subject, rows=detail["rows_written"], expected=expected))
    if status == "complete" and detail["rows_written"] != expected:
        issues.append(alert("completed_unit_missing_results", "error", "Completed unit has incomplete result rows",
                            subject, rows=detail["rows_written"], expected=expected))
    if status == "running":
        content = tail_current_log(Path(detail["folder"]) / "infer.log", started)
        # Worker appends an explicit boundary before each attempt; do not flag
        # traceback text left by an older attempt in the same append-only log.
        if "CARD_POOL_START " in content:
            content = content.rsplit("CARD_POOL_START ", 1)[1]
        marker = FATAL.search(content)
        if marker:
            issues.append(alert("current_unit_fatal_log", "error", "Fatal marker in current inference log",
                                subject, path=detail["folder"] + "/infer.log", marker=marker.group()))
        activity = max([value for value in [detail["result_mtime"], detail["log_mtime"], started] if value is not None],
                       default=None)
        if activity is not None and current_time - activity > stall_seconds:
            issues.append(alert("unit_no_recent_output_activity", "warning",
                                "No new output/log activity; slow media decoding or prefill requires inspection",
                                subject, seconds_since_activity=current_time - activity))


def snapshot(root, previous=None, stall_seconds=1800, heartbeat_seconds=180):
    current = time.time()
    tasks = load_json(root / "tasks.json", [])
    expected = count_jsonl(root / "data/worldsense.labels.jsonl")["rows"]
    if expected != 518:
        raise ValueError(f"Expected frozen 518 question labels, got {expected}")
    state = load_json(root / "card_pool_state.json", {})
    queue = load_json(root / "queue_state.json", {})
    controller = load_json(root / "card_pool_controller.json", {})
    result = dict(time=now(), timestamp=current, root=str(root.resolve()), monitor_pid=os.getpid(),
                  interval_seconds=60, issues=[], workers={}, cards={}, units=[], legacy_jobs=[],
                  controller=controller, automatic_retry_enabled=False,
                  liveness_method="Read only unit state, worker/controller heartbeat files, results and current logs; no signals, SSH or reservation changes.")
    issues = result["issues"]
    outputs_by_task = {}
    occupied = {}
    running_workers = set()
    running_tasks = set()
    heartbeat_slots = {"owned_unit": 0, "busy_legacy_or_external": 0, "idle": 0, "unknown_blocked": 0}
    observed_busy_slots = set()
    status_counts = {name: 0 for name in ("pending", "running", "complete", "failed")}
    for unit_id, unit in units_of(state):
        status = unit.get("status")
        status_counts[status] = status_counts.get(status, 0) + 1
        kind = unit.get("kind")
        task_key = unit.get("task_key", "")
        device = unit.get("assigned_device", unit.get("device"))
        folder = absolute(root, unit["folder"])
        result_path = absolute(root, unit["result_path"]) if unit.get("result_path") else folder / "results.jsonl"
        detail = output_detail(folder, result_path, unit.get("expected_rows"))
        record = dict(id=unit_id, status=status, worker=unit.get("worker"), device=device,
                      kind=kind, task_key=task_key, pid=unit.get("pid"), **detail)
        started = timestamp(unit.get("started_at", unit.get("claimed_at")))
        record["seconds_since_start"] = current - started if started else None
        record["phase"] = "loading_or_prefill" if status == "running" and detail["rows_written"] == 0 else status
        inspect_output(issues, unit_id, detail, status, started, current, stall_seconds)
        if status == "failed":
            issues.append(alert("unit_failed", "error", "Card-pool unit failed; operator repair required",
                                unit_id, worker=unit.get("worker"), device=device,
                                task=task_key, exit_code=unit.get("exit_code"), error=unit.get("error")))
        if status == "running":
            slot = f"{unit.get('worker')}:{device}"
            if slot in occupied:
                issues.append(alert("duplicate_running_card_claim", "error", "Two units claim the same card",
                                    slot, unit_ids=[occupied[slot], unit_id]))
            occupied[slot] = unit_id
            running_workers.add(unit.get("worker"))
            running_tasks.add(task_key)
            result["cards"][slot] = dict(unit_id=unit_id, task_key=task_key, kind=kind, status="running",
                                        rows_written=detail["rows_written"], expected_rows=detail["expected_rows"])
        outputs_by_task.setdefault(task_key, {})[str(result_path)] = detail["rows_written"]
        result["units"].append(record)

    # Old node-level runners can finish safely while card-pool workers wait on
    # their cards. Keep their progress/log diagnostics visible during migration.
    for key, item in queue.items():
        if item.get("status") == "failed":
            issues.append(alert("task_failed", "error", "Evaluation task state is failed",
                                key, exit_code=item.get("exit_code")))
        if item.get("status") != "running" or key in running_tasks or item.get("scheduler") == "card-pool":
            continue
        mode, label = key.split("/", 1)
        if mode not in ("mcq", "openqa", "score"):
            continue
        base = root / ("openqa" if mode == "score" else mode) / label
        source = base / ("judge_v2" if mode == "score" else "shards")
        started = timestamp(item.get("started_at"))
        job = dict(task_key=key, worker=item.get("worker"), status="running", cards=[])
        for folder in sorted(source.glob("card_*")):
            detail = output_detail(folder)
            inspect_output(issues, "legacy/" + key + "/" + folder.name, detail,
                           "running", started, current, stall_seconds)
            outputs_by_task.setdefault(key, {})[str(folder / "results.jsonl")] = detail["rows_written"]
            job["cards"].append(detail)
        job["rows_written"] = sum(card["rows_written"] for card in job["cards"])
        result["legacy_jobs"].append(job)

    unresolved = 0
    completed_summaries = dict(mcq=0, openqa=0)
    completed_generations = dict(mcq=0, openqa=0)
    generation_rows = 0
    judges_completed = 0
    for task in tasks:
        for mode in ("mcq", "openqa"):
            key = mode + "/" + task["label"]
            merged = count_jsonl(root / mode / task["label"] / "results.jsonl")
            written = max(sum(outputs_by_task.get(key, {}).values()), merged["rows"])
            generation_rows += written
            if queue.get(key, {}).get("status") in ("generated", "complete"):
                completed_generations[mode] += 1
            summary = load_json(root / mode / task["label"] / "summary.json")
            if summary is not None:
                if summary.get("total") != expected:
                    issues.append(alert("summary_wrong_total", "error", "Summary does not contain all 518 questions", key))
                else:
                    # The open-QA prepare-only stage publishes a v2 summary
                    # before judge inference. Prefix/row count alone cannot
                    # distinguish that temporary table from a final grade.
                    final_grade = queue.get(key, {}).get("status") == "complete"
                    if mode == "openqa":
                        final_grade = (final_grade
                                       and queue.get("score/" + task["label"], {}).get("status") == "complete"
                                       and summary.get("scoring_pipeline_reliable_on_calibration") is True)
                    if not final_grade:
                        continue
                    scoring = str(summary.get("scoring", ""))
                    prefix = "mcq-v2:" if mode == "mcq" else "v2:"
                    if scoring.startswith(prefix):
                        completed_summaries[mode] += 1
                        unresolved += int(summary.get("uncertain", 0))
                    else:
                        issues.append(alert("summary_not_v2", "warning", "Summary is awaiting current v2 grading", key))
        judges_completed += queue.get("score/" + task["label"], {}).get("status") == "complete"
    all_complete = all(queue.get(mode + "/" + task["label"], {}).get("status") == "complete"
                       for task in tasks for mode in ("mcq", "openqa"))
    all_complete = bool(tasks) and all_complete and completed_summaries == {"mcq": len(tasks), "openqa": len(tasks)}

    for path in sorted((root / "card_pool_workers").glob("*/heartbeat.json")):
        heartbeat = load_json(path, {})
        worker = heartbeat.get("worker", path.parent.name)
        stamp = timestamp(heartbeat.get("heartbeat_at"))
        age = current - stamp if stamp is not None else None
        result["workers"][worker] = dict(**heartbeat, heartbeat_age_seconds=age)
        devices = heartbeat.get("devices", heartbeat.get("slots", {}))
        if isinstance(devices, dict):
            for device, value in devices.items():
                slot = f"{worker}:{device}"
                result["cards"].setdefault(slot, dict(worker=worker, device=str(device), state=value))
                slot_status = value.get("status") if isinstance(value, dict) else value
                if slot_status in heartbeat_slots:
                    heartbeat_slots[slot_status] += 1
                if slot_status in ("owned_unit", "busy_legacy_or_external"):
                    observed_busy_slots.add(slot)
        if heartbeat.get("probe_ok") is False:
            issues.append(alert("worker_npu_probe_failed", "warning",
                                "Worker cannot establish current card availability and blocks new allocation", worker))
        if not all_complete and age is not None and age > heartbeat_seconds:
            severity = "error" if worker in running_workers else "warning"
            issues.append(alert("worker_heartbeat_stale", severity, "Card-pool worker heartbeat is stale",
                                worker, seconds_since_heartbeat=age, pid=heartbeat.get("worker_pid", heartbeat.get("pid"))))
        if stamp is None:
            issues.append(alert("worker_heartbeat_missing_timestamp", "warning", "Worker heartbeat lacks heartbeat_at", worker))
    for worker in running_workers - result["workers"].keys():
        issues.append(alert("running_worker_heartbeat_missing", "warning",
                            "Running unit has no registered card-pool worker heartbeat", str(worker)))
    if controller.get("status") == "failed":
        issues.append(alert("controller_failed", "error", "CPU card-pool controller reported a failure",
                            "controller", error=controller.get("error")))
    heartbeat = timestamp(controller.get("heartbeat_at"))
    if heartbeat is not None and not all_complete and current - heartbeat > heartbeat_seconds:
        issues.append(alert("controller_heartbeat_stale", "error", "CPU controller heartbeat is stale",
                            "controller", seconds_since_heartbeat=current - heartbeat))
    if controller and not all_complete:
        controller_log = root / "logs/card_pool_controller.log"
        if not controller_log.exists():
            controller_log = root / "controller_worldsense_card_pool.log"
        marker = FATAL.search(tail_current_log(controller_log,
                                              timestamp(controller.get("started_at", controller.get("heartbeat_at")))))
        if marker:
            issues.append(alert("controller_fatal_log", "error", "Fatal marker in current CPU controller log",
                                "controller", marker=marker.group(), path=str(controller_log)))
    result["progress"] = dict(unit_status_counts=status_counts,
                              active_cards=len(set(occupied) | observed_busy_slots),
                              active_pool_cards=len(occupied), heartbeat_slot_status_counts=heartbeat_slots,
                              mcq_generated=completed_generations["mcq"], mcq_total=len(tasks),
                              openqa_generated=completed_generations["openqa"], openqa_total=len(tasks),
                              openqa_scoring_completed=judges_completed,
                              summary_files_complete=completed_summaries,
                              generation_rows_written=generation_rows,
                              generation_rows_expected=expected * len(tasks) * 2,
                              unresolved_semantic_grades=unresolved)
    if previous and current > previous.get("timestamp", current):
        seconds = current - previous["timestamp"]
        delta = generation_rows - previous.get("progress", {}).get("generation_rows_written", generation_rows)
        if delta > 0:
            rate = delta / seconds
            remaining = max(0, expected * len(tasks) * 2 - generation_rows)
            result["rough_generation_eta"] = dict(recent_rows_per_second=rate,
                                                  remaining_seconds_range=[remaining / rate * 0.5, remaining / rate * 2],
                                                  note="Generation only; excludes scoring and changes in media/task workload.")
    result["all_26_tasks_and_summaries_complete"] = all_complete
    result["all_semantic_grades_resolved"] = all_complete and unresolved == 0
    result["healthy"] = not any(item["severity"] == "error" for item in issues)
    return result


def persist(root, observation, previous=None):
    atomic_write(root / "card_pool_health.json", observation)
    with (root / "card_pool_monitor_history.jsonl").open("a") as handle:
        handle.write(json.dumps(observation, ensure_ascii=False, allow_nan=False) + "\n")
    old = {item["id"]: item["severity"] for item in (previous or {}).get("issues", [])}
    current = {item["id"]: item["severity"] for item in observation["issues"]}
    new = [item for item in observation["issues"] if old.get(item["id"]) != item["severity"]]
    cleared = sorted(old.keys() - current.keys())
    if new or cleared:
        with (root / "card_pool_alerts.jsonl").open("a") as handle:
            handle.write(json.dumps(dict(time=observation["time"], issues=new, cleared_issue_ids=cleared),
                                    ensure_ascii=False) + "\n")


def run_monitor(args):
    root = args.root.resolve()
    previous = load_json(root / "card_pool_health.json")
    while True:
        try:
            observation = snapshot(root, previous, args.stall_seconds, args.heartbeat_seconds)
            observation["interval_seconds"] = args.interval
        except Exception as error:
            observation = dict(time=now(), timestamp=time.time(), root=str(root),
                               monitor_pid=os.getpid(), healthy=False,
                               all_26_tasks_and_summaries_complete=False,
                               issues=[alert("monitor_observation_error", "error",
                                             f"{type(error).__name__}: {error}", "monitor")])
        persist(root, observation, previous)
        print(json.dumps(dict(time=observation["time"], progress=observation.get("progress"),
                              healthy=observation["healthy"], issues=observation["issues"]), ensure_ascii=False), flush=True)
        if args.once:
            return
        if observation.get("all_26_tasks_and_summaries_complete"):
            command = [sys.executable, str(REPO / "training_code/scripts/aggregate_worldsense_training_matched_eval.py"),
                       "--root", str(root)]
            with (root / "logs/card_pool_final_aggregate.log").open("a") as log:
                completed = subprocess.run(command, stdout=log, stderr=log)
            if completed.returncode:
                observation["issues"].append(alert("final_aggregation_failed", "error",
                                                   "Final aggregation failed; inspect card_pool_final_aggregate.log",
                                                   "aggregate", exit_code=completed.returncode))
                observation["healthy"] = False
                persist(root, observation, previous)
                (root / "logs/card_pool_monitor.exit").write_text(str(completed.returncode) + "\n")
                return
            atomic_write(root / "card_pool_monitor_completion.json",
                         dict(time=now(), exit_code=0, all_26_tasks_and_summaries_complete=True,
                              all_semantic_grades_resolved=observation["all_semantic_grades_resolved"],
                              unresolved_semantic_grades=observation["progress"]["unresolved_semantic_grades"],
                              final_aggregation=str(root / "comparison/summary.json")))
            (root / "logs/card_pool_monitor.exit").write_text("0\n")
            return
        previous = observation
        time.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--stall-seconds", type=int, default=1800)
    parser.add_argument("--heartbeat-seconds", type=int, default=180)
    args = parser.parse_args()
    if args.interval < 10:
        raise ValueError("Monitoring interval must be at least 10 seconds")
    root = args.root.resolve()
    (root / "logs").mkdir(exist_ok=True)
    if args.once:
        run_monitor(args)
    else:
        with DirectoryLock(root / "logs/card_pool_monitor.lock.d", timeout=0):
            (root / "logs/card_pool_monitor.pid").write_text(str(os.getpid()) + "\n")
            run_monitor(args)


if __name__ == "__main__":
    main()
