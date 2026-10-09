#!/usr/bin/env python3
"""Run independent frozen work units as each selected NPU becomes free.

The controller supplies commands and input/output paths. This worker never
changes input rows, seeds, decoding limits, or commands; it only assigns one
physical NPU and a port unique to that card. Legacy and unrelated NPU processes
are observed and left running. Failed units remain failed until the controller
explicitly repairs them.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training_code/scripts"))
from distributed_directory_lock import DirectoryLock, DirectoryLockTimeout  # noqa: E402
from run_worldsense_training_matched_eval import env_for  # noqa: E402


class ClaimBlocked(RuntimeError):
    """This card must remain reserved until an operator resolves ownership."""


def transition_failure(root: Path, phase: str, error: Exception, trace: str,
                       *, recovered: bool, **details) -> None:
    """Keep the original lock/write error visible even after a safe recovery."""
    record = {"at": now(), "kind": phase, "message": str(error), "traceback": trace,
              "recovered_committed_transition": recovered, "worker_pid": os.getpid(),
              "hostname": socket.gethostname(), **details}
    append_event(root / "card_pool_alerts.jsonl", record)
    worker = details.get("worker")
    if worker and (root / "card_pool_workers" / worker).is_dir():
        append_event(root / "card_pool_workers" / worker / "alerts.jsonl", record)
    print(json.dumps(record, ensure_ascii=False), file=sys.stderr, flush=True)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def atomic_save(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("w") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_state(path: Path) -> dict:
    if not path.exists():
        return {"units": {}}
    state = json.loads(path.read_text())
    if not isinstance(state.get("units"), dict):
        raise ValueError("card_pool_state.json must contain a units mapping")
    return state


def append_event(path: Path, value: dict) -> None:
    payload = (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if os.write(fd, payload) != len(payload):
            raise OSError("Incomplete event journal append")
    finally:
        os.close(fd)


def parse_npu_processes(output: str, devices: list[str]) -> dict[str, dict]:
    """Only authoritative process-table rows establish that a card is empty."""
    result = {device: {"status": "unknown", "processes": []} for device in devices}
    header = re.search(r"^\|\s*NPU\s+Chip\s*\|\s*Process id\b", output, re.M | re.I)
    if not header:
        return result
    section = output[header.end():]
    empties = set(re.findall(r"No running processes found in NPU\s+(\d+)(?=\s|\|)", section))
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) < 4:
            continue
        device_chip = re.fullmatch(r"(\d+)\s+(\d+)", cells[0])
        if not device_chip or not re.fullmatch(r"\d+", cells[1]):
            continue
        device, chip = device_chip.groups()
        if device in result:
            result[device]["processes"].append({"pid": int(cells[1]), "chip": int(chip),
                                                  "name": cells[2], "memory": cells[3]})
    for device, item in result.items():
        if item["processes"] and device in empties:
            # A contradictory/partly copied report must never grant the card.
            item["status"] = "unknown"
        elif item["processes"]:
            item["status"] = "busy"
        elif device in empties:
            item["status"] = "empty"
    return result


def probe_npus(devices: list[str]) -> dict:
    try:
        process = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"observed_at": now(), "ok": False, "error": f"{type(error).__name__}: {error}",
                "devices": {device: {"status": "unknown", "processes": []} for device in devices}}
    if process.returncode:
        return {"observed_at": now(), "ok": False, "returncode": process.returncode,
                "error": process.stderr[-2000:],
                "devices": {device: {"status": "unknown", "processes": []} for device in devices}}
    return {"observed_at": now(), "ok": True, "returncode": 0,
            "devices": parse_npu_processes(process.stdout, devices)}


def validate_unit(root: Path, unit: dict) -> None:
    command = unit.get("command")
    if not isinstance(command, list) or not command or any(not isinstance(arg, str) for arg in command):
        raise ValueError("Unit command must be a nonempty argument list")
    folder = Path(unit.get("folder", ""))
    if not folder.is_absolute() or not folder.resolve().is_relative_to(root.resolve()):
        raise ValueError("Unit folder must be an absolute path under this evaluation root")
    if unit.get("kind") not in ("generation", "judge") or not isinstance(unit.get("task_key"), str):
        raise ValueError("Unit kind/task_key is missing or invalid")
    overrides = unit.get("env_overrides", {})
    if not isinstance(overrides, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in overrides.items()):
        raise ValueError("env_overrides must map strings to strings")


def claim_pending(root: Path, worker: str, device: str, verified_status: str,
                  lock_timeout: float = 10) -> tuple[str, dict] | None:
    if verified_status != "empty":
        return None
    path = root / "card_pool_state.json"
    proposed = None
    reservations = []
    try:
        with DirectoryLock(root / "card_pool.lock.d", timeout=lock_timeout):
            state = read_state(path)
            reservations = [key for key, unit in state["units"].items()
                            if unit.get("status") == "running" and unit.get("worker") == worker
                            and unit.get("assigned_device") == device]
            if reservations:
                raise ClaimBlocked("Existing running reservation for this worker/card: " + str(reservations))
            if state.get("stop_workers") is True or state.get("finished") is True or state.get("controller_complete") is True:
                return None
            pending = [(index, key, unit) for index, (key, unit) in enumerate(state["units"].items()) if unit.get("status") == "pending"]
            pending.sort(key=lambda item: (float(item[2].get("priority", 0)), item[0]))
            if not pending:
                return None
            _, key, unit = pending[0]
            validate_unit(root, unit)
            unit.update(status="running", claim_id=uuid.uuid4().hex, worker=worker,
                        worker_pid=os.getpid(), hostname=socket.gethostname(), assigned_device=device,
                        claimed_at=now(), claimed_epoch=time.time(), attempt=int(unit.get("attempt", 0)) + 1,
                        allocation_stage="claimed-not-started")
            proposed = key, copy.deepcopy(unit)
            atomic_save(path, state)
        return proposed
    except Exception as error:
        trace = traceback.format_exc()
        if proposed is None:
            if reservations:
                raise ClaimBlocked("Existing reservation; card blocked: " + str(reservations)) from error
            raise
        key, expected = proposed
        try:
            committed = read_state(path)["units"].get(key, {})
            recovered = (committed.get("status") == "running"
                         and all(committed.get(field) == expected[field]
                                 for field in ("claim_id", "worker", "assigned_device", "worker_pid", "hostname"))
                         and committed.get("allocation_stage") == "claimed-not-started"
                         and committed.get("pid") is None)
        except Exception:
            trace += "\nRecovery state read also failed:\n" + traceback.format_exc()
            recovered = False
        try:
            transition_failure(root, "unit_claim_transition_failed", error, trace, recovered=recovered,
                               worker=worker, device=device, unit_id=key, claim_id=expected["claim_id"])
        except Exception as journal_error:
            raise ClaimBlocked("Claim transition audit failed; card blocked: " + key) from journal_error
        if recovered:
            # The atomic state write committed before context-manager exit
            # failed. Adopt only this exact unstarted nonce; never claim again.
            return key, copy.deepcopy(committed)
        raise ClaimBlocked("Claim commit could not be recovered safely; card blocked: " + key) from error


def update_owned(root: Path, key: str, claim_id: str, changes: dict, lock_timeout: float = 10) -> None:
    with DirectoryLock(root / "card_pool.lock.d", timeout=lock_timeout):
        path = root / "card_pool_state.json"
        state = read_state(path)
        unit = state["units"].get(key)
        if unit is None or unit.get("claim_id") != claim_id or unit.get("status") != "running":
            raise RuntimeError("Work unit ownership changed: " + key)
        unit.update(changes)
        atomic_save(path, state)


def process_identity(pid: int) -> dict | None:
    try:
        folder = Path("/proc") / str(pid)
        fields = (folder / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start_ticks": fields[19], "state": fields[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def unit_environment(device: str, overrides: dict) -> dict:
    env = env_for(device)
    env.update(overrides)
    # Assignment is an infrastructure choice, never an ML hyperparameter.
    env.update(ASCEND_RT_VISIBLE_DEVICES=device, NPROC_PER_NODE="1", NNODES="1",
               MASTER_ADDR="127.0.0.1", MASTER_PORT=str(30170 + int(device)))
    for key in ("RANK_TABLE_FILE", "RANK_TABLE_FILE_V_1_0", "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        env.pop(key, None)
    return env


def complete_process(root: Path, key: str, claim_id: str, returncode: int,
                     lock_timeout: float = 10, reason: str | None = None) -> dict:
    changes = {"status": "complete" if returncode == 0 else "failed", "exit_code": returncode,
               "finished_at": now(), "allocation_stage": "process-exited"}
    if reason:
        changes["failure_reason"] = reason
    path = root / "card_pool_state.json"

    def committed_completion(unit):
        return (unit.get("claim_id") == claim_id and unit.get("status") == changes["status"]
                and unit.get("exit_code") == returncode and unit.get("allocation_stage") == "process-exited"
                and bool(unit.get("finished_at")) and (reason is None or unit.get("failure_reason") == reason))

    def existing_changes(unit):
        return {field: unit[field] for field in changes if field in unit}

    try:
        with DirectoryLock(root / "card_pool.lock.d", timeout=lock_timeout):
            state = read_state(path)
            unit = state["units"].get(key, {})
            if committed_completion(unit):
                # A prior call may have committed then failed while releasing
                # the lock. Preserve its original completion timestamp.
                changes = existing_changes(unit)
            else:
                if unit.get("claim_id") != claim_id or unit.get("status") != "running":
                    raise RuntimeError("Work unit ownership changed: " + key)
                unit.update(changes)
                atomic_save(path, state)
        return changes
    except Exception as error:
        trace = traceback.format_exc()
        try:
            unit = read_state(path)["units"].get(key, {})
            recovered = committed_completion(unit)
        except Exception:
            trace += "\nRecovery state read also failed:\n" + traceback.format_exc()
            recovered = False
        transition_failure(root, "unit_completion_transition_failed", error, trace, recovered=recovered,
                           unit_id=key, claim_id=claim_id, worker=unit.get("worker") if recovered else None,
                           device=unit.get("assigned_device") if recovered else None, exit_code=returncode)
        if recovered:
            return existing_changes(unit)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--lock-timeout", type=float, default=10)
    args = parser.parse_args()
    root = args.root.resolve()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.worker):
        raise ValueError("Invalid worker identifier")
    devices = args.devices.split(",")
    if any(device not in set("01234567") for device in devices) or len(set(devices)) != len(devices):
        raise ValueError("Devices must be distinct IDs 0..7")
    if args.poll_seconds < 1:
        raise ValueError("poll-seconds must be at least one")
    worker_dir = root / "card_pool_workers" / args.worker
    worker_dir.mkdir(parents=True, exist_ok=True)
    leases = root / "card_pool_worker_leases"
    leases.mkdir(exist_ok=True)
    hostname = socket.gethostname()
    safe_hostname = re.sub(r"[^A-Za-z0-9_.-]", "_", hostname)
    own = {}
    blocked = {}
    alerts_seen = set()
    drain_requested = False

    def alert(kind: str, message: str, **details):
        fingerprint = (kind, details.get("unit_id"), details.get("device"))
        if fingerprint in alerts_seen:
            return
        alerts_seen.add(fingerprint)
        rec = {"at": now(), "worker": args.worker, "worker_pid": os.getpid(), "hostname": hostname,
               "kind": kind, "message": message, **details}
        append_event(root / "card_pool_alerts.jsonl", rec)
        append_event(worker_dir / "alerts.jsonl", rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    def event(name: str, **details):
        append_event(root / "card_pool_events.jsonl", {"at": now(), "event": name, "worker": args.worker,
                                                      "worker_pid": os.getpid(), "hostname": hostname, **details})

    def request_drain(signum, frame):
        nonlocal drain_requested
        drain_requested = True
        event("drain_requested", signal=signum)

    signal.signal(signal.SIGTERM, request_drain)
    signal.signal(signal.SIGINT, request_drain)
    with DirectoryLock(leases / f"{safe_hostname}.lock.d", timeout=0):
        (worker_dir / "pid").write_text(str(os.getpid()) + "\n")
        event("worker_started", devices=devices)
        while True:
            for device, slot in list(own.items()):
                returncode = slot["process"].poll()
                if returncode is None:
                    continue
                try:
                    changes = complete_process(root, slot["unit_id"], slot["unit"]["claim_id"], returncode, args.lock_timeout)
                except (DirectoryLockTimeout, OSError, RuntimeError) as error:
                    alert("unit_exit_state_update_failed", str(error), unit_id=slot["unit_id"], device=device, exit_code=returncode,
                          traceback=traceback.format_exc())
                    continue
                slot["log"].close()
                event("unit_finished", unit_id=slot["unit_id"], device=device, pid=slot["process"].pid, **changes)
                if returncode != 0:
                    alert("unit_failed", "Owned inference process exited unsuccessfully; no automatic retry", unit_id=slot["unit_id"], device=device,
                          task_key=slot["unit"]["task_key"], exit_code=returncode, folder=slot["unit"]["folder"])
                del own[device]
            probe = probe_npus(devices)
            if not probe["ok"]:
                alert("npu_probe_failed", "Cannot establish free NPUs; no new commands launched", error=probe.get("error"))
            state = read_state(root / "card_pool_state.json")
            stop_requested = drain_requested or any(state.get(key) is True for key in ("stop_workers", "finished", "controller_complete"))
            if not stop_requested:
                for device in devices:
                    if device in own or device in blocked or probe["devices"][device]["status"] != "empty":
                        continue
                    try:
                        claim = claim_pending(root, args.worker, device, probe["devices"][device]["status"], args.lock_timeout)
                    except ClaimBlocked as error:
                        blocked[device] = {"error": str(error), "blocked_at": now()}
                        alert("unit_claim_ownership_uncertain", str(error), device=device, traceback=traceback.format_exc())
                        continue
                    except (DirectoryLockTimeout, OSError, ValueError) as error:
                        alert("unit_claim_blocked", str(error), device=device, traceback=traceback.format_exc())
                        continue
                    if claim is None:
                        continue
                    key, unit = claim
                    # A legacy/external process may have claimed the physical
                    # card after the earlier probe. Recheck before Popen.
                    confirmation = probe_npus([device])
                    if not confirmation["ok"] or confirmation["devices"][device]["status"] != "empty":
                        update_owned(root, key, unit["claim_id"], {"status": "pending", "allocation_stage": "not-started-device-became-busy",
                                                                 "returned_at": now()}, args.lock_timeout)
                        event("claim_returned_without_launch", unit_id=key, device=device)
                        if not confirmation["ok"]:
                            alert("confirmation_probe_failed", "Card could not be confirmed empty; claim returned without launch",
                                  unit_id=key, device=device, error=confirmation.get("error"))
                        probe["devices"][device] = confirmation["devices"][device]
                        continue
                    folder = Path(unit["folder"])
                    folder.mkdir(parents=True, exist_ok=True)
                    log = (folder / "infer.log").open("a")
                    log.write("\nCARD_POOL_START " + json.dumps({"at": now(), "claim_id": unit["claim_id"], "unit_id": key, "device": device}) + "\n")
                    log.flush()
                    try:
                        process = subprocess.Popen(unit["command"], env=unit_environment(device, unit.get("env_overrides", {})),
                                                   stdin=subprocess.DEVNULL, stdout=log, stderr=log, cwd=str(REPO))
                    except OSError as error:
                        log.close()
                        complete_process(root, key, unit["claim_id"], 127, args.lock_timeout, f"Launch failed: {error}")
                        alert("unit_launch_failed", str(error), unit_id=key, device=device, folder=unit["folder"])
                        continue
                    identity = process_identity(process.pid)
                    metadata = {"unit_id": key, "claim_id": unit["claim_id"], "worker": args.worker,
                                "worker_pid": os.getpid(), "hostname": hostname, "assigned_device": device,
                                "pid": process.pid, "pid_start_ticks": identity["start_ticks"] if identity else None,
                                "started_at": now(), "command": unit["command"],
                                "command_sha256": hashlib.sha256(json.dumps(unit["command"]).encode()).hexdigest(),
                                "master_port": str(30170 + int(device))}
                    own[device] = {"unit_id": key, "unit": unit, "process": process, "log": log, "metadata": metadata}
                    (folder / "pid").write_text(str(process.pid) + "\n")
                    atomic_save(folder / "command.json", {"command": unit["command"]})
                    atomic_save(folder / "assigned_device.json", metadata)
                    try:
                        update_owned(root, key, unit["claim_id"], {key: value for key, value in metadata.items() if key not in ("unit_id", "command")}
                                     | {"allocation_stage": "process-running"}, args.lock_timeout)
                    except (DirectoryLockTimeout, OSError, RuntimeError) as error:
                        # Keep supervising this owned child. Never launch a
                        # second process onto its slot after a state-write error.
                        alert("unit_start_state_update_failed", str(error), unit_id=key, device=device, pid=process.pid,
                              traceback=traceback.format_exc())
                    event("unit_started", **metadata)
            slots = {}
            for device in devices:
                physical = probe["devices"][device]
                if device in own:
                    slot = own[device]
                    unit = slot["unit"]
                    slots[device] = {"status": "owned_unit", "unit_id": slot["unit_id"], "task_key": unit["task_key"],
                                     "kind": unit["kind"], "pid": slot["process"].pid,
                                     "started_at": slot["metadata"]["started_at"], "folder": unit["folder"],
                                     "expected_rows": unit.get("expected_rows"), "result_path": unit.get("result_path", str(Path(unit["folder"]) / "results.jsonl"))}
                elif device in blocked:
                    slots[device] = {"status": "unknown_blocked", "reservation_error": blocked[device],
                                     "observed_npu_processes": physical["processes"]}
                else:
                    slots[device] = {"status": {"empty": "idle", "busy": "busy_legacy_or_external", "unknown": "unknown_blocked"}[physical["status"]],
                                     "observed_npu_processes": physical["processes"]}
            heartbeat = {"worker": args.worker, "worker_pid": os.getpid(), "pid": os.getpid(), "hostname": hostname,
                         "heartbeat_at": now(), "devices": slots, "probe_at": probe["observed_at"],
                         "probe_ok": probe["ok"], "poll_seconds": args.poll_seconds,
                         "draining": stop_requested, "status": "stopping" if stop_requested else "running"}
            atomic_save(worker_dir / "heartbeat.json", heartbeat)
            if stop_requested and not own:
                heartbeat.update(status="complete", finished_at=now(), exit_code=0)
                atomic_save(worker_dir / "heartbeat.json", heartbeat)
                (worker_dir / "exit").write_text("0\n")
                event("worker_finished", exit_code=0)
                return
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
