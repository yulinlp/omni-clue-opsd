#!/usr/bin/env python3
"""Shared-filesystem queue with cross-host directory locks and safe adoption."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'training_code/scripts'))
from distributed_directory_lock import DirectoryLock

PYTHON = '/home/ma-user/anaconda3/envs/PyTorch-2.9.0/bin/python'


def now():
    return datetime.now().astimezone().isoformat()


def save(path, state):
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(path)


def read_state(path):
    return json.loads(path.read_text()) if path.exists() else {}


def journal(root, event, **details):
    row = dict(at=now(), event=event, hostname=socket.gethostname(), queue_pid=os.getpid(), **details)
    payload = (json.dumps(row, sort_keys=True) + '\n').encode()
    fd = os.open(root / 'queue_events.jsonl', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if os.write(fd, payload) != len(payload):
            raise OSError('Incomplete queue journal append')
    finally:
        os.close(fd)


def process_identity(pid):
    """Return a local PID identity; zombies cannot be running inference."""
    folder = Path('/proc') / str(pid)
    try:
        fields = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
        cmdline = [part.decode() for part in (folder / 'cmdline').read_bytes().split(b'\0') if part]
        return dict(pid=int(pid), uid=folder.stat().st_uid, state=fields[0],
                    start_ticks=fields[19], cmdline=cmdline)
    except (FileNotFoundError, ProcessLookupError):
        return None


def live(identity):
    return identity is not None and identity['state'] not in ('Z', 'X')


def task_output(root, mode, task):
    return root / ('openqa' if mode == 'score' else mode) / task['label']


def runner_command(root, mode, task, devices, runner):
    command = [PYTHON, str(runner), '--root', str(root), '--label', task['label'],
               '--mode', 'openqa' if mode == 'score' else mode, '--model', task['model'],
               '--adapter', task['adapter'], '--devices', devices]
    if mode == 'score':
        command.append('--score-only')
    return command


def validate_adoption(root, mode, task, entry, worker, devices):
    """Only adopt the exact local runner of the trusted queue-state reservation."""
    if entry['worker'] != worker or entry['devices'] != devices:
        raise RuntimeError('Adoption worker/devices differ from the running reservation')
    previous_pid = int(entry['pid'])
    old_queue = process_identity(previous_pid)
    if live(old_queue) and any(Path(arg).name == 'queue_worldsense_training_matched_eval.py'
                               for arg in old_queue['cmdline']):
        raise RuntimeError(f'Old queue {previous_pid} is still live; stop only that queue before adoption')
    output = task_output(root, mode, task)
    runner = output / f'runner_{mode}_{worker}_{previous_pid}.py'
    # The shared NFS maps file ownership to 65534 even when the owning host
    # process runs as UID 1000. Verify the live /proc UID and exact command below.
    if not runner.is_file():
        raise RuntimeError(f'Missing immutable runner provenance: {runner}')
    runner_pid = int((output / f'{mode}.runner.pid').read_text().strip())
    identity = process_identity(runner_pid)
    expected = runner_command(root, mode, task, devices, runner)
    if live(identity):
        if identity['uid'] != os.getuid() or identity['cmdline'][1:] != expected[1:]:
            raise RuntimeError(f'Refusing unrelated runner PID {runner_pid}: command/UID mismatch')
    else:
        # A runner can finish between stopping its queue and launching the replacement.
        # Reuse only complete output; never launch over possibly orphaned shard work.
        for path in (output / 'shards').glob('card_*/pid'):
            child = process_identity(int(path.read_text().strip()))
            if live(child):
                raise RuntimeError(f'Runner already exited but shard PID {child["pid"]} is live; refusing overlap')
        merged = output / 'results.jsonl'
        if not merged.is_file() or sum(bool(line.strip()) for line in merged.open()) != 518:
            raise RuntimeError('Exited runner has no complete merged result; manual recovery required')
    return dict(previous_queue_pid=previous_pid, runner_pid=runner_pid,
                runner_start_ticks=identity['start_ticks'] if identity else None,
                runner_cmdline=expected, runner_live_verified=live(identity))


def wait_adopted(root, key, adoption):
    pid = adoption['runner_pid']
    while adoption['runner_live_verified']:
        identity = process_identity(pid)
        if not live(identity) or identity['start_ticks'] != adoption['runner_start_ticks']:
            break
        if identity['uid'] != os.getuid() or identity['cmdline'][1:] != adoption['runner_cmdline'][1:]:
            raise RuntimeError(f'Adopted runner PID {pid} changed command unexpectedly')
        time.sleep(2)
    journal(root, 'adopted_runner_exited', key=key, runner_pid=pid)


def claim_owned(state, key, claim_id):
    if state.get(key, {}).get('claim_id') != claim_id:
        raise RuntimeError(f'Queue reservation ownership changed: {key}')


def pending_score_claim(state, tasks):
    for task in tasks:
        key = 'score/' + task['label']
        if state.get('openqa/' + task['label'], {}).get('status') in ('generated', 'complete') and key not in state:
            return ('score', task, key)
    return None


def execute(a, mode, task, key, claim_id, lock, statepath):
    root = a.root
    output = task_output(root, mode, task)
    output.mkdir(parents=True, exist_ok=True)
    runner = output / f'runner_{mode}_{a.worker}_{os.getpid()}.py'
    shutil.copyfile(REPO / 'training_code/scripts/run_worldsense_training_matched_eval.py', runner)
    command = runner_command(root, mode, task, a.devices, runner)
    print(now(), 'START', key, flush=True)
    with (root / 'logs' / f'{mode}_{task["label"]}.log').open('a') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=log)
        (output / f'{mode}.runner.pid').write_text(str(proc.pid))
        with lock:
            state = read_state(statepath)
            claim_owned(state, key, claim_id)
            state[key].update(runner_pid=proc.pid, runner_path=str(runner), runner_started_at=now())
            save(statepath, state)
        journal(root, 'runner_started', key=key, claim_id=claim_id, runner_pid=proc.pid, worker=a.worker)
        rc = proc.wait()
    with lock:
        state = read_state(statepath)
        claim_owned(state, key, claim_id)
        status = 'generated' if mode == 'openqa' else 'complete'
        state[key].update(status=status if rc == 0 else 'failed', exit_code=rc, finished_at=now())
        if mode == 'score' and rc == 0:
            state['openqa/' + task['label']].update(status='complete', scored_at=now())
        save(statepath, state)
    journal(root, 'runner_finished', key=key, claim_id=claim_id, worker=a.worker, exit_code=rc)
    print(now(), 'END', key, rc, flush=True)
    return rc


def adopt_existing(a, tasks, lock, statepath):
    with lock:
        state = read_state(statepath)
        running = [(key, dict(entry)) for key, entry in state.items()
                   if entry.get('status') == 'running' and entry.get('worker') == a.worker]
    if not running:
        return 0
    if not a.adopt_running:
        raise RuntimeError(f'{a.worker} already owns running work; use --adopt-running after stopping its old queue')
    if len(running) != 1:
        raise RuntimeError(f'Multiple running reservations for {a.worker}; manual audit required')
    key, original = running[0]
    mode, label = key.split('/', 1)
    if (a.role == 'judge' and mode != 'score') or (a.role == 'generate' and mode == 'score'):
        raise RuntimeError('Adopted task conflicts with the requested queue role')
    task = next(task for task in tasks if task['label'] == label)
    adoption = validate_adoption(a.root, mode, task, original, a.worker, a.devices)
    claim_id = uuid.uuid4().hex
    with lock:
        state = read_state(statepath)
        if state.get(key) != original:
            raise RuntimeError(f'Running reservation changed during adoption validation: {key}')
        state[key].update(pid=os.getpid(), claim_id=claim_id, adopted_at=now(), adoption=adoption,
                          hostname=socket.gethostname())
        save(statepath, state)
    journal(a.root, 'adoption_verified', key=key, worker=a.worker, claim_id=claim_id, **adoption)
    print(now(), 'ADOPT_WAIT', key, adoption['runner_pid'], flush=True)
    wait_adopted(a.root, key, adoption)
    # Normal runner verifies/reuses complete merged/shard results and reruns scoring.
    return execute(a, mode, task, key, claim_id, lock, statepath)


def run_queue(a):
    root = a.root
    logroot = root / 'logs'
    tasks = json.loads((root / 'tasks.json').read_text())
    statepath = root / 'queue_state.json'
    lock = DirectoryLock(root / 'queue.lock.d', timeout=a.lock_timeout)
    rc = adopt_existing(a, tasks, lock, statepath)
    if rc:
        (logroot / f'{a.worker}.exit').write_text(str(rc) + '\n')
        return
    first_mode_pending = a.first_mode
    while True:
        with lock:
            state = read_state(statepath)
            claim = None
            if a.prioritize_scoring and a.role != 'generate' and (root / 'SCORING_READY').exists():
                claim = pending_score_claim(state, tasks)
            if claim is None and a.role != 'judge':
                modes = [first_mode_pending] + [m for m in ['mcq', 'openqa'] if m != first_mode_pending] if first_mode_pending else ['mcq', 'openqa']
                for mode in modes:
                    for task in tasks:
                        key = mode + '/' + task['label']
                        if key not in state:
                            claim = (mode, task, key)
                            break
                    if claim:
                        break
            if claim is None and a.role != 'generate' and (root / 'SCORING_READY').exists():
                claim = pending_score_claim(state, tasks)
            if claim:
                mode, task, key = claim
                claim_id = uuid.uuid4().hex
                state[key] = dict(status='running', worker=a.worker, pid=os.getpid(), devices=a.devices,
                                  hostname=socket.gethostname(), started_at=now(), attempt=1,
                                  claim_id=claim_id, claimed_mode=mode)
                save(statepath, state)
                if mode != 'score':
                    first_mode_pending = None
            complete = all(state.get(mode + '/' + task['label'], {}).get('status') == 'complete'
                           for task in tasks for mode in ['mcq', 'openqa'])
            # No state write on idle polls: another host's new claim cannot be clobbered.
        if complete:
            (logroot / f'{a.worker}.exit').write_text('0\n')
            return
        if not claim:
            time.sleep(15)
            continue
        mode, task, key = claim
        rc = execute(a, mode, task, key, claim_id, lock, statepath)
        if rc:
            (logroot / f'{a.worker}.exit').write_text(str(rc) + '\n')
            return


def main(a):
    a.root = a.root.resolve()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', a.worker):
        raise ValueError('Invalid worker identifier')
    if not re.fullmatch(r'[0-7](,[0-7])*', a.devices) or len(set(a.devices.split(','))) != len(a.devices.split(',')):
        raise ValueError('Devices must be distinct card IDs 0..7')
    logroot = a.root / 'logs'
    logroot.mkdir(parents=True, exist_ok=True)
    with DirectoryLock(logroot / f'{a.worker}.lease.lock.d', timeout=0):
        (logroot / f'{a.worker}.pid').write_text(str(os.getpid()))
        run_queue(a)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--worker', required=True)
    p.add_argument('--devices', default='0,1,2,3,4,5,6,7')
    p.add_argument('--role', choices=['both', 'generate', 'judge'], default='both')
    p.add_argument('--adopt-running', action='store_true', help='Verify and wait for this worker\'s preserved runner, then reuse its results')
    p.add_argument('--first-mode', choices=['mcq', 'openqa'], help='Prefer this mode for the first new generation claim only')
    p.add_argument('--prioritize-scoring', action='store_true', help='Grade available open answers before claiming more generation work')
    p.add_argument('--lock-timeout', type=float, default=30.0)
    main(p.parse_args())
