#!/usr/bin/env python3
"""Replace this run's queue coordinator, leaving its active inference runner alive.

The original queue did not install a signal cleanup handler. Only its exact PID
is terminated, never its process group or children. The new queue's adoption
checks then wait for the existing runner and reuse its predictions.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

REPO = Path('/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd')
PYTHON = '/home/ma-user/anaconda3/envs/PyTorch-2.9.0/bin/python'


def command_of(pid):
    try: return Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')[:-1]
    except FileNotFoundError: return []


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--worker', required=True)
    p.add_argument('--stop-only', action='store_true')
    p.add_argument('--prioritize-scoring', action='store_true')
    a = p.parse_args()
    a.root = a.root.resolve()
    logs = a.root/'logs'
    journal = logs/f'{a.worker}.migration.json'
    pidpath = logs/f'{a.worker}.pid'
    old_pid = int(pidpath.read_text())
    old_command = command_of(old_pid)
    expected_script = str(REPO/'training_code/scripts/queue_worldsense_training_matched_eval.py')
    if old_command:
        if (expected_script not in old_command or '--root' not in old_command or
                old_command[old_command.index('--root')+1] != str(a.root) or
                '--worker' not in old_command or
                old_command[old_command.index('--worker')+1] != a.worker or
                Path(f'/proc/{old_pid}').stat().st_uid != os.getuid()):
            raise RuntimeError('Old queue ownership/command cannot be verified; no signal sent')
        # Signal exactly one coordinator PID. Active runner/model processes persist.
        os.kill(old_pid, signal.SIGTERM)
        for _ in range(100):
            if command_of(old_pid) != old_command:
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Old queue has not exited; replacement refused')
        record = dict(worker=a.worker, previous_queue_pid=old_pid,
                      previous_command=old_command, children_signalled=False,
                      previous_queue_stopped_at=datetime.now().astimezone().isoformat())
        journal.write_text(json.dumps(record, indent=2)+'\n')
    elif journal.exists():
        record = json.loads(journal.read_text())
    else:
        raise RuntimeError('Queue already absent without migration journal; inspect manually')
    if a.stop_only:
        print(json.dumps(record)); raise SystemExit(0)
    source = Path(expected_script).read_text()
    if '--adopt-running' not in source:
        raise RuntimeError('Replacement adoption support is not ready')
    # A terminated coordinator cannot run DirectoryLock's finally clause.
    # Remove only its independently verified stale lock on this same host.
    for lockdir in [logs/f'{a.worker}.lease.lock.d', a.root/'queue.lock.d']:
        if not lockdir.exists():
            continue
        owner_path = lockdir/'owner.json'
        owner = json.loads(owner_path.read_text())
        if owner['pid'] == old_pid and owner['hostname'] == socket.gethostname():
            if command_of(old_pid):
                raise RuntimeError('Old queue PID is live; stale-lock cleanup refused')
            owner_path.unlink(); lockdir.rmdir()
            record.setdefault('released_verified_stale_locks', []).append(dict(path=str(lockdir),owner=owner))
        elif lockdir == logs/f'{a.worker}.lease.lock.d':
            raise RuntimeError('Node lease belongs to another coordinator; replacement refused')
    env = dict(os.environ, PYTHONPATH=':'.join([
        '/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_npu24_deps',
        str(REPO/'training_code/src'), '/opt/huawei/dataset/hyl_ulan/ylhu/third_party/ms-swift']),
        PYTHONUNBUFFERED='1')
    command = [PYTHON, expected_script, '--root', str(a.root), '--worker', a.worker,
               '--devices', '0,1,2,3,4,5,6,7', '--adopt-running']
    if a.prioritize_scoring:
        command.append('--prioritize-scoring')
    with (logs/f'{a.worker}.log').open('a') as log:
        child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, start_new_session=True, cwd=str(REPO))
    record.update(replacement_queue_pid=child.pid, replacement_command=command,
                  replaced_at=datetime.now().astimezone().isoformat())
    journal.write_text(json.dumps(record, indent=2)+'\n')
    (logs/f'{a.worker}.launch.json').write_text(json.dumps(dict(worker=a.worker,pid=child.pid,
        command=command, launched_at=record['replaced_at'], adopted_inference=True), indent=2)+'\n')
    print(json.dumps(record))
