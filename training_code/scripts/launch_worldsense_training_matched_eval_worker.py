#!/usr/bin/env python3
"""Launch a detached worker only after verifying the selected NPUs are empty."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime

REPO = Path('/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd')
PYTHON = '/home/ma-user/anaconda3/envs/PyTorch-2.9.0/bin/python'

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--worker', required=True)
    p.add_argument('--devices', default='0,1,2,3,4,5,6,7')
    p.add_argument('--role', choices=['both', 'generate', 'judge'], default='both')
    p.add_argument('--first-mode', choices=['mcq', 'openqa'])
    a = p.parse_args()
    logroot = a.root/'logs'; logroot.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=30)
    if probe.returncode:
        raise RuntimeError('NPU status cannot be verified: '+probe.stderr)
    for device in a.devices.split(','):
        if 'No running processes found in NPU '+device+' ' not in probe.stdout:
            raise RuntimeError('Selected NPU '+device+' is not empty; launch refused.')
    pidpath = logroot/f'{a.worker}.pid'
    if pidpath.exists():
        try: os.kill(int(pidpath.read_text()), 0)
        except ProcessLookupError: pass
        else: raise RuntimeError('Worker already running: '+a.worker)
    env = dict(os.environ, PYTHONPATH=':'.join([
        '/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_npu24_deps',
        str(REPO/'training_code/src'),
        '/opt/huawei/dataset/hyl_ulan/ylhu/third_party/ms-swift']), PYTHONUNBUFFERED='1')
    command = [PYTHON, str(REPO/'training_code/scripts/queue_worldsense_training_matched_eval.py'),
               '--root', str(a.root), '--worker', a.worker, '--devices', a.devices, '--role', a.role]
    if a.first_mode:
        command += ['--first-mode', a.first_mode]
    with (logroot/f'{a.worker}.log').open('a') as log:
        child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 start_new_session=True, cwd=str(REPO))
    rec = dict(worker=a.worker, pid=child.pid, command=command,
               launched_at=datetime.now().astimezone().isoformat(), empty_cards_confirmed=True)
    (logroot/f'{a.worker}.launch.json').write_text(json.dumps(rec, indent=2)+'\n')
    (logroot/f'{a.worker}.prelaunch_npu.txt').write_text(probe.stdout)
    print(json.dumps(rec))
