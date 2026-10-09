#!/usr/bin/env python3
"""Detach one pool component on an actual node SSH session."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

REPO = Path('/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd')
PYTHON = '/home/ma-user/anaconda3/envs/PyTorch-2.9.0/bin/python'

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--component', choices=('controller', 'worker', 'monitor'), required=True)
    p.add_argument('--worker')
    a = p.parse_args()
    a.root = a.root.resolve()
    if a.component == 'worker' and a.worker not in ('npu24-worker-0', 'npu24-worker-1', 'npu24-worker-2'):
        raise ValueError('Expected one of the three designated evaluation workers')
    scripts = {'controller': 'controller_worldsense_card_pool.py', 'worker': 'worker_worldsense_card_pool.py',
               'monitor': 'watch_worldsense_card_pool.py'}
    source = REPO / 'training_code/scripts' / scripts[a.component]
    if not source.is_file():
        raise FileNotFoundError(source)
    name = 'card_pool_' + (a.worker if a.component == 'worker' else a.component)
    logs = a.root / 'logs'
    logs.mkdir(exist_ok=True)
    pidpath = logs / (name + '.pid')
    if pidpath.exists():
        old = Path('/proc') / pidpath.read_text().strip()
        if old.exists() and (old / 'cmdline').read_bytes():
            raise RuntimeError('Recorded component PID still exists; inspect before relaunching')
    command = [PYTHON, str(source), '--root', str(a.root)]
    if a.component == 'worker':
        command += ['--worker', a.worker, '--devices', '0,1,2,3,4,5,6,7']
    elif a.component == 'monitor':
        command += ['--interval', '60']
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONPATH=':'.join([
        '/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_npu24_deps',
        str(REPO / 'training_code/src'), str(REPO / 'training_code/scripts'),
        '/opt/huawei/dataset/hyl_ulan/ylhu/third_party/ms-swift']))
    with (logs / (name + '.log')).open('a') as log:
        child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, start_new_session=True, cwd=str(REPO))
    pidpath.write_text(str(child.pid) + '\n')
    record = dict(pid=child.pid, hostname=socket.gethostname(), component=a.component, worker=a.worker,
                  command=command, launched_at=datetime.now().astimezone().isoformat(),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  existing_inference_left_alive=True)
    (logs / (name + '.launch.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))
