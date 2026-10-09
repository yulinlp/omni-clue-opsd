#!/usr/bin/env python3
"""Inventory or stop the explicitly authorized npu96 compute workload."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time

from worker_worldsense_card_pool import probe_npus, atomic_save

JOB = 'ma-job-7b525feb-b862-4352-8403-9c3a018c05e8'
PYTHON = '/home/ma-user/anaconda3/envs/PyTorch-2.9.0/bin/python'

def processes():
    result = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            if p.stat().st_uid != os.getuid():
                continue
            argv = (p / 'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
            stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            if argv == ['']:
                continue
            result[int(p.name)] = dict(pid=int(p.name), ppid=int(stat[1]), starttime=stat[19], argv=argv)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return result

def redacted(item):
    item = dict(item)
    safe, hide = [], False
    for arg in item.pop('argv'):
        sensitive = re.search(r'(?:api.?key|auth.?token|password|secret|access.?token)', arg, re.I)
        if hide:
            safe.append('[REDACTED]'); hide = False
        elif sensitive:
            safe.append(arg.split('=', 1)[0] + '=[REDACTED]' if '=' in arg else arg)
            hide = '=' not in arg
        else:
            safe.append(arg)
    item['command'] = shlex.join(safe)[:1200]
    return item

def snapshot():
    host = socket.gethostname()
    if not re.fullmatch(re.escape(JOB) + r'-worker-(?:[0-9]|1[01])', host):
        raise RuntimeError('Refusing operation outside the designated npu96 job: ' + host)
    probe = probe_npus([str(i) for i in range(8)])
    allp = processes()
    npu_pids = {p['pid'] for card in probe['devices'].values() for p in card['processes']}
    candidates = {}
    for pid, item in allp.items():
        cmd = ' '.join(item['argv'])
        compute = pid in npu_pids or bool(re.search(r'sglang|vllm|torchrun|swift.cli|AdaPersona/(?:ops|scripts)|(?:train|infer|eval).*\.py', cmd, re.I))
        infrastructure = pid < 3000 or any(s in cmd for s in ('jupyter', 'codex', 'npu96_generalization_fleet.py', 'sshd', 'modelarts/ma-training-toolkit'))
        if compute and not infrastructure:
            candidates[pid] = item
    # Include children of workload supervisors, even when their command is only
    # a multiprocessing bootstrap. SSH/platform ancestors are never included.
    changed = True
    while changed:
        changed = False
        for pid, item in allp.items():
            if item['ppid'] in candidates and pid not in candidates and pid != os.getpid():
                candidates[pid] = item; changed = True
    return dict(hostname=host, time=datetime.now().astimezone().isoformat(), probe=probe,
                processes=[redacted(v) for v in candidates.values()]), candidates

def local(action, root):
    before, candidates = snapshot()
    record = dict(action=action, before=before)
    if action == 'stop':
        actions = []
        for sig in (signal.SIGTERM, signal.SIGKILL):
            current = processes()
            for pid, item in candidates.items():
                if pid in current and current[pid]['starttime'] == item['starttime']:
                    try:
                        os.kill(pid, sig); actions.append(dict(pid=pid, signal=sig.name))
                    except ProcessLookupError:
                        pass
            time.sleep(8 if sig == signal.SIGTERM else 3)
        record['signals'] = actions
        record['after'], _ = snapshot()
    folder = root / 'fleet'
    folder.mkdir(parents=True, exist_ok=True)
    atomic_save(folder / (action + '_' + socket.gethostname() + '.json'), record)
    view = record.get('after', before)
    pids = {p['pid'] for p in view['processes']}
    print(json.dumps(dict(hostname=view['hostname'], action=action,
                         devices={k:v['status'] for k,v in view['probe']['devices'].items()},
                         process_count=len(view['processes']),
                         workload_root_pids=[p['pid'] for p in view['processes'] if p['ppid'] not in pids]), ensure_ascii=False))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--action', choices=['snapshot', 'stop'], default='snapshot')
    p.add_argument('--fleet', action='store_true')
    a = p.parse_args()
    if not a.fleet:
        return local(a.action, a.root)
    def remote(i):
        host = f'{JOB}-worker-{i}.{JOB}'
        cmd = [PYTHON, str(Path(__file__).resolve()), '--root', str(a.root), '--action', a.action]
        r = subprocess.run(['ssh','-F','/dev/null','-p','2222','-o','BatchMode=yes','-o','ConnectTimeout=8',
                            'ma-user@' + host, shlex.join(cmd)], capture_output=True, text=True, timeout=90)
        return dict(worker=i, exit=r.returncode, output=r.stdout, error=r.stderr[-2000:])
    with ThreadPoolExecutor(max_workers=12) as pool:
        for f in as_completed([pool.submit(remote, i) for i in range(12)]):
            print(json.dumps(f.result(), ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
