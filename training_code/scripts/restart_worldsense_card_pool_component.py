#!/usr/bin/env python3
"""Replace a verified idle CPU coordinator/monitor; never signal model workers."""
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


def command(pid):
    try:
        return [part.decode() for part in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if part]
    except FileNotFoundError:
        return []


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--component', choices=('controller', 'monitor'), required=True)
    a = p.parse_args()
    a.root = a.root.resolve()
    pid = int((a.root / 'logs' / f'card_pool_{a.component}.pid').read_text())
    script = REPO / 'training_code/scripts' / ('controller_worldsense_card_pool.py' if a.component == 'controller'
                                               else 'watch_worldsense_card_pool.py')
    expected = command(pid)
    if expected:
        if (str(script) not in expected or '--root' not in expected
                or expected[expected.index('--root') + 1] != str(a.root)
                or Path(f'/proc/{pid}').stat().st_uid != os.getuid()):
            raise RuntimeError('Component PID identity does not match this evaluation; no signal sent')
        # Finalizers may have a CPU scoring child. Wait for it to finish so the
        # replacement cannot concurrently overwrite the same score files.
        for _ in range(300):
            children = Path(f'/proc/{pid}/task/{pid}/children')
            if not children.exists() or not children.read_text().strip():
                break
            time.sleep(.1)
        else:
            raise RuntimeError('CPU finalizer has active children; restart deferred')
        os.kill(pid, signal.SIGTERM)
        for _ in range(100):
            if not command(pid):
                break
            time.sleep(.1)
        else:
            raise RuntimeError('CPU component did not exit')
    locks = [a.root / 'card_pool.lock.d', a.root / 'card_pool_controller.lease.lock.d'] if a.component == 'controller' else [a.root / 'logs/card_pool_monitor.lock.d']
    released = []
    for folder in locks:
        if not folder.exists():
            continue
        owner = json.loads((folder / 'owner.json').read_text())
        if owner['pid'] == pid and owner['hostname'] == socket.gethostname() and not command(pid):
            (folder / 'owner.json').unlink()
            folder.rmdir()
            released.append(str(folder))
        elif folder.name != 'card_pool.lock.d':
            raise RuntimeError('Component lease belongs to another live owner; no cleanup permitted')
    record = dict(component=a.component, stopped_pid=pid, children_signalled=False,
                  released_verified_stale_locks=released, at=datetime.now().astimezone().isoformat())
    (a.root / 'logs' / f'card_pool_{a.component}.restart.json').write_text(json.dumps(record, indent=2) + '\n')
    subprocess.run([PYTHON, str(REPO / 'training_code/scripts/launch_worldsense_card_pool.py'),
                    '--root', str(a.root), '--component', a.component], check=True)
    print(json.dumps(record))
