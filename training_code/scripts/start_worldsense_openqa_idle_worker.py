#!/usr/bin/env python3
"""Switch an idle queue worker to open QA without interrupting any inference."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--worker',type=int,required=True);a=p.parse_args()
    root=a.root
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads((root/'queue_state.json').read_text())
        assert not any(x['status']=='running' and x['worker']==a.worker for x in state.values()), 'Worker still has active inference'
        tasks=json.loads((root/'tasks.json').read_text())
        assert all('mcq/'+t['label'] in state for t in tasks)
        pid=int((root/f'logs/worker-{a.worker}.pid').read_text())
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        assert any(b'queue_worldsense_dual_eval.py' in x for x in cmd), 'PID is not the known queue'
        assert cmd[cmd.index(b'--worker')+1]==str(a.worker).encode()
        children=Path(f'/proc/{pid}/task/{pid}/children').read_text().strip()
        assert not children, 'Queue still has a child process'
        os.kill(pid,signal.SIGTERM)
        with (root/'scheduling_events.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=time.time(),worker=a.worker,old_queue_pid=pid,action='idle-only switch to open QA'))+'\n')
    repo=Path(__file__).resolve().parents[2]
    with (root/f'logs/worker-{a.worker}.log').open('a') as log:
        proc=subprocess.Popen(['python3',str(repo/'training_code/scripts/queue_worldsense_dual_eval.py'),
                               '--root',str(root),'--worker',str(a.worker),'--phase','openqa'],
                              stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    print(json.dumps(dict(worker=a.worker,new_queue_pid=proc.pid,phase='openqa')))

if __name__=='__main__':main()
