#!/usr/bin/env python3
"""Repair only the known uncached full-SFT open-QA process tree on worker-0."""
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'training_runs/worldsense_observation_eval_20261001'
KEY='openqa/sft_full_epoch3'

def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ')[1][0]!='Z'
    except FileNotFoundError:return False

def main():
    with (ROOT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=ROOT/'queue_state.json';state=json.loads(path.read_text())
        claim=state[KEY];assert claim['status']=='running' and claim['worker']==0
        pid=int((ROOT/'logs/worker-0.pid').read_text());assert pid==claim['pid']
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes();assert b'queue_worldsense_dual_eval.py' in cmd
        parent={}
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                values=p.joinpath('stat').read_text().split(') ')[1].split()
                parent[int(p.name)]=int(values[1])
            except (FileNotFoundError,ProcessLookupError):pass
        selected={pid}
        while True:
            bigger=selected|{p for p,pp in parent.items() if pp in selected}
            if bigger==selected:break
            selected=bigger
        os.kill(pid,signal.SIGTERM)
        for p in selected-{pid}:
            try:os.kill(p,signal.SIGTERM)
            except ProcessLookupError:pass
        deadline=time.time()+30
        while any(alive(p) for p in selected) and time.time()<deadline:time.sleep(.5)
        for p in selected:
            if alive(p):
                try:os.kill(p,signal.SIGKILL)
                except ProcessLookupError:pass
        time.sleep(1)
        assert not any(alive(p) for p in selected),'Known process tree has not stopped'
        folder=ROOT/'openqa/sft_full_epoch3';archived=ROOT/f'openqa/sft_full_epoch3.uncached.{int(time.time())}'
        try:
            folder.rename(archived)
        except PermissionError:
            # Some shared mounts refuse renaming a nonempty directory.
            shutil.copytree(folder,archived)
            for p in folder.glob('shards/card_*/results.jsonl'):
                p.rename(p.with_name('results.uncached.'+str(int(time.time()))+'.jsonl'))
        state.pop(KEY)
        tmp=ROOT/'queue_state.cache-repair.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
        exitfile=ROOT/'logs/worker-0.exit'
        if exitfile.exists():exitfile.rename(ROOT/f'logs/worker-0.exit.cache-repair.{int(time.time())}')
        with (ROOT/'repairs.jsonl').open('a') as f:
            f.write(json.dumps(dict(task=KEY,reason='Full checkpoint disabled inference KV cache',
                                    original_claim=claim,stopped_processes=sorted(selected),archive=str(archived)))+'\n')
    with (ROOT/'logs/worker-0.log').open('a') as log:
        proc=subprocess.Popen(['python3',str(REPO/'training_code/scripts/queue_worldsense_dual_eval.py'),
                               '--root',str(ROOT),'--worker','0'],stdin=subprocess.DEVNULL,
                              stdout=log,stderr=log,start_new_session=True)
    print(json.dumps(dict(worker=0,new_queue_pid=proc.pid,archive=str(archived))))

if __name__=='__main__':main()
