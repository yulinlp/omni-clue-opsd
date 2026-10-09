#!/usr/bin/env python3
"""Poll actual node queue PIDs and per-card JSONL progress every minute."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import re
import time

HOSTS={0:'172.16.2.181',1:'172.16.12.237',2:'172.16.12.70'}

def snapshot(root):
    state=json.loads((root/'queue_state.json').read_text())
    result={'time':datetime.now().astimezone().isoformat(),'jobs':[], 'workers':{},'issues':[]}
    for w,host in HOSTS.items():
        p=root/f'logs/worker-{w}.pid'
        if not p.exists():
            result['issues'].append(f'worker-{w}: missing queue PID')
            continue
        pid=int(p.read_text())
        proc=subprocess.run(['ssh','-F','/dev/null','-p','2222','-o','BatchMode=yes','-o','ConnectTimeout=8',
                             'ma-user@'+host, f'kill -0 {pid}'],capture_output=True,text=True,timeout=20)
        exitfile=root/f'logs/worker-{w}.exit'
        exitcode=int(exitfile.read_text()) if exitfile.exists() else None
        result['workers'][w]={'pid':pid,'probe_returncode':proc.returncode,'exit_code':exitcode}
        if proc.returncode==255:
            result['issues'].append(f'worker-{w}: SSH observation failed, re-poll before concluding stopped')
        elif proc.returncode!=0 and exitcode!=0:
            result['issues'].append(f'worker-{w}: queue process exited with {exitcode}')
    for key,item in state.items():
        mode,label=key.split('/')
        folder=root/mode/label
        counts=[]
        latest=0
        for p in sorted(folder.glob('shards/card_*/results.jsonl')):
            count=0
            for line in p.open():
                try:
                    if line.strip():json.loads(line);count+=1
                except json.JSONDecodeError:
                    # One in-progress final write may not yet be complete.
                    pass
            counts.append(count)
            latest=max(latest,p.stat().st_mtime)
        job=dict(task=key,status=item['status'],worker=item['worker'],predictions=sum(counts),
                 shard_counts=counts,seconds_since_prediction=int(time.time()-latest) if latest else None)
        judge_counts=[]
        for p in folder.glob('judge/card_*/results.jsonl'):
            judge_counts.append(sum(bool(l.strip()) for l in p.open()))
        job['judge_predictions']=sum(judge_counts)
        if item['status']=='running':
            # A single inference shard can fail while the other seven still
            # make progress. Inspect current per-card errors immediately.
            fatal=[]
            started=datetime.fromisoformat(item['started_at']).timestamp()
            for p in list(folder.glob('shards/card_*/infer.log'))+list(folder.glob('judge/card_*/infer.log')):
                if p.stat().st_mtime < started:continue
                with p.open('rb') as f:
                    f.seek(max(0,p.stat().st_size-32768))
                    tail=f.read().decode(errors='replace')
                if re.search(r'Traceback \(most recent call last\)|RuntimeError:|OutOfMemoryError|ChildFailedError',tail):
                    fatal.append(str(p.relative_to(root)))
            job['fatal_current_log_paths']=fatal
            if fatal:result['issues'].append(key+': current shard error: '+', '.join(fatal))
        if item['status']=='failed':result['issues'].append(key+': failed, inspect logs before repairing')
        if item['status']=='running' and latest and time.time()-latest>1800 and not judge_counts:
            result['issues'].append(key+': no new predictions for 30 minutes; inspect actual process')
        result['jobs'].append(job)
    tasks=json.loads((root/'tasks.json').read_text())
    result['complete']=len(state)==2*len(tasks) and all(x['status']=='complete' for x in state.values())
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--once',action='store_true');a=p.parse_args()
    if not a.once:(a.root/'logs/watch.pid').write_text(str(os.getpid()))
    previous=None
    while True:
        try:s=snapshot(a.root)
        except Exception as e:s={'time':datetime.now().astimezone().isoformat(),'issues':['monitor observation error: '+str(e)]}
        if a.once:
            print(json.dumps(s));return
        temp=a.root/'health_status.tmp';temp.write_text(json.dumps(s,indent=2)+'\n');temp.replace(a.root/'health_status.json')
        if s['issues'] and s['issues']!=previous:
            with (a.root/'alerts.jsonl').open('a') as f:f.write(json.dumps(s)+'\n')
        previous=s['issues']
        with (a.root/'monitor_history.jsonl').open('a') as f:f.write(json.dumps(s)+'\n')
        if s.get('complete'):return
        time.sleep(60)

if __name__=='__main__':main()
