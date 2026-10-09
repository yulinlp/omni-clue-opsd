#!/usr/bin/env python3
"""120-card open QA: one writer per node, immutable controller inboxes."""
import argparse
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback

REPO=Path('/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd')
sys.path.insert(0,str(REPO/'training_code/src'))
sys.path.insert(0,str(REPO/'training_code/scripts'))
from run_worldsense_training_matched_eval import PYENV, env_for
from worker_worldsense_card_pool import atomic_save
from controller_worldsense_card_pool import generation_command
from score_worldsense_openqa import join
from score_worldsense_openqa_v2 import calibration_cases, judge_source, STRONG_JUDGE

WORKERS=[f'npu24-worker-{i}' for i in range(3)]+[f'npu96-worker-{i}' for i in range(12)]
PARTS=16

def now():return datetime.now().astimezone().isoformat()
def read(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jsonl(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    body=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows)
    if p.exists():
        if p.read_text()!=body:raise ValueError('Immutable data differs: '+str(p))
        return
    temp=p.with_suffix('.tmp.'+str(os.getpid()));temp.write_text(body);temp.replace(p)

class LocalLock:
    """All queue writes happen on its owner host; flock lives on local /tmp."""
    def __init__(self,path,timeout=30,**unused):
        name=hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()
        self.path=Path('/tmp')/('openqa_local_'+name+'.lock');self.timeout=timeout
    def __enter__(self):
        self.f=self.path.open('a+');deadline=time.monotonic()+self.timeout
        while True:
            try:fcntl.flock(self.f,fcntl.LOCK_EX|fcntl.LOCK_NB);return self
            except BlockingIOError:
                if time.monotonic()>=deadline:raise TimeoutError('Local owner still alive')
                time.sleep(.05)
    def __exit__(self,*exc):fcntl.flock(self.f,fcntl.LOCK_UN);self.f.close()

def judge_command(source,target):
    return [str(PYENV/'swift'),'infer','--model',STRONG_JUDGE,'--model_type','qwen3_5',
            '--val_dataset',str(source),'--result_path',str(target),'--infer_backend','transformers',
            '--max_batch_size','1','--write_batch_size','1','--max_new_tokens','256','--temperature','0',
            '--stream','false','--torch_dtype','bfloat16','--attn_impl','sdpa','--max_length','4096',
            '--dataset_num_proc','1','--val_dataset_shuffle','false','--seed','20261003','--enable_thinking','false']

def worker(root,name):
    import worker_worldsense_card_pool as pool
    wr=root/'worker_pools'/name
    pool.DirectoryLock=LocalLock
    original_probe=pool.probe_npus
    processed=set()
    def ingest():
        for p in sorted((wr/'inbox').glob('*.json')):
            if p.name in processed:continue
            message=load(p);state=pool.read_state(wr/'card_pool_state.json')
            if message['action']=='add':
                key=message['key'];unit=message['unit']
                if key not in state['units']:state['units'][key]=unit
                elif state['units'][key]['command']!=unit['command']:raise ValueError('Unit command changed')
            elif message['action']=='stop':state['stop_workers']=True
            elif message['action']=='retry':
                key=message['key'];unit=state['units'][key]
                if unit['status']=='failed' and unit.get('attempt',0)<3:
                    folder=Path(unit['folder']);archive=folder/('failed_attempt_'+str(unit['attempt']));archive.mkdir(exist_ok=True)
                    for basename in ['results.jsonl','infer.log','assigned_device.json','pid']:
                        file=folder/basename
                        if file.exists():shutil.move(str(file),str(archive/basename))
                    unit.update(status='pending',retry_at=now())
            atomic_save(wr/'card_pool_state.json',state)
            processed.add(p.name)
        return
    def probe(devices):
        ingest();return original_probe(devices)
    pool.probe_npus=probe
    sys.argv=['worker','--root',str(wr),'--worker',name,'--devices','0,1,2,3,4,5,6,7','--poll-seconds','5']
    pool.main()

def score_command(root,bench,model,prepare=False,judge_paths=(),cal_paths=()):
    out=root/'eval'/bench/model
    command=[str(PYENV/'python'),str(root/'code/score_external_openqa.py'),
             '--results',str(out/'results.jsonl'),'--dataset',str(root/'data'/bench/'inputs.jsonl'),
             '--labels',str(root/'data'/bench/'labels.jsonl'),'--output',str(out/'summary.json'),
             '--arm',model,'--judge-model',STRONG_JUDGE,'--judge-model-type','qwen3_5']
    if prepare:command+=['--prepare-only']
    for p in judge_paths:command+=['--judge-results',str(p)]
    for p in cal_paths:command+=['--calibration-results',str(p)]
    return command

def init(root):
    if (root/'controller_state.json').exists():return
    mapping=load(root/'inference_cache_views.json')
    tasks=load(root/'tasks.json');state={'created_at':now(),'units':{},'tasks':{},'calibration':{}}
    for name in WORKERS:
        wr=root/'worker_pools'/name;(wr/'inbox').mkdir(parents=True,exist_ok=True)
        if not (wr/'card_pool_state.json').exists():atomic_save(wr/'card_pool_state.json',{'units':{}})
    def add(parts,rows,kind,key,priority,**more):
        ids=[]
        for i in range(min(parts,len(rows))):
            uid=f'{kind}/{key}/{i}';path=root/'staging'/uid/'input.jsonl';jsonl(path,rows[i::parts])
            state['units'][uid]=dict(kind=kind,task_key=key,partition=i,source=str(path),
                                     expected_rows=len(rows[i::parts]),priority=priority+i,status='unassigned',**more)
            ids.append(uid)
        return ids
    for bench in ['omnivideobench','dailyomni']:
        source=read(root/'data'/bench/'inputs.jsonl');labels=read(root/'data'/bench/'labels.jsonl')
        m=load(root/'data'/bench/'manifest.json')
        assert sha(root/'data'/bench/'inputs.jsonl')==m['input_sha256']
        assert sha(root/'data'/bench/'labels.jsonl')==m['labels_sha256']
        cal=calibration_cases(labels)
        cal_inputs=[judge_source(c['case_id'],c,c['candidate'],j) for c in cal for j in (0,1)]
        state['calibration'][bench]=add(8,cal_inputs,'judge','calibration/'+bench,-1000)
        for order,t in enumerate(tasks):
            key=bench+'/'+t['label'];model=mapping.get(t['model'],t['model'])
            ids=add(PARTS,source,'generation',key,order*100,model=model,adapter=t['adapter'])
            state['tasks'][key]={'generation_units':ids,'stage':'generating','benchmark':bench,'model':t['label']}
            folder=root/'eval'/bench/t['label'];folder.mkdir(parents=True,exist_ok=True)
            atomic_save(folder/'generation_protocol.json',dict(t,actual_model=model,partitions=PARTS,
                temperature=1.,top_p=1.,effective_top_k=50,max_new_tokens=512,seed=20260904,
                max_length=32768,audio_in_video=True,input_sha256=m['input_sha256']))
    atomic_save(root/'controller_state.json',state)

def inbox(root,name,filename,message):
    p=root/'worker_pools'/name/'inbox'/filename
    if p.exists():
        if load(p)!=message:raise ValueError('Immutable inbox changed: '+str(p))
    else:atomic_save(p,message)

def publish_unit(root,uid,u):
    name=u['worker'];folder=root/'worker_pools'/name/'units'/uid
    folder.mkdir(parents=True,exist_ok=True)
    source=folder/'input.jsonl';jsonl(source,read(Path(u['source'])))
    result=folder/'results.jsonl'
    command=(generation_command(u['model'],u['adapter'],source,result,'openqa') if u['kind']=='generation'
             else judge_command(source,result))
    record=dict(kind=u['kind'],task_key=u['task_key'],partition=u['partition'],expected_rows=u['expected_rows'],
                folder=str(folder),result_path=str(result),command=command,priority=u['priority'],
                env_overrides={} if u['kind']=='generation' else {'USE_AUDIO_IN_VIDEO':'0'},
                status='pending',created_at=u['assigned_at'])
    inbox(root,name,hashlib.sha256(uid.encode()).hexdigest()+'.json',dict(action='add',key=uid,unit=record))

def merge(root,ids,state,target):
    combined=[];sources=[]
    for uid in ids:
        u=state['units'][uid];path=Path(u['result_path']);src=read(Path(u['source']))
        rows=read(path);join(rows,src)
        combined+=rows;sources+=src
    join(combined,sources);jsonl(target,combined)
    return [Path(state['units'][uid]['result_path']) for uid in ids]

def aggregate(root,state):
    table=[]
    for key,t in state['tasks'].items():
        if t['stage']!='complete':continue
        s=load(root/'eval'/key/'summary.json')
        table.append(dict(benchmark=t['benchmark'],model=t['model'],total=s['total'],correct=s['correct'],
                          incorrect=s['incorrect'],uncertain=s['uncertain'],
                          accuracy_lower_bound=s['accuracy_lower_bound'],accuracy_upper_bound=s['accuracy_upper_bound'],
                          reference_insufficient=s['reference_insufficient_count'],
                          reference_sufficient_scores=s['scores_on_reference_sufficient_questions']))
    atomic_save(root/'comparison.json',dict(completed_tasks=len(table),total_tasks=26,rows=table))

def controller(root):
    with LocalLock(root/'controller.lease',timeout=0):
        init(root)
        while True:
            state=load(root/'controller_state.json');issues=[];workers={};loads={}
            for name in WORKERS:
                wr=root/'worker_pools'/name
                local=load(wr/'card_pool_state.json')['units']
                hpath=wr/'card_pool_workers'/name/'heartbeat.json'
                if hpath.exists():
                    h=load(hpath);age=time.time()-hpath.stat().st_mtime
                    workers[name]=dict(pid=h['pid'],age=age,devices=h['devices'],status=h['status'])
                    if age>180 and h['status']!='complete':issues.append(name+':stale-heartbeat')
                else:workers[name]={'status':'starting'}
                for uid,u in state['units'].items():
                    if u.get('worker')!=name:continue
                    if uid not in local:
                        publish_unit(root,uid,u);continue
                    lu=local[uid]
                    u.update(status=lu['status'],result_path=lu['result_path'],attempt=lu.get('attempt',0))
                    if lu['status']=='failed':
                        issues.append('failed:'+uid)
                        attempt=lu.get('attempt',0)
                        if attempt<3:
                            inbox(root,name,f'retry_{hashlib.sha256(uid.encode()).hexdigest()}_{attempt}.json',
                                  dict(action='retry',key=uid,attempt=attempt))
                loads[name]=sum(u.get('worker')==name and u['status'] in ['queued','pending','running'] for u in state['units'].values())
            # Prepare/scoring is CPU-only; model jobs continue independently.
            for key,t in state['tasks'].items():
                try:
                    out=root/'eval'/key
                    if t['stage']=='generating' and all(state['units'][uid]['status']=='complete' for uid in t['generation_units']):
                        merge(root,t['generation_units'],state,out/'results.jsonl')
                        with (out/'scoring.log').open('a') as log:
                            subprocess.run(score_command(root,t['benchmark'],t['model'],prepare=True),
                                           env=env_for(''),stdout=log,stderr=log,check=True,timeout=180)
                        inputs=read(out/'judge_v2_pending_inputs.jsonl');ids=[]
                        count=min(PARTS,len(inputs))
                        for i in range(count):
                            uid=f'judge/{key}/{i}';source=root/'staging'/uid/'input.jsonl';jsonl(source,inputs[i::count])
                            state['units'][uid]=dict(kind='judge',task_key=key,partition=i,source=str(source),
                                expected_rows=len(inputs[i::count]),priority=-100+i,status='unassigned')
                            ids.append(uid)
                        t.update(stage='judging',judge_units=ids)
                        atomic_save(root/'controller_state.json',state)
                    cal_ids=state['calibration'][t['benchmark']]
                    if t['stage']=='judging' and all(state['units'][uid]['status']=='complete' for uid in t['judge_units']+cal_ids):
                        jp=merge(root,t['judge_units'],state,out/'judge_merged.jsonl')
                        cp=merge(root,cal_ids,state,root/'calibration'/t['benchmark']/'results.jsonl')
                        with (out/'scoring.log').open('a') as log:
                            subprocess.run(score_command(root,t['benchmark'],t['model'],judge_paths=jp,cal_paths=cp),
                                           env=env_for(''),stdout=log,stderr=log,check=True,timeout=180)
                        s=load(out/'summary.json')
                        assert s['judge_calibration']['evaluated'] and s['scoring_pipeline_reliable_on_calibration']
                        t.update(stage='complete',completed_at=now())
                        atomic_save(root/'controller_state.json',state)
                except Exception as e:
                    issues.append(key+':'+repr(e))
                    with (root/'controller_errors.jsonl').open('a') as f:f.write(json.dumps(dict(at=now(),task=key,error=traceback.format_exc()))+'\n')
            pending=sorted(((k,u) for k,u in state['units'].items() if u['status']=='unassigned'),key=lambda x:(x[1]['priority'],x[0]))
            for uid,u in pending:
                name=min(WORKERS,key=lambda n:loads[n])
                if loads[name]>=9:break
                u.update(worker=name,status='queued',assigned_at=now());loads[name]+=1
                atomic_save(root/'controller_state.json',state)
                publish_unit(root,uid,u)
            counts=Counter(u['status'] for u in state['units'].values())
            complete=sum(t['stage']=='complete' for t in state['tasks'].values())
            rows_written=0
            for u in state['units'].values():
                if u['kind']=='generation' and u.get('result_path') and Path(u['result_path']).exists():
                    rows_written+=sum(bool(x.strip()) for x in Path(u['result_path']).open())
            health=dict(at=now(),controller_pid=os.getpid(),completed_tasks=complete,total_tasks=26,
                        generation_tasks_complete=sum(t['stage']!='generating' for t in state['tasks'].values()),
                        generation_rows=rows_written,total_generation_rows=13000,unit_statuses=dict(counts),
                        active_cards=sum(u['status']=='running' for u in state['units'].values()),
                        workers=workers,issues=issues,healthy=not issues)
            atomic_save(root/'controller_state.json',state);atomic_save(root/'health.json',health)
            aggregate(root,state)
            print(json.dumps({k:v for k,v in health.items() if k!='workers'}),flush=True)
            if complete==26:
                for name in WORKERS:inbox(root,name,'zz_stop.json',dict(action='stop'))
                atomic_save(root/'COMPLETE.json',dict(at=now(),tasks=26,original_responses_preserved=True))
                return
            time.sleep(15)

def launch(root,action,name=None):
    logs=root/'logs';logs.mkdir(exist_ok=True)
    label=name or 'controller';pidfile=logs/(label+'.pid')
    if pidfile.exists() and (Path('/proc')/pidfile.read_text().strip()/'cmdline').exists():
        raise ValueError('Recorded component still exists: '+label)
    script=root/'code/generalization_openqa_fleet.py'
    cmd=[str(PYENV/'python'),str(script),'--root',str(root),'--action',action]
    if name:cmd+=['--worker',name]
    with (logs/(label+'.log')).open('a') as log:
        child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,cwd=REPO,env=env_for(''))
    pidfile.write_text(str(child.pid)+'\n')
    atomic_save(logs/(label+'.launch.json'),dict(at=now(),pid=child.pid,hostname=socket.gethostname(),command=cmd,sha256=sha(script)))
    print(json.dumps(dict(component=label,pid=child.pid,hostname=socket.gethostname())))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--action',choices=['init','controller','worker','launch-controller','launch-worker'],required=True)
    p.add_argument('--worker',choices=WORKERS);a=p.parse_args();root=a.root.resolve()
    if a.action=='init':init(root)
    elif a.action=='controller':controller(root)
    elif a.action=='worker':worker(root,a.worker)
    elif a.action=='launch-controller':launch(root,'controller')
    else:launch(root,'worker',a.worker)

if __name__=='__main__':main()
