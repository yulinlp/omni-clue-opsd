#!/usr/bin/env python3
"""Schedule frozen external MCQ+analysis shards across all 96 NPUs."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime
import hashlib
import io
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
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'training_code/src'))
from distributed_directory_lock import DirectoryLock
from worker_worldsense_card_pool import atomic_save, read_state, append_event
from run_worldsense_training_matched_eval import env_for, PYENV
from score_worldsense_mcq_v2 import summarize
from score_generalization_analysis_mcq import score_rows as scored_mcq_v2_rows
from aggregate_worldsense_training_matched_eval import VideoBootstrap, paired_comparison
from npu96_generalization_fleet import JOB

PRIOR = REPO/'training_runs/worldsense_training_matched_eval_20261003'

def now():
    return datetime.now().astimezone().isoformat()

def read(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def jsonl(path, rows):
    temporary=path.with_name(path.name+f'.tmp.{os.getpid()}')
    temporary.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    temporary.replace(path)

def validate_data(root, benchmark):
    folder=root/'data'/benchmark
    m=json.loads((folder/'manifest.json').read_text())
    for name,key in [('inputs.jsonl','input_sha256'),('labels.jsonl','labels_sha256')]:
        if sha(folder/name)!=m[key]:raise ValueError('Frozen data changed: '+str(folder/name))
    rows,labels=read(folder/'inputs.jsonl'),read(folder/'labels.jsonl')
    if len(rows)!=500 or len(labels)!=500 or len({r['case_id'] for r in rows})!=500:
        raise ValueError('Expected 500 unique QA: '+benchmark)
    if {r['case_id'] for r in rows}!={r['sample_id'] for r in labels}:raise ValueError('Input/label IDs differ')
    for row in rows:
        if {'answer','solution','gold_answer','teacher_prompt','teacher_videos','observation'} & row.keys():
            raise ValueError('Answer leakage: '+row['case_id'])
        if len(row['messages'])!=1 or row['messages'][0]['role']!='user':raise ValueError('Invalid model messages')
        if not row['sampling_contract']['use_audio_in_video']:raise ValueError('Audio disabled')
        if row['dynamic_student_budget']['max_checked_tokens']>32768:raise ValueError('Context budget overflow')
    return rows, labels

def initialize(root, benchmarks):
    root.mkdir(parents=True,exist_ok=True)
    models=json.loads((PRIOR/'tasks.json').read_text())
    mapping=json.loads((PRIOR/'inference_cache_views.json').read_text())
    planned=[];new_units={}
    existing=json.loads((root/'tasks.json').read_text()) if (root/'tasks.json').exists() else []
    existing_keys={t['key'] for t in existing}
    for bi,bench in enumerate(benchmarks):
        rows,_=validate_data(root,bench)
        for mi,model in enumerate(models):
            key=bench+'/'+model['label']
            if key in existing_keys:continue
            task=dict(model,key=key,benchmark=bench)
            planned.append(task)
            directory=root/'eval'/bench/model['label']
            directory.mkdir(parents=True,exist_ok=True)
            actual_model=mapping.get(model['model'],model['model'])
            atomic_save(directory/'generation_protocol.json',dict(task,model_source=model['model'],
                        actual_model=actual_model,input_sha256=sha(root/'data'/bench/'inputs.jsonl'),
                        temperature=1.,top_p=1.,requested_top_k=-1,effective_top_k=50,
                        max_new_tokens=512,max_length=32768,seed=20260904,audio_in_video=True,
                        torch_dtype='bfloat16',attn_impl='sdpa',batch_size=1))
            for i in range(8):
                folder=directory/'shards'/f'card_{i}';folder.mkdir(parents=True,exist_ok=True)
                source=folder/'input.jsonl';target=folder/'results.jsonl'
                jsonl(source,rows[i::8])
                command=[str(PYENV/'swift'),'infer','--model',actual_model,'--model_type','qwen2_5_omni',
                         '--val_dataset',str(source),'--result_path',str(target),'--infer_backend','transformers',
                         '--max_batch_size','1','--write_batch_size','1','--max_new_tokens','512',
                         '--temperature','1','--top_p','1','--top_k','-1','--repetition_penalty','1',
                         '--num_beams','1','--stream','false','--torch_dtype','bfloat16','--attn_impl','sdpa',
                         '--max_length','32768','--dataset_num_proc','1','--val_dataset_shuffle','false',
                         '--seed','20260904']
                if model['adapter']!='-':command+=['--adapters',model['adapter']]
                new_units['generation/'+key+'/'+str(i)]=dict(status='pending',kind='generation',task_key=key,
                         partition=i,folder=str(folder),result_path=str(target),expected_rows=len(rows[i::8]),
                         priority=mi*100+bi*10+i,command=command,env_overrides={},created_at=now())
    with DirectoryLock(root/'card_pool.lock.d'):
        state=read_state(root/'card_pool_state.json')
        if new_units.keys() & state['units'].keys():raise ValueError('Unit already exists')
        state['units'].update(new_units)
        state.update(controller_complete=False,stop_workers=False)
        atomic_save(root/'card_pool_state.json',state)
        atomic_save(root/'tasks.json',existing+planned)
    sources=[Path(__file__),REPO/'training_code/scripts/worker_worldsense_card_pool.py',
             REPO/'training_code/scripts/prepare_worldsense_generalization_eval.py',
             REPO/'training_code/scripts/score_worldsense_mcq_v2.py']
    atomic_save(root/'experiment_manifest.json',dict(created_at=now(),models=models,
                ready_benchmarks=sorted({t['benchmark'] for t in existing+planned}),requested_benchmarks=['omnivideobench','dailyomni','video_odyssey'],
                video_odyssey_policy='awaiting-user-selection',expected_workers=12,expected_cards=96,
                output='analysis<=120 English words plus final A-D option; 512 new tokens',
                scoring='Conservative explicit final option; no option inferred from analysis; unparsed counts wrong',
                sample_seed=20261003,generation_seed=20260904,
                sources={str(p):sha(p) for p in sources}))
    print(json.dumps(dict(added_tasks=len(planned),added_units=len(new_units),total_tasks=len(existing+planned))))

def score_task(root, task, units):
    rows,labels=validate_data(root,task['benchmark'])
    if len(units)!=8 or {u['partition'] for u in units}!=set(range(8)):raise ValueError('Shard set differs')
    merged=[]
    for u in sorted(units,key=lambda u:u['partition']):
        shard=rows[u['partition']::8]
        shard_ids={r['case_id'] for r in shard}
        predictions=read(u['result_path'])
        scored_mcq_v2_rows(predictions,[r for r in labels if r['sample_id'] in shard_ids],shard)
        merged.extend(predictions)
    scored=scored_mcq_v2_rows(merged,labels,rows)
    summary=summarize(scored)
    summary.update(benchmark=task['benchmark'],model=task['label'],epoch=task['epoch'],completed_at=now(),
                   parser_sha256=sha(REPO/'training_code/scripts/score_generalization_analysis_mcq.py'),
                   underlying_mcq_parser_sha256=sha(REPO/'training_code/scripts/score_worldsense_mcq_v2.py'))
    strict=0;unclosed=0;over=0;repetitive=0
    for r in scored:
        response=r['response']
        analysis=re.search(r'<analysis>(.*?)</analysis>',response,re.S|re.I)
        answer=re.search(r'<answer>\s*([ABCD])\s*</answer>',response,re.I)
        strict+=bool(analysis and answer)
        unclosed+=not bool(analysis)
        words=(analysis.group(1) if analysis else response.split('<answer>',1)[0]).split()
        over+=len(words)>120
        grams=[tuple(words[i:i+20]) for i in range(max(0,len(words)-19))]
        repetitive+=bool(len(words)>=60 and grams and 1-len(set(grams))/len(grams)>=.4)
    summary.update(analysis_plus_answer_format_count=strict,analysis_plus_answer_format_rate=strict/500,
                   unclosed_analysis_count=unclosed,analysis_over_120_count=over,repetitive_analysis_count=repetitive,
                   format_definition='Closed analysis block and closed answer block containing one A-D letter',
                   strict_format_compliant=strict,strict_format_rate=strict/500,
                   analysis_factuality_evaluated=False)
    folder=root/'eval'/task['benchmark']/task['label']
    jsonl(folder/'results.jsonl',merged);jsonl(folder/'scored.jsonl',scored)
    atomic_save(folder/'summary.json',summary)

def partition_workers(root):
    """Create one single-writer queue per host before launching any worker.

    NFS directory-owner reads were inconsistent under cross-host contention.
    No host writes another host's queue during this run.
    """
    if list((root/'worker_pools').glob('*/card_pool_workers/*/heartbeat.json')):
        raise RuntimeError('Cannot repartition queues after workers have started')
    state=read_state(root/'card_pool_state.json')
    pools=[dict(units={}) for _ in range(12)]
    loads=[0.]*12
    units=[]
    for key,u in state['units'].items():
        source=Path(u['folder'])/'input.jsonl'
        rows=read(source)
        # Greedy balancing uses media duration plus a common response cost;
        # labels and model correctness never influence assignment.
        cost=sum(r['dynamic_student_budget']['audio_seconds']+60 for r in rows)
        units.append((cost,key,u,rows))
    for cost,key,u,rows in sorted(units,reverse=True,key=lambda x:(x[0],x[1])):
        worker=min(range(12),key=lambda i:(loads[i],len(pools[i]['units']),i))
        worker_root=root/'worker_pools'/f'npu96-worker-{worker}'
        folder=worker_root/'units'/key
        folder.mkdir(parents=True,exist_ok=True)
        source=folder/'input.jsonl';target=folder/'results.jsonl'
        if target.exists():raise ValueError('Partition already has inference output: '+str(target))
        jsonl(source,rows)
        command=list(u['command'])
        command[command.index('--val_dataset')+1]=str(source)
        command[command.index('--result_path')+1]=str(target)
        pools[worker]['units'][key]=dict(status='pending',kind='generation',task_key=u['task_key'],partition=u['partition'],
                   folder=str(folder),result_path=str(target),expected_rows=len(rows),priority=u['priority'],
                   command=command,env_overrides={},created_at=now())
        loads[worker]+=cost
    for i,pool in enumerate(pools):
        folder=root/'worker_pools'/f'npu96-worker-{i}';folder.mkdir(parents=True,exist_ok=True)
        atomic_save(folder/'card_pool_state.json',pool)
    atomic_save(root/'worker_partition.json',dict(created_at=now(),single_writer_per_queue=True,
                estimated_costs=loads,unit_counts=[len(p['units']) for p in pools],
                mapping={key:i for i,p in enumerate(pools) for key in p['units']}))
    manifest=json.loads((root/'experiment_manifest.json').read_text())
    manifest.update(scheduler='12 independent single-writer host queues; 8 asynchronous slots per host',
                    scheduler_revision_at=now(),cross_host_queue_mutations=False)
    manifest['sources'][str(Path(__file__))]=sha(Path(__file__))
    atomic_save(root/'experiment_manifest.json',manifest)
    print(json.dumps(dict(partitioned_units=sum(len(p['units']) for p in pools),workers=12,unit_counts=[len(p['units']) for p in pools])))

def combined_state(root):
    state={'units':{}}
    for i in range(12):
        pool=read_state(root/'worker_pools'/f'npu96-worker-{i}'/'card_pool_state.json')
        if state['units'].keys() & pool['units'].keys():raise ValueError('Duplicate cross-worker unit')
        state['units'].update(pool['units'])
    return state

def aggregate(root,tasks):
    table=[];detailed={}
    for bench in sorted({t['benchmark'] for t in tasks}):
        labels=read(root/'data'/bench/'labels.jsonl');labelmap={r['sample_id']:r for r in labels};ids=sorted(labelmap)
        bootstrap=VideoBootstrap(ids,labelmap)
        loaded={}
        for task in tasks:
            if task['benchmark']!=bench:continue
            folder=root/'eval'/bench/task['label']
            if not (folder/'summary.json').exists():continue
            summary=json.loads((folder/'summary.json').read_text());scored={r['sample_id']:r for r in read(folder/'scored.jsonl')}
            if set(scored)!=set(ids):raise ValueError('Scored IDs differ')
            loaded[task['label']]=(task,summary,[scored[k]['correct'] for k in ids])
        detailed[bench]={}
        for name,(task,summary,values) in loaded.items():
            pair=paired_comparison(values,loaded['base'][2],bootstrap,name=='base') if 'base' in loaded else {}
            ci=pair.get('ci95_video_bootstrap') or [None,None]
            detailed[bench][name]=dict(summary=summary,paired_vs_base=pair)
            table.append(dict(benchmark=bench,model=name,epoch=task['epoch'],total=summary['total'],correct=summary['correct'],
                         accuracy=summary['accuracy'],parse_failures=summary['parse_failures'],
                         analysis_answer_format_rate=summary['analysis_plus_answer_format_rate'],
                         delta_vs_base=pair.get('delta_accuracy'),delta_ci95_low=ci[0],delta_ci95_high=ci[1]))
    availability=json.loads((root/'dataset_availability.json').read_text()) if (root/'dataset_availability.json').exists() else {}
    atomic_save(root/'comparison.json',dict(updated_at=now(),completed_tasks=len(table),expected_tasks=len(tasks),
                benchmarks=detailed,dataset_availability=availability,one_decoding_seed=True,
                ci_scope='paired video bootstrap conditional on generated responses'))
    if table:
        s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
        p=root/'comparison.csv';temp=p.with_suffix('.csv.tmp');temp.write_text(s.getvalue());temp.replace(p)

def controller(root):
    with DirectoryLock(root/'generalization_controller_v2.lease.lock.d',timeout=1):
        previous=None
        while True:
            try:
                state=combined_state(root);tasks=json.loads((root/'tasks.json').read_text())
                changed=False;completed=0
                for task in tasks:
                    summary=root/'eval'/task['benchmark']/task['label']/'summary.json'
                    units=[u for u in state['units'].values() if u['task_key']==task['key']]
                    if not summary.exists() and units and all(u['status']=='complete' for u in units):
                        score_task(root,task,units);changed=True
                    completed+=summary.exists()
                if changed:aggregate(root,tasks)
                statuses=Counter(u['status'] for u in state['units'].values());issues=[];active=0;workers={}
                for i in range(12):
                    name=f'npu96-worker-{i}';p=root/'worker_pools'/name/'card_pool_workers'/name/'heartbeat.json'
                    if not p.exists():workers[name]={'status':'not-started'};continue
                    h=json.loads(p.read_text());age=time.time()-p.stat().st_mtime
                    workers[name]=dict(pid=h['pid'],age_seconds=age,status=h['status'],devices=h['devices'])
                    active+=sum(d['status']=='owned_unit' for d in h['devices'].values())
                    if age>180 and h['status']!='finished':issues.append(f'{name}: stale heartbeat {age:.0f}s')
                for key,u in state['units'].items():
                    if u['status']=='failed':issues.append('failed unit: '+key)
                rows_written=0
                for u in state['units'].values():
                    p=Path(u['result_path'])
                    if p.exists():rows_written+=sum(bool(x.strip()) for x in p.open())
                health=dict(time=now(),controller_pid=os.getpid(),healthy=not issues,issues=issues,workers=workers,
                            active_cards=active,unit_statuses=dict(statuses),completed_tasks=completed,total_tasks=len(tasks),
                            rows_written=rows_written,total_rows=len(tasks)*500)
                atomic_save(root/'generalization_health.json',health)
                signature=(completed,rows_written,tuple(issues))
                if signature!=previous:
                    print(json.dumps({k:v for k,v in health.items() if k!='workers'},ensure_ascii=False),flush=True)
                    previous=signature
                if completed==len(tasks):
                    aggregate(root,tasks)
                    availability=json.loads((root/'dataset_availability.json').read_text()) if (root/'dataset_availability.json').exists() else {}
                    atomic_save(root/'READY_DATASETS_COMPLETE.json',dict(at=now(),tasks=completed,dataset_availability=availability))
                    return
            except Exception as e:
                append_event(root/'controller_alerts.jsonl',dict(at=now(),error=str(e),traceback=traceback.format_exc()))
                print(traceback.format_exc(),flush=True)
            time.sleep(15)

def launch(root,worker=None):
    if not re.fullmatch(re.escape(JOB)+r'-worker-(?:[0-9]|1[01])',socket.gethostname()):
        raise ValueError('Launch only on npu96')
    name=f'npu96-worker-{worker}' if worker is not None else 'controller'
    folder=root/'logs_independent_queues';folder.mkdir(exist_ok=True)
    pidfile=folder/(name+'.pid')
    if pidfile.exists():
        p=Path('/proc')/pidfile.read_text().strip()
        if p.exists() and (p/'cmdline').read_bytes():raise RuntimeError('Recorded component remains alive: '+name)
    if worker is not None:
        if not socket.gethostname().endswith(f'-worker-{worker}'):raise ValueError('Wrong worker host')
        cmd=[str(PYENV/'python'),str(REPO/'training_code/scripts/worker_worldsense_card_pool.py'),
             '--root',str(root/'worker_pools'/name),'--worker',name,'--devices','0,1,2,3,4,5,6,7']
    else:cmd=[str(PYENV/'python'),str(Path(__file__).resolve()),'--root',str(root),'--action','controller']
    with (folder/(name+'.log')).open('a') as log:
        child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,cwd=REPO,env=env_for('0'))
    pidfile.write_text(str(child.pid)+'\n')
    record=dict(at=now(),pid=child.pid,hostname=socket.gethostname(),command=cmd,source_sha256=sha(cmd[1]))
    atomic_save(folder/(name+'.launch.json'),record);print(json.dumps(record),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--action',choices=['init','partition','controller','launch-controller','restart-controller','launch-worker','launch-fleet','aggregate'],required=True)
    p.add_argument('--benchmark',nargs='+',default=['omnivideobench','dailyomni']);p.add_argument('--worker',type=int)
    a=p.parse_args();root=a.root.resolve()
    if a.action=='init':initialize(root,a.benchmark)
    elif a.action=='partition':partition_workers(root)
    elif a.action=='controller':controller(root)
    elif a.action=='launch-controller':launch(root)
    elif a.action=='restart-controller':
        pid=int((root/'logs_independent_queues/controller.pid').read_text())
        proc=Path('/proc')/str(pid)
        expected=[str(PYENV/'python'),str(Path(__file__).resolve()),'--root',str(root),'--action','controller']
        if proc.exists():
            actual=(proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
            if actual!=expected:raise ValueError('Controller PID identity differs')
            os.kill(pid,signal.SIGTERM)
            for _ in range(50):
                if not proc.exists() or not (proc/'cmdline').read_bytes():break
                time.sleep(.1)
            else:raise RuntimeError('Controller did not exit')
        lease=root/'generalization_controller_v2.lease.lock.d'
        if lease.exists():
            owner=json.loads((lease/'owner.json').read_text())
            if owner['pid']!=pid or owner['hostname']!=socket.gethostname():raise ValueError('Controller lease differs')
            (lease/'owner.json').unlink();lease.rmdir()
        launch(root)
    elif a.action=='launch-worker':launch(root,a.worker)
    elif a.action=='aggregate':aggregate(root,json.loads((root/'tasks.json').read_text()))
    else:
        launch(root)
        def remote(i):
            cmd=[str(PYENV/'python'),str(Path(__file__).resolve()),'--root',str(root),'--action','launch-worker','--worker',str(i)]
            r=subprocess.run(['ssh','-F','/dev/null','-p','2222','-o','BatchMode=yes','-o','ConnectTimeout=8',
                              f'ma-user@{JOB}-worker-{i}.{JOB}',shlex.join(cmd)],capture_output=True,text=True,timeout=45)
            return dict(worker=i,exit=r.returncode,output=r.stdout,error=r.stderr[-1000:])
        with ThreadPoolExecutor(max_workers=12) as pool:
            for f in as_completed([pool.submit(remote,i) for i in range(12)]):print(json.dumps(f.result()),flush=True)

if __name__=='__main__':main()
