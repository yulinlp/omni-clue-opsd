#!/usr/bin/env python3
"""Evaluate all formal checkpoints with frozen training prompts, safe single-card queues."""
import argparse, copy, json, os, re, shlex, struct, subprocess, sys, time, traceback
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
REPO=Path('/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd')
sys.path[:0]=[str(REPO/'training_code/scripts'),str(REPO/'training_code/src')]
import generalization_openqa_fleet as f
from worldsense_mcq_training_suite import ssh_args
from prepare_worldsense_inference_cache_views import enable_cache
from score_generalization_analysis_mcq_v3 import score_rows
from score_generalization_analysis_mcq_v4 import parse_analysis_option, VERSION
from omni_opsd.data.dynamic_budget import dynamic_budget_for
TRAIN=REPO/'training_runs/worldsense_mcq8_latest1500_20261004'
OLD=REPO/'training_runs/worldsense_generalization_mcq500_npu96_20261003'
PARTS=16
BENCHES=['omnivideobench','dailyomni']
load,save,read,now=f.load,f.atomic_save,f.read,f.now

def variant(arm):
    return 'distill_analysis' if arm.startswith(('clue_','opsd_')) else ('sft_analysis' if 'observation' in arm else 'answer')

def prepare(root):
    root.mkdir(parents=True,exist_ok=True);suite=load(TRAIN/'suite.json')
    instructions=dict(answer=suite['student_answer_instruction'],sft_analysis=suite['student_analysis_instruction'],distill_analysis=suite['distillation_analysis_instruction'])
    audit={}
    for bench in BENCHES:
        old=OLD/'data'/bench;rows=read(old/'inputs.jsonl');labels=read(old/'labels.jsonl');media=read(old/'media_budget_audit.jsonl')
        lm={x['sample_id']:x for x in labels};mm={x['case_id']:x for x in media}
        assert len(rows)==len(lm)==len(mm)==500
        for row in rows:
            a=mm[row['case_id']];m=a['media'];v=Path(row['videos'][0]['video']);st=v.stat()
            assert 0<m['duration']<300 and m['audio_streams']>0
            assert st.st_size==m['size'] and st.st_mtime_ns==m['mtime_ns'],str(v)
            budget=dynamic_budget_for(dict(duration=m['duration'],metadata=dict(resolution=f"{m['width']}x{m['height']}")))
            assert budget==row['dynamic_student_budget'],row['case_id']
        for kind,instruction in instructions.items():
            data=copy.deepcopy(rows)
            for row in data:
                label=lm[row['case_id']]
                text='<video>\n'+label['question']+'\nOptions:\n'+'\n'.join(f'{chr(65+i)}. {c}' for i,c in enumerate(label['choices']))+'\n'+instruction
                row['messages']=[dict(role='user',content=text)]
                assert not {'answer','teacher_prompt','observation','teacher_videos','gold_answer_text'}&row.keys()
            f.jsonl(root/'data'/bench/kind/'inputs.jsonl',data)
        f.jsonl(root/'data'/bench/'labels.jsonl',labels);f.jsonl(root/'data'/bench/'media_budget_audit.jsonl',media)
        audit[bench]=dict(rows=500,max_duration=max(x['media']['duration'] for x in media),strictly_below_300=True,source=str(old),all_media_size_mtime_unchanged=True,all_dynamic_budgets_recomputed_equal=True)
    tasks=[]
    for kind in instructions:
        tasks.append(dict(label='base_'+kind,arm='base',step=0,prompt=kind,source=suite['model'],adapter='-',priority=0))
    for arm,spec in suite['arms'].items():
        steps=[32,64,96] if arm.startswith('sft') else list(range(3,31,3))+[32]
        for step in reversed(steps):
            cp=str(Path(spec['output_dir'])/f'checkpoint-{step}')
            tasks.append(dict(label=f'{arm}_step{step}',arm=arm,step=step,prompt=variant(arm),source=suite['model'] if spec['tuner_type']=='lora' else cp,adapter=cp if spec['tuner_type']=='lora' else '-',checkpoint=cp,priority=10+(max(steps)-step)))
    assert len(tasks)==59
    save(root/'tasks.json',tasks)
    save(root/'protocol.json',dict(at=now(),training_root=str(TRAIN),sample_audit=audit,instructions=instructions,checkpoints=56,base_prompt_conditions=3,total_tasks=118,total_responses=59000,max_new_tokens=512,temperature=1.,top_p=1.,requested_top_k=-1,effective_top_k=50,seed=20260904,max_length=32768,use_audio_in_video=True,audio_sample_rate=16000,video_reader='pyav_seek',parser=VERSION))
    for name in f.WORKERS:
        wr=root/'worker_pools'/name;(wr/'inbox').mkdir(parents=True,exist_ok=True)
        if not (wr/'card_pool_state.json').exists():save(wr/'card_pool_state.json',{'units':{}})
    if not (root/'controller_state.json').exists():save(root/'controller_state.json',dict(tasks={},units={}))

def ready(task,health):
    if task['arm']=='base':return True
    # Wait for training exit, including the second final save, before reading weights.
    if health['arms'][task['arm']]['status']!='complete':return False
    cp=Path(task['checkpoint']);marker=cp/'MCQ_CHECKPOINT_COMPLETE.json'
    if not marker.exists():return False
    assert load(marker)['global_step']==task['step']
    files=list(cp.glob('adapter_model.safetensors')) if task['adapter']!='-' else list(cp.glob('model-*.safetensors'))
    assert len(files)==(1 if task['adapter']!='-' else 4),str(cp)
    for path in files:
        with path.open('rb') as file:
            size=struct.unpack('<Q',file.read(8))[0];header=json.loads(file.read(size))
        expected=8+size+max(v['data_offsets'][1] for k,v in header.items() if k!='__metadata__')
        assert path.stat().st_size==expected,str(path)
    return True

def model_view(root,task):
    source=Path(task['source']);target=root/'inference_models'/task['label']
    if (target/'READY.json').exists():return str(target)
    target.mkdir(parents=True,exist_ok=True)
    for p in source.iterdir():
        if not p.is_file() or p.name in ['config.json','generation_config.json','ema_teacher.safetensors','optimizer.pt','scheduler.pt','training_args.bin','args.json'] or p.name.startswith('rng_state'):continue
        dest=target/p.name
        if not dest.exists():dest.symlink_to(p.resolve())
    cfg=load(source/'config.json');enable_cache(cfg);cfg['use_cache']=True;save(target/'config.json',cfg)
    gen=load(source/'generation_config.json');gen.update(use_cache=True,top_k=50);save(target/'generation_config.json',gen)
    save(target/'READY.json',dict(source=str(source),source_config_sha256=f.sha(source/'config.json'),weights_unchanged=True,use_cache=True))
    return str(target)

def add_ready(root,state):
    health=load(TRAIN/'health.json')
    for task in load(root/'tasks.json'):
        if all(bench+'/'+task['label'] in state['tasks'] for bench in BENCHES):continue
        if not ready(task,health):continue
        model=model_view(root,task)
        for bench in BENCHES:
            key=bench+'/'+task['label']
            if key in state['tasks']:continue
            data=read(root/'data'/bench/task['prompt']/'inputs.jsonl');ids=[]
            for i in range(PARTS):
                uid=key+'/'+str(i);source=root/'staging'/uid/'input.jsonl';part=data[i::PARTS];f.jsonl(source,part)
                state['units'][uid]=dict(kind='generation',task_key=key,partition=i,source=str(source),expected_rows=len(part),priority=task['priority']*100+i,status='unassigned',model=model,adapter=task['adapter'])
                ids.append(uid)
            state['tasks'][key]=dict(**task,benchmark=bench,stage='generating',generation_units=ids)

def publish(root,uid,u):
    wr=root/'worker_pools'/u['worker'];folder=wr/'units'/uid;source=folder/'input.jsonl';target=folder/'results.jsonl'
    f.jsonl(source,read(Path(u['source'])))
    command=f.generation_command(u['model'],u['adapter'],source,target,'openqa')
    unit=dict(kind='generation',task_key=u['task_key'],partition=u['partition'],expected_rows=u['expected_rows'],folder=str(folder),result_path=str(target),command=command,priority=u['priority'],env_overrides={},status='pending')
    import hashlib
    f.inbox(root,u['worker'],'add_'+hashlib.sha256(uid.encode()).hexdigest()+'.json',dict(action='add',key=uid,unit=unit))

def score(root,state,key,t):
    out=root/'eval'/key;rp=out/'results.jsonl';f.merge(root,t['generation_units'],state,rp)
    labels=read(root/'data'/t['benchmark']/'labels.jsonl');lm={x['sample_id']:x for x in labels}
    records=score_rows(read(rp),labels,read(root/'data'/t['benchmark']/t['prompt']/'inputs.jsonl'))
    for row in records:
        row.update(parse_analysis_option(row['response'],lm[row['sample_id']]['choices']))
        row['correct']=bool(row['parsed'] and row['prediction']==row['gold_answer'])
        pattern=r'\s*<answer>\s*[A-D]\s*</answer>\s*' if t['prompt']=='answer' else r'\s*<analysis>.+?</analysis>\s*<answer>\s*[A-D]\s*</answer>\s*'
        row['training_format_compliant']=bool(re.fullmatch(pattern,row['response'],re.S))
    assert len(records)==500
    f.jsonl(out/'scored.jsonl',records)
    save(out/'summary.json',dict(benchmark=t['benchmark'],model=t['label'],arm=t['arm'],step=t['step'],prompt=t['prompt'],total=500,correct=sum(x['correct'] for x in records),accuracy_percent=sum(x['correct'] for x in records)/5,unparsed=sum(not x['parsed'] for x in records),format_compliant=sum(x['training_format_compliant'] for x in records),parser=VERSION))
    t.update(stage='complete',completed_at=now())

def controller(root):
    with f.LocalLock(root/'controller.lease',timeout=0):
        while True:
            state=load(root/'controller_state.json');issues=[];workers={};loads={};add_ready(root,state)
            training=load(TRAIN/'health.json')
            busy={n for a in training['arms'].values() if a['status']!='complete' for n in a['nodes']}
            for name in f.WORKERS:
                wr=root/'worker_pools'/name;local=load(wr/'card_pool_state.json')['units'];hp=wr/'card_pool_workers'/name/'heartbeat.json'
                workers[name]=dict(status='starting')
                if hp.exists():
                    h=load(hp);workers[name]=dict(age=time.time()-hp.stat().st_mtime,status=h['status'],devices=h['devices'])
                    if workers[name]['age']>180:issues.append(name+':stale-heartbeat')
                for uid,u in state['units'].items():
                    if u.get('worker')!=name:continue
                    if uid not in local:publish(root,uid,u);continue
                    lu=local[uid];u.update(status=lu['status'],result_path=lu['result_path'],attempt=lu.get('attempt',0))
                    if u['status']=='failed':
                        issues.append('failed:'+uid)
                        if u['attempt']<3:f.inbox(root,name,'retry_'+uid.replace('/','_')+'_'+str(u['attempt'])+'.json',dict(action='retry',key=uid))
                loads[name]=sum(u.get('worker')==name and u['status'] in ['queued','pending','running'] for u in state['units'].values())
            available=[n for n in f.WORKERS if n not in busy and workers[n].get('age',0)<180]
            for uid,u in sorted(state['units'].items(),key=lambda kv:(kv[1]['priority'],kv[0])):
                if u['status']!='unassigned' or not available:continue
                name=min(available,key=lambda n:loads[n])
                if loads[name]>=9:break
                u.update(worker=name,status='queued');loads[name]+=1;publish(root,uid,u)
            for key,t in state['tasks'].items():
                if t['stage']=='generating' and all(state['units'][u]['status']=='complete' for u in t['generation_units']):
                    try:score(root,state,key,t)
                    except Exception:issues.append('scoring:'+key);(root/'errors.log').open('a').write(traceback.format_exc())
            completed=sum(t['stage']=='complete' for t in state['tasks'].values());rows=0
            for u in state['units'].values():
                p=Path(u.get('result_path','/nonexistent'))
                if p.is_file():rows+=sum(bool(x.strip()) for x in p.open())
            health=dict(at=now(),completed_tasks=completed,total_tasks=118,registered_tasks=len(state['tasks']),generation_rows=rows,total_generation_rows=59000,unit_statuses=dict(Counter(u['status'] for u in state['units'].values())),active_cards=sum(u['status']=='running' for u in state['units'].values()),workers=workers,issues=issues)
            save(root/'controller_state.json',state);save(root/'health.json',health)
            summaries=[load(root/'eval'/k/'summary.json') for k,t in state['tasks'].items() if t['stage']=='complete'];save(root/'comparison.json',summaries)
            print(json.dumps({k:v for k,v in health.items() if k!='workers'}),flush=True)
            if completed==118:
                for name in f.WORKERS:f.inbox(root,name,'zz_stop.json',dict(action='stop'))
                save(root/'COMPLETE.json',dict(at=now(),tasks=118,responses=59000));return
            time.sleep(15)

def launch(root,action,name=None):
    logs=root/'logs';logs.mkdir(exist_ok=True);label=name or 'controller';pidfile=logs/(label+'.pid')
    if pidfile.exists():
        p=Path('/proc')/pidfile.read_text().strip()/'cmdline'
        if p.exists() and str(root).encode() in p.read_bytes():return
    cmd=[str(f.PYENV/'python'),str(Path(__file__).resolve()),'--root',str(root),'--action',action]
    if name:cmd+=['--worker',name]
    with (logs/(label+'.log')).open('a') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,cwd=REPO,env=f.env_for(''))
    pidfile.write_text(str(proc.pid));print(json.dumps(dict(component=label,pid=proc.pid)))

def launch_all(root):
    script=str(Path(__file__).resolve())
    def one(name):
        cmd=shlex.join([str(f.PYENV/'python'),script,'--root',str(root),'--action','launch-worker','--worker',name])
        for attempt in range(3):
            p=subprocess.run(ssh_args(name,cmd),capture_output=True,text=True,timeout=90)
            if p.returncode==0:return dict(worker=name,output=p.stdout[-500:])
            time.sleep(2)
        return dict(worker=name,error=p.stderr[-1000:])
    with ThreadPoolExecutor(max_workers=4) as ex:
        for result in ex.map(one,f.WORKERS):print(json.dumps(result),flush=True)
    launch(root,'controller')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--action',required=True,choices=['prepare','launch-all','controller','worker','launch-worker']);p.add_argument('--worker');a=p.parse_args();root=a.root.resolve()
    if a.action=='prepare':prepare(root)
    elif a.action=='launch-all':launch_all(root)
    elif a.action=='controller':controller(root)
    elif a.action=='worker':f.worker(root,a.worker)
    else:launch(root,'worker',a.worker)
