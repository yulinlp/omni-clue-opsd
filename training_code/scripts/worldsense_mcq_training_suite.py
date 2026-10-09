#!/usr/bin/env python3
"""Three full-parameter training jobs, gated on completed and drained MCQ eval."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import traceback

REPO = Path('/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd')
sys.path.insert(0, str(REPO/'training_code/scripts'))
import generalization_openqa_fleet as fleet
from worker_worldsense_card_pool import probe_npus

PYTHON = str(fleet.PYENV/'python')
SCRIPT_NAME = 'worldsense_mcq_training_suite.py'
load, save, now = fleet.load, fleet.atomic_save, fleet.now


def ssh_args(node, command):
    opts = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8']
    i = int(node.rsplit('-', 1)[1])
    if node.startswith('npu24'):
        return opts + ['-p', '2222', 'ma-user@'+['172.16.2.181','172.16.12.237','172.16.12.70'][i], command]
    host = f'ma-job-7b525feb-b862-4352-8403-9c3a018c05e8-worker-{i}.ma-job-7b525feb-b862-4352-8403-9c3a018c05e8'
    nested = shlex.join(opts + ['-p','2222','ma-user@'+host,command])
    return opts + ['-p','31445','-i','/home/ma-user/.ssh/ulan_31445_npu96.pem',
                   'ma-user@dev-modelarts-cnnorth9.huaweicloud.com',nested]


def remote(node, args, timeout=90):
    p = subprocess.run(ssh_args(node, shlex.join([PYTHON]+args)), capture_output=True, text=True, timeout=timeout)
    if p.returncode: raise RuntimeError(node+': '+p.stderr[-3000:]+' '+p.stdout[-1000:])
    return json.loads(next(line for line in reversed(p.stdout.splitlines()) if line.startswith('{')))


def inventory():
    table = load(Path('/user/config/jobstart_hccl.json'))
    return dict(hostname=socket.gethostname(), pod_ip=socket.gethostbyname(socket.gethostname()),
                rank_table=table, memory=Path('/proc/meminfo').read_text().splitlines()[:3])


def prepare_hosts(root):
    manifest=load(root/'suite.json'); nodes=[n for a in manifest['arms'].values() for n in a['workers']]
    assert len(nodes)==len(set(nodes))==15
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures={n:ex.submit(remote,n,[str(root/'code'/SCRIPT_NAME),'--root',str(root),'--action','inventory']) for n in nodes}
        hosts={n:f.result() for n,f in futures.items()}
    save(root/'hosts.json',hosts)
    for arm, spec in manifest['arms'].items():
        source=hosts[spec['workers'][0]]['rank_table']; servers=[]
        for i,node in enumerate(spec['workers']):
            number=int(node.rsplit('-',1)[1])
            # Original job tables are ordered by original global rank, which
            # has been checked against the previously working subset tables.
            server=copy.deepcopy(source['server_list'][number])
            assert [int(d['rank_id']) for d in server['device']]==list(range(number*8,number*8+8))
            assert hosts[node]['rank_table']==source
            assert hosts[node]['hostname'].endswith('worker-'+str(number))
            for j,d in enumerate(server['device']):d['rank_id']=str(i*8+j)
            servers.append(server)
        table=dict(source,server_count=str(len(servers)),server_list=servers)
        save(root/arm/'data/ranktable.json',table)
    save(root/'host_preflight.json',dict(at=now(),nodes=15,cards=120,
        groups={a:dict(workers=s['workers'],master=hosts[s['workers'][0]]['pod_ip'],
                       ranktable_sha256=fleet.sha(root/a/'data/ranktable.json')) for a,s in manifest['arms'].items()}))
    print(json.dumps(dict(hosts=15,cards=120,groups=list(manifest['arms']))))


def prerequisite_ready(root):
    previous=Path(load(root/'suite.json')['prerequisite'])
    if not (previous/'COMPLETE.json').exists():return False,'waiting_for_mcq_evaluation'
    if load(previous/'comparison.json').get('completed_tasks')!=26:return False,'waiting_for_26_mcq_scores'
    for name in fleet.WORKERS:
        wr=previous/'worker_pools'/name;hp=wr/'card_pool_workers'/name/'heartbeat.json'
        if not hp.exists() or load(hp).get('status')!='complete':return False,'waiting_for_eval_worker_exit:'+name
        if any(u['status']!='complete' for u in load(wr/'card_pool_state.json')['units'].values()):
            return False,'waiting_for_eval_units:'+name
    return True,'all_mcq_scores_complete_and_workers_drained'


def verify(root):
    m=load(root/'suite.json')
    for name,digest in m['code_sha256'].items():
        assert fleet.sha(root/'code'/name)==digest, 'Training code snapshot changed: '+name
    for arm,spec in m['arms'].items():
        assert fleet.sha(Path(spec['dataset']))==spec['dataset_sha256']
        assert fleet.sha(Path(spec['smoke_dataset']))==spec['smoke_sha256']
        assert fleet.sha(root/arm/'data/ranktable.json')==load(root/'host_preflight.json')['groups'][arm]['ranktable_sha256']


def output_dir(root, arm, phase, attempt):
    return root/arm/'outputs'/('formal' if phase=='formal' else f'smoke_attempt{attempt}')


def attempt_dir(root, arm, phase, attempt):return root/arm/'logs'/f'{phase}_attempt{attempt}'


def train_command(root,arm,rank,phase,attempt,resume=None):
    m=load(root/'suite.json');s=m['arms'][arm];clue=arm=='clue_no_observation'
    entry=root/'code'/('worldsense_mcq_gkd_entry.py' if clue else 'worldsense_mcq_sft_entry.py')
    master=load(root/'hosts.json')[s['workers'][0]]['pod_ip'];output=output_dir(root,arm,phase,attempt)
    args=[PYTHON,'-m','torch.distributed.run','--nproc_per_node','8','--nnodes',str(len(s['workers'])),
          '--node_rank',str(rank),'--master_addr',master,'--master_port',str(s['master_port']),str(entry),
          '--model',m['model'],'--model_type','qwen2_5_omni','--dataset',s['smoke_dataset'] if phase=='smoke' else s['dataset'],
          '--output_dir',str(output),'--add_version','false','--split_dataset_ratio','0',
          '--tuner_type','full','--freeze_llm','false','--freeze_vit','false','--freeze_aligner','false',
          '--deepspeed',str(root/'code/zero2.json'),'--optim','adamw_torch','--torch_dtype','bfloat16',
          '--num_train_epochs',str(s['epochs']),'--per_device_train_batch_size','1',
          '--gradient_accumulation_steps',str(s['gradient_accumulation_steps']),
          '--learning_rate',str(s['learning_rate']),'--lr_scheduler_type','cosine','--warmup_ratio','0.03',
          '--logging_steps','1','--max_length','32768','--use_logits_to_keep','true',
          '--gradient_checkpointing','true','--gradient_checkpointing_kwargs','{"use_reentrant":false}',
          '--attn_impl','sdpa','--dataset_num_proc','1','--dataloader_num_workers','0','--dataloader_drop_last','false','--no_dataset_shuffle',
          '--seed','20260904','--report_to','none','--save_only_model','false','--save_total_limit','0']
    if clue:
        args+=['--rlhf_type','gkd','--lmbda','1.0','--sft_alpha','0','--beta','0.5','--temperature','1.0',
               '--clue_ema_alpha','0.05','--max_completion_length','512','--use_vllm','false','--log_completions','true',
               '--save_strategy','steps','--save_steps','3']
    else:
        args+=['--loss_type','mcq_region_weighted','--average_tokens_across_devices','false',
               '--max_grad_norm','0','--save_strategy','epoch']
    if phase=='smoke':args+=['--max_steps','1','--save_strategy','steps','--save_steps','1']
    if resume:args+=['--resume_from_checkpoint',str(resume)]
    env=fleet.env_for('0,1,2,3,4,5,6,7')
    env.update(PYTHONPATH=':'.join([str(root/'code'),'/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_npu24_deps',
        str(REPO/'training_code/src'),'/opt/huawei/dataset/hyl_ulan/ylhu/third_party/ms-swift']),
        RANK_TABLE_FILE=str(root/arm/'data/ranktable.json'),RANK_TABLE_FILE_V_1_0=str(root/arm/'data/ranktable.json'),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OMNI_FULL_CPU_THREADS='4',WORLDSENSE_LOGITS_TO_KEEP='1',
        WORLDSENSE_CKPT_LM_HEAD='1',HCCL_CONNECT_TIMEOUT='1200',HCCL_EXEC_TIMEOUT='1800',
        NPROC_PER_NODE='8',NNODES=str(len(s['workers'])),NODE_RANK=str(rank),MASTER_ADDR=master,
        MASTER_PORT=str(s['master_port']),MCQ_ANSWER_WEIGHT=str(m['answer_weight']),MCQ_RESUME_PATH=str(resume or ''))
    return args,env


def checkpoints(root,arm,phase,attempt):
    out=output_dir(root,arm,phase,attempt); found=[]
    for p in out.glob('checkpoint-*/MCQ_CHECKPOINT_COMPLETE.json'):
        value=load(p)
        if arm=='clue_no_observation' and not value['has_ema_teacher']:continue
        if not (p.parent/'trainer_state.json').exists():continue
        found.append((value['global_step'],p.parent,value))
    return sorted(found)


def validate_smoke_audits(folder, workers, arm):
    """Swift INFO is emitted by LOCAL_RANK=0 on each host, not all eight ranks.

    The input assertions execute on every rank before the filtered logger call.
    Successful torchrun exits and the distributed checkpoint barrier are checked
    separately. Validate the actual local-master records instead of counting
    messages that Swift deliberately suppresses.
    """
    tag='MULTIMODAL_INPUT_AUDIT' if arm=='clue_no_observation' else 'MCQ_SFT_AUDIT'
    records=[]
    for rank,node in enumerate(workers):
        text=(folder/(node+'.train.log')).read_text()
        audits=[json.JSONDecoder().raw_decode(line.split(tag,1)[1].lstrip())[0]
                for line in text.splitlines() if tag in line]
        matches=[a for a in audits if a.get('rank')==rank*8]
        assert matches,(node,'missing local-master input audit',rank*8)
        audit=matches[-1]
        views=[audit['student'],audit['teacher']] if arm=='clue_no_observation' else [audit['input_shapes']]
        for view in views:
            for field in ['input_features','pixel_values_videos']:
                shape=view.get(field,[])
                assert shape and all(x>0 for x in shape),(node,'missing media input',field)
        if arm=='clue_no_observation':
            assert audit['source']=='student',(node,'wrong GKD branch')
        else:
            assert all(v>0 for v in audit['trainable_parameters'].values()),(node,'frozen model component')
            assert audit['loss']=='region_means_then_weighted_average'
        records.append(audit)
    return records


def read_cmd(pid):
    try:return [x.decode() for x in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if x]
    except (OSError,UnicodeError):return []


def stop_local(root,arm,phase,attempt):
    target=str(output_dir(root,arm,phase,attempt));victims={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        args=read_cmd(p.name)
        if '--output_dir' in args and args[args.index('--output_dir')+1]==target and any(str(root/'code') in x for x in args):
            victims[int(p.name)]=args
    for pid,args in victims.items():
        try:os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:pass
    deadline=time.time()+10
    while time.time()<deadline and any(read_cmd(pid)==args for pid,args in victims.items()):time.sleep(.5)
    for pid,args in victims.items():
        if read_cmd(pid)==args:
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass
    return dict(stopped_pids=list(victims),output_dir=target)


def node(root,arm,rank,phase,attempt):
    if (root/'USER_STOP_REQUEST.json').exists():return
    assert prerequisite_ready(root)[0], 'Evaluation has not drained'
    verify(root);m=load(root/'suite.json');name=m['arms'][arm]['workers'][rank]
    folder=attempt_dir(root,arm,phase,attempt);folder.mkdir(parents=True,exist_ok=True)
    statepath=folder/(name+'.json')
    with fleet.LocalLock(root/arm/('node_'+name+'.lease'),timeout=0):
        state=dict(at=now(),supervisor_pid=os.getpid(),worker=name,rank=rank,phase=phase,attempt=attempt,status='waiting_for_free_cards')
        save(statepath,state);deadline=time.time()+900
        while True:
            if (folder/'STOP.json').exists():return
            probe=probe_npus(list('01234567'))
            if probe['ok'] and all(v['status']=='empty' for v in probe['devices'].values()):break
            if time.time()>deadline:raise RuntimeError('Assigned cards remain busy; no unrelated processes were stopped')
            state.update(at=now(),status='waiting_for_free_cards');save(statepath,state);time.sleep(10)
        state.update(at=now(),status='ready');save(statepath,state)
        while not (folder/'GO.json').exists():
            if (folder/'STOP.json').exists():return
            state.update(at=now());save(statepath,state);time.sleep(3)
        request=load(folder/'GO.json');args,env=train_command(root,arm,rank,phase,attempt,request.get('resume'))
        with (folder/(name+'.train.log')).open('a') as log:
            process=subprocess.Popen(args,env=env,stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True,cwd=REPO)
        state.update(at=now(),status='running',train_pid=process.pid,command=args);save(statepath,state)
        while process.poll() is None:
            if (folder/'STOP.json').exists():stop_local(root,arm,phase,attempt)
            state.update(at=now());save(statepath,state);time.sleep(10)
        state.update(at=now(),status='complete' if process.returncode==0 else 'failed',exit_code=process.returncode)
        save(statepath,state)


def launch_node(root,arm,rank,phase,attempt):
    if (root/'USER_STOP_REQUEST.json').exists():return dict(status='stopped_by_user',launched=False)
    name=load(root/'suite.json')['arms'][arm]['workers'][rank];folder=attempt_dir(root,arm,phase,attempt)
    folder.mkdir(parents=True,exist_ok=True);pidfile=folder/(name+'.pid')
    if pidfile.exists():
        pid=int(pidfile.read_text());args=read_cmd(pid)
        if str(root/'code'/SCRIPT_NAME) in args and '--action' in args and args[args.index('--action')+1]=='node':
            return dict(pid=pid,worker=name,already_alive=True)
    command=[PYTHON,str(root/'code'/SCRIPT_NAME),'--root',str(root),'--action','node','--arm',arm,'--rank',str(rank),
             '--phase',phase,'--attempt',str(attempt)]
    with (folder/(name+'.supervisor.log')).open('a') as log:
        p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,cwd=REPO)
    pidfile.write_text(str(p.pid)+'\n');return dict(pid=p.pid,worker=name,command=command)


def event(root,kind,**kwargs):
    record=dict(at=now(),kind=kind,**kwargs)
    with (root/'events.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    print(json.dumps(record,ensure_ascii=False),flush=True)


def remote_nodes(root,arm,phase,attempt,action):
    workers=load(root/'suite.json')['arms'][arm]['workers']; results={}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures={n:ex.submit(remote,n,[str(root/'code'/SCRIPT_NAME),'--root',str(root),'--action',action,
            '--arm',arm,'--rank',str(i),'--phase',phase,'--attempt',str(attempt)]) for i,n in enumerate(workers)}
        for n,f in futures.items():
            try:results[n]=f.result()
            except Exception as e:results[n]=dict(error=repr(e))
    event(root,action,arm=arm,phase=phase,attempt=attempt,results=results)
    return results


def controller(root):
    if (root/'USER_STOP_REQUEST.json').exists():return
    verify(root)
    with fleet.LocalLock(root/'training_controller.lease',timeout=0):
        m=load(root/'suite.json');sp=root/'controller_state.json'
        if not sp.exists():save(sp,dict(arms={a:dict(phase='smoke',attempt=1,status='waiting') for a in m['arms']}))
        while True:
            ok,reason=prerequisite_ready(root)
            save(root/'health.json',dict(at=now(),controller_pid=os.getpid(),stage='waiting_for_evaluations',reason=reason))
            if ok:break
            time.sleep(30)
        event(root,'evaluation_handoff_verified',prerequisite=m['prerequisite'])
        while True:
            if (root/'USER_STOP_REQUEST.json').exists():return
            state=load(sp); issues=[]
            for arm,s in state['arms'].items():
                if s['status'] in ['complete','failed']:continue
                phase,attempt=s['phase'],s['attempt'];folder=attempt_dir(root,arm,phase,attempt);spec=m['arms'][arm]
                try:
                    if s['status']=='waiting':
                        s.update(status='launching',launched_at=now());save(sp,state)
                        remote_nodes(root,arm,phase,attempt,'launch-node')
                    nodes={}
                    for name in spec['workers']:
                        p=folder/(name+'.json')
                        if p.exists():nodes[name]=load(p)
                    # Keep terminal node states visible even when validation fails.
                    s['nodes']={n:{k:v for k,v in x.items() if k!='command'} for n,x in nodes.items()}
                    elapsed=(datetime.now().astimezone()-datetime.fromisoformat(s['launched_at'])).total_seconds()
                    stale=[n for n,v in nodes.items() if v['status'] not in ['complete','failed'] and
                           (datetime.now().astimezone()-datetime.fromisoformat(v['at'])).total_seconds()>180]
                    failure=any(v['status']=='failed' for v in nodes.values()) or bool(stale) or (len(nodes)<len(spec['workers']) and elapsed>180)
                    if s['status']=='launching' and len(nodes)==len(spec['workers']) and all(v['status']=='ready' for v in nodes.values()):
                        latest=checkpoints(root,arm,'formal',attempt) if phase=='formal' else []
                        resume=str(latest[-1][1]) if latest else None
                        save(folder/'GO.json',dict(at=now(),resume=resume))
                        s.update(status='running',started_at=now(),resume=resume);event(root,'training_started',arm=arm,phase=phase,attempt=attempt,resume=resume)
                    if len(nodes)==len(spec['workers']) and all(v['status']=='complete' for v in nodes.values()):
                        assert all(v.get('exit_code')==0 for v in nodes.values()), 'Nonzero completed node exit'
                        saved=checkpoints(root,arm,phase,attempt)
                        assert saved,'No complete checkpoint after successful exits'
                        trainer_state=load(saved[-1][1]/'trainer_state.json')
                        expected=1 if phase=='smoke' else trainer_state['max_steps']
                        assert saved[-1][0]>=expected,(arm,saved[-1][0],expected)
                        assert saved[-1][2]['world_size']==8*len(spec['workers']), 'Wrong distributed world size'
                        assert any('loss' in row and math.isfinite(float(row['loss']))
                                   for row in trainer_state['log_history']), 'No finite training loss'
                        if phase=='formal':
                            assert saved[-1][2]['epoch']>=spec['epochs']-1e-4,(arm,'insufficient epochs',saved[-1][2])
                            if arm=='clue_no_observation':
                                required=set(range(3,expected+1,3))
                                assert required.issubset({x[0] for x in saved}), 'Missing requested three-step checkpoints'
                        if phase=='smoke':
                            audits=validate_smoke_audits(folder,spec['workers'],arm)
                            checkpoint=saved[-1][1]
                            expected_ranks=set(range(8*len(spec['workers'])))
                            rng_ranks={int(p.stem.rsplit('_',1)[1]) for p in checkpoint.glob('rng_state_*.pth')}
                            assert rng_ranks==expected_ranks, 'Incomplete per-rank RNG checkpoint set'
                            save(folder/'validation.json',dict(at=now(),passed=True,
                                log_scope='one INFO record per local master; assertions execute on every rank',
                                nodes=len(nodes),world_size=len(expected_ranks),audits=audits,
                                checkpoint=str(checkpoint),all_node_exits_zero=True,all_rng_ranks_present=True))
                            s.update(phase='formal',attempt=1,status='waiting',smoke_passed_at=now())
                            event(root,'smoke_passed',arm=arm)
                        else:
                            s.update(status='complete',finished_at=now(),final_checkpoint=str(saved[-1][1]))
                            save(root/arm/'COMPLETE.json',dict(at=now(),checkpoint=str(saved[-1][1]),steps=saved[-1][0],epochs=spec['epochs']))
                            event(root,'training_complete',arm=arm,checkpoint=str(saved[-1][1]))
                    elif failure:
                        issues.append(arm+':failed_or_stale_node')
                        save(folder/'STOP.json',dict(at=now(),reason='failed_or_stale_node',stale=stale))
                        remote_nodes(root,arm,phase,attempt,'stop-node')
                        if attempt<3:s.update(status='waiting',attempt=attempt+1)
                        else:s.update(status='failed',failure_reason='three attempts failed; inspect logs and repair before resume')
                        event(root,'attempt_failed',arm=arm,phase=phase,attempt=attempt,next_status=s['status'])
                    s['nodes']={n:{k:v for k,v in x.items() if k!='command'} for n,x in nodes.items()}
                    progress=checkpoints(root,arm,phase,attempt)
                    s['saved_steps']=[x[0] for x in progress]
                except Exception as e:
                    # Do not report a finished-but-rejected attempt as running
                    # forever. Preserve its files and surface a terminal gate error.
                    if s.get('nodes') and len(s['nodes'])==len(spec['workers']) and all(
                            v['status']=='complete' for v in s['nodes'].values()):
                        s.update(status='failed',failure_reason='completed_attempt_validation_failed: '+repr(e),failed_at=now())
                    issues.append(arm+':'+repr(e));event(root,'controller_arm_error',arm=arm,error=traceback.format_exc())
                save(sp,state)
            completed=sum(s['status']=='complete' for s in state['arms'].values())
            save(root/'health.json',dict(at=now(),controller_pid=os.getpid(),stage='training',completed_arms=completed,
                arms=state['arms'],issues=issues))
            if completed==3:
                save(root/'COMPLETE.json',dict(at=now(),completed_arms=3));event(root,'all_three_training_arms_complete');return
            time.sleep(20)


def launch_controller(root):
    if (root/'USER_STOP_REQUEST.json').exists():return dict(status='stopped_by_user',launched=False,root=str(root))
    verify(root);p=root/'controller.pid'
    if p.exists():
        args=read_cmd(p.read_text().strip())
        if str(root/'code'/SCRIPT_NAME) in args and '--action' in args and args[args.index('--action')+1]=='controller':
            return dict(pid=int(p.read_text()),already_alive=True)
    with (root/'controller.log').open('a') as log:
        child=subprocess.Popen([PYTHON,str(root/'code'/SCRIPT_NAME),'--root',str(root),'--action','controller'],
             stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,cwd=REPO)
    p.write_text(str(child.pid)+'\n');return dict(pid=child.pid,root=str(root))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--action',choices=['inventory','prepare-hosts','controller','launch-controller','node','launch-node','stop-node'],required=True)
    p.add_argument('--arm',choices=['sft_answer','sft_observation_answer','clue_no_observation'])
    p.add_argument('--rank',type=int,default=0);p.add_argument('--phase',choices=['smoke','formal'],default='smoke');p.add_argument('--attempt',type=int,default=1)
    a=p.parse_args();root=a.root.resolve()
    if a.action=='inventory':print(json.dumps(inventory()))
    elif a.action=='prepare-hosts':prepare_hosts(root)
    elif a.action=='launch-controller':print(json.dumps(launch_controller(root)))
    elif a.action=='controller':controller(root)
    elif a.action=='launch-node':print(json.dumps(launch_node(root,a.arm,a.rank,a.phase,a.attempt)))
    elif a.action=='stop-node':print(json.dumps(stop_local(root,a.arm,a.phase,a.attempt)))
    else:
        try:node(root,a.arm,a.rank,a.phase,a.attempt)
        except Exception:
            name=load(root/'suite.json')['arms'][a.arm]['workers'][a.rank]
            folder=attempt_dir(root,a.arm,a.phase,a.attempt);folder.mkdir(parents=True,exist_ok=True)
            save(folder/(name+'.json'),dict(at=now(),status='failed',error=traceback.format_exc(),supervisor_pid=os.getpid()))
            raise


if __name__=='__main__':main()
