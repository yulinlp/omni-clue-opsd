#!/usr/bin/env python3
"""Full <=300 s WorldSense original-clue verification followed by agentic repairs."""
import argparse, collections, concurrent.futures as cf, fcntl, hashlib, json, os, pathlib, re, sys, tarfile, threading, time, traceback
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'training_code/src'),str(ROOT/'training_runs/worldsense_mcq_annotation_pilot100_20261003/api_deps')]
from omni_opsd.worldsense import quality_gate as q
from omni_opsd.worldsense import clue_only_agentic as c
from run_worldsense_quality_pilot import configure_credentials

def rows(p):return [json.loads(l) for l in pathlib.Path(p).open() if l.strip()]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def jsonl(p,rs):
 tmp=p.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs));os.replace(tmp,p)
def code_files():return [pathlib.Path(__file__).resolve(),pathlib.Path(c.__file__),pathlib.Path(q.__file__),ROOT/'training_code/src/omni_opsd/worldsense/api_client.py',ROOT/'training_code/scripts/run_worldsense_quality_pilot.py']

def prepare(args):
 root=args.root
 if (root/'manifest.json').exists():verify(root);return
 qa_path=args.video_root.parent/'worldsense_qa.json';qa=json.loads(qa_path.read_text())
 annotation_path=ROOT/'data/annotation/merged.evidence.jsonl';annotations={r['question_id']:r for r in rows(annotation_path)}
 inventory=json.loads((root/'media_inventory.json').read_text())
 training_ids={r['case_id'] for r in rows(ROOT/'training_runs/worldsense_observation_sft_lora_20260930/data/sft_observation_openqa.jsonl')}
 heldout={r['sample_id'] for r in rows(ROOT/'training_runs/worldsense_training_matched_eval_20261003/data/worldsense.labels.jsonl')}
 captions={}
 for r in rows(ROOT/'data/annotation/captions.jsonl'):
  if r.get('caption'):captions[r['question_id']]=dict(caption=r['caption'],source='data/annotation/captions.jsonl',verified=False)
 # Prefer the exact last bootstrap caption from the archived per-question episode.
 with tarfile.open(ROOT/'data/annotation/traces.tar.gz') as archive:
  for member in archive:
   if not member.isfile():continue
   for line in archive.extractfile(member):
    r=json.loads(line)
    for event in r.get('events',[]):
     if event.get('kind')=='bootstrap' and event.get('caption'):
      captions[r['question_id']]=dict(caption=event['caption'],source='data/annotation/traces.tar.gz/'+member.name,verified=False,
                                    caption_valid=event.get('caption_valid'),caption_truncated=event.get('caption_truncated'))
 selected=[];excluded=[]
 for vid,entry in sorted(qa.items()):
  meta=inventory[vid]
  for task,record in sorted(entry.items()):
   if not task.startswith('task') or not isinstance(record,dict):continue
   qid=vid+'::'+task
   if 'error_type' in meta or meta['duration']>300:
    excluded.append(dict(question_id=qid,video_id=vid,reason='probe_error' if 'error_type' in meta else 'source_over_300_seconds',media=meta));continue
   choices=[re.sub(r'^[A-D][.)]\s*','',v).strip() for v in record['candidates']]
   if len(choices) not in (3,4) or record['answer'] not in 'ABCD'[:len(choices)]:raise ValueError('Invalid question schema '+qid)
   selected.append(dict(sample_id=qid,video_id=vid,video_path=str(args.video_root/(vid+'.mp4')),media=meta,
    question=record['question'],choices=choices,answer=record['answer'],question_type=record.get('task_type'),
    annotation=annotations.get(qid),caption_candidate=captions.get(qid),
    split_provenance={'previous_training_1453':qid in training_ids,'previous_heldout_518':qid in heldout},
    duplicate_option_texts=len(set(x.lower().rstrip('.') for x in choices))<len(choices)))
 assert len(selected)+len(excluded)==3172
 jsonl(root/'samples.jsonl',selected);jsonl(root/'excluded.jsonl',excluded)
 frozen=[qa_path,annotation_path,ROOT/'data/annotation/captions.jsonl',ROOT/'data/annotation/traces.tar.gz',root/'media_inventory.json',root/'samples.jsonl',root/'excluded.jsonl',args.tokenizer,ROOT/'training_runs/worldsense_observation_sft_lora_20260930/data/sft_observation_openqa.jsonl',ROOT/'training_runs/worldsense_training_matched_eval_20261003/data/worldsense.labels.jsonl']+code_files()
 manifest=dict(version=c.VERSION,model=c.MODEL,total_questions=3172,eligible_questions=len(selected),
  excluded_questions=len(excluded),eligible_videos=len({r['video_id'] for r in selected}),
  excluded_videos=len({r['video_id'] for r in excluded}),max_source_seconds=max(r['media']['duration'] for r in selected),
  original_status_counts=dict(collections.Counter((r['annotation'] or {}).get('status','missing') for r in selected)),
  missing_captions=sum(not r['caption_candidate'] for r in selected),duplicate_option_rows=sum(r['duplicate_option_texts'] for r in selected),
  tokenizer=str(args.tokenizer),token_limit_contract='API max_tokens=120 for the ENTIRE XML reply; analysis also checked with local Qwen2.5 tokenizer <=120',
  repair_budgets={'max_inspects':10,'max_checks':3,'max_turns':24},
  gate_inputs='selected AV + question/options only; no gold, caption, observation, or error history',
  repair_inputs='question/options + unverified timestamped caption + exact failed answer/analysis + requested AV views; correct answer withheld',
  observation_verification=False,automatic_training_admission=False,training_and_evaluation_splits_preserved=True,
  frozen_sha256={str(p):sha(p) for p in frozen})
 q.write_json(root/'manifest.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k!='frozen_sha256'},ensure_ascii=False),flush=True)

def verify(root):
 m=json.loads((root/'manifest.json').read_text())
 for p,h in m['frozen_sha256'].items():
  if sha(p)!=h:raise ValueError('Frozen file changed: '+p)
 return m

def run(args):
 root=args.root;manifest=verify(root)
 missing=configure_credentials(args.env_file)
 if missing:raise RuntimeError('Missing credential setting names: '+','.join(missing))
 from omni_opsd.worldsense.api_client import QwenAPIOmniClient
 from tokenizers import Tokenizer
 tokenizer=Tokenizer.from_file(manifest['tokenizer']);count=lambda text:len(tokenizer.encode(text,add_special_tokens=False).ids)
 q.encoder_binary()
 samples=rows(root/'samples.jsonl')
 if args.ids:samples=[s for s in samples if s['sample_id'] in set(args.ids.split(','))]
 if args.limit:samples=samples[:args.limit]
 assert samples
 lock=threading.Lock();hashes=json.loads((root/'source_hashes.json').read_text()) if (root/'source_hashes.json').exists() else {}
 video_locks={r['video_id']:threading.Lock() for r in samples}
 stop=threading.Event()
 def setup(item):
  item=dict(item);src=pathlib.Path(item['video_path']);st=src.stat();expected=item['media']
  if st.st_size!=expected['bytes'] or st.st_mtime_ns!=expected['mtime_ns']:raise ValueError('Source media changed')
  with video_locks[item['video_id']]:
   if item['video_id'] not in hashes:
    digest=sha(src)
    with lock:
     hashes[item['video_id']]=digest;q.write_json(root/'source_hashes.json',hashes)
   item['video_sha256']=hashes[item['video_id']]
  folder=root/'items'/item['sample_id'].replace('::','__');folder.mkdir(parents=True,exist_ok=True)
  cache=root/'media'/item['sample_id'].replace('::','__')
  client=QwenAPIOmniClient(model=c.MODEL,caption_reuse=False,request_timeout=180,max_retries=2,clip_cache_dir=cache)
  return item,folder,cache,c.RecordedAPI(client,folder/'requests')
 def work(item,phase):
  folder=root/'items'/item['sample_id'].replace('::','__');folder.mkdir(parents=True,exist_ok=True)
  target=folder/(phase+'.json')
  if target.exists():
   old=json.loads(target.read_text())
   if old['status'] not in ['technical_error','interrupted']:return old
  if stop.is_set():return dict(question_id=item['sample_id'],status='interrupted')
  api=None
  try:
   item,folder,cache,api=setup(item)
   if phase=='original':
    ann=item.get('annotation') or {}
    try:spans=q.intervals_checked(ann.get('clue_intervals'),item['media']['duration'])
    except (ValueError,KeyError,TypeError):
     result=dict(question_id=item['sample_id'],status='original_unavailable',check={'passed':False,'reason':'Original annotation has no usable clue intervals; no answer was fabricated.'})
    else:
     check=c.blind_verify(item,spans,api,cache,count)
     result=dict(question_id=item['sample_id'],status='original_clue_pass' if check['passed'] else 'original_clue_fail',check=check)
   else:
    original=json.loads((folder/'original.json').read_text())
    assert original['status'] in ['original_clue_fail','original_unavailable']
    result=dict(question_id=item['sample_id'],**c.repair(item,original['check'],api,cache,count))
   result.update(observation_checked=False,human_verified=False,source_video_sha256=item['video_sha256'],version=c.VERSION,
                 completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
  except Exception as exc:
   result=dict(question_id=item['sample_id'],status='content_blocked' if getattr(exc,'kind','')=='content_blocked' else 'technical_error',
    error_type=type(exc).__name__,error_kind=getattr(exc,'kind',None),http_status=getattr(exc,'http_status',None),
    observation_checked=False,human_verified=False,version=c.VERSION,
    error_location=[dict(file=f.filename,line=f.lineno,function=f.name) for f in traceback.extract_tb(exc.__traceback__)])
   if result['http_status'] in (401,403) and result['status']!='content_blocked':stop.set()
  finally:
   if api is not None:api.client.client.close()
  q.write_json(target,result);return result
 def phase_run(phase,records):
  started=time.time();results={}
  for r in records:
   p=root/'items'/r['sample_id'].replace('::','__')/(phase+'.json')
   if p.exists():results[r['sample_id']]=json.loads(p.read_text())
  start_completed=sum(r['status'] not in ['technical_error','interrupted'] for r in results.values())
  def save(state='running'):
   ordered=[results[s['sample_id']] for s in records if s['sample_id'] in results]
   terminal=sum(r['status'] not in ['technical_error','interrupted'] for r in ordered);new=terminal-start_completed
   eta=(time.time()-started)*(len(records)-terminal)/new if new>0 else None
   status=dict(phase=phase,status=state,target=len(records),completed=len(ordered),terminal=terminal,
      counts=dict(collections.Counter(r['status'] for r in ordered)),elapsed_seconds=round(time.time()-started,1),
      eta_seconds=round(eta) if eta is not None else None,workers=args.workers,pid=os.getpid(),
      updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),observation_checked=False)
   q.write_json(root/(phase+'_status.json'),status);q.write_json(root/'status.json',status)
   jsonl(root/(phase+'_results.jsonl'),ordered)
   print(json.dumps(status,ensure_ascii=False),flush=True)
  save()
  for retry_pass in range(3):
   todo=[s for s in records if s['sample_id'] not in results or results[s['sample_id']]['status'] in ['technical_error','interrupted']]
   if not todo or stop.is_set():break
   with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
    pending={pool.submit(work,s,phase):s for s in todo}
    last_save=0
    while pending:
     done,_=cf.wait(pending,timeout=15,return_when=cf.FIRST_COMPLETED)
     for future in done:
      item=pending.pop(future)
      try:results[item['sample_id']]=future.result()
      except Exception as exc:
       result=dict(question_id=item['sample_id'],status='technical_error',error_type=type(exc).__name__)
       results[item['sample_id']]=result
       q.write_json(root/'items'/item['sample_id'].replace('::','__')/(phase+'.json'),result)
     if time.time()-last_save>=15 or not pending:save();last_save=time.time()
   if retry_pass<2 and any(r['status']=='technical_error' for r in results.values()):time.sleep(5)
  save('complete' if len(results)==len(records) and not any(r['status'] in ['technical_error','interrupted'] for r in results.values()) else 'incomplete_technical_errors')
  return results
 if args.phase in ['original','all']:originals=phase_run('original',samples)
 else:
  originals={r['sample_id']:json.loads((root/'items'/r['sample_id'].replace('::','__')/'original.json').read_text()) for r in samples}
 if args.phase in ['repair','all'] and not stop.is_set():
  todo=[r for r in samples if originals.get(r['sample_id'],{}).get('status') in ['original_clue_fail','original_unavailable']]
  phase_run('repair',todo)
 summarize(root,samples)
 return 2 if stop.is_set() else 0

def summarize(root,samples=None):
 samples=samples or rows(root/'samples.jsonl');all_results=[];passed=[]
 for s in samples:
  folder=root/'items'/s['sample_id'].replace('::','__');orig=folder/'original.json';repair=folder/'repair.json'
  if not orig.exists():continue
  o=json.loads(orig.read_text());r=json.loads(repair.read_text()) if repair.exists() else None
  final=r or o
  state=final['status']
  if state in ['original_clue_fail','original_unavailable']:state='pending_agentic_repair'
  all_results.append(dict(question_id=s['sample_id'],status=state,original=o,repair=r,split_provenance=s['split_provenance']))
  if state in ['original_clue_pass','repaired_clue_pass']:
   check=o['check'] if state=='original_clue_pass' else r['checks'][-1]
   passed.append(dict(question_id=s['sample_id'],video_id=s['video_id'],clue_intervals=check['clue_intervals'],
    verification_analysis=check['analysis'],verification_answer=check['answer'],gold_answer=s['answer'],
    status=state,observation_checked=False,human_verified=False,split_provenance=s['split_provenance'],
    automatic_training_admission=False))
 jsonl(root/'results.jsonl',all_results);jsonl(root/'clue_only_verified_candidates.jsonl',passed)
 summary=dict(target=len(samples),completed_records=len(all_results),counts=dict(collections.Counter(r['status'] for r in all_results)),
              verified_clue_candidates=len(passed),observation_checked=False,updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
 q.write_json(root/'summary.json',summary)
 print(json.dumps(summary),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--root',type=pathlib.Path,default=ROOT/'training_runs/worldsense_clue_only_agentic_20261004')
 p.add_argument('--action',choices=['prepare','run','summarize'],required=True)
 p.add_argument('--phase',choices=['original','repair','all'],default='all')
 p.add_argument('--workers',type=int,default=24);p.add_argument('--limit',type=int);p.add_argument('--ids')
 p.add_argument('--video-root',type=pathlib.Path,default=pathlib.Path('/opt/huawei/dataset/hyl_ulan/ylhu/OmniFold/data/benchmarks/WorldSense/videos'))
 p.add_argument('--tokenizer',type=pathlib.Path,default=pathlib.Path('/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-assets/models/Qwen2.5-Omni-7B/tokenizer.json'))
 p.add_argument('--env-file',type=pathlib.Path,default=ROOT/'.env')
 args=p.parse_args();args.root=args.root.resolve();args.root.mkdir(parents=True,exist_ok=True)
 if args.action=='summarize':summarize(args.root);return 0
 with (args.root/'runner.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if args.action=='prepare':prepare(args);return 0
  return run(args)
if __name__=='__main__':raise SystemExit(main())
