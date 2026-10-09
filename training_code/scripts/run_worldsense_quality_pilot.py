#!/usr/bin/env python3
"""Prepare/run a fixed 100-question WorldSense MCQ verification/repair pilot."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tarfile
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'training_code/src'))
from omni_opsd.worldsense.quality_gate import MODEL, VERSION, digest, probe, process_item, write_json


def read_rows(path):
    return [json.loads(s) for s in Path(path).open() if s.strip()]


def prepare(args):
    root = args.root
    root.mkdir(parents=True,exist_ok=True)
    path = root/'samples.jsonl'
    if (root/'manifest.json').exists():
        manifest = json.loads((root/'manifest.json').read_text())
        if manifest['sample_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('Frozen samples changed')
        if manifest['seed'] != args.seed or manifest['sample_size'] != args.sample_size:
            raise ValueError('Existing frozen sample configuration differs')
        print(json.dumps({'prepared':True,'sample_size':manifest['sample_size'],'root':str(root)}),flush=True)
        return
    source = REPO/'training_runs/worldsense_observation_sft_lora_20260930/data/sft_observation_openqa.jsonl'
    training = read_rows(source)
    train_ids = {r.get('case_id',r.get('prompt_id')) for r in training}
    assert len(train_ids)==1453
    candidates = {r['sample_id']:r for r in read_rows(REPO/'data/screening/candidates.jsonl')}
    annotations = {r['question_id']:r for r in read_rows(REPO/'data/annotation/merged.evidence.jsonl')}
    selected = random.Random(args.seed).sample(sorted(train_ids),args.sample_size)
    trace_captions = {}
    with tarfile.open(REPO/'data/annotation/traces.tar.gz') as archive:
        for member in archive.getmembers():
            if not member.isfile():continue
            for line in archive.extractfile(member):
                if not line.strip():continue
                row = json.loads(line)
                if row['question_id'] not in selected:continue
                matches = [e for e in row.get('events',[]) if e.get('caption')]
                if matches:
                    event = matches[-1]
                    trace_captions[row['question_id']] = {k:event.get(k) for k in ['caption','caption_valid','caption_truncated','caption_key']}
                    trace_captions[row['question_id']]['source'] = 'data/annotation/traces.tar.gz/'+member.name
    def build(qid):
        c = dict(candidates[qid]);video=args.video_root/(c['video_id']+'.mp4')
        c['video_path']=str(video.resolve());c['video_sha256']=hashlib.sha256(video.read_bytes()).hexdigest()
        c['media']=probe(video)
        if c['media']['duration']>300:raise ValueError(qid+': source video/audio over 300 seconds')
        if not 2<=len(c['choices'])<=8 or c['answer'] not in 'ABCDEFGH'[:len(c['choices'])]:raise ValueError('Invalid MCQ')
        c['annotation']=annotations[qid]
        c['caption_candidate']=trace_captions.get(qid)
        return c
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(build,selected))
    # Explicitly keep the pilot disjoint from the frozen 518-question evaluation videos.
    heldout = read_rows(REPO/'training_runs/worldsense_training_matched_eval_20261003/data/worldsense.labels.jsonl')
    heldout_videos={r['sample_id'].split('::')[0] for r in heldout}
    assert not ({r['video_id'] for r in rows}&heldout_videos)
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    path.write_text(payload)
    manifest={'version':VERSION,'model':MODEL,'seed':args.seed,'sample_size':len(rows),
              'sample_sha256':hashlib.sha256(payload.encode()).hexdigest(),
              'sampling_frame':'1453 questions actually used in observation training; fixed random sample',
              'training_data_source':str(source),'question_ids':selected,
              'task_type_counts':dict(Counter(r['question_type'] for r in rows)),
              'unique_videos':len({r['video_id'] for r in rows}),
              'maximum_actual_duration':max(r['media']['duration'] for r in rows),
              'evaluation_video_overlap':0,'choice_mode':True,'gold_labels_changed':False,
              'gate1':'all original clue intervals together; no gold, observation, or caption',
              'gate2':'full-video observation claims and reference consistency; only after correct gate1',
              'repair':'reuse unverified caption; full AV timestamped facts; new intervals; independent clue answer; concise observation; re-audit',
              'max_repair_attempts':args.max_repairs,'human_audit':'not performed; export is auto-verified candidates, not automatic training admission',
              'files_sha256':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 [REPO/'training_code/src/omni_opsd/worldsense/quality_gate.py',Path(__file__)]}}
    write_json(root/'manifest.json',manifest)
    print(json.dumps({'prepared':True,'samples':len(rows),'videos':manifest['unique_videos'],
                      'max_seconds':manifest['maximum_actual_duration'],'root':str(root)}),flush=True)


def configure_credentials(env_file):
    if env_file:
        # Parse only these named settings; never execute shell configuration.
        for line in Path(env_file).read_text().splitlines():
            line=line.strip()
            if line.startswith('export '):line=line[7:].strip()
            if not line or line.startswith('#') or '=' not in line:continue
            name,value=line.split('=',1)
            if name.strip() in ['DASHSCOPE_API_KEY','DASHSCOPE_BASE_URL']:
                os.environ[name.strip()]=value.strip().strip('\"\'')
    missing=[k for k in ['DASHSCOPE_API_KEY','DASHSCOPE_BASE_URL'] if not os.environ.get(k)]
    return missing


def run(args):
    from omni_opsd.worldsense.api_client import QwenAPIOmniClient
    from omni_opsd.worldsense.quality_gate import ReviewAPI
    root=args.root
    manifest=json.loads((root/'manifest.json').read_text())
    samples=root/'samples.jsonl'
    if manifest['sample_sha256']!=hashlib.sha256(samples.read_bytes()).hexdigest():
        raise ValueError('Frozen samples changed')
    for name,expected in manifest['files_sha256'].items():
        if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Code differs from frozen manifest: '+name)
    if args.max_repairs!=manifest['max_repair_attempts']:raise ValueError('Repair budget changed')
    missing=configure_credentials(args.env_file)
    if missing:
        write_json(root/'status.json',{'status':'blocked_credentials','missing_environment_variables':missing,
                                      'completed':0,'model':MODEL,'api_requests_started':0})
        print(json.dumps({'status':'blocked_credentials','missing':missing}),flush=True)
        return 2
    records=read_rows(samples)
    if args.limit:records=records[:args.limit]
    results={}
    for item in records:
        path=root/'items'/item['sample_id'].replace('::','__')/'result.json'
        if path.exists():
            result=json.loads(path.read_text())
            if result['status']!='technical_error':results[item['sample_id']]=result
    def work(item):
        folder=root/'items'/item['sample_id'].replace('::','__');folder.mkdir(parents=True,exist_ok=True)
        if hashlib.sha256(Path(item['video_path']).read_bytes()).hexdigest()!=item['video_sha256']:
            raise ValueError('Original source video changed')
        client=QwenAPIOmniClient(model=MODEL,caption_reuse=False,clip_cache_dir=root/'media',request_timeout=180,max_retries=2)
        api=ReviewAPI(client,folder/'requests')
        try:
            result=process_item(item,api,root/'media'/item['sample_id'].replace('::','__'),folder,args.max_repairs)
        except Exception as exc:
            result={'question_id':item['sample_id'],'status':'technical_error',
                    'error_type':type(exc).__name__,'annotation':None,'human_verified':False}
        write_json(folder/'result.json',result)
        return result
    def save():
        ordered=[results[r['sample_id']] for r in records if r['sample_id'] in results]
        (root/'results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in ordered))
        passed=[r for r in ordered if r['status'] in ['retained_original','reannotated_verified']]
        (root/'auto_verified_candidates.jsonl').write_text(''.join(json.dumps(r['annotation'],ensure_ascii=False)+'\n' for r in passed))
        counts=dict(Counter(r['status'] for r in ordered))
        status={'status':'complete' if len(ordered)==len(records) else 'running',
                'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'pid':os.getpid(),'target':len(records),
                'completed':len(ordered),'counts':counts,'model':MODEL,'max_repairs':args.max_repairs,
                'human_verified':False,'source_annotations_unchanged':True}
        if len(ordered)==len(records) and counts.get('technical_error'):status['status']='incomplete_technical_errors'
        write_json(root/'status.json',status)
        print(json.dumps(status),flush=True)
    save()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(work,r):r['sample_id'] for r in records if r['sample_id'] not in results}
        for future in as_completed(futures):
            try:results[futures[future]]=future.result()
            except Exception as exc:results[futures[future]]={'question_id':futures[future],'status':'technical_error','error_type':type(exc).__name__,'annotation':None}
            save()
    return 1 if any(r['status']=='technical_error' for r in results.values()) else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--action',choices=['prepare','run'],required=True)
    p.add_argument('--video-root',type=Path,default=Path('/opt/huawei/dataset/hyl_ulan/ylhu/OmniFold/data/benchmarks/WorldSense/videos'))
    p.add_argument('--sample-size',type=int,default=100)
    p.add_argument('--seed',type=int,default=20261003)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--max-repairs',type=int,default=2)
    p.add_argument('--env-file',type=Path)
    p.add_argument('--limit',type=int,help='Run only first N frozen samples for smoke validation')
    args=p.parse_args();args.root=args.root.resolve();args.root.mkdir(parents=True,exist_ok=True)
    with (args.root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='prepare':prepare(args);return 0
        return run(args)


if __name__=='__main__':
    raise SystemExit(main())
