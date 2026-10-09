#!/usr/bin/env python3
"""Freeze and rerun exactly the 23 uncertainty-stopped pilot questions."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'training_code/src'))
from omni_opsd.worldsense import quality_gate as q
from omni_opsd.worldsense import quality_gate_v2 as v2
from run_worldsense_quality_pilot import configure_credentials, read_rows


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))


def code_paths():
    return [Path(__file__).resolve(), REPO/'training_code/scripts/run_worldsense_quality_pilot.py',
            REPO/'training_code/src/omni_opsd/worldsense/quality_gate.py',
            REPO/'training_code/src/omni_opsd/worldsense/quality_gate_v2.py']


def verify(root):
    manifest = json.loads((root/'manifest.json').read_text())
    for path, expected in manifest['frozen_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Frozen source/code changed: '+path)
    if sha(root/'samples.jsonl') != manifest['sample_sha256']:
        raise ValueError('Rerun samples changed')
    return manifest


def prepare(args):
    root, source = args.root, args.source
    if (root/'manifest.json').exists():
        verify(root); return
    original = read_rows(source/'results.jsonl')
    selected = [r for r in original if r['status']=='needs_review' and r['checks'][-1]['reason']=='unresolved_facts']
    assert len(original)==50 and len(selected)==23
    ids = [r['question_id'] for r in selected]
    smap = {r['sample_id']: r for r in read_rows(source/'samples.jsonl')}
    samples = [smap[k] for k in ids]
    assert len(set(ids))==23 and max(r['media']['duration'] for r in samples)<=300
    jsonl(root/'samples.jsonl', samples)
    jsonl(root/'previous_results.jsonl', selected)
    sources = [source/'samples.jsonl', source/'results.jsonl', source/'auto_verified_candidates.jsonl']
    manifest = dict(version=v2.VERSION, model=q.MODEL, source=str(source), question_ids=ids,
                    sample_size=23, max_repairs=2, temperature=0., workers=args.workers,
                    scope='Only original terminal unresolved_facts; remaining 27 pilot questions untouched',
                    original_results_sha256=sha(source/'results.jsonl'), sample_sha256=sha(root/'samples.jsonl'),
                    frozen_sha256={str(p): sha(p) for p in sources+code_paths()+[root/'previous_results.jsonl']},
                    human_reviewed=False, automatically_update_training_data=False,
                    focused_media='AV <=60 s; prefer 4fps/1280x720 area, payload fallback recorded; full context retained',
                    uncertainty_policy='Independent answer-relevance triage; blocking -> focused AV -> full facts -> triage; up to 2 repair rounds; both gates mandatory')
    q.write_json(root/'manifest.json', manifest)
    # Reuse validated immutable encodings in the NEW directory; never write
    # cache metadata or temporary files to the old pilot directory.
    for sample_id in ids:
        dirname = sample_id.replace('::','__')
        target = root/'media'/dirname; target.mkdir(parents=True, exist_ok=True)
        for path in (source/'media'/dirname).glob('*'):
            if path.suffix not in ['.mp4','.json'] or '.tmp.' in path.name:
                continue
            destination = target/path.name
            if not destination.exists():
                try: os.link(path, destination)
                except OSError: shutil.copy2(path, destination)
    print(json.dumps({'prepared':True, 'count':23, 'root':str(root)}), flush=True)


def run(args):
    root = args.root
    manifest = verify(root)
    missing = configure_credentials(args.env_file)
    if missing:
        raise ValueError('Missing credential setting names: '+','.join(missing))
    from omni_opsd.worldsense.api_client import QwenAPIOmniClient
    samples = read_rows(root/'samples.jsonl')
    previous = {r['question_id']:r for r in read_rows(root/'previous_results.jsonl')}
    original = read_rows(Path(manifest['source'])/'results.jsonl')
    results = {}
    for item in samples:
        saved = root/'items'/item['sample_id'].replace('::','__')/'result.json'
        if saved.exists():
            row = json.loads(saved.read_text())
            if row['status']!='technical_error': results[item['sample_id']] = row
    def save():
        ordered = [results[r['sample_id']] for r in samples if r['sample_id'] in results]
        counts = dict(Counter(r['status'] for r in ordered))
        state = 'running' if len(ordered)<23 else ('incomplete_technical_errors' if counts.get('technical_error') else 'complete')
        jsonl(root/'results.jsonl', ordered)
        jsonl(root/'auto_verified_candidates.jsonl', [r['annotation'] for r in ordered if r['status']=='reannotated_verified'])
        combined = [results.get(r['question_id'], r) for r in original]
        jsonl(root/'pilot50_combined_candidates.jsonl', [r['annotation'] for r in combined if r['status'] in ['retained_original','reannotated_verified']])
        status = dict(status=state, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), pid=os.getpid(),
                      target=23, completed=len(ordered), counts=counts, version=v2.VERSION,
                      original_50_combined_counts=dict(Counter(r['status'] for r in combined)),
                      combined_is_final=state=='complete', human_verified=False)
        q.write_json(root/'status.json', status)
        with (root/'review.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['question_id','status','rounds','last_reason','detail']); writer.writeheader()
            for row in ordered:
                last = row.get('checks',[{}])[-1]
                writer.writerow(dict(question_id=row['question_id'],status=row['status'],rounds=row.get('reannotation_attempts'),
                                     last_reason=last.get('reason',row.get('error_type')),
                                     detail=json.dumps(last,ensure_ascii=False)))
        print(json.dumps(status), flush=True)
    def work(item):
        folder=root/'items'/item['sample_id'].replace('::','__'); folder.mkdir(parents=True,exist_ok=True)
        try:
            if sha(item['video_path'])!=item['video_sha256']: raise ValueError('Source video changed')
            client=QwenAPIOmniClient(model=q.MODEL,caption_reuse=False,clip_cache_dir=root/'media',request_timeout=180,max_retries=2)
            result=v2.process_item(item,previous[item['sample_id']],q.ReviewAPI(client,folder/'requests'),
                                   root/'media'/item['sample_id'].replace('::','__'),max_repairs=2)
        except Exception as exc:
            # Do not persist exception messages that may contain API secrets.
            result=dict(question_id=item['sample_id'],status='technical_error',error_type=type(exc).__name__,
                        annotation=None,human_verified=False,version=v2.VERSION)
        q.write_json(folder/'result.json',result)
        return result
    save()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(work,r):r['sample_id'] for r in samples if r['sample_id'] not in results}
        for future in as_completed(futures):
            results[futures[future]]=future.result(); save()
    verify(root)
    return 1 if any(r['status']=='technical_error' for r in results.values()) else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--action',choices=['prepare','run'],required=True)
    p.add_argument('--env-file',type=Path)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args(); args.root=args.root.resolve(); args.source=args.source.resolve()
    if args.root==args.source: raise ValueError('Use a separate rerun directory')
    args.root.mkdir(parents=True,exist_ok=True)
    with (args.root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return prepare(args) if args.action=='prepare' else run(args)


if __name__=='__main__':
    raise SystemExit(main())
