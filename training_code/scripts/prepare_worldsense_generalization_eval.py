#!/usr/bin/env python3
"""Freeze 500 external MCQs with the WorldSense training media budget."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
BASE = Path('/opt/huawei/dataset/hyl_ulan/ylhu')
sys.path.insert(0, str(REPO / 'training_code/src'))
from omni_opsd.data.dynamic_budget import dynamic_budget_for, dynamic_video_spec, dynamic_sampling_contract

INSTRUCTION = ('Briefly analyze the video and audio evidence in English using at most 120 words. '
               'Write your analysis inside <analysis>...</analysis>, then select exactly one option '
               'and write only its letter (A, B, C, or D) inside <answer>...</answer>.')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    path.write_text(body)

def freeze_jsonl(path, rows):
    body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    if path.exists() and path.read_text() != body:
        raise ValueError('Refusing to change frozen evaluation input: ' + str(path))
    path.write_text(body)

def seconds(text):
    out = 0.
    for part in str(text).rstrip('s').split(':'):
        out = out * 60 + float(part)
    return out

def sources(benchmark):
    if benchmark == 'omnivideobench':
        import pyarrow.parquet as pq
        root = BASE / 'omni-opsd-assets/datasets/OmniVideoBench'
        source = root / 'data.parquet'
        rows = [dict(case_id=f'omnivideobench_{i:04d}', video_id=x['video'], video_path=str(root/x['video']),
                     question=x['question'], choices=x['options'], answer=x['correct_option'],
                     question_type=x['question_type'], declared_duration=seconds(x['duration']), source_index=i)
                for i,x in enumerate(pq.read_table(source).to_pylist())]
    elif benchmark == 'dailyomni':
        root = BASE / 'OmniFold/data/benchmarks/DailyOmni'
        source = root / 'qa.json'
        rows = [dict(case_id=f'dailyomni_{i:04d}', video_id=x['video_id'],
                     video_path=str(root/'Videos'/x['video_id']/(x['video_id']+'_video.mp4')),
                     question=x['Question'], choices=x['Choice'], answer=x['Answer'],
                     question_type=x['Type'], declared_duration=seconds(x['video_duration']), source_index=i)
                for i,x in enumerate(json.loads(source.read_text()))]
    else:
        raise ValueError(benchmark)
    return source, rows

def probe(path):
    p = Path(path)
    try:
        if not p.is_file() or not p.stat().st_size:
            return dict(error='missing-or-empty-video')
        r = subprocess.run(['ffprobe','-v','error','-show_entries',
                            'stream=codec_type,width,height,duration:format=duration','-of','json',path],
                           capture_output=True, text=True, timeout=30)
        if r.returncode:
            return dict(error='ffprobe-error', detail=r.stderr[-500:])
        d = json.loads(r.stdout)
        video = next(x for x in d['streams'] if x['codec_type']=='video')
        audios = [x for x in d['streams'] if x['codec_type']=='audio']
        duration = max([float(d['format']['duration'])] +
                       [float(x['duration']) for x in d['streams'] if x.get('duration','N/A')!='N/A'])
        return dict(duration=duration, width=video['width'], height=video['height'],
                    audio_streams=len(audios), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
    except Exception as e:
        return dict(error=type(e).__name__, detail=str(e)[:500])

def prepare(root, benchmark, size, seed):
    out = root / 'data' / benchmark
    out.mkdir(parents=True, exist_ok=True)
    source, original = sources(benchmark)
    cache_path = out/'media_probe.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    paths = sorted({r['video_path'] for r in original if r['declared_duration'] <= 300})
    missing = [p for p in paths if p not in cache]
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(probe,p): p for p in missing}
        for f in as_completed(futures):
            cache[futures[f]] = f.result()
    write(cache_path, cache)
    excluded, eligible, seen = [], [], {}
    for row in original:
        reason = None
        choices = []
        for letter, text in zip('ABCD',row['choices']):
            choices.append(re.sub(r'^'+letter+r'(?:[.):]|\s)\s*','',str(text).strip()))
        if len(row['choices']) != 4 or row['answer'] not in 'ABCD' or len(row['answer']) != 1 or not all(choices):
            raise ValueError('Invalid four-choice source: '+row['case_id'])
        row = dict(row, choices=choices)
        if row['declared_duration'] > 300:
            reason='declared-duration-over-300s'
        else:
            media=cache[row['video_path']]
            if media.get('error'):reason=media['error']
            elif media['duration'] > 300:reason='actual-duration-over-300s'
            elif not media['audio_streams']:reason='no-audio-stream'
        identity=(row['video_id'],row['question'],tuple(choices))
        if not reason and identity in seen:
            if seen[identity]['answer'] != row['answer']:
                raise ValueError('Conflicting labels for identical input: '+row['case_id'])
            reason='duplicate-identical-question'
        if reason:
            excluded.append(dict(case_id=row['case_id'],reason=reason));continue
        seen[identity]=row;eligible.append(row)
    eligibility=dict(benchmark=benchmark, source=str(source), source_sha256=sha(source), source_rows=len(original),
                     eligible_unique_questions=len(eligible), excluded_counts=dict(Counter(r['reason'] for r in excluded)),
                     excluded=excluded, eligible_ids=[r['case_id'] for r in eligible])
    write(out/'eligibility.json',eligibility)
    print(json.dumps({k:v for k,v in eligibility.items() if k not in ['excluded','eligible_ids']},ensure_ascii=False),flush=True)
    if len(eligible) < size:
        raise ValueError(f'{benchmark}: only {len(eligible)} eligible unique QA, requested {size}')
    selected=sorted(random.Random(seed).sample(eligible,size),key=lambda r:r['source_index'])
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(BASE/'omni-opsd-assets/models/Qwen2.5-Omni-7B/tokenizer.json'))
    inputs,labels,audit=[],[],[]
    train=REPO/'training_runs/worldsense_observation_sft_lora_20260930/data/sft_observation_openqa.jsonl'
    train_rows=[json.loads(x) for x in train.read_text().splitlines() if x.strip()]
    train_paths={str(Path(v['video']).resolve()) for r in train_rows for v in r['videos']}
    train_ids={r.get('video_id') for r in train_rows}
    for row in selected:
        media=cache[row['video_path']]
        if str(Path(row['video_path']).resolve()) in train_paths or row['video_id'] in train_ids:
            raise ValueError('Training media ID/path overlap: '+row['case_id'])
        dynamic=dict(row,duration=media['duration'],metadata={'resolution':f"{media['width']}x{media['height']}"})
        budget=dynamic_budget_for(dynamic); spec=dynamic_video_spec(dynamic,budget)
        contract=dynamic_sampling_contract(budget)
        contract.pop('teacher_view',None);contract.pop('teacher_frame_cap_total',None)
        contract.update(held_out_evaluation=True,audio_sample_rate=16000,audio_max_seconds=300,video_reader='pyav_seek')
        prompt='<video>\nQuestion: '+row['question']+'\nOptions:\n'+'\n'.join(f'{a}. {b}' for a,b in zip('ABCD',row['choices']))+'\n'+INSTRUCTION
        text_tokens=len(tokenizer.encode(prompt,add_special_tokens=False).ids)
        if text_tokens+512+128>2048:raise ValueError('Text reserve exceeded: '+row['case_id'])
        inputs.append(dict(case_id=row['case_id'],prompt_id=row['case_id'],video_id=row['video_id'],benchmark=benchmark,
                           question_type=row['question_type'],messages=[dict(role='user',content=prompt)],videos=[spec],
                           dynamic_student_budget=budget,sampling_contract=contract))
        labels.append(dict(sample_id=row['case_id'],video_id=row['video_id'],answer=row['answer'],
                           question=row['question'],choices=row['choices'],question_type=row['question_type']))
        audit.append(dict(case_id=row['case_id'],media=media,budget=budget,video=spec,prompt_tokens=text_tokens))
    freeze_jsonl(out/'inputs.jsonl',inputs);freeze_jsonl(out/'labels.jsonl',labels);freeze_jsonl(out/'media_budget_audit.jsonl',audit)
    manifest=dict(benchmark=benchmark,rows=size,seed=seed,source=str(source),source_sha256=sha(source),
                  eligible_unique_questions=len(eligible),unique_videos=len({r['video_id'] for r in selected}),
                  input_sha256=sha(out/'inputs.jsonl'),labels_sha256=sha(out/'labels.jsonl'),instruction=INSTRUCTION,
                  max_new_tokens=512,training_media_budget_source=str(REPO/'training_code/src/omni_opsd/data/dynamic_budget.py'),
                  training_media_budget_sha256=sha(REPO/'training_code/src/omni_opsd/data/dynamic_budget.py'),
                  max_duration=max(r['media']['duration'] for r in audit),media_scope='full-video-and-audio',
                  training_id_or_path_overlap=0,content_reencoding_overlap_not_audited=True,
                  duplicate_policy='deduplicate-identical-video-question-choices; conflicting-gold-fails')
    write(out/'manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--benchmark',nargs='+',choices=['omnivideobench','dailyomni'],required=True)
    p.add_argument('--sample-size',type=int,default=500);p.add_argument('--seed',type=int,default=20261003)
    a=p.parse_args()
    for b in a.benchmark:prepare(a.root.resolve(),b,a.sample_size,a.seed)
