#!/usr/bin/env python3
"""Freeze the previous 2x500 videos for answer-free open QA on both clusters."""
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil

import pyarrow.parquet as pq
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'training_runs/generalization_openqa_npu120_20261003'
PRIOR = REPO / 'training_runs/worldsense_generalization_mcq500_npu96_20261003'
INSTRUCTION = ('Briefly analyze the video and audio evidence in English using at most 120 words. '
               'Write your analysis inside <analysis>...</analysis>, then give the answer in natural language '
               'inside <answer>...</answer>. Do not output an option letter.')
REWRITES = {
    'omnivideobench_0012': 'What is expressed between the second and third times the person in black speaks?',
    'omnivideobench_0248': 'Describe the sequence of steps a Duduk player might take when dealing with a closed reed during dry winter months.',
    'omnivideobench_0282': 'When Nick said that no snakes had come to Zootopia for a long time, what was the number on the drink closest to Nick?',
    'omnivideobench_0716': "Who is 'Uncle Dude'?",
    'omnivideobench_0535': 'In the video, which game sound effect sounds the most distinctive?',
}
# Set before any predictions. These references need the omitted candidate set
# to define an absence, ranking, or selected sequence; do not blame the model.
DEPENDENT = {f'omnivideobench_{n:04d}' for n in [11,19,27,29,30,40,503,535,629,732,851]}
DEPENDENT |= {f'dailyomni_{n:04d}' for n in [412,535,632]}
REFERENCE_CONFLICTS = {f'omnivideobench_{n:04d}' for n in [633,822,857,881,983]}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def freeze(p, body):
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.read_text() != body:
        raise ValueError('Refusing to change frozen data: '+str(p))
    p.write_text(body)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    source_path = Path('/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-assets/datasets/OmniVideoBench/data.parquet')
    omni = pq.read_table(source_path).to_pylist()
    tok = Tokenizer.from_file('/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-assets/models/Qwen2.5-Omni-7B/tokenizer.json')
    summary = {}
    for bench in ['omnivideobench','dailyomni']:
        previous = PRIOR/'data'/bench
        old_inputs, old_labels = read(previous/'inputs.jsonl'), read(previous/'labels.jsonl')
        labelmap = {x['sample_id']: x for x in old_labels}
        inputs, labels, audits = [], [], []
        for old in old_inputs:
            sid = old['case_id']; label = labelmap[sid]
            choices = label['choices']
            option_gold = choices['ABCD'.index(label['answer'])]
            gold = omni[int(sid.split('_')[-1])]['answer'].strip() if bench=='omnivideobench' else option_gold
            original_gold = gold
            gold_source = 'original-natural-answer' if bench=='omnivideobench' else 'correct-option-content'
            if not gold:
                gold=option_gold;gold_source='correct-option-content-fallback-empty-original'
            q=REWRITES.get(sid,label['question'])
            prompt='<video>\nQuestion: '+q+'\n'+INSTRUCTION
            assert len(tok.encode(prompt,add_special_tokens=False).ids)+512+128 <= 2048
            row=copy.deepcopy(old);row['messages']=[{'role':'user','content':prompt}]
            assert row['videos']==old['videos'] and row['dynamic_student_budget']==old['dynamic_student_budget']
            assert row['sampling_contract']==old['sampling_contract']
            assert row['videos'][0]['video_end']<=300
            assert not {'answer','gold_answer_text','teacher_prompt','teacher_videos','observation'} & row.keys()
            issues=[]
            if sid in DEPENDENT:issues.append('question requires omitted candidate set or has underdetermined open-ended reference')
            if sid in REFERENCE_CONFLICTS:issues.append('original natural answer conflicts with the correct option content')
            labels.append(dict(sample_id=sid,video_id=old['video_id'],benchmark=bench,tier='external-benchmark',
                               question=q,original_question=label['question'],choices=choices,
                               question_type=label['question_type'],gold_answer_text=gold,
                               original_natural_answer=original_gold,correct_option_content=option_gold,
                               reference_source=gold_source,options_dependent_wording=sid in DEPENDENT,
                               openqa_reference_issues=issues))
            inputs.append(row)
            audits.append(dict(sample_id=sid,question_rewritten=q!=label['question'],
                               reference_issues=issues,media_identical_to_mcq=True,
                               reference_source=gold_source,original_natural_answer=original_gold,
                               correct_option_content=option_gold))
        assert len(inputs)==len(labels)==500 and len({x['case_id'] for x in inputs})==500
        folder=ROOT/'data'/bench
        for name,rows in [('inputs.jsonl',inputs),('labels.jsonl',labels),('conversion_audit.jsonl',audits)]:
            freeze(folder/name,''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
        manifest=dict(rows=500,input_sha256=sha(folder/'inputs.jsonl'),labels_sha256=sha(folder/'labels.jsonl'),
                      previous_inputs_sha256=sha(previous/'inputs.jsonl'),same_500_ids=True,media_unchanged=True,
                      instruction=INSTRUCTION,reference_insufficient_count=sum(bool(x['reference_issues']) for x in audits),
                      question_rewrites=sum(x['question_rewritten'] for x in audits),max_new_tokens=512,
                      original_omni_source_sha256=sha(source_path) if bench=='omnivideobench' else None)
        freeze(folder/'manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        summary[bench]=manifest
    for name in ['tasks.json','inference_cache_views.json']:
        source = PRIOR/name if name=='tasks.json' else REPO/'training_runs/worldsense_training_matched_eval_20261003'/name
        # The MCQ tasks are duplicated across benchmarks; use the 13-model source.
        if name=='tasks.json':source=REPO/'training_runs/worldsense_training_matched_eval_20261003/tasks.json'
        freeze(ROOT/name,source.read_text())
    code=ROOT/'code';code.mkdir(exist_ok=True)
    for name in ['score_worldsense_openqa.py','score_worldsense_openqa_v2.py','score_external_openqa.py',
                 'worker_worldsense_card_pool.py','generalization_openqa_fleet.py']:
        source=REPO/'training_code/scripts'/name
        if source.exists():freeze(code/name,source.read_text())
    freeze(ROOT/'data_summary.json',json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
