#!/usr/bin/env python3
"""Freeze eight experiments using the selected 1500 and latest observation edits."""
import copy
import hashlib
import json
import math
import random
from pathlib import Path
import re
import sys
from collections import Counter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'training_code/src'))
sys.path.insert(0, str(REPO/'training_code/scripts'))
from omni_opsd.data.dynamic_budget import dynamic_budget_for, dynamic_clue_budget_for, dynamic_video_spec, dynamic_sampling_contract
from prepare_worldsense_clue_opsd_npu import allocate_frames
ROOT = REPO/'training_runs/worldsense_mcq8_latest1500_20261004'
SOURCE = REPO/'training_runs/worldsense_train1500_ab810_d690_20261004/train_1500.canonical.jsonl'
OBS = REPO/'training_runs/worldsense_clue_only_agentic_20261004/samples.jsonl'
EDITS = [REPO/'training_runs/worldsense_train671_observation_review_20261004/rewrite35/results.jsonl',
         REPO/'training_runs/worldsense_train829_observation_rewrite_20261004_v2/results.jsonl']
ANALYSIS = ('Briefly analyze the video and audio evidence in no more than 120 words, then select the correct option. '
            'Return exactly <analysis>your concise analysis</analysis>\n<answer>A</answer>. '
            'Replace A with one uppercase option letter from the available options. Do not add text outside these tags.')
DISTILL_ANALYSIS = 'Briefly analyze the video and audio evidence in English using at most 120 words.\nPut ALL analysis inside <analysis>...</analysis>.\nThen select exactly ONE option.\nBetween <answer> and </answer>, write ONLY ONE uppercase option letter: A, B, C, or D.\nDo NOT put analysis, explanations, option text, punctuation, or any other text inside <answer>...</answer>.\nAn analysis alone is incomplete: after </analysis>, you MUST write <answer>, your chosen letter, and </answer>.\nClose the answer with </answer> and do not write any text after it.'
ANSWER = ('Select the correct option. Output only <answer>A</answer>. '
          'Replace A with one uppercase option letter from the available options. Do not output analysis or any other text.')

def read(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def jl(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x))
def shorten(s):
    s=s.strip()
    if len(s.split())<=120:return s
    prefix=' '.join(s.split()[:120]);ends=list(re.finditer(r'[.!?](?=\s|$)',prefix))
    return prefix[:ends[-1].end()] if ends and ends[-1].end()>len(prefix)/2 else prefix.rstrip(',;:')+'.'

def main():
    assert not (ROOT/'suite.json').exists(), 'Frozen suite already exists'
    originals={r['sample_id']:r for r in read(OBS)}
    edits={r['sample_id']:r for p in EDITS for r in read(p)}
    specs={
      'sft_lora_answer':dict(workers=['npu96-worker-0'],tuner_type='lora',observation=False),
      'sft_lora_observation':dict(workers=['npu96-worker-1'],tuner_type='lora',observation=True),
      'sft_full_answer':dict(workers=['npu96-worker-2','npu96-worker-3'],tuner_type='full',observation=False),
      'sft_full_observation':dict(workers=['npu96-worker-4','npu96-worker-5'],tuner_type='full',observation=True),
      'clue_full_answer':dict(workers=['npu96-worker-6','npu96-worker-7'],tuner_type='full',observation=False),
      'clue_full_observation':dict(workers=['npu96-worker-8','npu96-worker-9'],tuner_type='full',observation=True),
      'opsd_full_answer':dict(workers=['npu96-worker-10','npu96-worker-11'],tuner_type='full',observation=False),
      'opsd_full_observation':dict(workers=['npu24-worker-0','npu24-worker-1','npu24-worker-2'],tuner_type='full',observation=True)}
    datasets={k:[] for k in specs};audit=[];merged=[]
    rows=read(SOURCE);assert len(rows)==len({r['sample_id'] for r in rows})==1500
    for r in rows:
        sid=r['sample_id'];gold=r['answer'];choices=r['choices']
        assert gold in 'ABCD'[:len(choices)] and 0<float(r['duration'])<=300 and Path(r['video_path']).is_file()
        old=originals[sid]['annotation'].get('observation','');edit=edits.get(sid)
        status=edit['status'] if edit else 'normal_original'
        latest=edit.get('observation','') if edit else old
        usable=status in ['rewritten','normal_original'] and bool(latest.strip())
        obs=shorten(latest) if usable else ''
        assert '<analysis>' not in obs and '<answer>' not in obs
        item=dict(sample_id=sid,status=status,observation_usable=usable,original_observation=old,
                  latest_observation=latest,training_observation=obs,original_words=len(latest.split()),used_words=len(obs.split()))
        audit.append(item);merged.append(dict(r,observation=latest,training_observation=obs,observation_usable=usable,observation_status=status))
        budgetrow=dict(r,metadata={'resolution':f"{r['source_width']}x{r['source_height']}"})
        budget=dynamic_budget_for(budgetrow);full=dynamic_video_spec(r,budget)
        cb=dynamic_clue_budget_for(budgetrow,r['evidence_spans'])
        caps=allocate_frames(cb['clue_intervals'],cb['nframes'])
        teacher=[dict(video=r['video_path'],video_start=a,video_end=b,nframes=n,
                      resized_height=cb['resized_height'],resized_width=cb['resized_width'],
                      min_pixels=3136,max_pixels=cb['resized_height']*cb['resized_width'])
                 for (a,b),n in zip(cb['clue_intervals'],caps)]
        qa=r['question']+'\nOptions:\n'+'\n'.join(f'{"ABCD"[i]}. {c}' for i,c in enumerate(choices))
        common=dict(case_id=sid,prompt_id=sid,video_id=r['video_id'],videos=[full],
                    dynamic_student_budget=budget,sampling_contract=dynamic_sampling_contract(budget),
                    source_option_letter=gold,observation_status=status,observation_usable=usable)
        for arm,s in specs.items():
            x=copy.deepcopy(common);sft=arm.startswith('sft_');use_obs=s['observation'] and usable
            instruction=(ANSWER if not s['observation'] else ANALYSIS) if sft else DISTILL_ANALYSIS
            x.update(messages=[dict(role='user',content='<video>\n'+qa+'\n'+instruction)],experiment_arm=arm)
            x['supervision_contract']=dict(answer_weight=3 if s['observation'] else 1,
                observation_supervised=use_obs,unresolved_observation_omitted=not usable,student_prompt_has_gold=False)
            if sft:
                completion=(f'<analysis>{obs}</analysis>\n' if use_obs else '')+f'<answer>{gold}</answer>'
                x['messages'].append(dict(role='assistant',content=completion))
            else:
                tv=teacher if arm.startswith('clue_') else [full]
                context=f'Correct option: {gold}. {choices["ABCD".index(gold)]}'
                if use_obs:context+='\nEvidence explanation: '+obs
                context+='\nUse the evidence to explain the answer. Do not mention the supplied correct option or evidence explanation.'
                x.update(teacher_videos=copy.deepcopy(tv),teacher_prompt='\n'.join(['<video>']*len(tv))+'\n'+qa+'\n'+context+'\n'+instruction,
                         solution=gold,gold_answer_text=choices['ABCD'.index(gold)],dynamic_teacher_budget=cb if arm.startswith('clue_') else budget)
                x['sampling_contract']['teacher_view']='golden-clue' if arm.startswith('clue_') else 'full-video-uniform'
                x['supervision_contract'].update(lmbda=1.,sft_alpha=0.,student_rollout_only=True,teacher_prompt_has_observation=use_obs)
            datasets[arm].append(x)
    for i,(arm,s) in enumerate(specs.items()):
        folder=ROOT/arm;cards=len(s['workers'])*8;sft=arm.startswith('sft_')
        s.update(cards=cards,epochs=3 if sft else 1,gradient_accumulation_steps=48//cards,effective_batch_size=48,
                 learning_rate=1e-4 if s['tuner_type']=='lora' else (1e-5 if sft else 2e-6),expected_steps=96 if sft else 32,
                 save_strategy='epoch' if sft else 'steps',save_steps=3,master_port=29801+i*10)
        random.Random(20260904).shuffle(datasets[arm])
        train=folder/'data/train.jsonl';smoke=folder/'data/smoke.jsonl'
        jl(train,datasets[arm]);jl(smoke,sorted(datasets[arm],key=lambda x:x['dynamic_student_budget']['max_checked_tokens'],reverse=True)[:48])
        s.update(dataset=str(train),dataset_sha256=sha(train),smoke_dataset=str(smoke),smoke_sha256=sha(smoke),output_dir=str(folder/'outputs'/('formal_complete1500' if sft else 'formal_complete1500_v2')))
    jl(ROOT/'data/observation_audit.jsonl',audit);jl(ROOT/'data/train_1500.updated_observations.jsonl',merged)
    write(ROOT/'suite.json',dict(arms=specs,model='/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-assets/models/Qwen2.5-Omni-7B',
          prerequisite='user_authorized_fresh_eight_arm_suite',train_samples=1500,answer_weight=3.,
          observation_status_counts=dict(Counter(a['status'] for a in audit)),unresolved_observation_count=sum(not a['observation_usable'] for a in audit),
          unresolved_policy='SFT omits analysis target (zero analysis supervision); distillation omits teacher observation; retain gold answer and all 1500 IDs',
          answer_weight_semantics='SFT and obs distillation: (mean_analysis_loss + 3*mean_answer_loss)/4; no-answer rollout uses mean loss, logged explicitly',
          student_analysis_instruction=ANALYSIS,student_answer_instruction=ANSWER,distillation_analysis_instruction=DISTILL_ANALYSIS,distillation_prompt_revision=2,
          source_sha256={str(p):sha(p) for p in [SOURCE,OBS]+EDITS},
          max_completion_length=512,max_length=32768,seed=20260904,lmbda=1.,sft_alpha=0.,beta=.5,clue_ema_alpha=.05,
          code_sha256={p.name:sha(p) for p in (ROOT/'code').iterdir() if p.is_file()}))
    print(json.dumps(dict(root=str(ROOT),counts=dict(Counter(a['status'] for a in audit)),unresolved=sum(not a['observation_usable'] for a in audit))))

if __name__=='__main__':main()
