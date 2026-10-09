#!/usr/bin/env python3
"""Freeze three MCQ full-parameter training arms after the external eval pair."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics

REPO = Path('/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd')
EVAL = REPO / 'training_runs/generalization_mcq_strict_answer_npu120_20261003'
SOURCE = REPO / 'training_runs/worldsense_clue_full_no_observation_npu96_w10w11_20260930/data/clue_openqa_thinking.jsonl'
BASE = '/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-assets/models/Qwen2.5-Omni-7B'


def read(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
def jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))


def prepare(root, weight):
    if (root / 'suite.json').exists(): raise ValueError('Suite already prepared; do not overwrite frozen inputs')
    root.mkdir(parents=True, exist_ok=True)
    instruction = json.loads((EVAL / 'experiment_manifest.json').read_text())['instruction']
    originals = {r['case_id']: r for r in read(REPO / 'data/sft/sft.jsonl')}
    annotations = {r['question_id']: r for r in read(REPO / 'data/annotation/merged.evidence.jsonl')}
    arms = {'sft_answer': [], 'sft_observation_answer': [], 'clue_no_observation': []}
    audit = []; lengths = []
    for source in read(SOURCE):
        case = source['case_id']; original = originals[case]
        question_options = original['messages'][0]['content'].split('\nBriefly analyze', 1)[0]
        choices = dict(re.findall(r'^([ABCD])\.\s*(.+)$', question_options, re.M))
        gold = source['source_option_letter']
        assert set(choices) in [set('ABC'), set('ABCD')] and gold in choices
        original_gold = re.findall(r'<answer>\s*([ABCD])\s*</answer>', original['messages'][-1]['content'])
        assert original_gold == [gold]
        assert '<video>' in question_options and '\nOptions:' in question_options
        prompt = question_options + '\n' + instruction
        assert prompt.endswith(instruction)
        obs = annotations[case]['observation'].strip()
        assert annotations[case]['status'] == 'submitted' and obs
        words = obs.split(); lengths.append(len(words))
        if len(words) > 120:
            # Keep an unchanged leading passage, preferably ending at a complete
            # sentence; record the exact edit instead of silently breaking the cap.
            prefix = ' '.join(words[:120])
            sentences = list(re.finditer(r'[.!?](?:[\"\']?)(?=\s|$)', prefix))
            obs = prefix[:sentences[-1].end()] if sentences and sentences[-1].end() >= len(prefix)//2 else prefix
        assert len(obs.split()) <= 120 and '<answer>' not in obs and '<analysis>' not in obs
        shared = {k: copy.deepcopy(source[k]) for k in ['case_id', 'prompt_id', 'video_id', 'videos', 'sampling_contract', 'dynamic_student_budget']}
        for key in list(shared['sampling_contract']):
            if key.startswith('teacher_'): shared['sampling_contract'].pop(key)
        assert all(0 < v['video_end'] <= 300 and Path(v['video']).is_file() for v in shared['videos'])
        assert shared['dynamic_student_budget']['max_checked_tokens'] <= 32768
        for arm, completion in [('sft_answer', f'<answer>{gold}</answer>'),
                               ('sft_observation_answer', f'<analysis>\n{obs}\n</analysis>\n<answer>{gold}</answer>')]:
            row = copy.deepcopy(shared)
            row.update(messages=[dict(role='user', content=prompt), dict(role='assistant', content=completion)],
                       experiment_arm=arm, response_format='mcq_analysis_and_option' if arm != 'sft_answer' else 'mcq_option_only',
                       supervision_contract=dict(student_prompt_has_gold=False, kind=arm, answer_weight=weight,
                           normalization='mean_each_region_then_weighted_average', analysis_word_limit=120))
            arms[arm].append(row)
        clue = copy.deepcopy(source)
        clue.update(messages=[dict(role='user', content=prompt)], solution=gold, gold_answer_text=choices[gold],
                    teacher_prompt=('\n'.join(['<video>'] * len(source['teacher_videos'])) +
                        question_options[len('<video>'):] + f'\nVerified correct option: {gold}. {choices[gold]}\n' + instruction),
                    experiment_arm='clue_no_observation', response_format='mcq_analysis_and_option',
                    supervision_contract=dict(student_prompt_has_gold=False, teacher_prompt_has_gold=True,
                         teacher_prompt_has_observation=False, student_rollout_only=True, lmbda=1., sft_alpha=0.))
        clue.pop('observation', None)
        arms['clue_no_observation'].append(clue)
        audit.append(dict(case_id=case, gold=gold, original_observation_words=len(words), used_observation_words=len(obs.split()),
                          truncated=len(words)>120, source_observation=annotations[case]['observation'], used_observation=obs,
                          previous_openqa_answer=source['gold_answer_text'], correct_option_text=choices[gold]))
    assert all(len(rows) == 1453 and len({r['case_id'] for r in rows}) == 1453 for rows in arms.values())
    specs = {
        'sft_answer': dict(workers=[f'npu24-worker-{i}' for i in range(3)], epochs=5, gradient_accumulation_steps=2,
                           learning_rate=1e-5, save_strategy='epoch', expected_steps=155),
        'sft_observation_answer': dict(workers=[f'npu96-worker-{i}' for i in range(3)], epochs=5, gradient_accumulation_steps=2,
                           learning_rate=1e-5, save_strategy='epoch', expected_steps=155),
        'clue_no_observation': dict(workers=[f'npu96-worker-{i}' for i in range(3,12)], epochs=3, gradient_accumulation_steps=1,
                           learning_rate=2e-6, save_strategy='steps', save_steps=3, expected_steps=63)}
    for name, rows in arms.items():
        folder = root / name; (folder/'logs').mkdir(parents=True, exist_ok=True)
        jsonl(folder/'data/train.jsonl', rows)
        # An entire distributed update containing the longest full-video inputs.
        smoke_size = len(specs[name]['workers']) * 8 * specs[name]['gradient_accumulation_steps']
        smoke = sorted(rows, key=lambda r: r['dynamic_student_budget']['max_checked_tokens'], reverse=True)[:smoke_size]
        jsonl(folder/'data/smoke.jsonl', smoke)
        specs[name].update(cards=len(specs[name]['workers'])*8, effective_batch_size=smoke_size,
            dataset=str(folder/'data/train.jsonl'), dataset_sha256=sha(folder/'data/train.jsonl'),
            smoke_dataset=str(folder/'data/smoke.jsonl'), smoke_sha256=sha(folder/'data/smoke.jsonl'),
            output_dir=str(folder/'outputs/formal'), master_port=29741+10*list(specs).index(name))
    jsonl(root/'data/observation_length_audit.jsonl', audit)
    scripts = ['worldsense_mcq_sft_entry.py', 'worldsense_mcq_gkd_entry.py', 'worldsense_full_gkd_entry.py', 'worldsense_mcq_checkpointing.py',
               'worldsense_mcq_training_suite.py', 'prepare_worldsense_mcq_full_suite.py']
    (root/'code').mkdir(exist_ok=True)
    for name in scripts: shutil.copy2(REPO/'training_code/scripts'/name, root/'code'/name)
    shutil.copy2(REPO/'training_code/configs/zero2_npu_cpu_offload.json', root/'code/zero2.json')
    write(root/'suite.json', dict(model=BASE, prerequisite=str(EVAL), arms=specs, instruction=instruction,
        instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(), answer_weight=weight,
        answer_weight_semantics='L=(mean_observation_ce+weight*mean_answer_region_ce)/(1+weight); answer-only uses mean_answer_ce',
        source_dataset=str(SOURCE), source_sha256=sha(SOURCE), train_samples=1453,
        epochs_sft=5, epochs_clue=3, tuner_type='full', freeze_llm=False, freeze_vit=False, freeze_aligner=False,
        observation_words_mean=statistics.mean(lengths), observation_over_120=sum(x>120 for x in lengths),
        seed=20260904, max_length=32768, max_completion_length=512, lmbda=1., sft_alpha=0., beta=.5, clue_ema_alpha=.05,
        sft_save_each_epoch=True, clue_save_every_optimizer_steps=3, retain_all_checkpoints=True,
        code_sha256={p.name: sha(p) for p in (root/'code').iterdir() if p.is_file()}))
    print(json.dumps(dict(root=str(root), arms={k:{a:b for a,b in v.items() if a in ['cards','effective_batch_size','epochs','expected_steps']} for k,v in specs.items()},
                         observation_over_120=sum(x>120 for x in lengths))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root', type=Path, required=True);parser.add_argument('--answer-weight', type=float, default=3.)
    a=parser.parse_args();prepare(a.root.resolve(),a.answer_weight)
