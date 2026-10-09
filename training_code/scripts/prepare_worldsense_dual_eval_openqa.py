#!/usr/bin/env python3
"""Reuse frozen MCQ IDs/media, removing all choices and isolating open-QA labels."""
import argparse
import hashlib
import json
import re
from pathlib import Path

INSTRUCTION = ('Analyze the video and its audio to answer the question. Write a concise analysis '
               'in English, at most 120 words, inside <analysis>...</analysis>. Then give a clear '
               'natural-language final answer inside <answer>...</answer>. Do not output an option '
               'letter. Use only evidence from the video and audio. Do not add text outside these tags.')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    data = a.root / 'data'
    read = lambda path: [json.loads(l) for l in path.open() if l.strip()]
    source = read(data / 'worldsense.answer_free.jsonl')
    candidates = {r['sample_id']: r for r in read(repo / 'data/atomic_eval_20260929/worldsense_candidates_3079.atomic.jsonl')}
    tiers = {r.get('sample_id', r.get('question_id')): r for r in read(repo / 'data/screening/per_question.jsonl')}
    rows, labels = [], []
    for old in source:
        r = candidates[old['case_id']]
        assert r['duration'] <= 300
        row = dict(old)
        row['messages'] = [dict(role='user', content='<video>\nQuestion: ' + r['question'] + '\n' + INSTRUCTION)]
        rows.append(row)
        answer = r['choices'][ord(r['answer']) - ord('A')]
        labels.append(dict(sample_id=r['sample_id'], video_id=r['video_id'], question=r['question'],
                           gold_answer_text=answer, original_answer_letter=r['answer'],
                           choices=r['choices'], question_type=r['question_type'],
                           tier=tiers[r['sample_id']]['tier'],
                           options_dependent_wording=bool(re.search(r'\bfollowing\b|\bwhich (?:statement|option)', r['question'], re.I))))
    assert len(rows) == len(labels) == 518
    for name, content in [('worldsense.openqa.jsonl', rows), ('worldsense.openqa.labels.jsonl', labels)]:
        (data / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in content))
    manifest = dict(rows=518, same_mcq_ids_and_media=True, choices_visible_to_respondent=False,
                    answers_visible_to_respondent=False, use_audio_in_video=True,
                    instruction=INSTRUCTION, generation_max_new_tokens=768,
                    options_dependent_wording_count=sum(r['options_dependent_wording'] for r in labels),
                    answer_free_sha256=hashlib.sha256((data / 'worldsense.openqa.jsonl').read_bytes()).hexdigest(),
                    labels_sha256=hashlib.sha256((data / 'worldsense.openqa.labels.jsonl').read_bytes()).hexdigest())
    (data / 'openqa_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))

if __name__ == '__main__':
    main()
