#!/usr/bin/env python3
"""Categorize explicitly unresolved scored rows; never adjudicate or compute a score."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

REPO = Path('/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd')
sys.path.insert(0, str(REPO / 'training_code/scripts'))
from score_worldsense_openqa_v2 import read, write


def classify(row, audit_flags):
    reasons = []
    reviews = row.get('judge_reviews', [])
    verdicts = [r.get('verdict') for r in reviews]
    text = ' '.join(str(r.get('reason', '')) for r in reviews).lower()
    if any(f['severity'] == 'manual-reference-required' for f in audit_flags):
        reasons.append('reference-manual-check')
    if 'YES' in verdicts and 'NO' in verdicts:
        reasons.append('reviewer-disagreement')
    if any(r.get('reason') in ('unparseable-judge-json', 'invalid-judge-schema', 'missing-judge-evidence') for r in reviews):
        reasons.append('judge-output-protocol')
    if re.search(r'insufficient|undefined|not defined|not provided|missing mapping|cannot map|ambiguous reference', text):
        reasons.append('possible-reference-insufficiency')
    if re.search(r'unclear|vague|ambiguous|cannot interpret', text):
        reasons.append('answer-interpretation-ambiguity')
    if not reviews:
        reasons.append('judge-not-yet-run')
    return reasons or ['semantic-review-needed']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    labels = {r['sample_id']: r for r in read(a.root / 'data/worldsense.openqa.labels.jsonl')}
    audit_path = a.root / 'reference_sufficiency_audit.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {'records': []}
    flags = {r['sample_id']: r['flags'] for r in audit['records']}
    queue, reference_queue, models = [], [], {}
    for scored_path in sorted((a.root / 'openqa').glob('*/scored.jsonl')):
        rows = read(scored_path)
        counts = Counter()
        unknown = 0
        reference_checks = 0
        for row in rows:
            if 'semantic_correct' not in row:
                raise ValueError('Missing semantic_correct; unscored generation is not a scored null: ' + str(scored_path))
            key = row['sample_id']
            if any(f['severity'] == 'manual-reference-required' for f in flags.get(key, [])):
                reference_checks += 1
                reference_queue.append({'model': scored_path.parent.name, 'sample_id': key,
                                        'question': labels[key]['question'],
                                        'gold_answer_text': labels[key]['gold_answer_text'],
                                        'candidate_final_answer': row['final_answer'],
                                        'existing_semantic_correct_unchanged': row['semantic_correct'],
                                        'reference_audit_flags': flags[key],
                                        'judge_reviews': row.get('judge_reviews', []),
                                        'needs_reference_verification_even_if_reviewers_agree': True,
                                        'automatic_adjudication': False})
            if row['semantic_correct'] is not None:
                if type(row['semantic_correct']) is not bool:
                    raise ValueError('Nonbinary resolved value: ' + str(row['sample_id']))
                continue
            unknown += 1
            reasons = classify(row, flags.get(key, []))
            counts.update(reasons)
            queue.append({'model': scored_path.parent.name, 'sample_id': key,
                          'video_id': row['video_id'], 'question': labels[key]['question'],
                          'gold_answer_text': labels[key]['gold_answer_text'],
                          'candidate_final_answer': row['final_answer'], 'review_categories': reasons,
                          'judge_reviews': row.get('judge_reviews', []), 'reference_audit_flags': flags.get(key, []),
                          'semantic_correct': None, 'scored_source': str(scored_path.resolve()),
                          'automatic_adjudication': False})
        models[scored_path.parent.name] = {'scored_rows': len(rows), 'unresolved': unknown,
                                          'category_counts': dict(counts),
                                          'reference_verification_items': reference_checks}
    report = {'models': models, 'unresolved_total': len(queue),
              'reference_verification_total': len(reference_queue),
              'score_changed': False, 'null_treated_as_wrong': False,
              'next_step': 'Review each candidate against question-essential facts and adequate reference; preserve explicit evidence/reason in human adjudications. Never infer the final answer from analysis.'}
    write(a.root / 'unresolved_review_queue.jsonl', queue)
    write(a.root / 'reference_verification_queue.jsonl', reference_queue)
    (a.root / 'unresolved_review_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
