#!/usr/bin/env python3
"""Open-QA SFT: complete annotation observation + gold option text.

Use the previous screened sample IDs and audio/video budgets unchanged.
Rewrite explicit option-letter references using the original choice text.
"""
import argparse
import copy
import json
import re
import statistics
from pathlib import Path

INSTRUCTION = ('Provide a concise analysis of the video and audio evidence relevant to the question '
               'inside <analysis>...</analysis>, then give the answer in natural language '
               'inside <answer>...</answer>. Do not output an option letter or refer to answer options.')
OPTION_REFERENCE = re.compile(r'\b(?:option|choice)\s+([A-D])\b', re.I)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--original-thinking', type=Path, required=True)
    parser.add_argument('--annotation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    annotations = {r['question_id']: r for line in args.annotation.open() if (r := json.loads(line))}
    originals = {r['case_id']: r for line in args.original_thinking.open() if (r := json.loads(line))}
    rows, lengths, changed = [], [], 0
    for line in args.source.open():
        row = copy.deepcopy(json.loads(line))
        case_id = row['case_id']
        observation = annotations[case_id]['observation'].strip()
        if not observation or annotations[case_id]['status'] != 'submitted':
            raise ValueError(f'Missing submitted observation: {case_id}')
        options = dict(re.findall(r'^([A-D])\.\s*(.+)$', originals[case_id]['messages'][0]['content'], re.M))
        def replace(match):
            letter = match.group(1).upper()
            return 'the answer "' + options[letter].strip().rstrip('.') + '"'
        analysis = OPTION_REFERENCE.sub(replace, observation)
        changed += analysis != observation
        question = row['messages'][0]['content'].split('\nAnswer the question directly', 1)[0]
        assert '\nOptions:' not in question
        answer = row['gold_answer_text']
        row['messages'] = [
            {'role': 'user', 'content': question + '\n' + INSTRUCTION},
            {'role': 'assistant', 'content': '<analysis>\n' + analysis + '\n</analysis>\n<answer>' + answer + '</answer>'},
        ]
        row['experiment_arm'] = 'worldsense_openqa_observation_sft_lora'
        row['response_format'] = 'analysis_and_natural_language_answer'
        row['sft_target_source'] = 'annotation_observation + gold_answer_text'
        row['observation_source'] = str(args.annotation)
        row['observation_option_references_rewritten'] = analysis != observation
        row['analysis_truncated'] = False
        row['supervision_contract'] = {'kind': 'observation-and-gold-answer-teacher-forcing',
                                       'student_prompt_has_gold': False,
                                       'supervise_analysis': True, 'supervise_answer': True,
                                       'analysis_word_limit': None}
        lengths.append(len(analysis.split()))
        rows.append(row)
    assert len({r['case_id'] for r in rows}) == len(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    report = {'samples': len(rows), 'analysis_words_mean': statistics.mean(lengths),
              'analysis_words_max': max(lengths), 'analysis_over_120_words': sum(n > 120 for n in lengths),
              'option_reference_rows_rewritten': changed, 'truncated_rows': 0}
    args.output.with_suffix('.stats.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
