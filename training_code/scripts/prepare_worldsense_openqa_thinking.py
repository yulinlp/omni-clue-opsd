#!/usr/bin/env python3
"""Preserve screened videos/clues and gold text; add open-QA reasoning prompts."""
import argparse
import copy
import json
from pathlib import Path

INSTRUCTION = ('Briefly analyze the video and audio evidence in English using at most 120 words. '
               'Write your analysis inside <analysis>...</analysis>, then give the answer '
               'in natural language inside <answer>...</answer>. Do not output an option letter.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open('w') as output:
        for line in args.source.open():
            row = copy.deepcopy(json.loads(line))
            question = row['messages'][0]['content'].split('\nAnswer the question directly', 1)[0]
            if '\nOptions:' in question:
                raise ValueError('Source must already be open QA.')
            gold = row['gold_answer_text']
            row['messages'] = [{'role': 'user', 'content': question + '\n' + INSTRUCTION}]
            row['teacher_prompt'] = ('\n'.join(['<video>'] * len(row['teacher_videos']))
                                     + '\n' + question.split('\n', 1)[1]
                                     + '\nVerified correct answer: ' + gold + '\n' + INSTRUCTION)
            row['response_format'] = 'analysis_and_natural_language_answer'
            row['experiment_arm'] = 'worldsense_openqa_thinking_full_onpolicy'
            row['supervision_contract'].update(dataset_completion_is_gold_answer_text=False,
                                               rollout_or_dataset_completion=False,
                                               student_rollout_only=True)
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
    print(f'wrote {count} open-QA thinking samples to {args.output}')


if __name__ == '__main__':
    main()
