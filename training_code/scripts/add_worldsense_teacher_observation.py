#!/usr/bin/env python3
"""Add the same cleaned observation as SFT to teacher prompt only."""
import argparse
import copy
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--sft-observations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sft = {r['case_id']: r for line in args.sft_observations.open() if (r := json.loads(line))}
    rows = []
    for line in args.source.open():
        before = json.loads(line)
        row = copy.deepcopy(before)
        reference = sft[row['case_id']]
        observation = re.fullmatch(r'<analysis>\n(.*?)\n</analysis>\n<answer>.*?</answer>',
                                   reference['messages'][1]['content'], re.S).group(1)
        if row['gold_answer_text'] != reference['gold_answer_text'] or not observation:
            raise ValueError(f'Missing/mismatched observation or answer: {row["case_id"]}')
        marker = '\nBriefly analyze the video and audio evidence'
        prompt, instruction = row['teacher_prompt'].split(marker, 1)
        row['teacher_prompt'] = (prompt + '\nReference evidence observation:\n' + observation
                                 + '\nUse this observation as an auxiliary reference together with the '
                                 'provided clue video and audio. Base your analysis on the available '
                                 'evidence and the verified correct answer. Do not treat the reference '
                                 'as the student response or simply copy it.' + marker + instruction)
        row['teacher_observation'] = observation
        row['teacher_observation_source'] = str(args.sft_observations)
        row['experiment_arm'] = 'worldsense_openqa_thinking_full_onpolicy_observation_teacher'
        row['supervision_contract']['teacher_prompt_has_observation'] = True
        assert row['messages'] == before['messages']
        assert row['videos'] == before['videos'] and row['teacher_videos'] == before['teacher_videos']
        assert all(m['role'] != 'assistant' for m in row['messages'])
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    print(f'Added teacher-only observations to {len(rows)} samples; student prompts and media unchanged.')


if __name__ == '__main__':
    main()
