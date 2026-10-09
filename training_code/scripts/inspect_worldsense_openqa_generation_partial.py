#!/usr/bin/env python3
"""Read live generation shards for formatting/length/repetition, without scoring."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

REPO = Path('/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd')
sys.path.insert(0, str(REPO / 'training_code/scripts'))
from score_worldsense_openqa_v2 import extract, read, write
from omni_opsd.evaluation import _input_signature


def live_rows(path):
    if not path.exists():
        return [], 0
    lines = path.read_text().splitlines()
    rows, incomplete = [], 0
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise ValueError('Malformed interior prediction line: ' + str(path))
            incomplete += 1
    return rows, incomplete


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', default='base')
    a = p.parse_args()
    output = a.root / 'openqa' / a.model
    protocol = json.loads((output / 'generation_protocol.json').read_text())
    cap = protocol['max_new_tokens']
    tokenizer = None
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file('/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-assets/models/Qwen2.5-Omni-7B/tokenizer.json')
    except ImportError:
        pass
    records, examples, seen, incomplete = [], [], set(), 0
    counts = Counter()
    for folder in sorted((output / 'shards').glob('card_*')):
        signatures = {_input_signature(r): r['case_id'] for r in read(folder / 'input.jsonl')}
        rows, skipped = live_rows(folder / 'results.jsonl')
        incomplete += skipped
        counts[folder.name] = len(rows)
        for row in rows:
            key = row.get('case_id') or signatures.get(_input_signature(row))
            if key is None or key in seen:
                raise ValueError('Unmapped/duplicate live prediction: ' + str(key))
            seen.add(key)
            response = str(row.get('response', ''))
            answer, analysis, valid, format_ok = extract(response)
            words = re.findall(r'\w+', analysis.lower())
            windows = [tuple(words[i:i + 20]) for i in range(max(0, len(words) - 19))]
            repetition = 1 - len(set(windows)) / len(windows) if windows else 0
            tokens = len(tokenizer.encode(response, add_special_tokens=False).ids) if tokenizer else None
            rec = {'sample_id': key, 'valid_final_answer': valid, 'strict_format_ok': format_ok,
                   'analysis_closed': bool(re.search(r'</analysis>', response, re.I)),
                   'analysis_words': len(analysis.split()), 'repeated_20gram_fraction': repetition,
                   'repetitive_analysis': len(words) >= 60 and repetition >= .4,
                   'decoded_response_tokens': tokens,
                   'near_generation_cap': tokens >= cap - 2 if tokens is not None else None}
            records.append(rec)
            if not valid or rec['repetitive_analysis'] or rec['analysis_words'] > 120:
                examples.append({**rec, 'final_answer': answer, 'response': response})
    n = len(records)
    report = {'model': a.model, 'prediction_rows_observed': n, 'total_expected': 518,
              'partial': n != 518, 'per_card': dict(counts),
              'live_last_incomplete_lines_skipped': incomplete,
              'invalid_final_answers': sum(not r['valid_final_answer'] for r in records),
              'unfinished_analysis': sum(not r['analysis_closed'] for r in records),
              'strict_format_ok': sum(r['strict_format_ok'] for r in records),
              'over_120_word_analysis': sum(r['analysis_words'] > 120 for r in records),
              'analysis_words_mean_including_unclosed': sum(r['analysis_words'] for r in records) / n if n else None,
              'repetitive_analysis': sum(r['repetitive_analysis'] for r in records),
              'near_generation_cap': sum(r['near_generation_cap'] is True for r in records) if tokenizer else None,
              'tokenizer_available': tokenizer is not None, 'generation_cap': cap,
              'cap_note': 'Retokenized decoded text, approximate; actual finish_reason not recorded.',
              'semantic_scoring_performed': False, 'existing_scores_modified': False,
              'warning': 'A live partial set is not a representative full benchmark; no accuracy or final comparison is inferred.'}
    diagnostic_root = a.root / 'generation_diagnostics_partial'
    diagnostic_root.mkdir(exist_ok=True)
    (diagnostic_root / (a.model + '.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    write(diagnostic_root / (a.model + '.rows.jsonl'), records)
    write(diagnostic_root / (a.model + '.examples.jsonl'), examples)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
