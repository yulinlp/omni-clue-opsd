"""Recover explicit final decisions in sentences and misplaced answer fields.

No semantic option matching: inputs are generated text only, never options/gold.
"""
import re

from score_generalization_analysis_mcq_v2 import parse_analysis_option as parse_v2, score_rows as score_v2
from score_generalization_analysis_mcq import normalize_explicit_option
from score_worldsense_mcq_v2 import parse_mcq_choice
from omni_opsd.rewards import completion_text

VERSION = 'external-analysis-mcq-explicit-formats-v3'
CHOICE = r'''[\s'"`*]*(?:\(([A-D])\)|([A-D]))[\s'"`*]*'''
END = r'''(?=$|[.,:;!?)'"`*]|\s+(?:because|as|since|which|for\s+the|due\s+to|given|based\s+on)\b)'''


def sentence_choices(body):
    found = []
    # Harmless paragraph/emphasis markup; keep content and all competing choices.
    body = re.sub(r'</?(?:p|b|strong|em|span)\s*>', '', body, flags=re.I).strip()
    patterns = [
        r'\b(?:(?:the\s+)?(?:correct|final|best)\s+)?answer\s*(?:is\s*:?[ \t]*|should\s+be\s*:?[ \t]*|[:=]\s*)' + CHOICE + END,
        r'\b(?:I\s+(?:will\s+)?(?:choose|select|pick)|(?:the\s+)?(?:correct|final|selected)\s+(?:option|choice)\s+is)\s+' + CHOICE + END,
        r"\banswer\s+['\"]([A-D])[.'\"]*\s+is\s+(?:the\s+)?correct\b",
        r'(?:(?<=^)|(?<=\n))\s*' + CHOICE + r'\s+is\s+(?:the\s+)?(?:correct|right|best)\s+(?:answer|option|choice|response)\b',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, body, re.I):
            letter = next(x for x in match.groups() if x is not None)
            found.append(letter.upper())
    # Canonical field can be a quoted single letter, including a final period.
    single = re.fullmatch(r'''[\s'"`*]*([A-D])[\s.'"`*]*''', body, re.I)
    if single:
        found.append(single.group(1).upper())
    for match in re.finditer(r'(?:^|\banswer\s+is\s+)([A-D])\s+(?:or|and)\s+([A-D])\b', body, re.I):
        found.extend(letter.upper() for letter in match.groups())
    standard = parse_mcq_choice('<answer>' + body + '</answer>')
    found.extend(c['letter'] for c in standard['choice_candidates'])
    return found


def parse_analysis_option(value):
    raw = completion_text(value)
    previous = parse_v2(raw)
    # Inspect explicit final fields before removing analysis: models sometimes
    # put the answer tag immediately before </analysis>.
    bodies = re.findall(r'<(?:answer|option|solution)\s*>(.*?)(?:</(?:answer|option|solution)\s*>|$)', raw, re.S | re.I)
    letters = [letter for body in bodies for letter in sentence_choices(body)]
    rules = []
    if letters:
        rules.append('explicit-answer-field-sentence-or-nested-field')
    else:
        # A distinct last "Answer: B" line may be misplaced inside analysis.
        # It must be the terminal declaration; earlier reasoning is not searched.
        tail = re.sub(r'(?:\s*</analysis\s*>\s*)+$', '', raw, flags=re.I)
        final = re.search(r'(?:^|\n)\s*((?:Final\s+answer|Answer)\s*(?:is\s+|[:=]\s*).*)$', tail, re.I)
        if final:
            letters = sentence_choices(final.group(1))
            if letters:
                rules.append('terminal-answer-declaration-inside-analysis')
    if not letters:
        return dict(previous, scoring_version=VERSION, v3_recovery_rules=[])
    parsed = parse_mcq_choice('\n'.join(f'<answer>{letter}</answer>' for letter in letters))
    # Do not silently replace an already recognized decision with another.
    if previous['parsed'] and parsed['parsed'] and previous['prediction'] != parsed['prediction']:
        letters.append(previous['prediction'])
        parsed = parse_mcq_choice('\n'.join(f'<answer>{letter}</answer>' for letter in letters))
    return dict(parsed, scoring_version=VERSION, v3_recovery_rules=rules,
                recovered_parser_input='\n'.join(f'<answer>{letter}</answer>' for letter in letters))


def score_rows(results, labels, sources):
    rows = score_v2(results, labels, sources)
    for row in rows:
        previous = {key: row[key] for key in ('prediction', 'parsed', 'correct', 'parse_reason')}
        parsed = parse_analysis_option(row['response'])
        parsed.pop('strict_format_compliant', None)
        parsed.pop('legacy_prediction', None)
        row.update(parsed)
        row['correct'] = row['prediction'] == row['gold_answer'] if row['parsed'] else False
        row['previous_v2_scoring'] = previous
        row['prediction_changed_from_v2'] = row['prediction'] != previous['prediction']
    return rows
