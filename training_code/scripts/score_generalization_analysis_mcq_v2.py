"""Versioned MCQ format recovery, based only on model output (never gold).

Keep the original scorer reproducible. This version adds explicit alternative
final fields, without turning mentions in reasoning into final decisions.
"""
import re
import unicodedata

from score_generalization_analysis_mcq import (
    normalize_explicit_option, parse_analysis_option as parse_v1,
    score_rows as score_rows_v1,
)
from score_worldsense_mcq_v2 import parse_mcq_choice
from omni_opsd.rewards import completion_text

VERSION = 'external-analysis-mcq-explicit-formats-v2'


def parse_analysis_option(value):
    raw = unicodedata.normalize('NFKC', completion_text(value)).strip()
    previous = parse_v1(value)
    text = raw
    recovered = []

    # A last analysis field containing ONLY a choice is a misplaced final
    # field. Ordinary analysis paragraphs, even with "answer is B", stay out.
    terminal_analysis = re.search(
        r'<analysis\s*>\s*([A-D])\s*[.]?\s*</analysis\s*>\s*[.]?\s*$',
        text, re.I)
    if terminal_analysis:
        recovered.append((terminal_analysis.group(1).upper(), 'terminal-letter-only-analysis'))
        text = text[:terminal_analysis.start()]

    # A solution field is an explicit final field just like an answer field.
    if re.search(r'<solution\s*>', text, re.I):
        text = re.sub(r'<(/?)solution\s*>', r'<\1answer>', text, flags=re.I)
        recovered.append((None, 'solution-field'))

    # The requested final-answer field remains authoritative, as in v1.
    # A stray <C> before <answer>B</answer> is not a second final answer.
    normalized = normalize_explicit_option(text)
    fields = re.findall(r'<answer>.*?</answer>', normalized, re.S | re.I)
    primary = parse_mcq_choice('\n'.join(fields))
    if primary['choice_candidates']:
        return dict(primary, scoring_version=VERSION,
                    recovery_rules=['solution-field'] if any(r == 'solution-field' for _, r in recovered) else [],
                    recovered_parser_input='\n'.join(fields))

    # Remove full reasoning blocks before looking for alternative final tags.
    outside = re.sub(r'<analysis\s*>.*?</analysis\s*>', '\n', text, flags=re.S | re.I)
    unfinished = re.search(r'<analysis\s*>', outside, re.I)
    if unfinished:
        # Only a separate final field / standalone terminal decision ends
        # unfinished analysis; inline rhetorical option mentions never do.
        tail = outside[unfinished.end():]
        boundary = re.search(
            r'<(?:answer|option)\s*>|(?:^|\n)\s*(?:<[A-D]\s*>|'
            r'(?:Final\s+answer|Answer)\s*[:=]\s*[A-D]\s*[.]?\s*$)',
            tail, re.I)
        outside = outside[:unfinished.start()] + (
            tail[boundary.start():] if boundary else '')

    # <B>, <B></B>, <B>...</B>, or <B> followed by prose are explicit
    # letter-tag decisions. <B>C</B> is conflicting and must not choose B.
    for match in re.finditer(r'<([A-D])\s*>', outside, re.I):
        letter = match.group(1).upper()
        recovered.append((letter, 'letter-tag'))
        tail = outside[match.end():]
        body = re.match(r'\s*([A-D])\s*[.]?\s*(?:</[A-D]\s*>|</answer\s*>|$)', tail, re.I)
        if body:
            recovered.append((body.group(1).upper(), 'letter-tag-body'))
        closing = re.match(r'[^<>]*</([A-D])\s*>', tail, re.I)
        if closing:
            recovered.append((closing.group(1).upper(), 'letter-tag-closing'))

    # A final line containing just the decision is a boundary, even when
    # earlier untagged prose discussed multiple choices.
    final_line = re.search(
        r'(?:^|\n)\s*(?:(?:Final\s+answer|Answer|Option|Choice)\s*(?:is\s+|[:=]\s*|\s+))?'
        r'([A-D])\s*[.]?\s*$', outside, re.I)
    if final_line:
        recovered.append((final_line.group(1).upper(), 'standalone-final-line'))

    if not recovered:
        return dict(previous, scoring_version=VERSION, recovery_rules=[])

    # No parseable canonical final field exists. Conflicting alternative
    # markers remain ambiguous; do not resolve them using the reference.
    supplemental = [f'<answer>{letter}</answer>' for letter, _ in recovered if letter]
    if previous['parsed']:
        supplemental.append(f"<answer>{previous['prediction']}</answer>")
    final_text = '\n'.join(fields + supplemental)
    parsed = parse_mcq_choice(final_text)
    return dict(parsed, scoring_version=VERSION,
                recovery_rules=sorted({rule for _, rule in recovered}),
                recovered_parser_input=final_text)


def score_rows(results, labels, sources):
    rows = score_rows_v1(results, labels, sources)
    for row in rows:
        previous = {key: row[key] for key in ('prediction', 'parsed', 'correct', 'parse_reason')}
        parsed = parse_analysis_option(row['response'])
        # Preserve the strict requested XML format diagnostic from v1.
        parsed.pop('strict_format_compliant', None)
        parsed.pop('legacy_prediction', None)
        row.update(parsed)
        row['correct'] = row['prediction'] == row['gold_answer'] if row['parsed'] else False
        row['previous_scoring'] = previous
        row['prediction_changed'] = row['prediction'] != previous['prediction']
    return rows
