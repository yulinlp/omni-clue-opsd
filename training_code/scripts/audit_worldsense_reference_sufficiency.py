#!/usr/bin/env python3
"""CPU-only reference-scope audit. Flags guide review and never change any score."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

REPO = Path('/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd')
sys.path.insert(0, str(REPO / 'training_code/scripts'))
from score_worldsense_openqa_v2 import EVENT_MARKER, integer_words, normalize, read, singular


def reference_letter_sequence(label):
    if label['question_type'] != 'Event Sorting':
        return None
    text = label['gold_answer_text'].strip().lower()
    compact = re.sub(r'[\s(),.;:\[\]{}→>\-]+', '', text)
    explicit = bool(re.search(r'\([a-z]\)', text))
    if (len(compact) >= 3 and len(set(compact)) == len(compact) and
            re.fullmatch(r'[a-k]+', compact) and
            (explicit or re.fullmatch(r'[a-k]+\.?', text))):
        return list(compact)
    return None


def count_reference(text):
    text = text.strip(' \n\t.!').lower()
    adverbs = {'once': 1, 'twice': 2, 'thrice': 3}
    if text in adverbs:
        return {'value': adverbs[text], 'units': ['time'], 'kind': 'count-adverb'}
    words = text.split()
    for end in range(len(words), 0, -1):
        value = integer_words(' '.join(words[:end]))
        if value is not None:
            return {'value': value, 'units': [singular(w) for w in words[end:]], 'kind': 'integer-prefix'}
    return None


def audit(labels):
    records, all_flags = [], Counter()
    for label in labels:
        q, ref = label['question'], label['gold_answer_text']
        flags = []
        details = {}
        def flag(name, severity, explanation):
            flags.append({'category': name, 'severity': severity, 'explanation': explanation})
            all_flags[name] += 1
        sequence = reference_letter_sequence(label)
        if sequence:
            definitions = {}
            markers = list(EVENT_MARKER.finditer(q))
            for i, marker in enumerate(markers):
                letter = (marker.group(1) or marker.group(2)).lower()
                end = markers[i + 1].start() if i + 1 < len(markers) else len(q)
                definitions[letter] = q[marker.end():end].strip(' \n\t,.;:')
            missing = sorted(set(sequence) - set(definitions))
            details.update(reference_sequence=sequence, question_event_definitions=definitions,
                           undefined_reference_letters=missing)
            if missing:
                flag('letter_sequence_without_question_mapping', 'manual-reference-required',
                     'The text-only judge cannot map natural-language events to these reference letters from the question.')
        if re.search(r'\b(?:which|what) of the following\b|\b(?:none|all) of (?:the )?above\b', q, re.I):
            flag('question_refers_to_removed_choices', 'scope-review',
                 'The question retains a reference to choices. A substantive reference may still be usable; review question scope.')
        elif label.get('options_dependent_wording'):
            flag('legacy_option_wording_flag', 'informational',
                 'Legacy metadata flagged this wording; explicit event definitions can make an order question self-contained.')
        if re.search(r'\b(?:none|all) of (?:the )?above\b|\b(?:option|choice) [A-D]\b', ref, re.I):
            flag('reference_depends_on_absent_choices', 'manual-reference-required',
                 'The reference itself points to choices that are absent from the open question.')
        if re.match(r'^(?:yes|no)\b', ref, re.I):
            bool_question = bool(re.search(r'(?:^|[,]\s*)(?:is|are|was|were|do|does|did|can|could|will|would|has|have|had)\b', q, re.I))
            needs_reason = bool(re.search(r'\b(?:why|explain|reason|how)\b', q, re.I))
            details.update(yes_no_reference=True, question_has_yes_no_form=bool_question,
                           question_explicitly_requests_reason=needs_reason)
            elaboration = bool(re.search(r'\b(?:because|but|increases|decreases|quieter|mentioned|constant)\b', ref, re.I)) or len(ref.split()) > 9
            if bool_question and not needs_reason and elaboration:
                flag('yes_no_reference_has_unasked_explanation', 'scope-review',
                     'Correct bare yes/no is sufficient for this question; the reference explanation must not become a mandatory extra target. Contradictory extra candidate claims still fail.')
            elif not bool_question:
                flag('yes_no_reference_for_nonbinary_question', 'scope-review',
                     'Check whether a bare yes/no meets the question requirements; regex form detection is only a review aid.')
        if re.match(r'^\s*how many\b', q, re.I):
            count = count_reference(ref)
            details['requested_count_reference'] = count
            if re.search(r"\bi (?:do not|don't) know\b|\bunknown\b|\bcannot (?:tell|determine)\b", ref, re.I):
                flag('unknown_reference_for_count_question', 'manual-reference-required',
                     'The reference supplies uncertainty instead of a count. Check the original annotation/video; do not invent a count or mark every concrete count wrong automatically.')
            elif count is None:
                flag('noncanonical_reference_for_count_question', 'scope-review',
                     'The count is not a simple integer/adverb prefix. This can be a valid paraphrase; review without auto-conversion.')
            elif count['units']:
                requested_kind = re.match(r'^\s*how many\s+(times|types|kinds)\b', q, re.I)
                q_words = {singular(w) for w in re.findall(r'\w+', q.lower())}
                unit_words = [u for u in count['units'] if u not in ('of', 'the', 'in', 'total')]
                if requested_kind and unit_words and unit_words[0] != singular(requested_kind.group(1).lower()):
                    flag('count_reference_unit_scope_mismatch', 'scope-review',
                         'The explicit unit may answer a different quantity from the question (e.g., occurrences versus kinds).')
                elif any(u not in q_words for u in unit_words):
                    flag('count_reference_unit_needs_semantic_check', 'scope-review',
                         'The explicit count unit is not named literally in the question; synonyms can be valid, so this is not an automatic error.')
        if flags:
            records.append({**label, 'flags': flags, 'details': details,
                            'automatic_score_change': False})
    return {'total': len(labels), 'flagged_items': len(records), 'category_counts': dict(all_flags),
            'manual_reference_required': [r['sample_id'] for r in records
                                          if any(f['severity'] == 'manual-reference-required' for f in r['flags'])],
            'yes_no_reference_count': sum(bool(re.match(r'^(?:yes|no)\b', r['gold_answer_text'], re.I)) for r in labels),
            'how_many_question_count': sum(bool(re.match(r'^\s*how many\b', r['question'], re.I)) for r in labels),
            'event_sorting_letter_reference_count': sum(reference_letter_sequence(r) is not None for r in labels),
            'records': records,
            'limitations': ['Flags are textual sufficiency/scope heuristics, not factual video-label validation.',
                            'Unflagged references are not certified correct.',
                            'This audit never changes frozen data, rubric, predictions or scores.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--labels', type=Path)
    a = p.parse_args()
    labels_path = a.labels or a.root / 'data/worldsense.openqa.labels.jsonl'
    labels = read(labels_path)
    if len({r['sample_id'] for r in labels}) != len(labels):
        raise ValueError('Duplicate label IDs')
    report = audit(labels)
    report['labels'] = str(labels_path.resolve())
    report['labels_sha256'] = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    path = a.root / 'reference_sufficiency_audit.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('total', 'flagged_items', 'category_counts', 'manual_reference_required')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
