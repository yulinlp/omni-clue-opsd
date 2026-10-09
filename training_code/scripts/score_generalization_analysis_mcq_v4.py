"""Recover damaged final tags and literal option text, without gold access.

No semantic judge or fuzzy similarity. Ambiguous final choices remain unresolved.
The original strict-format diagnostic must remain separate from this parser.
"""
import re
import unicodedata

from score_generalization_analysis_mcq_v3 import parse_analysis_option as parse_v3, sentence_choices
from omni_opsd.rewards import completion_text

VERSION = 'external-analysis-mcq-final-format-and-option-text-v4'
FINAL = r'(?:answer|option|solution|answertopic)'


def normalized_text(value):
    text = unicodedata.normalize('NFKC', value).casefold()
    text = text.translate(str.maketrans({'’': "'", '‘': "'", '“': '"', '”': '"', '–': '-', '—': '-'}))
    text = re.sub(r'</?(?:p|b|strong|em|span)\s*>', '', text, flags=re.I)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.strip(' \t\r\n.!?;:,"\'`*')


def option_key(value):
    """Only orthographic changes: articles and standalone numerical notation."""
    text = normalized_text(value)
    text = re.sub(r'^(?:a|an|the)\s+', '', text)
    number = re.fullmatch(r'\d{1,3}(?:,\d{3})+', text)
    if number: text = text.replace(',', '')
    cardinals = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split()
    ordinals = 'first second third fourth fifth sixth seventh eighth ninth tenth'.split()
    if text in cardinals: return str(cardinals.index(text))
    ordinal = re.fullmatch(r'(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th))(?:\s+(?:position|place))?', text)
    if ordinal:
        part = ordinal.group(1)
        return 'ordinal:' + str(ordinals.index(part)+1 if part in ordinals else int(re.match(r'\d+', part).group()))
    return text


def option_text_choices(body, choices):
    text = body.strip()
    # An entire final field, possibly prefixed by an explicit answer declaration.
    text = re.sub(r'^(?:(?:the\s+)?(?:correct|final)\s+)?answer\s*(?:is\s*:?[ \t]*|[:=]\s*)', '', text, flags=re.I)
    key = option_key(text)
    if not key: return []
    return [chr(65+i) for i,c in enumerate(choices) if key == option_key(c)]


def outside_analysis_segments(raw):
    """Handle nested analysis tags without leaking their option discussions."""
    parts=[]; depth=0; last=0
    for match in re.finditer(r'<(/?)analysis\s*>', raw, re.I):
        if depth==0: parts.append(raw[last:match.start()])
        if match.group(1): depth=max(0,depth-1)
        else: depth+=1
        last=match.end()
    if depth==0: parts.append(raw[last:])
    return [part.strip() for part in parts if part.strip()]


def final_bodies(raw):
    bodies=[]
    # Limit an unfinished field at the next tag; never scan later analysis as
    # though it were part of that final answer.
    for match in re.finditer(r'<'+FINAL+r'\s*>', raw, re.I):
        tail=raw[match.end():]
        end=re.search(r'</?'+FINAL+r'\s*>|</?analysis\s*>',tail,re.I)
        bodies.append((tail[:end.start()] if end else tail,'explicit-final-field'))
    # A real final field has priority over untagged prose after its closing tag.
    if bodies: return bodies
    segments=outside_analysis_segments(raw)
    outside='\n'.join(segments)
    # A duplicated closing tag is a frequent misspelling of the opening tag.
    for match in re.finditer(r'</'+FINAL+r'\s*>([^<>]+)(?=</'+FINAL+r'\s*>|$)',outside,re.I):
        bodies.append((match.group(1),'closing-tag-used-as-opening'))
    # A stray closing tag after a standalone final decision contributes no text.
    clean=lambda text:re.sub(r'</(?:answer|option|solution|answertopic|analysis)\s*>','\n',text,flags=re.I).strip()
    # Separate earlier standalone decisions remain competing answers. Earlier
    # untagged prose, however, is not scanned for incidental option mentions.
    for segment in segments[:-1]:
        short=clean(segment)
        if re.fullmatch(r'[A-D][.!?,;:]?',short,re.I):
            bodies.append((short,'earlier-standalone-final-decision'))
    tail=clean(segments[-1]) if segments else ''
    unknown=re.fullmatch(r'<([a-z][\w-]*)\s*>\s*([ABCD])\s*</\1\s*>',tail,re.I)
    if unknown and unknown.group(1).casefold() not in ['a','b','c','d','analysis']:
        return bodies+[(unknown.group(2),'terminal-single-letter-in-other-tag')]
    nonempty=[line.strip() for line in tail.splitlines() if line.strip() and line.strip()!='.']
    if nonempty and '<' not in tail and not all(re.fullmatch(r'[A-D][.!?,;:]?',line,re.I) for line in nonempty):
        # A distinct final decision line has priority over preceding untagged
        # reasoning. Do not treat its compared alternatives as final votes.
        if re.fullmatch(r'(?:(?:final\s+answer|answer|option|choice)\s*(?:is\s+|[:=]\s*))?[A-D][.!?,;:]?',nonempty[-1],re.I):
            tail=nonempty[-1]
    if tail: bodies.append((tail,'outside-analysis-final-text'))
    return bodies


def parse_analysis_option(value, choices):
    if not 2 <= len(choices) <= 4: raise ValueError('Expected two to four input options')
    raw=unicodedata.normalize('NFKC',completion_text(value)).strip()
    # Repair escaped angle brackets only; keep literal model wording intact.
    raw=re.sub(r'\\([<>/])',r'\1',raw)
    previous=parse_v3(value)
    candidates=[]; evidence=[]
    for body,scope in final_bodies(raw):
        letters=sentence_choices(body)
        alternatives=re.match(r'^[\s,]*([A-D](?:\s*(?:,\s*(?:(?:and|or)\s+)?|/\s*|\s+(?:and|or)\s+)[A-D]\b)+)',body,re.I)
        if alternatives:
            letters.extend(x.upper() for x in re.findall(r'\b[A-D]\b',alternatives.group(1),re.I))
        if scope=='outside-analysis-final-text':
            # <D> C </D> contains competing explicit choices even if C sits
            # on its own line. Do not accidentally discard the tag's D.
            letters.extend(x.upper() for x in re.findall(r'</?([A-D])\s*>',body,re.I))
            letters.extend(x.upper() for x in re.findall(r'<[A-D]\s*>\s*([A-D])[.]?\s*</[A-D]\s*>',body,re.I))
        single=re.fullmatch(r'''[\s'"`*]*([A-D])[\s.,:;!?'"`*]*''',body,re.I)
        if single: letters.append(single.group(1).upper())
        leading=re.match(r'^\s*([A-D])\s*[,\-]\s+',body)
        if leading: letters.append(leading.group(1))
        # Repeated standalone decisions such as B </answer> B are still one
        # decision only when all recovered letters agree.
        for line in body.splitlines():
            match=re.fullmatch(r'\s*(?:(?:option|choice)\s+)?([A-D])[.!?,;:]?\s*',line,re.I)
            if match: letters.append(match.group(1).upper())
        # A bare B is an option letter even when an option's content is the
        # musical note B. Content matching is only a fallback without a letter.
        matched=option_text_choices(body,choices) if not letters else []
        if letters:
            candidates.extend(letters)
            evidence.append(dict(rule=scope+'-explicit-letter',letters=sorted(set(letters)),body=body))
        if matched:
            candidates.extend(matched)
            evidence.append(dict(rule=scope+'-unique-option-text' if len(matched)==1 else 'duplicate-option-text',letters=matched,body=body))
    if not candidates:
        return dict(previous,scoring_version=VERSION,v4_evidence=[],v4_recovery_rules=[])
    # Earlier untagged analysis may have misled v3. Decide from these final
    # scopes and retain every competing decision found in those scopes.
    letters=sorted(set(candidates))
    valid=len(letters)==1 and letters[0] in 'ABCD'[:len(choices)]
    return dict(prediction=letters[0] if valid else None,parsed=valid,
                parse_reason='explicit-final-decision-v4' if valid else 'conflicting-or-invalid-final-options-v4',
                conflicting_letters=letters if not valid else [],scoring_version=VERSION,
                v4_evidence=evidence,v4_recovery_rules=sorted({x['rule'] for x in evidence}))
