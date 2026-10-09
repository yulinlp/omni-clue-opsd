"""Accept explicit option wrappers, while preserving conservative MCQ grading."""
import copy
import re

from score_worldsense_mcq_v2 import scored_mcq_v2_rows, parse_mcq_choice
from omni_opsd.rewards import completion_text, extract_mcq_answer

def normalize_explicit_option(text):
    text=re.sub(r'<(/?)option\s*>',r'<\1answer>',text,flags=re.I)
    text=re.sub(r'<analysis\s*>.*?</analysis\s*>','\n',text,flags=re.S|re.I)
    opening=re.search(r'<analysis\s*>',text,re.I)
    if opening:
        # An unfinished analysis may itself say "the answer is C". Only a
        # separate explicit XML final-answer field establishes its boundary.
        final=re.search(r'<answer\s*>',text[opening.end():],re.I)
        text=text[:opening.start()]+(text[opening.end()+final.start():] if final else '')
    # Only an explicit final-answer tag supplies this context. Never replace
    # general mentions such as 'Option B looks plausible' inside an analysis.
    text=re.sub(r'(<answer\s*>\s*)(?:option|choice)\s*(?:is\s+|[:=]\s*|(?=[A-D](?!\w)))',
                r'\1',text,flags=re.I)
    fields=[]
    for opening in re.finditer(r'<answer\s*>',text,re.I):
        tail=text[opening.end():]
        close=re.search(r'</answer\s*>',tail,re.I)
        body=tail[:close.start()] if close else tail
        fields.append('<answer>'+body+'</answer>')
    # The requested final-answer field is authoritative. Plain reasoning may
    # compare A/B/C/D outside well-formed analysis tags; it is not a final vote.
    # Multiple conflicting final fields remain ambiguous under the base parser.
    return '\n'.join(fields) if fields else text

def parse_analysis_option(text):
    return parse_mcq_choice(normalize_explicit_option(completion_text(text)))

def score_rows(results,labels,sources):
    normalized=copy.deepcopy(results)
    for raw,row in zip(results,normalized):
        response=completion_text(raw.get('response',''))
        row['response']=normalize_explicit_option(response)
    scored=scored_mcq_v2_rows(normalized,labels,sources)
    # Rejoin original results by their model-facing input signature, not by text.
    from omni_opsd.evaluation import _input_signature
    signatures={_input_signature(r):r.get('case_id') or r.get('prompt_id') for r in sources}
    response_by_id={(r.get('case_id') or r.get('prompt_id') or signatures.get(_input_signature(r))):
                    completion_text(r.get('response','')) for r in results}
    for row in scored:
        original_response=response_by_id[row['sample_id']]
        row['parser_input_normalized']=original_response!=row['response']
        row['parser_input']=row['response']
        row['response']=original_response
        row['legacy_prediction']=extract_mcq_answer(original_response)
        row['legacy_parsed']=row['legacy_prediction'] is not None
        row['legacy_correct']=row['legacy_prediction']==row['gold_answer']
        row['strict_format_compliant']=bool(re.search(r'<analysis>.*?</analysis>',original_response,re.S|re.I)
                                            and re.search(r'<answer>\s*[ABCD]\s*</answer>',original_response,re.I))
    return scored
