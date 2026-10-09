#!/usr/bin/env python3
"""Include unfinished analyses in diagnostics; never modify primary answer grades."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from tokenizers import Tokenizer
from score_worldsense_openqa import extract

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    tokenizer=Tokenizer.from_file('/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-assets/models/Qwen2.5-Omni-7B/tokenizer.json')
    reports={}
    for task in json.loads((a.root/'tasks.json').read_text()):
        out=a.root/'openqa'/task['label'];summary=json.loads((out/'summary.json').read_text())
        rows=[json.loads(l) for l in (out/'scored.jsonl').open() if l.strip()]
        before=[(r['sample_id'],r['semantic_correct'],r['final_answer']) for r in rows]
        diagnostics=[]
        for row in rows:
            answer,analysis,valid,format_ok=extract(row['response'])
            assert answer==row['final_answer'] and valid==row['valid_answer'] and format_ok==row['format_ok']
            row['analysis']=analysis;row['analysis_words']=len(analysis.split())
            words=re.findall(r'\w+',analysis.lower())
            windows=[tuple(words[i:i+20]) for i in range(max(0,len(words)-19))]
            repetition=1-len(set(windows))/len(windows) if windows else 0.0
            tokens=len(tokenizer.encode(row['response'],add_special_tokens=False).ids)
            diagnostics.append(dict(sample_id=row['sample_id'],analysis_words=row['analysis_words'],
                                    analysis_closed=bool(re.search(r'</analysis>',row['response'],re.I)),
                                    valid_final_answer=valid,decoded_response_tokens=tokens,
                                    near_generation_cap=tokens>=766,repeated_20gram_fraction=repetition,
                                    repetitive_analysis=len(words)>=60 and repetition>=.4))
        assert before==[(r['sample_id'],r['semantic_correct'],r['final_answer']) for r in rows]
        n=len(rows)
        report=dict(total=n,analysis_words_mean_including_unclosed=sum(x['analysis_words'] for x in diagnostics)/n,
                    analysis_over_120_fraction=sum(x['analysis_words']>120 for x in diagnostics)/n,
                    unfinished_analysis_count=sum(not x['analysis_closed'] for x in diagnostics),
                    repetitive_analysis_count=sum(x['repetitive_analysis'] for x in diagnostics),
                    invalid_final_answers=sum(not x['valid_final_answer'] for x in diagnostics),
                    decoded_response_near_cap_count=sum(x['near_generation_cap'] for x in diagnostics),
                    cap_detection_note='Retokenized decoded text >=766 for max_new_tokens=768; approximate, not recorded finish_reason.',
                    primary_answer_scores_unchanged=True)
        summary.setdefault('analysis_words_mean_closed_only_previous',summary['analysis_words_mean'])
        summary['analysis_words_mean']=report['analysis_words_mean_including_unclosed']
        summary['analysis_over_120_fraction']=report['analysis_over_120_fraction']
        summary['analysis_word_metrics_include_unclosed']=True
        summary['generation_diagnostics']=report
        (out/'scored.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        (out/'analysis_diagnostics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in diagnostics))
        (out/'review_candidates.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows
                                                        if x['scoring_method']=='local-judge' or not x['format_ok']))
        reports[task['label']]=report
    (a.root/'analysis_diagnostics_summary.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports))

if __name__=='__main__':main()
