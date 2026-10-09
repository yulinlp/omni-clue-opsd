#!/usr/bin/env python3
"""Paired model comparison; bootstrap videos to keep same-video questions together."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from omni_opsd.evaluation import scored_mcq_rows

def read(p):
    return [json.loads(l) for l in p.open() if l.strip()]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--partial', action='store_true')
    a = p.parse_args()
    tasks = json.loads((a.root/'tasks.json').read_text())
    labels = read(a.root/'data/worldsense.openqa.labels.jsonl')
    labelmap = {r['sample_id']:r for r in labels}
    ids = sorted(labelmap)
    videos = sorted({r['video_id'] for r in labels})
    vi = {v:i for i,v in enumerate(videos)}
    counts = np.bincount([vi[labelmap[k]['video_id']] for k in ids], minlength=len(videos))
    rng = np.random.default_rng(20261001)
    draws = rng.integers(0,len(videos),size=(5000,len(videos)))
    denominator = counts[draws].sum(axis=1)
    report = {'n_questions':len(ids),'n_videos':len(videos),'bootstrap_unit':'video',
              'bootstrap_resamples':5000,'seed':20261001,'modes':{}}
    table=[]
    for mode in ('mcq','openqa'):
        sets={}
        summaries={}
        for task in tasks:
            out=a.root/mode/task['label']
            if not (out/'summary.json').exists():
                if a.partial:continue
                raise ValueError('Incomplete model: '+str(out))
            summary=json.loads((out/'summary.json').read_text())
            if mode=='mcq':
                rows=scored_mcq_rows(read(out/'results.jsonl'), read(a.root/'data/worldsense.labels.jsonl'),
                                    read(a.root/'data/worldsense.answer_free.jsonl'))
                scores={r['sample_id']:bool(r['correct']) for r in rows}
            else:
                rows=read(out/'scored.jsonl')
                scores={r['sample_id']:bool(r['semantic_correct']) for r in rows}
                assert len(scores)==len(rows)
            assert set(scores)==set(ids)
            sets[task['label']]=np.array([scores[k] for k in ids],dtype=float)
            summaries[task['label']]=summary
        comparisons={}
        if 'base' in sets:
            for name,y in sets.items():
                d=y-sets['base']
                sums=np.bincount([vi[labelmap[k]['video_id']] for k in ids],weights=d,minlength=len(videos))
                boot=sums[draws].sum(axis=1)/denominator
                comparisons[name]=dict(delta_accuracy=float(d.mean()),
                                       ci95_video_bootstrap=[float(x) for x in np.quantile(boot,[.025,.975])],
                                       wrong_to_correct=int((d>0).sum()),correct_to_wrong=int((d<0).sum()))
        for name,y in sets.items():
            tiers={t:dict(total=sum(labelmap[k]['tier']==t for k in ids),
                          accuracy=float(np.mean([y[i] for i,k in enumerate(ids) if labelmap[k]['tier']==t])))
                   for t in ('A','B','D')}
            summary=summaries[name]
            row=dict(mode=mode,model=name,total=len(ids),correct=int(y.sum()),accuracy=float(y.mean()),
                     delta_vs_base=comparisons.get(name,{}).get('delta_accuracy'),
                     ci95_low=comparisons.get(name,{}).get('ci95_video_bootstrap',[None,None])[0],
                     ci95_high=comparisons.get(name,{}).get('ci95_video_bootstrap',[None,None])[1],
                     format_or_parse_rate=summary.get('format_rate',summary.get('parse_rate')),
                     judge_calibration_accuracy=summary.get('judge_calibration',{}).get('accuracy'),
                     tier_A_accuracy=tiers['A']['accuracy'],tier_B_accuracy=tiers['B']['accuracy'],tier_D_accuracy=tiers['D']['accuracy'])
            table.append(row)
        report['modes'][mode]=dict(complete_models=list(sets),summaries=summaries,paired_vs_base=comparisons)
    out=a.root/('comparison_partial' if a.partial else 'comparison');out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if table:
        with (out/'table.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    print(json.dumps({'models':{m:len(r['complete_models']) for m,r in report['modes'].items()},'output':str(out)}))

if __name__=='__main__':
    main()
