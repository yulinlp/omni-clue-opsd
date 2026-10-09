#!/usr/bin/env python3
"""Publish partial/final open-QA tables and paired uncertainty as scoring finishes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

REPO=Path('/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd')
sys.path.insert(0,str(REPO/'training_code/src'))
sys.path.insert(0,str(REPO/'training_code/scripts'))

from aggregate_worldsense_training_matched_eval import VideoBootstrap, paired_comparison
from rescore_generalization_explicit_formats import name
from worker_worldsense_card_pool import atomic_save


def read(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def report(root):
    comparison=json.loads((root/'comparison.json').read_text());table=[];audits={}
    models=[t['label'] for t in json.loads((root/'tasks.json').read_text())]
    for bench in ['omnivideobench','dailyomni']:
        labels={r['sample_id']:r for r in read(root/'data'/bench/'labels.jsonl')};ids=sorted(labels)
        bootstrap=VideoBootstrap(ids,labels)
        available={r['model']:r for r in comparison['rows'] if r['benchmark']==bench}
        loaded={}
        for model in available:
            p=root/'eval'/bench/model/'scored.jsonl';rows=read(p)
            mapping={r['sample_id']:r for r in rows}
            if len(rows)!=500 or set(mapping)!=set(ids):raise ValueError('Incomplete scored rows')
            loaded[model]=[mapping[i]['semantic_correct'] for i in ids]
            audits[bench+'/'+model]={'scored_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':500}
        for model in models:
            if model not in available:continue
            summary=available[model]
            row=dict(benchmark=bench,model=model,total=summary['total'],correct=summary['correct'],
                     incorrect=summary['incorrect'],uncertain=summary['uncertain'],
                     lower_percent=summary['accuracy_lower_bound']*100,upper_percent=summary['accuracy_upper_bound']*100,
                     reference_insufficient=summary['reference_insufficient'],
                     delta_lower_pp=None,delta_upper_pp=None,ci95_lower_pp=None,ci95_upper_pp=None)
            if 'base' in loaded:
                paired=paired_comparison(loaded[model],loaded['base'],bootstrap,model=='base')
                row.update(delta_lower_pp=100*paired['delta_accuracy_lower_bound'],delta_upper_pp=100*paired['delta_accuracy_upper_bound'],
                           ci95_lower_pp=100*paired['ci95_video_bootstrap_conservative_envelope'][0],
                           ci95_upper_pp=100*paired['ci95_video_bootstrap_conservative_envelope'][1])
            table.append(row)
    folder=root/'report';folder.mkdir(exist_ok=True)
    atomic_save(folder/'summary.json',dict(completed_tasks=len(table),total_tasks=26,rows=table,validation=audits))
    if table:
        with (folder/'scores.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    by={(r['benchmark'],r['model']):r for r in table}
    doc=['# OmniVideoBench / DailyOmni 开放式评测结果','',
         f"语义评分已完成 {len(table)}/26 组。尚未完成的模型不列为最终分数。",'',
         '每组分母 500；准确率下界=确认正确/500，上界=(确认正确+未决)/500。这里的上下界不是置信区间。', '',
         '模型输入完整音视频和问题，无选择题选项。主分数仅评价最终自然语言答案，分析事实正确性未单独评分。', '',
         'OmniVideoBench 有16题、DailyOmni有3题在生成前统一标记为参考不足，保留未决。参考充分子集的分数另见每个模型 summary.json。', '',
         '| 模型 | OmniVideoBench | DailyOmni |','|---|---:|---:|']
    for m in models:
        cells=[]
        for b in ['omnivideobench','dailyomni']:
            r=by.get((b,m));cells.append(f"{r['lower_percent']:.2f}–{r['upper_percent']:.2f}%" if r else '评分未完成')
        doc.append('| '+name(m)+' | '+' | '.join(cells)+' |')
    doc+=['','## 对比原模型','',
          '差值与95%区间按视频分组配对bootstrap计算（5,000次），保留未决项的不确定性；仅针对本次固定回答，不含换随机种子的波动或裁判误差。', '',
          '| 数据集 | 模型 | 差值上下界（百分点） | 保守95%区间 |','|---|---|---:|---:|']
    for r in table:
        if r['model']=='base' or r['delta_lower_pp'] is None:continue
        doc.append(f"| {r['benchmark']} | {name(r['model'])} | {r['delta_lower_pp']:+.2f}～{r['delta_upper_pp']:+.2f} | [{r['ci95_lower_pp']:+.2f}, {r['ci95_upper_pp']:+.2f}] |")
    (folder/'RESULTS.zh-CN.md').write_text('\n'.join(doc)+'\n')
    return len(table)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--watch',action='store_true')
    a=p.parse_args();previous=None
    while True:
        try:
            f=a.root/'comparison.json'
            if f.exists():
                source=json.loads(f.read_text());signature=source['completed_tasks']
                if signature!=previous:
                    count=report(a.root);previous=signature;print(json.dumps({'scored_tasks':count}),flush=True)
                    if count==26:return
        except Exception as e:
            print(json.dumps({'report_error':repr(e)}),flush=True)
            if not a.watch:raise
        if not a.watch:return
        time.sleep(20)

if __name__=='__main__':main()
