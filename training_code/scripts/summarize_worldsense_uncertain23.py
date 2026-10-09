#!/usr/bin/env python3
"""Validate and summarize the completed uncertainty repair experiment."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    root=p.parse_args().root.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    status=json.loads((root/'status.json').read_text())
    assert status['status']=='complete' and status['completed']==23
    results=read(root/'results.jsonl'); samples=read(root/'samples.jsonl')
    original=read(Path(manifest['source'])/'results.jsonl')
    assert len(results)==len({r['question_id'] for r in results})==23
    assert {r['question_id'] for r in results}==set(manifest['question_ids'])
    assert all(r['status']!='technical_error' for r in results)
    for path,expected in manifest['frozen_sha256'].items():assert sha(Path(path))==expected
    passed=[r for r in results if r['status']=='reannotated_verified']
    for r in passed:
        last=r['checks'][-1]
        sample=next(s for s in samples if s['sample_id']==r['question_id'])
        assert last['passed'] and last['reason']=='both_pass' and last['blind']['answer']==sample['answer']
        assert all(c['verdict']=='PASS' for c in last['observation_audit']['claims'])
        assert 1<=len(r['annotation']['observation'].split())<=120
    expected_candidates=[r['annotation'] for r in original if r['status'] in ['retained_original','reannotated_verified']]+[r['annotation'] for r in passed]
    combined=read(root/'pilot50_combined_candidates.jsonl')
    assert {r['question_id']:r for r in combined}=={r['question_id']:r for r in expected_candidates}
    request_files=list(root.glob('items/*/requests/*.json'))
    requests=[json.loads(p.read_text()) for p in request_files]
    summary=dict(completed_at=status['updated_at'],rerun_count=23,
                 counts=dict(Counter(r['status'] for r in results)),
                 final_failure_reasons=dict(Counter(r['checks'][-1]['reason'] for r in results if r['status']=='needs_review')),
                 rounds=dict(Counter(r['reannotation_attempts'] for r in results)),
                 cases_with_focused_recheck=sum(any('focused_recheck' in c for c in r['checks']) for r in results),
                 cases_with_nonblocking_classification=sum(any(any(d['severity']=='non_blocking' for d in c['uncertainty_review']['decisions']) for c in r['checks']) for r in results),
                 newly_verified_ids=[r['question_id'] for r in passed],
                 combined_original50_counts=status['original_50_combined_counts'],
                 combined_candidate_count=len(combined),human_reviewed=0,
                 source_files_unchanged=True,training_data_updated=False,
                 request_stages=len(requests),recorded_model_responses=sum(sum('response' in a for a in r.get('attempts',[])) for r in requests),
                 technical_failure_fixed='Two oversized focus-range responses recovered by lossless <=60s view splitting; only failed items resumed',
                 hashes={name:sha(root/name) for name in ['results.jsonl','auto_verified_candidates.jsonl','pilot50_combined_candidates.jsonl','manifest.json']})
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    reasons={'repaired_clue_answer_wrong_or_unknown':'新 clue 盲答仍错误或无法确定',
             'observation_or_gold_not_verified':'observation / 标准答案一致性核验未通过',
             'blocking_uncertainty_after_focus':'片段复核后仍有关键事实疑点',
             'observation_blocking_uncertainty':'observation 写作仍有关键疑点',
             'both_pass':'盲答和全片 observation 核验均通过'}
    text=['# 原23条疑点样本重跑结果','',f"完成时间：{status['updated_at']}。23/23 完成，技术错误遗留为 0。",'',
          f"新增自动核验通过 {len(passed)} 条，仍待复核 {23-len(passed)} 条。与原先23条通过候选合并后，原50题共有 {len(combined)} 条通过候选。尚未人工核验，未覆盖或更新训练数据。",'',
          '## 实施的改进','',
          '- 独立按是否影响选答案/事实支持来区分关键疑点与附带疑点；附带疑点不直接停止。',
          '- 关键疑点先复查相关AV，再放回全片上下文重新整理事实；两轮预算内继续重试。',
          '- 超过60秒的复核范围自动拆分，每次不超过60秒，完整保留选中证据及原时间映射。实际画质记录在请求元数据中。',
          '- 下一轮接收具体疑点、失败声明、证据及此前复核发现。',
          '- 提示明确区分数据集“I don’t know”选项与模型看不清/听不清导致的UNKNOWN。',
          '- 通过标准不变：独立clue盲答正确，并通过全片逐声明一致性核验。','',
          '## 最终未通过原因','', '| 原因 | 数量 |','|---|---:|']
    for reason,count in summary['final_failure_reasons'].items():text.append(f"| {reasons.get(reason,reason)} | {count} |")
    text+=['','## 逐题结果','', '| 题目ID | 结果 | 使用轮次 | 最终原因 |','|---|---|---:|---|']
    for r in results:text.append(f"| {r['question_id']} | {'通过' if r['status']=='reannotated_verified' else '待复核'} | {r['reannotation_attempts']} | {reasons.get(r['checks'][-1]['reason'],r['checks'][-1]['reason'])} |")
    text+=['','## 新增通过的 observation','']
    for r in passed:text += [f"### {r['question_id']}",'',r['annotation']['observation'],'']
    text+=['## 如何解读','',
           '待复核并不等于已证明原标注或标准答案错误。同一模型在不同片段、完整视频、不同提示下可能判断不一致；这套自动核验受输入画质和模型能力限制。',
           '此次修正消除了部分程序性过度拦截；未通过的题没有靠反复采样直到碰巧答对来强行通过，也没有更改标准答案。',
           '原运行目录保持不变；修复前代码及两条技术失败记录在 code_versions/before_focus_split/。新代码与全部原始请求、回复、媒体映射可追溯。',
           'summary.json 为机器可读汇总，review.csv 为逐题记录；pilot50_combined_candidates.jsonl 为合并后的候选集。']
    (root/'RESULTS.zh-CN.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
