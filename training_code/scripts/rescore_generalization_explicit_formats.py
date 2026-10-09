#!/usr/bin/env python3
"""Regrade all saved external benchmark answers; preserve original artifacts."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

from score_generalization_analysis_mcq import score_rows as score_v1
from score_generalization_analysis_mcq_v2 import score_rows, VERSION
from aggregate_worldsense_training_matched_eval import VideoBootstrap, paired_comparison

MODELS=['base']+[f'{m}_epoch{e}' for m in ['sft_lora','sft_full','clue_obs','clue_noobs'] for e in [1,2,3]]
NAMES={'base':'原模型','sft_lora':'LoRA SFT','sft_full':'全参数 SFT',
       'clue_obs':'全参数 CLUE（含 observation）','clue_noobs':'全参数 CLUE（不含 observation）'}


def name(model):
    if model=='base':return NAMES[model]
    m,e=model.rsplit('_epoch',1);return NAMES[m]+' 第'+e+'轮'


def read(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def rows(path, values):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve();out=root/'rescore_explicit_formats_v2';out.mkdir(exist_ok=True)
    table=[];changes=[];unparsed=[];audit={};checksums={}
    for bench in ['omnivideobench','dailyomni']:
        source_path=root/'data'/bench/'inputs.jsonl';label_path=root/'data'/bench/'labels.jsonl'
        sources,labels=read(source_path),read(label_path)
        checksums[str(source_path)]=sha(source_path);checksums[str(label_path)]=sha(label_path)
        ids=sorted(r['sample_id'] for r in labels)
        assert len(ids)==len(set(ids))==len(sources)==500
        bootstrap=VideoBootstrap(ids,{r['sample_id']:r for r in labels})
        for model in MODELS:
            folder=root/'eval'/bench/model
            raw_path=folder/'results.jsonl';old_path=folder/'scored.jsonl'
            checksums[str(raw_path)]=sha(raw_path);checksums[str(old_path)]=sha(old_path)
            raw,old=read(raw_path),read(old_path)
            oldmap={r['sample_id']:r for r in old}
            recomputed=score_v1(raw,labels,sources)
            assert len(raw)==len(oldmap)==len(recomputed)==500
            assert all(r==oldmap[r['sample_id']] for r in recomputed), 'Original scores not reproducible'
            new=score_rows(raw,labels,sources);newmap={r['sample_id']:r for r in new}
            assert sorted(newmap)==ids
            values=[newmap[k]['correct'] for k in ids]
            if model=='base':base_values=values
            pair=paired_comparison(values,base_values,bootstrap,model=='base')
            correct=sum(values); old_correct=sum(r['correct'] for r in old)
            recovered=sum(r['parsed'] and not oldmap[r['sample_id']]['parsed'] for r in new)
            revoked=sum(not r['parsed'] and oldmap[r['sample_id']]['parsed'] for r in new)
            summary=dict(benchmark=bench,model=model,total=500,old_correct=old_correct,correct=correct,
                         old_accuracy_percent=old_correct/5,accuracy_percent=correct/5,
                         old_unparsed=sum(not r['parsed'] for r in old),unparsed=sum(not r['parsed'] for r in new),
                         recovered=recovered,revoked_as_ambiguous=revoked,
                         delta_vs_base_pp=pair['delta_accuracy']*100,
                         paired_ci95_lower_pp=pair['ci95_video_bootstrap'][0]*100,
                         paired_ci95_upper_pp=pair['ci95_video_bootstrap'][1]*100)
            table.append(summary)
            for r in new:
                before=oldmap[r['sample_id']]
                if r['prediction']!=before['prediction']:
                    changes.append(dict(benchmark=bench,model=model,sample_id=r['sample_id'],
                                        old_prediction=before['prediction'],prediction=r['prediction'],gold=r['gold_answer'],
                                        old_correct=before['correct'],correct=r['correct'],rules=r['recovery_rules'],
                                        reason=r['parse_reason'],response=r['response']))
                if not r['parsed']:unparsed.append(dict(benchmark=bench,model=model,**r))
            dest=out/bench/model
            rows(dest/'scored.jsonl',new);write(dest/'summary.json',dict(summary,scoring_version=VERSION,paired_vs_base=pair))
            audit[bench+'/'+model]=dict(rows=500,old_scores_exactly_reproduced=True,identical_unique_ids=True,
                                      exact_raw_input_signature_join=True,raw_sha256=checksums[str(raw_path)],
                                      old_scored_sha256=checksums[str(old_path)],new_scored_sha256=sha(dest/'scored.jsonl'))
    for path,expected in checksums.items():assert sha(Path(path))==expected,'Original artifact changed'
    rows(out/'changed_predictions.jsonl',changes);rows(out/'remaining_unparsed.jsonl',unparsed)
    with (out/'scores.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    write(out/'validation.json',dict(version=VERSION,time=datetime.now().astimezone().isoformat(),
                                   total_tasks=26,total_rows=13000,original_files_unchanged=True,
                                   rules_depend_on_gold=False,source_sha256=checksums,tasks=audit,
                                   code_sha256={str(Path(__file__)):sha(Path(__file__)),
                                                str(Path(__file__).with_name('score_generalization_analysis_mcq_v2.py')):sha(Path(__file__).with_name('score_generalization_analysis_mcq_v2.py'))}))
    by={(r['benchmark'],r['model']):r for r in table}
    text=['# 选择题明确格式修正后的评分（v2）','',
          '统一重新评分 26 组、13,000 条已保存回答；未重新生成回答，原始文件和旧评分未修改。每组固定 500 题。','',
          '## 新规则','',
          '- 保留原先明确 answer/option 字段的优先级；兼容 solution 字段。多个明确最终字段冲突时不选。',
          '- 无有效最终字段时，识别 <B>、<B></B> 等字母标签，末尾只含字母的 analysis 字段，以及单独最终选项行。',
          '- <B>C</B> 等互相矛盾的备用格式不选；不会从分析段中的普通选项讨论或自然语言答案反推选项。',
          '- 标准答案只用于比较提取后的字母，绝不用于决定怎样提取。剩余未解析回答仍计错；分析内容不单独评分。','',
          '## 正确率：旧评分 → 修正评分','', '| 模型 | OmniVideoBench | DailyOmni |','|---|---:|---:|']
    for model in MODELS:
        values=[by[(b,model)] for b in ['omnivideobench','dailyomni']]
        text.append('| '+name(model)+' | '+' | '.join(f"{r['old_accuracy_percent']:.1f}% → **{r['accuracy_percent']:.1f}%**（{r['correct']}/500）" for r in values)+' |')
    text+=['','## 未解析数量与相对原模型差值','',
           '| 数据集 | 模型 | 未解析：旧→新 | 相对修正后原模型（百分点） | 配对95%区间 |','|---|---|---:|---:|---:|']
    for r in table:
        text.append(f"| {r['benchmark']} | {name(r['model'])} | {r['old_unparsed']} → {r['unparsed']} | {r['delta_vs_base_pp']:+.1f} | [{r['paired_ci95_lower_pp']:+.2f}, {r['paired_ci95_upper_pp']:+.2f}] |")
    text+=['','## 解释与限制','',
           '原评分明显漏识别原模型及 LoRA SFT 的替代格式，应以本次修正后的表格比较模型。旧表中的提升幅度不能继续沿用。',
           '配对区间按视频聚类 bootstrap 5,000 次、seed=20261003；只反映固定回答的抽样不确定性，不包含解码随机种子变化或标准答案错误，也未进行多重比较校正。',
           '选择题分数仍衡量正确最终选项，不等同于单独证明音视频理解能力提高。未改提示、媒体预算、temperature 或生成长度。VideoOdyssey 没有符合 300 秒限制的本地样本，仍无结果。','',
           '逐题变化见 changed_predictions.jsonl；剩余未解析见 remaining_unparsed.jsonl；完整校验见 validation.json；每模型新评分位于本目录的数据集/模型/scored.jsonl。']
    (out/'RESULTS.zh-CN.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'output':str(out),'tasks':26,'rows':13000,'changed_predictions':len(changes),'remaining_unparsed':len(unparsed)}))
    for r in table:print(json.dumps(r))


if __name__=='__main__':main()
