#!/usr/bin/env python3
"""Uniform CPU-only regrading of saved MCQ answers, preserving old artifacts."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

from score_generalization_analysis_mcq_v3 import score_rows as score_v3
from score_generalization_analysis_mcq_v4 import parse_analysis_option, VERSION
from rescore_generalization_explicit_formats import MODELS, name, read, sha, write, rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    root=p.parse_args().root.resolve();out=root/'rescore_formats_and_option_text_v4';out.mkdir(exist_ok=True)
    checksums={};table=[];changes=[];unparsed=[];checks={};recovery=Counter()
    for benchmark in ['omnivideobench','dailyomni']:
        lp=root/'data'/benchmark/'labels.jsonl';sp=root/'data'/benchmark/'inputs.jsonl'
        labels,sources=read(lp),read(sp);labelmap={x['sample_id']:x for x in labels}
        assert len(labelmap)==len(labels)==len(sources)==500
        for path in [lp,sp]:checksums[str(path)]=sha(path)
        source_by_id={x.get('case_id') or x.get('prompt_id'):x for x in sources}
        assert set(source_by_id)==set(labelmap)
        for key,label in labelmap.items():
            prompt='\n'.join(str(m['content']) for m in source_by_id[key]['messages'] if m['role']=='user')
            assert all(f'{chr(65+i)}. {choice}' in prompt for i,choice in enumerate(label['choices'])),key
        for model in MODELS:
            folder=root/'eval'/benchmark/model;rp=folder/'results.jsonl';op=folder/'scored.jsonl'
            for path in [rp,op]:checksums[str(path)]=sha(path)
            raw,old=read(rp),read(op);oldmap={x['sample_id']:x for x in old}
            reproduced=score_v3(raw,labels,sources)
            assert len(raw)==len(oldmap)==len(reproduced)==500
            assert set(oldmap)==set(labelmap)
            for x in reproduced:
                assert all(x[k]==oldmap[x['sample_id']][k] for k in ['response','prediction','parsed','correct']),x['sample_id']
            new=[]
            for before in old:
                key=before['sample_id'];choices=labelmap[key]['choices']
                parsed=parse_analysis_option(before['response'],choices)
                record=dict(before)
                record['previous_v3_scoring']={k:before[k] for k in ['prediction','parsed','correct','parse_reason']}
                record.update(parsed)
                record['correct']=bool(record['parsed'] and record['prediction']==before['gold_answer'])
                record['choices']=choices
                record['prediction_changed_from_v3']=record['prediction']!=before['prediction']
                new.append(record)
                if record['prediction_changed_from_v3']:
                    changes.append(dict(benchmark=benchmark,model=model,**record))
                    if record['parsed'] and not before['parsed']:
                        for rule in record['v4_recovery_rules']:recovery[rule]+=1
                if not record['parsed']:unparsed.append(dict(benchmark=benchmark,model=model,**record))
            total=len(new);correct=sum(x['correct'] for x in new);oldcorrect=sum(x['correct'] for x in old)
            summary=dict(benchmark=benchmark,model=model,total=total,old_correct=oldcorrect,correct=correct,
                old_accuracy_percent=100*oldcorrect/total,accuracy_percent=100*correct/total,
                old_unparsed=sum(not x['parsed'] for x in old),unparsed=sum(not x['parsed'] for x in new),
                recovered=sum(x['parsed'] and not x['previous_v3_scoring']['parsed'] for x in new),
                revoked=sum(not x['parsed'] and x['previous_v3_scoring']['parsed'] for x in new),
                changed_previously_parsed=sum(x['parsed'] and x['previous_v3_scoring']['parsed'] and x['prediction_changed_from_v3'] for x in new),
                strict_answer_letter=sum(x.get('strict_answer_letter',False) for x in old),
                strict_complete_format=sum(x.get('strict_complete_format',False) for x in old),scoring_version=VERSION)
            dest=out/benchmark/model;rows(dest/'scored.jsonl',new);write(dest/'summary.json',summary)
            table.append(summary);checks[benchmark+'/'+model]=dict(rows=total,old_v3_reproduced=True,
                input_signature_join_checked=True,option_text_present_in_prompt=True,new_scored_sha256=sha(dest/'scored.jsonl'))
    for path,digest in checksums.items():assert sha(Path(path))==digest,'Original file changed: '+path
    rows(out/'changed_predictions.jsonl',changes);rows(out/'remaining_unparsed.jsonl',unparsed)
    with (out/'scores.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    write(out/'comparison.json',dict(completed_tasks=26,rows=table))
    write(out/'validation.json',dict(at=datetime.now().astimezone().isoformat(),version=VERSION,tasks=checks,
        total_rows=13000,original_files_unchanged=True,rules_depend_on_gold=False,
        parser_arguments=['model_response','input_choices'],original_sha256=checksums,
        recovery_rule_counts=dict(recovery),
        code_sha256={filename:sha(Path(__file__).with_name(filename)) for filename in
            ['rescore_generalization_mcq_v4.py','score_generalization_analysis_mcq_v4.py']}))
    code=out/'code';code.mkdir(exist_ok=True)
    for filename in ['rescore_generalization_mcq_v4.py','score_generalization_analysis_mcq_v4.py']:
        shutil.copy2(Path(__file__).with_name(filename),code/filename)
    by={(x['benchmark'],x['model']):x for x in table}
    text=['# 明确选项提示评测：格式修复与选项原文映射（v4）','',
        '对 26 项评测的 13,000 条已保存回答统一重评分；未调用模型重新推理。每组仍为 500 题，未改变样本、生成参数、媒体输入或原文件。','',
        '## 解析规则','',
        '旧评分已使用 v3 字母解析。新增修复误写的 answer 起止标签、尾部多余关闭标签、转义标签、仅含一个字母的替代标签，以及单字母后的逗号等标点。',
        '最终答案字段直接写出选项内容时，进行唯一的选项原文匹配。仅统一大小写、空白、外围标点、冠词和独立数字/序数写法（如 Three 与 3、25,000 与 25000、third position 与 3rd）。不会将选项文本在 analysis 中的普通出现当成最终答案。',
        '解析函数只接收回答和输入选项，不能访问正确答案。多个最终答案冲突、选项文本重复、没有明确最终答案、无法唯一匹配的改写继续计为无法解析。未使用相似度、语义模型或人工逐题指定答案。',
        '严格格式遵循率保持原值；恢复一个选项并不代表原回答遵守了 prompt。无法解析仍按错误计分。','',
        '## 准确率：v3 → v4','',
        '| 模型 | OmniVideoBench | DailyOmni |','|---|---:|---:|']
    for model in MODELS:
        values=[by[b,model] for b in ['omnivideobench','dailyomni']]
        text.append('| '+name(model)+' | '+' | '.join(f"{x['old_accuracy_percent']:.1f}% → **{x['accuracy_percent']:.1f}%**" for x in values)+' |')
    text+=['','## 无法解析数量','', '| 模型 | OmniVideoBench（原→现） | DailyOmni（原→现） |','|---|---:|---:|']
    for model in MODELS:
        text.append('| '+name(model)+' | '+' | '.join(f"{by[b,model]['old_unparsed']} → {by[b,model]['unparsed']}" for b in ['omnivideobench','dailyomni'])+' |')
    text+=['',f"共恢复 {sum(x['recovered'] for x in table)} 条原未解析回答；撤销 {sum(x['revoked'] for x in table)} 条有冲突的原解析；修正 {sum(x['changed_previously_parsed'] for x in table)} 条原字母误提取；剩余 {len(unparsed)} 条无法解析。",'',
        '## 审计与限制','',
        '分数变化只来自解析规则变化，不能解释为模型能力提升。此处“原→现”比较的是同一轮严格 prompt 的相同回答，不是上一轮较短 prompt 的重新生成结果。',
        '自然语言同义表达可能仍无法匹配，例如 Grateful 与 Thankful。剩余未解析中也可能有正确答案，因此本结果仍受到格式与表达方式影响；不能将未解析数量直接当作音视频理解错误数量。',
        '逐题变化：changed_predictions.jsonl；剩余未解析：remaining_unparsed.jsonl；哈希与完整性核验：validation.json；每组逐题新评分位于 数据集/模型/scored.jsonl。']
    (out/'RESULTS.zh-CN.md').write_text('\n'.join(text)+'\n')
    write(root/'LATEST_RESCORE.json',dict(at=datetime.now().astimezone().isoformat(),version=VERSION,
        directory=str(out),report=str(out/'RESULTS.zh-CN.md'),comparison=str(out/'comparison.json'),
        original_scoring_preserved=True))
    print(json.dumps(dict(output=str(out),tasks=26,rows=13000,recovered=sum(x['recovered'] for x in table),
        revoked=sum(x['revoked'] for x in table),remaining_unparsed=len(unparsed),recovery_rules=dict(recovery))))


if __name__=='__main__':main()
