#!/usr/bin/env python3
"""Audit every remaining MCQ parse failure, and uniformly apply v3 recovery."""
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import unicodedata

from tokenizers import Tokenizer
from score_generalization_analysis_mcq_v2 import score_rows as score_v2
from score_generalization_analysis_mcq_v3 import score_rows, VERSION
from aggregate_worldsense_training_matched_eval import VideoBootstrap, paired_comparison
from rescore_generalization_explicit_formats import (
    MODELS, read as read_rows, sha, write as write_json, rows as write_jsonl, name as model_name,
)

REPO = Path(__file__).resolve().parents[2]
BENCHMARKS = ['omnivideobench', 'dailyomni']
ROOT = REPO / 'training_runs/worldsense_generalization_mcq500_npu96_20261003'
OUT = ROOT / 'rescore_explicit_formats_v3'
CATEGORIES = {
    'answer_text_without_option': '答案栏有文字，但未明确选择唯一 A–D',
    'no_final_field': '没有最终答案字段或独立最终选项',
    'malformed_final_tag': '结尾出现异常/残缺标签',
    'conflicting_choices': '多个选项或标记冲突',
    'invalid_choice': '选择范围外字母',
    'empty_final_field': '最终答案字段为空',
}


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def answer_bodies(text):
    return [x.strip() for x in re.findall(r'<(?:answer|option|solution)\s*>(.*?)(?:</(?:answer|option|solution)\s*>|$)', text, re.S | re.I)]


def category(row):
    bodies = answer_bodies(row['response'])
    if row['parse_reason'] == 'conflicting-explicit-options' or any(
            re.fullmatch(r'[A-D](?:\s*[,/&]?\s*[A-D])+[.,]?', b) for b in bodies):
        return 'conflicting_choices'
    if any(re.fullmatch(r'[E-Z][.]?', b) for b in bodies):
        return 'invalid_choice'
    if bodies:
        visible = [re.sub(r'<[^>]*>', '', b).strip() for b in bodies]
        return 'answer_text_without_option' if any(visible) else 'empty_final_field'
    outside = re.sub(r'<analysis\s*>.*?</analysis\s*>', '', row['response'], flags=re.S | re.I)
    tail = outside.strip()
    if re.search(r'<(?!/?analysis\b)[^<>\n]{0,90}>?\s*$', tail, re.I):
        return 'malformed_final_tag'
    return 'no_final_field'


def normalize_content(text):
    text = unicodedata.normalize('NFKC', text).lower().strip()
    return re.sub(r'\s+', ' ', text).strip(' .,!?:;\"\'')


def choices(source):
    prompt = source['messages'][0]['content']
    options = prompt.split('Options:\n', 1)[1].split('\nBriefly analyze', 1)[0]
    return dict(re.findall(r'^([ABCD])\. (.*)$', options, re.M))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path('/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-assets/models/Qwen2.5-Omni-7B/tokenizer.json')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    audit = {'created_at': datetime.now().astimezone().isoformat(), 'scoring_version': VERSION,
             'sources': {str(tokenizer_path): sha(tokenizer_path)}, 'groups': {},
             'regenerated_answers': False, 'same_parser_for_all_models': True,
             'cap_note': 'Retokenized decoded text >=510 is approximate evidence of reaching the 512-token cap; original finish_reason is unavailable.'}
    for name in ['score_generalization_analysis_mcq.py', 'score_generalization_analysis_mcq_v2.py',
                 'score_generalization_analysis_mcq_v3.py', 'score_worldsense_mcq_v2.py', 'audit_generalization_unparsed.py']:
        p = Path(__file__).with_name(name)
        audit['sources'][str(p)] = sha(p)
    summaries, details, changed, all_remaining = [], [], [], []
    for bench in BENCHMARKS:
        sources = read_rows(ROOT / 'data' / bench / 'inputs.jsonl')
        labels = read_rows(ROOT / 'data' / bench / 'labels.jsonl')
        smap = {r['case_id']: r for r in sources}
        lmap = {r['sample_id']: r for r in labels}
        ids = sorted(lmap)
        assert len(ids) == len(sources) == len(labels) == 500
        bootstrap = VideoBootstrap(ids, lmap)
        by_model = {}
        for file in [ROOT / 'data' / bench / 'inputs.jsonl', ROOT / 'data' / bench / 'labels.jsonl']:
            audit['sources'][str(file)] = sha(file)
        for model in MODELS:
            raw_path = ROOT / 'eval' / bench / model / 'results.jsonl'
            prior_path = ROOT / 'rescore_explicit_formats_v2' / bench / model / 'scored.jsonl'
            for file in [raw_path, prior_path]:
                audit['sources'][str(file)] = sha(file)
            raw = read_rows(raw_path)
            prior = read_rows(prior_path)
            pmap = {r['sample_id']: r for r in prior}
            recomputed = score_v2(raw, labels, sources)
            assert len(raw) == len(prior) == len(pmap) == 500
            assert all(r == pmap[r['sample_id']] for r in recomputed)
            rows = score_rows(raw, labels, sources)
            assert {r['sample_id'] for r in rows} == set(ids)
            assert all(r['response'] == pmap[r['sample_id']]['response'] for r in rows)
            by_model[model] = {r['sample_id']: r for r in rows}
            write_jsonl(OUT / bench / model / 'scored.jsonl', rows)
            group_details = []
            for r in rows:
                old = pmap[r['sample_id']]
                if r['prediction_changed_from_v2']:
                    changed.append(dict(benchmark=bench, model=model, sample_id=r['sample_id'],
                                        previous_prediction=old['prediction'], prediction=r['prediction'],
                                        previous_correct=old['correct'], correct=r['correct'],
                                        gold=r['gold_answer'], rules=r['v3_recovery_rules'], response=r['response']))
                if old['parsed'] and r['parsed']:
                    continue
                bodies = answer_bodies(r['response'])
                tokens = len(tokenizer.encode(r['response'], add_special_tokens=False).ids)
                options = choices(smap[r['sample_id']])
                matches = {letter for b in bodies for letter, text in options.items()
                           if normalize_content(b) == normalize_content(text)}
                d = dict(benchmark=bench, model=model, sample_id=r['sample_id'],
                         v2_unparsed=not old['parsed'], v3_unparsed=not r['parsed'],
                         category=category(r) if not r['parsed'] else 'parser_omission_recovered',
                         decoded_tokens=tokens, near_512_token_cap=tokens >= 510,
                         analysis_closed='</analysis>' in r['response'].lower(),
                         answer_closed='</answer>' in r['response'].lower(),
                         mentions_reference=bool(re.search(r'provided reference|reference answer|verified correct answer', r['response'], re.I)),
                         exact_option_content_match=next(iter(matches)) if len(matches) == 1 else None,
                         exact_match_is_diagnostic_only=True,
                         question_and_options=smap[r['sample_id']]['messages'][0]['content'],
                         response=r['response'], answer_fields=bodies,
                         prediction=r['prediction'], gold=r['gold_answer'], correct=r['correct'])
                details.append(d)
                if not r['parsed']:
                    group_details.append(d)
                    all_remaining.append(d)
            summary = dict(benchmark=bench, model=model, total=500,
                           previous_unparsed=sum(not r['parsed'] for r in prior), unparsed=len(group_details),
                           recovered=sum(not pmap[r['sample_id']]['parsed'] and r['parsed'] for r in rows),
                           now_conflicting=sum(pmap[r['sample_id']]['parsed'] and not r['parsed'] for r in rows),
                           correct=sum(r['correct'] for r in rows),
                           previous_correct=sum(r['correct'] for r in prior),
                           near_cap_unparsed=sum(d['near_512_token_cap'] for d in group_details),
                           reference_mentions_unparsed=sum(d['mentions_reference'] for d in group_details),
                           exact_option_content_match_unparsed=sum(d['exact_option_content_match'] is not None for d in group_details))
            summary['categories'] = dict(Counter(d['category'] for d in group_details))
            summary['accuracy_percent'] = summary['correct'] / 5
            summaries.append(summary)
            audit['groups'][bench+'/'+model] = {'v2_reproduced': True, 'unique_rows': 500,
                                              'raw_response_unchanged': True,
                                              'v3_scored_sha256': sha(OUT / bench / model / 'scored.jsonl')}
        for s in [s for s in summaries if s['benchmark'] == bench]:
            m = s['model']
            paired = paired_comparison([by_model[m][i]['correct'] for i in ids],
                                       [by_model['base'][i]['correct'] for i in ids], bootstrap, m == 'base')
            s['delta_vs_base_pp'] = 100 * paired['delta_accuracy']
            s['ci95_pp'] = [100*x for x in paired['ci95_video_bootstrap']]
            write_json(OUT / bench / m / 'summary.json', dict(s, paired_vs_base=paired))
    assert all(sha(Path(p)) == digest for p, digest in audit['sources'].items())
    audit.update(validated_groups=26, rows=13000, previous_unparsed_audited=sum(d['v2_unparsed'] for d in details),
                 recovered=sum(s['recovered'] for s in summaries),
                 now_conflicting=sum(s['now_conflicting'] for s in summaries),
                 remaining_unparsed=len(all_remaining), originals_unchanged=True)
    write_json(OUT / 'validation.json', audit)
    write_json(OUT / 'comparison.json', {'rows': summaries, 'scoring_version': VERSION})
    write_jsonl(OUT / 'unparsed_audit.jsonl', details)
    write_jsonl(OUT / 'remaining_unparsed.jsonl', all_remaining)
    write_jsonl(OUT / 'changed_answers.jsonl', changed)
    write_csv(OUT / 'scores.csv', [{k: v for k, v in s.items() if k not in ['categories', 'ci95_pp']} for s in summaries])
    write_csv(OUT / 'unparsed_categories.csv', [dict(benchmark=s['benchmark'], model=s['model'],
              **{k:s['categories'].get(k, 0) for k in CATEGORIES}, near_cap=s['near_cap_unparsed'],
              exact_content_match=s['exact_option_content_match_unparsed']) for s in summaries])
    by = {(s['benchmark'], s['model']): s for s in summaries}
    doc = ['# 剩余无法解析回答的逐题审查与 v3 修正', '',
           f"时间：{audit['created_at']}。审查 v2 剩余 {audit['previous_unparsed_audited']} 条，同时对全部 13,000 条应用统一 v3。", '',
           f"新增恢复 {audit['recovered']} 条明确选项，另有 {audit['now_conflicting']} 条旧解析发现多选冲突；最终剩余 {audit['remaining_unparsed']} 条。原始回答和 v2 结果均未修改。", '',
           '## 1. 具体是哪几类问题', '',
           '- 评分器遗漏：答案字段中的 `The correct answer is D because...`、`B is the correct response...`、引号中的字母、HTML段落包装，以及嵌套在 analysis 内的明确 answer 字段。v3 已统一补充。',
           '- 自然语言答案：SFT 常写 `<answer>Angry.</answer>`、`<answer>Third.</answer>`；它有答案内容，却没给字母。只有唯一标准化完全匹配选项文字的情况单独统计为诊断，不纳入当前字母评分。',
           '- 答案栏继续分析：CLUE 一部分输出把解释、与 reference 对齐等话语写进 answer，甚至整段都没有选项决策；长文字不等于完成回答。',
           '- 没有最终答案：一些回答仍在分析便结束，部分接近 512 token；也有很短的回答只写 analysis 就停止。',
           '- 畸形输出：`<DONE>`、`<D D>`、`<A` 等不是规范字母标签，不能猜测。',
           '- 多选或越界：`AC`、`A and B`、多个不同 answer 字段、`E/G/N` 等不满足单选 A–D。', '',
           '## 2. 每组数量', '',
           '| 数据集 | 模型 | v2未解析 | 此次恢复 | v3未解析 | 答案栏文字 | 无最终字段 | 畸形标签 | 冲突多选 | 越界 | 空字段 | 接近512token* |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        c = s['categories']
        counts = [c.get(k, 0) for k in CATEGORIES]
        doc.append('| '+s['benchmark']+' | '+model_name(s['model'])+' | '+ ' | '.join(str(x) for x in [s['previous_unparsed'],s['recovered'],s['unparsed']]+counts+[s['near_cap_unparsed']])+' |')
    doc += ['', '*接近上限是对已解码文本重新分词得到 ≥510 tokens，属于交叉统计，不与各原因类别相加。原始文件未保存 finish_reason，不能把接近上限全部断言为截断。', '',
            '类别是基于可见输出结构的可复现归类；“答案栏文字”包括真正答案内容、冗长解释及不确定性表达，不声称每条都语义正确。', '',
            '## 3. 与训练方式的关系', '',
            'SFT 的实际训练 prompt 写了 “Do not output an option letter or refer to answer options.”；监督目标是 observation 分析加自然语言答案。CLUE 的 student/teacher prompt 也明确 “Do not output an option letter.”。本次评测却要求选项字母。因此训练并未直接教会这项输出任务。',
            '全参数 SFT 仍输出自然语言答案，与其训练目标一致；这能解释格式偏移，但不能仅凭这些日志证明 full parameter 必然比 LoRA 更差。',
            'CLUE teacher 在训练时可见标准答案，含 observation 版本还可见参考观察；评测 student 看不到这些。部分输出却仍声称“provided reference”或“reference answer”，呈现参考答案口吻。它与教师话语被学生模仿的假设一致，但仅凭这些回答不能证明唯一因果。',
            '训练沿 student 生成序列做分布蒸馏，且 sft_alpha=0，没有固定的标准输出全文交叉熵项来确保最终答案格式。对这类长分析任务，120词是软提示，512 token 是硬生成上限，分析或答案栏解释占满预算会使最终选项缺失。', '',
            '## 4. 最新统一评分', '',
            '| 模型 | OmniVideoBench | DailyOmni |', '|---|---:|---:|']
    for m in MODELS:
        doc.append('| '+model_name(m)+' | '+' | '.join(f"{by[b,m]['accuracy_percent']:.1f}%" for b in BENCHMARKS)+' |')
    doc += ['', '## 5. 具体原始例子', '']
    examples = [('sft_full_epoch3','omnivideobench_0009'), ('sft_full_epoch3','omnivideobench_0007'),
                ('clue_obs_epoch3','dailyomni_0055'), ('clue_obs_epoch3','omnivideobench_0008'),
                ('sft_full_epoch3','omnivideobench_0021'), ('sft_lora_epoch3','omnivideobench_0613'),
                ('clue_obs_epoch1','omnivideobench_0319'), ('sft_lora_epoch3','omnivideobench_0881')]
    for model, sid in examples:
        matching = [d for d in details if d['model']==model and d['sample_id']==sid]
        if not matching:
            continue
        d = matching[0]
        doc += [f"### {model} / {sid}", '',
                f"类别：{CATEGORIES.get(d['category'], d['category'])}；重新分词 tokens={d['decoded_tokens']}。", '',
                '```text', d['question_and_options'], '```', '', '原始回答：', '', '```text', d['response'], '```', '']
    doc += ['## 6. 文件', '',
            '- `unparsed_audit.jsonl`：所有 v2 未解析题及新增冲突题的完整原回答、题目选项、结构类别和长度。',
            '- `remaining_unparsed.jsonl`：v3 后仍未解析的全部回答。',
            '- `changed_answers.jsonl`：此次解析修正改变的所有选项及得分。',
            '- `scores.csv`、`comparison.json`：最新统一主分数，含重新计算的配对区间。',
            '- `validation.json`：全部旧评分复现、源文件哈希及原始文件未修改证据。', '']
    (OUT / 'RESULTS.zh-CN.md').write_text('\n'.join(doc))
    print(json.dumps({k:v for k,v in audit.items() if k not in ['sources','groups']},ensure_ascii=False))
    for s in summaries:
        if s['model'].endswith('epoch3') or s['model']=='base':
            print(s['benchmark'],s['model'],s['categories'],'near_cap',s['near_cap_unparsed'],'exact_text_match',s['exact_option_content_match_unparsed'])


if __name__ == '__main__':
    main()
