#!/usr/bin/env python3
"""Create a versioned MCQ dataset with exact duplicate options merged.

Preserve answer text and split membership. Do not invent distractors, modify
frozen experiments, or call a model. Saved responses only receive a separate
historical interpretation; they are not evaluations of the repaired prompts.
"""
import argparse
import copy
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LETTERS = 'ABCD'
VERSION = 'worldsense-exact-option-dedup-v1'


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(text)
    os.replace(temporary, path)


def jsonl(path, values):
    write(path, ''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values))


def repair_options(sample):
    choices, gold = sample['choices'], sample['answer']
    if len(choices) not in (3, 4) or gold not in LETTERS[:len(choices)]:
        raise ValueError(f"Invalid source schema: {sample['sample_id']}")
    unique = list(dict.fromkeys(choices))
    mapping = {LETTERS[i]: LETTERS[unique.index(text)] for i, text in enumerate(choices)}
    gold_text = choices[LETTERS.index(gold)]
    changed = len(unique) != len(choices)
    usable = len(unique) in (3, 4)
    fixed = copy.deepcopy(sample)
    patch = None
    if changed:
        new_gold = mapping[gold] if usable else None
        patch = dict(question_id=sample['sample_id'], question=sample['question'],
                     original_choices=choices, original_answer=gold, original_answer_text=gold_text,
                     repaired_choices=unique, repaired_answer=new_gold, old_to_new_letter=mapping,
                     action='merge_exact_duplicates' if usable else 'quarantine_too_few_distinct_options',
                     valid_mcq=usable, requires_reverification=usable, original_gold_truth_revalidated=False,
                     split_provenance=sample['split_provenance'])
        fixed.update(choices=unique, answer=new_gold, duplicate_option_texts=False,
                     option_repair=dict(version=VERSION, **patch),
                     option_schema_valid=usable, requires_reverification=usable, training_ready=False)
    return fixed, patch


def interpret_old_answer(sample, patch, result):
    answer = (result.get('response') or {}).get('answer')
    if not patch['valid_mcq']:
        status = 'invalid_original_mcq_not_scored'
    elif not isinstance(answer, str) or len(answer) != 1 or answer not in LETTERS[:len(sample['choices'])]:
        status = 'no_valid_historical_answer'
    elif sample['choices'][LETTERS.index(answer)] == patch['original_answer_text']:
        status = 'same_answer_text_different_letter' if answer != sample['answer'] else 'same_answer_text_same_letter'
    else:
        status = 'different_answer_text'
    return dict(question_id=sample['sample_id'], original_status=result['status'],
                original_gold=sample['answer'], original_model_answer=answer,
                mapped_model_answer=patch['old_to_new_letter'].get(answer) if patch['valid_mcq'] else None,
                repaired_gold=patch['repaired_answer'], interpretation=status,
                new_prompt_evaluated=False, result_promoted_to_passed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=REPO / 'training_runs/worldsense_clue_only_agentic_20261004/samples.jsonl')
    parser.add_argument('--review-root', type=Path, default=REPO / 'training_runs/worldsense_codex_reannotation_20261004')
    parser.add_argument('--output', type=Path, default=REPO / 'training_runs/worldsense_option_repair_20261004')
    args = parser.parse_args()
    source_sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
    source = read_rows(args.source)
    samples = {s['sample_id']: s for s in source}
    if len(samples) != len(source):
        raise ValueError('Repeated source question IDs')
    unresolved = {s['question_id']: s for s in read_rows(args.review_root / 'verification/unresolved_candidates.jsonl')}
    fixed, quarantined, patches = [], [], []
    for sample in source:
        revised, patch = repair_options(sample)
        if patch:
            patch['in_previous_unresolved_666'] = sample['sample_id'] in unresolved
            patches.append(patch)
        if patch and not patch['valid_mcq']:
            quarantined.append(revised)
        else:
            fixed.append(revised)
    # Semantic invariants: preserve the correct answer's TEXT and data splits.
    for item in fixed:
        old = samples[item['sample_id']]
        assert len(item['choices']) in (3, 4)
        assert len(item['choices']) == len(set(item['choices']))
        assert item['choices'][LETTERS.index(item['answer'])] == old['choices'][LETTERS.index(old['answer'])]
        assert item['split_provenance'] == old['split_provenance']
        assert item['question'] == old['question'] and item['video_path'] == old['video_path']
        assert item['media']['duration'] <= 300
    assert {x['sample_id'] for x in fixed + quarantined} == set(samples)
    assert len(fixed) + len(quarantined) == len(source)
    audit = [interpret_old_answer(samples[p['question_id']], p, unresolved[p['question_id']])
             for p in patches if p['question_id'] in unresolved]
    model_error_candidates = []
    for qid, result in unresolved.items():
        annotation = json.loads(Path(result['review_document']).with_name('annotation.json').read_text())
        assessment = annotation.get('answer_assessment', {})
        if (result['status'] == 'answer_mismatch' and annotation['review_status'] == 'reannotated'
                and assessment.get('supported_option') == result['original_gold']):
            model_error_candidates.append(dict(question_id=qid, question=samples[qid]['question'],
                original_gold=result['original_gold'], qwen_answer=result['response']['answer'],
                proposed_clue_intervals=result['proposed_clue_intervals'],
                review_reason=assessment.get('reason_zh'), limitations=annotation.get('limitations_zh', []),
                review_document=result['review_document'], confirmed_model_capability_failure=False,
                note='Review supports gold; input adequacy and causal attribution are not independently established.'))
    assert hashlib.sha256(args.source.read_bytes()).hexdigest() == source_sha
    out = args.output
    jsonl(out / 'samples.fixed.jsonl', fixed)
    jsonl(out / 'quarantined.jsonl', quarantined)
    jsonl(out / 'option_patches.jsonl', patches)
    jsonl(out / 'historical_answer_audit.jsonl', audit)
    jsonl(out / 'suspected_model_errors.jsonl', model_error_candidates)
    # Compact QA export for consumers that do not need captions or annotation history.
    jsonl(out / 'qa.fixed.jsonl', [dict(question_id=s['sample_id'], video_id=s['video_id'],
        video_path=s['video_path'], question=s['question'], choices=s['choices'], answer=s['answer'],
        split_provenance=s['split_provenance'], option_repaired='option_repair' in s,
        requires_reverification=bool(s.get('requires_reverification')), training_ready=False) for s in fixed])
    summary = dict(version=VERSION, created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        source=str(args.source), source_sha256=source_sha, source_records=len(source),
        duplicate_option_questions=len(patches), repaired_questions=sum(p['valid_mcq'] for p in patches),
        quarantined_questions=len(quarantined), usable_records=len(fixed),
        duplicates_in_unresolved=sum(p['in_previous_unresolved_666'] for p in patches),
        repaired_in_unresolved=sum(p['in_previous_unresolved_666'] and p['valid_mcq'] for p in patches),
        quarantined_in_unresolved=sum(p['in_previous_unresolved_666'] and not p['valid_mcq'] for p in patches),
        historical_answer_audit=dict(Counter(x['interpretation'] for x in audit)),
        suspected_model_error_candidates=len(model_error_candidates), confirmed_model_error_count=None,
        candidate_definition='answer_mismatch AND review_status=reannotated AND supported_option=original_gold',
        validation=dict(no_duplicate_options_in_usable=True, answer_text_preserved=True,
                        split_membership_preserved=True, original_source_unchanged=True),
        api_calls=0, historical_verification_results_modified=False,
        output_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.jsonl')})
    write(out / 'summary.json', json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = ['# WorldSense 重复选项修复与失败类型澄清', '',
        f"生成时间：{summary['created_at']}。", '', '## 1. 修复结果', '',
        f'- ≤300秒的 {len(source)} 题中，{len(patches)} 题含完全相同的重复选项；其中 {summary["duplicates_in_unresolved"]} 题属于此前666题未解决集。',
        f'- {summary["repaired_questions"]} 题合并为三个不同选项，同步调整正确答案字母；答案文本保持不变。',
        f'- {len(quarantined)} 题四个选项完全相同，无法构成有效选择题，移入隔离文件；不凭空补造干扰项。',
        f'- 可读取的修复版共 {len(fixed)} 题；这是选项格式修复，不代表其视频、原答案和 observation 已通过质量核验。',
        '- 新版支持三选一；下游提示与解析必须按实际 choices 生成 A/B/C，不能强制四选一。现有 clue-only 的 question_data 按实际选项数量生成标签。', '',
        '|题目|属于此前666题|旧选项 → 新选项|旧答案 → 新答案|处理|', '|---|---|---|---|---|']
    for p in patches:
        mapping = ', '.join(f'{a}→{b}' for a, b in p['old_to_new_letter'].items())
        lines.append(f"|{p['question_id']}|{'是' if p['in_previous_unresolved_666'] else '否'}|{mapping}|{p['original_answer']}→{p['repaired_answer'] or '隔离'}|{'合并重复项' if p['valid_mcq'] else '无有效选择题'}|")
    lines += ['', '## 2. 旧回答应如何解释', '',
        '旧实验的168题通过、655题答案不一致等原始结果保持可追溯；没有把旧字母直接拿到新选项上评分。',
        '- TRmFtiSv::task3、fGXbLpbs::task1、kstmjYAV::task2：旧模型所选文本与原答案完全相同，属于字母不同导致的误判。',
        '- cEtyreWh::task1：四个选项都是“There was no cheer.”，原题无辨别能力。应排除该题，不能仅凭选中相同文本就计为模型理解正确。',
        '- 其他四道未解决重复选项题，模型所选文本仍与原答案不同；去重不会自动解决其理解或证据问题。',
        '- 修改选项会改变提示，因此修复版的模型表现需要重新核验。本次没有新增API调用，历史审计不能当作新一轮核验分数。', '',
        '## 3. 第四类有多少：需要区分候选与已证实原因', '',
        f'严格筛选“复查状态reannotated、复查支持原答案、Qwen却选其他选项”，得到 **{len(model_error_candidates)}题**。完整清单为 suspected_model_errors.jsonl。',
        '这些是疑似模型识别/理解错误的候选，不是已经证明全部由模型能力导致。复查记录仍可能含题意边界、抽帧遗漏、声音未直接听辨、编码与服务端取帧等限制。',
        '例如机器人颜色题 AErOVvGi::task0 有较明确的颜色对应反证；但 BQkzFmeP::task0 涉及“拥抱/抱持”定义，CFfWQggQ::task3 涉及主观情绪解释，不能一并断言为纯模型能力错误。',
        '因此，已证实且唯一归因于模型能力的总数目前没有可靠统计；不能用146直接替代。', '',
        '## 4. 前两个例子是否属于数据集问题', '',
        '- 天气“阴沉/寒冷”例子：选项可能不互斥，属于题目设计疑点，但尚未完成独立人工核验。',
        '- 烧杯“最后位置”例子：原答案可能对应了早先状态，属于原标签或时间指向疑点。',
        '- 这两个具体例子不能推出所有“证据不确定”的题都是数据集错误。', '',
        '## 5. 文件', '',
        '- [修复后完整样本](samples.fixed.jsonl)：保留原视频、标注、caption和数据划分信息，更新选项与答案字母。',
        '- [精简QA](qa.fixed.jsonl)：供后续加载，含问题、选项、答案、视频路径与划分。',
        '- [逐题选项变更](option_patches.jsonl)',
        '- [隔离题](quarantined.jsonl)',
        '- [旧回答解释](historical_answer_audit.jsonl)',
        '- [146题疑似模型错误候选](suspected_model_errors.jsonl)',
        '- [统计与文件哈希](summary.json)', '']
    write(out / 'REPORT.zh-CN.md', '\n'.join(lines))
    print(json.dumps({k: v for k, v in summary.items() if k != 'output_sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
