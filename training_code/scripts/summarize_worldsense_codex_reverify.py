#!/usr/bin/env python3
"""Report current review versions; never submit API requests or change source labels."""
import collections
import hashlib
import json
import os
import time
from pathlib import Path

from worldsense_codex_review import PARENT, ROOT, parent_ended


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(content)
    os.replace(tmp, path)


def write_jsonl(path, records):
    atomic_text(path, ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))


def main():
    samples = {s['sample_id']: s for s in rows(PARENT / 'samples.jsonl')}
    queue = read(ROOT / 'queue.json')
    proposals = rows(ROOT / 'proposals.jsonl')
    by_id = {p['question_id']: p for p in proposals}
    parent_pass = set()
    parent_counts = collections.Counter()
    for qid in samples:
        folder = PARENT / 'items' / qid.replace('::', '__')
        original = read(folder / 'original.json') if (folder / 'original.json').exists() else {}
        repair = read(folder / 'repair.json') if (folder / 'repair.json').exists() else {}
        state = repair.get('status', original.get('status', 'missing_original'))
        parent_counts[state] += 1
        if original.get('status') == 'original_clue_pass' or repair.get('status') == 'repaired_clue_pass':
            parent_pass.add(qid)
    queue_ids = {row['question_id'] for row in queue}
    missing_scope = set(samples) - parent_pass - queue_ids
    overlapping_scope = parent_pass & queue_ids
    results = {}
    stale_proposals = []
    for proposal in proposals:
        qid = proposal['question_id']
        digest = hashlib.sha256(Path(proposal['annotation']).read_bytes()).hexdigest()
        if digest != proposal['annotation_sha256']:
            stale_proposals.append(qid)
            continue
        target = ROOT / 'verification/items' / qid.replace('::', '__') / digest[:16] / 'result.json'
        if target.exists():
            record = read(target)
            if record.get('annotation_sha256') == digest:
                results[qid] = record
    counts = collections.Counter(r['status'] for r in results.values())
    pending_retries = [qid for qid, r in results.items()
                       if r['status'] == 'technical_error' and r.get('execution_runs', 1) < 3]
    ended = parent_ended()
    closeout_path = ROOT / 'review_closeout.json'
    closeout = read(closeout_path) if closeout_path.exists() else {}
    review_writes_finished = bool(closeout.get('all_review_writes_finished'))
    complete = bool(ended and review_writes_finished and not missing_scope and not overlapping_scope
                    and queue_ids == set(by_id) == set(results)
                    and not stale_proposals and not pending_retries)
    cross = collections.defaultdict(collections.Counter)
    same = collections.defaultdict(collections.Counter)
    split = collections.defaultdict(collections.Counter)
    exports = []
    duplicate_option_ids = sorted(qid for qid in queue_ids
                                  if len(set(samples[qid]['choices'])) < len(samples[qid]['choices']))
    equivalent_letter_mismatches = []
    for qid, result in sorted(results.items()):
        proposal = by_id[qid]
        cross[proposal['review_status']][result['status']] += 1
        same['same_old_intervals' if proposal['same_as_a_previous_clue'] else 'different_intervals'][result['status']] += 1
        provenance = samples[qid]['split_provenance']
        split_name = ('previous_training_1453' if provenance['previous_training_1453'] else
                      'previous_heldout_518' if provenance['previous_heldout_518'] else 'other_eligible')
        split[split_name][result['status']] += 1
        predicted = (result.get('response') or {}).get('answer')
        choices = samples[qid]['choices']
        gold_text = choices[ord(samples[qid]['answer']) - ord('A')]
        predicted_text = choices[ord(predicted) - ord('A')] if predicted and predicted in 'ABCD'[:len(choices)] else None
        text_matches = predicted_text == gold_text if predicted_text is not None else None
        if result['status'] == 'answer_mismatch' and text_matches:
            equivalent_letter_mismatches.append(qid)
        exports.append(dict(question_id=qid, video_id=samples[qid]['video_id'],
                            video_path=samples[qid]['video_path'],
                            proposed_clue_intervals=proposal['proposed_clue_intervals'],
                            original_gold=samples[qid]['answer'], status=result['status'],
                            original_gold_text=gold_text, predicted_option_text=predicted_text,
                            selected_option_text_equals_gold=text_matches,
                            response=result.get('response'), review_status=proposal['review_status'],
                            review_document=proposal['document'], annotation_sha256=proposal['annotation_sha256'],
                            same_as_a_previous_clue=proposal['same_as_a_previous_clue'],
                            split_provenance=provenance, human_verified=False, observation_checked=False,
                            training_ready=False,
                            note='Clue-only model check only; ambiguity and observation remain separate review gates.'))
    metrics = dict(updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), complete=complete,
                   review_writes_finished=review_writes_finished,
                   source_eligible=len(samples), parent_passed=len(parent_pass), parent_counts=dict(parent_counts),
                   failed_scope=len(queue), reviewed=len(proposals), results=len(results), counts=dict(counts),
                   combined_clue_passed=len(parent_pass) + counts['passed'],
                   remaining_unresolved=len(queue) - counts['passed'],
                   pending_retries=len(pending_retries), stale_proposals=len(stale_proposals),
                   missing_scope=sorted(missing_scope), overlapping_scope=sorted(overlapping_scope),
                   exact_duplicate_option_questions=duplicate_option_ids,
                   different_letter_identical_answer_text=equivalent_letter_mismatches,
                   review_status_by_result={k: dict(v) for k, v in cross.items()},
                   old_interval_comparison={k: dict(v) for k, v in same.items()},
                   split_by_result={k: dict(v) for k, v in split.items()})
    output = ROOT / 'verification'
    atomic_text(output / 'report_metrics.json', json.dumps(metrics, ensure_ascii=False, indent=2) + '\n')
    write_jsonl(output / 'passed_candidates.jsonl', [e for e in exports if e['status'] == 'passed'])
    write_jsonl(output / 'unresolved_candidates.jsonl', [e for e in exports if e['status'] != 'passed'])
    total_answers = counts['passed'] + counts['answer_mismatch']
    accuracy = f"{counts['passed'] / total_answers:.1%}" if total_answers else '尚无有效回答'
    lines = ['# WorldSense：逐题重标后的独立 clue-only 核验', '',
             f"更新时间：{metrics['updated_at']}。状态：**{'本轮完成' if complete else '进行中'}**。", '',
             '## 范围与进度', '',
             f'- 时长 ≤300 秒的原始范围：{len(samples)} 题。',
             f'- 父 Qwen 流程已通过：{len(parent_pass)} 题；最终待复查：{len(queue)} 题。',
             f'- 当前有逐题复查记录：{len(proposals)} 题；当前候选版本有核验结果：{len(results)} 题。',
             f"- 补充证据备注：{'已全部结束' if review_writes_finished else '仍在收尾，尚未冻结最终文档版本'}。",
             f'- 仍待技术重试：{len(pending_retries)} 题；等待刷新文档哈希：{len(stale_proposals)} 题。',
             f'- 范围核对：漏项 {len(missing_scope)}，与父通过集重复 {len(overlapping_scope)}。', '',
             '## 独立核验结果', '', '|状态|题数|含义|', '|---|---:|---|',
             f"|passed|{counts['passed']}|本次选项字母与保留的原始答案相同|",
             f"|answer_mismatch|{counts['answer_mismatch']}|有合法回答，但选项不同|",
             f"|content_blocked|{counts['content_blocked']}|接口拦截，没有可计分回答|",
             f"|technical_error|{counts['technical_error']}|请求或解析失败，不能当作答错或通过|",
             f"|not_evaluable|{counts['not_evaluable']}|题目有明确不可核验原因，没有发送模型请求|", '',
             f'在有合法选项回答的 {total_answers} 题中，严格字母匹配为 {accuracy}。这不是随机测试集准确率：本轮专门处理此前的困难题。', '',
             f"连同父流程已经通过的 {len(parent_pass)} 题，目前共有 {len(parent_pass) + counts['passed']} 题至少在相应 clue-only 流程中答对原选项；本次失败队列仍有 {len(queue) - counts['passed']} 题未解决。这是标注筛选数量，不是训练模型评测分数，也不表示 observation 或标签已经人工验收。", '',
             '## 复查判断与核验结果', '', '|复查判断|通过|答案不同|其他结果|', '|---|---:|---:|---:|']
    for name in ['reannotated', 'uncertain', 'possible_dataset_issue']:
        c = cross[name]
        lines.append(f"|{name}|{c['passed']}|{c['answer_mismatch']}|{sum(c.values()) - c['passed'] - c['answer_mismatch']}|")
    lines += ['', '复查发现题意、选项或标签疑点的题，即使本次猜中原字母，也不能因此消除疑点。未通过不一定意味着新区间错误；也可能是模型识别失败、声音证据不足或数据集标签有问题。', '',
              f'本队列有 **{len(duplicate_option_ids)} 题存在完全重复的选项文本**。当前已返回结果中，{len(equivalent_letter_mismatches)} 题虽字母不同、所选答案文本却与原答案完全相同。这些仍保留原始字母计分结果，并单独标注，不能通过重新裁剪视频解决选项重复。详情见 [选项重复清单](exact_duplicate_options.json)。', '',
              '## 与以前区间及数据划分的关系', '', '|区间关系|通过|答案不同|其他|', '|---|---:|---:|---:|']
    for name in ['different_intervals', 'same_old_intervals']:
        c = same[name]
        lines.append(f"|{name}|{c['passed']}|{c['answer_mismatch']}|{sum(c.values()) - c['passed'] - c['answer_mismatch']}|")
    lines += ['', '|保留的原划分|通过|答案不同|其他|', '|---|---:|---:|---:|']
    for name in ['previous_training_1453', 'previous_heldout_518', 'other_eligible']:
        c = split[name]
        lines.append(f"|{name}|{c['passed']}|{c['answer_mismatch']}|{sum(c.values()) - c['passed'] - c['answer_mismatch']}|")
    lines += ['', '## 如何理解本轮核验', '',
              '- 模型为 qwen3.8-omni-flash，temperature=0，整段 XML 回答上限120 tokens，要求短分析和一个最终选项。',
              '- 核验只接收 clue 音视频、问题、选项、原片总长与裁片到原片的时间映射。标准答案、复查判断、旧回答、caption、observation 和本地ASR均未发送。',
              '- 完整协议见 [protocol.json](protocol.json)。短片采用较高编码帧率并保留较高分辨率，但接口实际取帧数量未知。',
              '- 因时间参照提示和媒体编码预算也有变化，不能把所有通过数归因于区间调整；区间未变的题已单列。',
              '- 同一盲核验输入在仅文档备注变化时复用原返回，不对同一输入反复抽取答案投票。',
              '- sYVBxvNj::task1、plOGuRmc::task0 在首次返回后，依据此前已排队的独立 ASR 补全遗漏语句区间，各对新输入再核验一次；前后区间、时间和旧结果均保留，最终表使用最新候选版本。',
              '- ASR 是未人工核验的机器转写；所有复查记录 human_verified=false，observation_checked=false。',
              '- 没有覆盖旧标注、改变原答案、改变训练/评测划分或自动投入训练。导出文件 training_ready=false。', '',
              '## 文件', '',
              '- [全部逐题证据与区间索引](../REPORT.zh-CN.md)',
              '- [通过候选 JSONL](passed_candidates.jsonl)',
              '- [未解决候选 JSONL](unresolved_candidates.jsonl)',
              '- [机器可读统计及范围核对](report_metrics.json)', '',
              '- [音频输入回执核对](AUDIO_INPUT_AUDIT.json)：有回执的请求中，检查服务端报告的 audio_tokens。',
              '- [三个具体案例：如何理解通过与失败](HOW_TO_READ_FAILURES.zh-CN.md)', '',
              '## 当前结果明细', '', '|题目|复查判断|原答案|本次回答|结果|逐题证据|', '|---|---|---|---|---|---|']
    for entry in exports:
        response = (entry.get('response') or {}).get('answer', '—')
        relative = os.path.relpath(entry['review_document'], output)
        lines.append(f"|{entry['question_id']}|{entry['review_status']}|{entry['original_gold']}|{response}|{entry['status']}|[记录]({relative})|")
    atomic_text(output / 'RESULTS.zh-CN.md', '\n'.join(lines) + '\n')
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == '__main__':
    main()
