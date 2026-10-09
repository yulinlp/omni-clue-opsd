#!/usr/bin/env python3
"""Add final verification links after every review author has handed off writes."""
import hashlib
import json
import os
import time
from pathlib import Path

from summarize_worldsense_codex_reverify import ROOT, atomic_text, read, rows

BEGIN = '<!-- BEGIN FINAL CLUE VERIFICATION -->'
END = '<!-- END FINAL CLUE VERIFICATION -->'


def main():
    closeout = read(ROOT / 'review_closeout.json')
    metrics = read(ROOT / 'verification/report_metrics.json')
    if not closeout.get('all_review_writes_finished') or not metrics.get('complete'):
        raise SystemExit('Wait for review handoffs and complete current-version verification.')
    proposals = rows(ROOT / 'proposals.jsonl')
    planned = []
    for p in proposals:
        annotation = Path(p['annotation'])
        digest = hashlib.sha256(annotation.read_bytes()).hexdigest()
        if digest != p['annotation_sha256']:
            raise SystemExit(f"Annotation changed after final verification: {p['question_id']}")
        result_file = ROOT / 'verification/items' / p['question_id'].replace('::', '__') / digest[:16] / 'result.json'
        result = read(result_file)
        if result.get('annotation_sha256') != digest:
            raise SystemExit(f"Wrong result version: {p['question_id']}")
        document = Path(p['document'])
        original = document.read_text()
        clean = original
        if BEGIN in clean:
            start = clean.index(BEGIN)
            finish = clean.index(END, start) + len(END)
            clean = clean[:start] + clean[finish:].lstrip('\n')
        head, separator, body = clean.partition('\n')
        selected = (result.get('response') or {}).get('answer', '无可计分回答')
        relative = os.path.relpath(result_file, document.parent)
        header = [BEGIN, '## 本轮最终核验结果', '',
                  f"- 结果：**{result['status']}**；Qwen 回答：`{selected}`。",
                  f"- 当前候选区间（原视频秒）：`{json.dumps(p['proposed_clue_intervals'])}`。",
                  f"- [对应此版本的完整核验记录]({relative})；返回时间：{result['completed_at']}。",
                  '- 下文保留了此前的复查过程；“尚未核验”等表述是当时状态，以本节为准。',
                  '- 通过仅指与原选项字母一致；所有记录 human_verified=false、observation_checked=false。',
                  END, '']
        updated = head + '\n\n' + '\n'.join(header) + '\n' + body.lstrip('\n')
        planned.append((p['question_id'], document, updated, digest))
    manifest = []
    for qid, document, updated, digest in planned:
        atomic_text(document, updated)
        manifest.append(dict(question_id=qid, document=str(document), annotation_sha256=digest,
                             document_sha256=hashlib.sha256(updated.encode()).hexdigest()))
    record = dict(completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  documents=len(manifest), annotations_modified=False, records=manifest)
    atomic_text(ROOT / 'verification/final_document_manifest.json', json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'records'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
