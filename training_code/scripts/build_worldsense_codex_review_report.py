#!/usr/bin/env python3
import collections,hashlib,json,os,time
from pathlib import Path
from worldsense_codex_review import parent_ended
ROOT=Path(__file__).resolve().parents[2]/'training_runs/worldsense_codex_reannotation_20261004'
queue=json.loads((ROOT/'queue.json').read_text());rows=[]
for row in queue:
 path=Path(row['annotation'])
 if not path.exists():continue
 try:a=json.loads(path.read_text())
 except json.JSONDecodeError:continue
 if not Path(row['document']).exists():continue
 packet=json.loads(Path(row['packet']).read_text());s=packet['sample'];spans=a['proposed_clue_intervals']
 old=[]
 if packet.get('original',{}).get('check',{}).get('clue_intervals'):old.append(packet['original']['check']['clue_intervals'])
 for check in (packet.get('repair') or {}).get('checks',[]):
  if check.get('clue_intervals'):old.append(check['clue_intervals'])
 same=any(spans==x for x in old)
 rows.append(dict(question_id=row['question_id'],owner=row['owner'],reviewer=a['reviewer'],review_status=a['review_status'],
  proposed_clue_intervals=spans,same_as_a_previous_clue=same,answer_assessment=a['answer_assessment'],
  document=row['document'],annotation=row['annotation'],annotation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
  parent_status=row['parent_status'],split_provenance=s['split_provenance'],observation_checked=False,human_verified=False))
now=time.strftime('%Y-%m-%d %H:%M:%S %z');counts=dict(collections.Counter(r['review_status'] for r in rows));ended=parent_ended()
verification={}
for row in rows:
 p=ROOT/'verification/items'/row['question_id'].replace('::','__')/row['annotation_sha256'][:16]/'result.json'
 if p.exists():
  try:verification[row['question_id']]=json.loads(p.read_text())
  except json.JSONDecodeError:pass
verification_counts=dict(collections.Counter(r['status'] for r in verification.values()))
report=['# WorldSense：Codex 并行逐题复查和重标','',f'更新时间：{now}。','',
 '## 当前进度','',f'- 当前已进入复查队列：**{len(queue)} 题**；'+('原 Qwen 已结束，队列包含最终仍未通过和技术未完成题目。' if ended else '原 Qwen 进程仍可能产生更多未通过题目。'),
 f'- 已写入逐题复查文档：**{len(rows)} 题**。',f'- 复查结论分布：`{json.dumps(counts,ensure_ascii=False)}`。',
 f'- 当前候选版本的独立核验结果：**{len(verification)} 题**，`{json.dumps(verification_counts,ensure_ascii=False)}`。技术错误仍可能处于有限重试中。',
 '- `reannotated` 表示已找到有依据的候选区间；`uncertain` 表示证据仍有不足；`possible_dataset_issue` 表示发现题意、选项或参考答案的疑点。都不等于独立核验通过。',
 '- 曾分派4个子 agent；平台线程限制使第4名完成8题后无法恢复，后续由主 agent 与3个子 agent 并行完成。主 agent 接手分片4、音频支持和核验调度，并按 review_assignments.json 将明确列出的分片4题目委派给其他复查员；实际作者逐题注明，避免重复修改。',
 '- 原 Qwen 重标进程结束前不进行本轮再次核验；之后只给新 clue 音视频、问题、选项及原视频时间映射，不给本轮解释、旧回答、caption、ASR或gold。',
 '- 相同区间会明确标记，不能把重复核验的变化归因于区间修改。保留原训练/评测归属，不覆盖原标注。','',
 '## 逐题索引','', '|题目|复查员|结论|新区间（原视频秒）|与某次旧区间相同|独立核验|详细文档|', '|---|---|---|---|---|---|---|']
for r in rows:
 rel=Path(r['document']).relative_to(ROOT)
 outcome=verification.get(r['question_id'],{}).get('status','pending')
 report.append(f'|{r["question_id"]}|{r["reviewer"]}|{r["review_status"]}|{json.dumps(r["proposed_clue_intervals"])}|{"是" if r["same_as_a_previous_clue"] else "否"}|{outcome}|[逐题证据]({rel})|')
report+=['','## 音频和证据边界','',
 '画面记录必须来自实际打开的原视频抽帧。稀疏帧不能证明两个采样点之间没有动作；对白或声音不能从静帧推断。旧 caption 和旧模型说法仅作线索。',
 '本地 ASR（如有）是辅助机器转写，单独保存模型、区间、文本和时间戳；不向转写模型提供问题、选项或标准答案。转写并不等于人工听辨。',
 '所有记录为 AI agent 复查，human_verified=false。本轮不审计 observation。争议题保留证据冲突，不为匹配 gold 剪掉反证。','']
(ROOT/'REPORT.zh-CN.md').write_text('\n'.join(report))
p=ROOT/'proposals.jsonl';t=p.with_suffix('.tmp');t.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));os.replace(t,p)
print(json.dumps(dict(candidates=len(queue),reviewed=len(rows),counts=counts,report=str(ROOT/'REPORT.zh-CN.md')),ensure_ascii=False))
