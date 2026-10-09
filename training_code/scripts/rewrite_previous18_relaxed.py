#!/usr/bin/env python3
"""Relaxed re-edit of the exact 18 former evidence_only records; update the 35-result bundle."""
import collections, concurrent.futures as cf, copy, fcntl, json, os, shutil
from pathlib import Path
import review_train671_observations as b
import rewrite_train829_observations as relaxed
import run_worldsense_observation_rewrite as api
PARENT=b.OUT/'rewrite35'
ROOT=PARENT/'relaxed18'
PROMPT=relaxed.WRITE+'''\nThe previous media-based observation and review notes are provided. Re-edit the NEW observation using its supported facts. A previous rejection can be mistaken: check the question and correct option rather than repeating the rejection. If the new facts already describe the correct action, accept that explanation even when the old annotation described another action. Do not require a distractor action in addition to the correct action. Do not turn "not observed" in sampled frames into proof of absence. Do not force agreement when relevant facts explicitly contradict the correct answer.'''

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with (ROOT/'run.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if not (ROOT/'before_results35.jsonl').exists():
   shutil.copyfile(PARENT/'results.jsonl',ROOT/'before_results35.jsonl')
   for name in ['RESULTS.zh-CN.md','COMPLETE.json','status.json','rewritten_overlay.jsonl','needs_attention.jsonl','evidence_only.jsonl']:
    if (PARENT/name).exists():shutil.copyfile(PARENT/name,ROOT/('before_'+name))
  old=b.read(ROOT/'before_results35.jsonl');selected=[r for r in old if r['status']=='evidence_only'];assert len(selected)==18
  inputs={r['sample_id']:r for r in b.read(b.OUT/'all_671.observations.jsonl')}
  sources={r['sample_id']:r for r in b.read(b.SOURCE)}
  hashes=json.loads((b.SOURCE.parent/'source_hashes.json').read_text())
  api.setup_env(b.REPO/'.env')
  assert all(not os.environ.get(k) for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'])
  api.save(ROOT/'manifest.json',dict(ids=[r['sample_id'] for r in selected],code_sha256=b.sha(Path(__file__)),before35_sha256=b.sha(ROOT/'before_results35.jsonl'),proxy_env_cleared=True,httpx_trust_env=False,model=api.q.MODEL,created_at=api.now()))
  (ROOT/'prompt.txt').write_text(PROMPT)
  def one(prev):
   sid=prev['sample_id'];folder=ROOT/'items'/sid.replace('::','__');out=folder/'result.json'
   if out.exists():
    v=json.loads(out.read_text())
    if v['status']!='technical_error':return v
   s=inputs[sid];stages=[]
   data={k:s[k] for k in ['sample_id','question','choices','answer']}
   data.update(observation=prev['observation'],previous_rejection=prev['issue_zh'],style_examples=b.EXAMPLES)
   try:
    v=api.cached_call(folder,'relaxed_text',[{'role':'system','content':PROMPT},{'role':'user','content':json.dumps(data,ensure_ascii=False)}],1000,relaxed.parse_rewrite);stages.append('relaxed_text')
    if v['status']!='rewritten':
     item=copy.deepcopy(sources[sid]);item['video_sha256']=hashes[item['video_id']]
     # Reuse the existing full-AV encoding. The earlier clue view was already provided.
     clip,meta=api.media(item,[[0,item['media']['duration']]],PARENT/'media'/item['video_id'],True)
     data.update(source_duration=item['media']['duration'],clip_mapping=meta['mapping'],media_scope='full',current_clue_intervals=s['current_evidence_spans'])
     prompt=relaxed.AV+'\nReconsider previous rejection using this FULL video. Do not insist on literal phrasing or direct proof of a future prediction. If facts support the reference with reasonable context, write the explanation. A disagreement between the OLD annotation and video is not a disagreement between the NEW observation and reference.'
     v=api.cached_call(folder,'relaxed_full_av',[{'role':'system','content':prompt},api.c.media_message(json.dumps(data,ensure_ascii=False),clip,meta)],1000,relaxed.parse_rewrite);stages.append('relaxed_full_av')
    result=dict(sample_id=sid,**v,old_observation=prev.get('old_observation',s['observation']),previous_observation=prev['observation'],previous_status=prev['status'],stages=stages,word_count=len(v['observation'].split()),human_verified=False,independently_verified=False)
   except Exception as e:
    result=dict(sample_id=sid,status='content_blocked' if getattr(e,'kind',None)=='content_blocked' else 'technical_error',error=api.safe_error(e),observation='',issue_zh='API调用未完成',stages=stages)
   api.save(out,result);return result
  results={}
  for retry in range(2):
   pending=[r for r in selected if r['sample_id'] not in results or results[r['sample_id']]['status']=='technical_error']
   with cf.ThreadPoolExecutor(max_workers=6) as pool:
    for f in cf.as_completed([pool.submit(one,r) for r in pending]):
     r=f.result();results[r['sample_id']]=r;counts=dict(collections.Counter(x['status'] for x in results.values()))
     api.save(ROOT/'status.json',dict(completed=len(results),total=18,counts=counts,at=api.now()));print(json.dumps(dict(completed=len(results),counts=counts)),flush=True)
  final=[results[r['sample_id']] for r in selected];api.jsonl(ROOT/'results.jsonl',final)
  api.jsonl(ROOT/'rewritten_overlay.jsonl',[r for r in final if r['status']=='rewritten'])
  api.jsonl(ROOT/'needs_attention.jsonl',[r for r in final if r['status']!='rewritten'])
  updated=[]
  for r in old:
   if r['sample_id'] in results:
    v=copy.deepcopy(results[r['sample_id']]);v['status']='needs_evidence' if v['status']=='needs_media' else v['status'];r=v
    dest=PARENT/'items'/r['sample_id'].replace('::','__')/'result.json'
    if not dest.with_name('before_relaxed18.json').exists():shutil.copyfile(dest,dest.with_name('before_relaxed18.json'))
    api.save(dest,r)
   updated.append(r)
  assert len(updated)==35
  api.jsonl(PARENT/'results.jsonl',updated)
  api.jsonl(PARENT/'rewritten_overlay.jsonl',[r for r in updated if r['status']=='rewritten'])
  api.jsonl(PARENT/'needs_attention.jsonl',[r for r in updated if r['status']!='rewritten'])
  api.jsonl(PARENT/'evidence_only.jsonl',[r for r in updated if r['status']=='evidence_only'])
  labels={'rewritten':'完成简洁改写','conflict':'模型仍判断与标准答案冲突','needs_media':'关键依据仍不足','needs_evidence':'关键依据仍不足','content_blocked':'API内容审核拒绝','technical_error':'调用错误','evidence_only':'仅有事实描述'}
  def report(path,rows,title):
   counts=dict(collections.Counter(r['status'] for r in rows))
   md=['# '+title,'','本轮按放宽要求重新处理原18条 evidence_only，允许上下文推理、近义名称和预测。API直连，HTTP客户端trust_env=False。','','| 结果 | 数量 |','|---|---:|']
   md += [f'| {labels[k]} | {v} |' for k,v in counts.items()]
   md += ['','结果是生成模型的判断，不是独立事实核验；原数据集答案未更改。35条中另外17条未重跑。旧结果保存在 relaxed18/before_results35.jsonl。','']
   for r in rows:
    s=inputs[r['sample_id']];md += ['## '+r['sample_id'],'','问题：'+s['question'],'','标准答案：'+s['answer']+' — '+s['choices'][ord(s['answer'])-65],'','状态：'+labels[r['status']],'',r.get('observation') or '（未生成）','',r.get('issue_zh',''),'']
   path.write_text('\n'.join(md))
  report(ROOT/'RESULTS.zh-CN.md',final,'原18条 observation 放宽要求后的重写结果')
  report(PARENT/'RESULTS.zh-CN.md',updated,'35条 observation 最新汇总（已更新原18条）')
  summary=dict(at=api.now(),total=18,counts=dict(collections.Counter(r['status'] for r in final)),phase='finished')
  api.save(ROOT/'COMPLETE.json',summary);api.save(ROOT/'status.json',summary)
  total=dict(at=api.now(),total=35,counts=dict(collections.Counter(r['status'] for r in updated)),phase='finished',latest_update='relaxed18')
  api.save(PARENT/'COMPLETE.json',total);api.save(PARENT/'status.json',total)
def repair_direction():
 api.setup_env(b.REPO/'.env')
 folder=ROOT/'items/YVAxgwZw__task1';old=json.loads((folder/'result.json').read_text())
 req=json.loads((folder/'requests/relaxed_full_av.json').read_text())
 messages=req['messages']+[{'role':'assistant','content':json.dumps({k:old[k] for k in ['status','observation','issue_zh']},ensure_ascii=False)},{'role':'user','content':'Your explanation says the camera is to HIS RIGHT and therefore on HIS LEFT. These are contradictory unless distinct reference frames are explicitly supported. Correct this using the video. Do not switch reference frames merely to match the answer. Distinguish screen positions from person-relative positions only when justified. If the direction cannot be established, retain supported facts with needs_media; if clearly opposite, use conflict. Return the same concise JSON schema.'}]
 v=api.cached_call(folder,'direction_consistency_fix',messages,1000,relaxed.parse_rewrite)
 api.save(folder/'before_direction_fix.json',old)
 old.update(v);old['word_count']=len(v['observation'].split());old['stages'].append('direction_consistency_fix');api.save(folder/'result.json',old)
if __name__=='__main__':
 import sys
 if '--repair-direction' in sys.argv:repair_direction()
 main()
