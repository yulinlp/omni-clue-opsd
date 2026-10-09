#!/usr/bin/env python3
"""Rewrite the selected 32 and generate the 3 missing observations, direct API only."""
import concurrent.futures as cf
import json, os, re, time, sys
from pathlib import Path
import review_train671_observations as base
import run_worldsense_observation_rewrite as api
ROOT=base.OUT/'rewrite35'
FOCUS='--focus' in sys.argv
MEDIA_PROMPT=base.PROMPT.replace('This is TEXT-ONLY editing: you have not watched the video. Do not claim to have watched or verified it.', 'Use the supplied video and audio to resolve errors in the old observation. Write only evidence supported by this media. Original full-video times are supplied. Do not discuss the act of watching.')

def parse(raw):
    o=api.q.parse_json(raw)
    assert o.get('status') in ('rewritten','needs_evidence','evidence_only')
    assert isinstance(o.get('observation'),str) and isinstance(o.get('issue_zh'),str)
    if o['status'] in ('rewritten','evidence_only'):
        assert o['observation'].strip() and len(o['observation'].split())<=120
        assert not re.search(r'\b(?:maybe|perhaps|probably|possibly|likely|uncertain|guess)\b|let me|let.s (?:check|look)|reference answer|verified correct answer|I (?:think|have|will|can|need)',o['observation'],re.I)
    else:
        assert not o['observation'] and o['issue_zh'].strip()
    return o

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    api.setup_env(base.REPO/'.env')
    assert all(not os.environ.get(k) for k in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'))
    rows=base.read(base.OUT/'rewrite_candidates.jsonl')+base.read(base.OUT/'missing_observation.jsonl')
    assert len(rows)==len({r['sample_id'] for r in rows})==35
    source={s['sample_id']:s for s in base.read(base.SOURCE)}
    hashes=json.loads((base.SOURCE.parent/'source_hashes.json').read_text())
    base.dump(ROOT/'network_and_inputs.json',dict(created_at=api.now(),proxy_env_cleared=True,httpx_trust_env=False,NO_PROXY='*',model=api.q.MODEL,source_sha256=base.sha(base.SOURCE),train_sha256=base.sha(base.TRAIN),count=35))
    def one(r):
        sid=r['sample_id']; folder=ROOT/'items'/sid.replace('::','__');folder.mkdir(parents=True,exist_ok=True)
        saved=folder/'result.json'
        if saved.exists():
            old=json.loads(saved.read_text())
            if old.get('status')=='technical_error' and old.get('error',{}).get('kind')=='content_blocked':
                old['status']='content_blocked';base.dump(saved,old);return old
            if old.get('status')!='technical_error' and not (FOCUS and old.get('status')=='needs_evidence'):return old
        payload={k:r[k] for k in ('sample_id','question','choices','answer','observation','reason_zh')}
        payload['style_examples']=base.EXAMPLES
        stages=[]
        try:
            out={'status':'needs_evidence'}
            if FOCUS and saved.exists():
                out=old if old.get('status')!='technical_error' else {'status':'needs_evidence'};stages=old.get('stages',[])[:]
            if r['observation'] and not FOCUS:
                out=api.cached_call(folder,'text',[{'role':'system','content':base.PROMPT},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}],900,parse)
                stages.append('text')
            if out['status']=='needs_evidence' and not FOCUS:
                item=dict(source[sid]);item['video_sha256']=hashes[item['video_id']]
                clip,meta=api.media(item,[[0.,item['media']['duration']]],ROOT/'media'/item['video_id'],True)
                payload['source_duration']=item['media']['duration'];payload['current_clue_intervals']=r['current_evidence_spans']
                out=api.cached_call(folder,'video',[{'role':'system','content':MEDIA_PROMPT},api.c.media_message(json.dumps(payload,ensure_ascii=False),clip,meta)],900,parse)
                stages.append('full_video_audio')
            if out['status']=='needs_evidence' and FOCUS:
                item=dict(source[sid]);item['video_sha256']=hashes[item['video_id']]
                clip,meta=api.media(item,r['current_evidence_spans'],ROOT/'media'/item['video_id'],False)
                payload['source_duration']=item['media']['duration'];payload['clip_mapping']=meta['mapping']
                prompt=MEDIA_PROMPT+' This media contains selected clue intervals only, mapped to original times in clip_mapping. Do not infer full-video absence or total counts from missing footage. Inspect the higher-detail clue evidence to write the explanation.'
                prompt += '\nFor this final rewrite, if the evidence does not support the reference answer, still describe the relevant facts that ARE supported in a concise observation, with no speculative conclusion. Return status=\"evidence_only\" and state the unresolved conflict separately in issue_zh. This overrides the earlier requirement to leave observation empty. Never invent missing evidence. If no relevant factual statement is possible, retain needs_evidence with empty observation.'
                out=api.cached_call(folder,'focused_clue',[{'role':'system','content':prompt},api.c.media_message(json.dumps(payload,ensure_ascii=False),clip,meta)],900,parse)
                stages.append('focused_clue_audio')
                if saved.exists():base.dump(folder/'before_focus.json',old)
            result=dict(sample_id=sid,**out,old_observation=r['observation'],stages=stages,word_count=len(out['observation'].split()),human_verified=False,independently_verified=False)
        except Exception as e:
            result=dict(sample_id=sid,status='content_blocked' if getattr(e,'kind',None)=='content_blocked' else 'technical_error',error=api.safe_error(e),stages=stages)
        base.dump(saved,result);return result
    results={}
    for attempt in range(2):
        pending=[r for r in rows if r['sample_id'] not in results or results[r['sample_id']]['status']=='technical_error']
        if not pending:break
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(one,pending):
                results[result['sample_id']]=result
                counts=dict(__import__('collections').Counter(r['status'] for r in results.values()))
                base.dump(ROOT/'status.json',dict(at=api.now(),completed=len(results),total=35,counts=counts))
                print(json.dumps(dict(completed=len(results),counts=counts)),flush=True)
    ordered=[results[r['sample_id']] for r in rows]
    base.lines(ROOT/'results.jsonl',ordered)
    base.lines(ROOT/'rewritten_overlay.jsonl',[r for r in ordered if r['status']=='rewritten'])
    base.lines(ROOT/'needs_attention.jsonl',[r for r in ordered if r['status']!='rewritten'])
    md=['# 35 条 observation 改写结果','','模型：qwen3.8-omni-flash。API 直连，清除代理环境变量，HTTP 客户端 trust_env=False。','', '先改写原文；原文为空或证据不足时，提供完整视频和音频再生成。未覆盖训练集，未进行独立事实核验。','',json.dumps(counts,ensure_ascii=False),'']
    for r in ordered:
        md.extend(['## '+r['sample_id'],'', '状态：'+r['status'],'',r.get('observation',''),'',r.get('issue_zh',''),''])
    (ROOT/'RESULTS.zh-CN.md').write_text('\n'.join(md))
    base.dump(ROOT/'COMPLETE.json',dict(at=api.now(),counts=counts,total=35))
if __name__=='__main__':main()
