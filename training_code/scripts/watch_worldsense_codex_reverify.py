#!/usr/bin/env python3
"""Wait for the old run, then independently verify reviewed batches as they arrive."""
import argparse,fcntl,json,subprocess,sys,time
from pathlib import Path
import run_worldsense_codex_reverify as v

def snapshot():
 proposals=v.rows(v.ROOT/'proposals.jsonl')
 queue=v.read(v.ROOT/'queue.json');status=v.read(v.ROOT/'status.json')
 ended=v.parent_finished()
 marker=v.PARENT/'monitor/final.json'
 refreshed=ended and (v.ROOT/'status.json').stat().st_mtime>=marker.stat().st_mtime
 pending=[p for p in proposals if v.still_pending(p)]
 valid=v.ROOT/'verification/results.jsonl'
 current=v.rows(valid) if valid.exists() else []
 return dict(updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),parent_finished=ended,
  queue_refreshed_after_parent=refreshed,failed_questions=len(queue),reviewed=len(proposals),
  pending_verification=len(pending),current_results=len(current),
  complete=bool(refreshed and len(proposals)==len(queue) and not pending and len(current)==len(proposals)),
  parent_status=status.get('parent_status',{}))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=16)
 p.add_argument('--batch-size',type=int,default=96);p.add_argument('--once',action='store_true');args=p.parse_args()
 root=v.ROOT/'verification';root.mkdir(exist_ok=True)
 with (root/'watch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  failures=0
  while not (root/'STOP_WATCH').exists():
   state=snapshot();v.q.write_json(root/'watch_status.json',state);print(json.dumps(state,ensure_ascii=False),flush=True)
   if state['complete']:
    v.q.write_json(root/'watch_complete.json',state);return
   if state['parent_finished'] and state['queue_refreshed_after_parent'] and state['pending_verification']:
    log=root/('batch_'+time.strftime('%Y%m%d-%H%M%S')+'.log')
    with log.open('w') as out:
     result=subprocess.run([sys.executable,str(Path(v.__file__)),'--action','run','--workers',str(args.workers),'--limit',str(args.batch_size)],stdout=out,stderr=subprocess.STDOUT)
    failures=failures+1 if result.returncode else 0
    if failures>=3:
     v.q.write_json(root/'watch_error.json',dict(status='needs_operator',reason='three consecutive runner failures',last_log=str(log),updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
     raise SystemExit(1)
   if args.once:return
   time.sleep(30)

if __name__=='__main__':main()
