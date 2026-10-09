#!/usr/bin/env python3
"""Coordinate separate reviewer shards and create timestamped original-video contact sheets."""
import argparse,collections,concurrent.futures,hashlib,json,math,os,subprocess,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
PARENT=REPO/'training_runs/worldsense_clue_only_agentic_20261004'
ROOT=REPO/'training_runs/worldsense_codex_reannotation_20261004'
FFMPEG='/opt/huawei/dataset/hyl_ulan/ylhu/conda-envs/omni-opsd-video-client/bin/ffmpeg'

def read(p):return json.loads(Path(p).read_text())
def write(p,r):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
def rows(p):return [json.loads(l) for l in Path(p).open() if l.strip()]
def slug(qid):return qid.replace('::','__')
def shard(qid):return int(hashlib.sha256(qid.encode()).hexdigest()[:8],16)%4+1

def parent_ended():
 marker=PARENT/'monitor/final.json'
 if not marker.exists() or read(marker).get('status') not in ['complete','needs_operator']:return False
 launch=read(PARENT/'launch.json')
 try:return b'run_worldsense_clue_only_agentic.py' not in (Path('/proc')/str(launch['pid'])/'cmdline').read_bytes()
 except FileNotFoundError:return True

def sync():
 samples={s['sample_id']:s for s in rows(PARENT/'samples.jsonl')};out=[];ended=parent_ended()
 for qid,s in sorted(samples.items()):
  folder=PARENT/'items'/slug(qid);o=read(folder/'original.json') if (folder/'original.json').exists() else None
  r=read(folder/'repair.json') if (folder/'repair.json').exists() else None
  if r and r['status'] in ['needs_review','technical_error','content_blocked']:
   final=r;phase='repair'
  elif o and o['status'] in ['technical_error','content_blocked']:
   final=o;phase='original'
  elif ended and (not o or o['status']!='original_clue_pass') and (not r or r['status']!='repaired_clue_pass'):
   # A newly recovered original failure may have no repair.json when an extra
   # resume is stopped. Keep it in scope without inventing a parent result.
   final=dict(status='pending_agentic_repair_after_parent_stop' if o else 'pending_original_after_parent_stop',
              reason='Parent ended without a passing result; recorded for review, not fabricated as a completed repair.')
   phase='repair' if o else 'original'
  else:continue
  owner=shard(qid);dest=ROOT/'reviewers'/f'agent_{owner}'/'cases'/slug(qid)
  dest.mkdir(parents=True,exist_ok=True)
  packet=dict(question_id=qid,owner=owner,sample=s,original=o,repair=r,parent_phase=phase,parent_status=final['status'],
              parent_reason=final.get('reason'),packet_updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
  write(dest/'packet.json',packet)
  out.append(dict(question_id=qid,owner=owner,parent_phase=phase,parent_status=final['status'],packet=str(dest/'packet.json'),
                  annotation=str(dest/'annotation.json'),document=str(dest/'REVIEW.zh-CN.md'),reviewed=(dest/'annotation.json').exists()))
 write(ROOT/'queue.json',out)
 status=dict(updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),parent_status=read(PARENT/'status.json'),current_failed_candidates=len(out),
             reviewed=sum(r['reviewed'] for r in out),by_shard={str(i):dict(total=sum(r['owner']==i for r in out),reviewed=sum(r['owner']==i and r['reviewed'] for r in out)) for i in range(1,5)},
             parent_counts=dict(collections.Counter(r['parent_status'] for r in out)),verification_started=(ROOT/'verification/protocol.json').exists())
 write(ROOT/'status.json',status);print(json.dumps(status,ensure_ascii=False))

def pending(owner,limit):
 records=read(ROOT/'queue.json')
 for r in records:
  if r['owner']!=owner or Path(r['annotation']).exists():continue
  p=read(r['packet']);s=p['sample']
  print(json.dumps(dict(question_id=r['question_id'],parent_status=r['parent_status'],question=s['question'],choices=s['choices'],gold=s['answer'],duration=s['media']['duration'],case_dir=str(Path(r['packet']).parent)),ensure_ascii=False))
  limit-=1
  if limit<=0:break

def packet(qid):
 return read(ROOT/'reviewers'/f'agent_{shard(qid)}'/'cases'/slug(qid)/'packet.json')

def frames(args):
 from PIL import Image,ImageDraw,ImageFont
 r=packet(args.id);s=r['sample'];duration=s['media']['duration'];end=args.end if args.end is not None else duration
 start=args.start
 if not 0<=start<end<=duration or not 1<=args.count<=48:raise ValueError('Invalid frame request')
 # Every panel is labelled with the requested ORIGINAL source timestamp.
 stamps=[start+(end-start)*(i+.5)/args.count for i in range(args.count)]
 if args.times:stamps=[float(x) for x in args.times.split(',')]
 if not all(0<=t<duration for t in stamps):raise ValueError('Timestamp outside source')
 crop=None
 if args.crop:
  crop=[int(v) for v in args.crop.split(',')]
  if len(crop)!=4:raise ValueError('Crop must be x,y,width,height in original pixels')
  x,y,cw,ch=crop
  if not (0<=x<x+cw<=s['media']['width'] and 0<=y<y+ch<=s['media']['height']):raise ValueError('Crop outside original image')
 out_width=min(args.width,crop[2] if crop else s['media']['width'])
 identity=hashlib.sha256(json.dumps([stamps,out_width,s['video_path'],crop]).encode()).hexdigest()[:16]
 folder=ROOT/'reviewers'/f'agent_{shard(args.id)}'/'cases'/slug(args.id)/'assets'/identity
 folder.mkdir(parents=True,exist_ok=True)
 def one(pair):
  i,t=pair;p=folder/f'frame_{i:03d}.jpg'
  if not p.exists():
   vf=(f'crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]},' if crop else '')+f'scale={out_width}:-2'
   cmd=[FFMPEG,'-hide_banner','-loglevel','error','-threads','1','-ss',str(t),'-i',s['video_path'],'-frames:v','1','-vf',vf,'-threads','1','-q:v','2','-y',str(p)]
   subprocess.run(cmd,capture_output=True,check=True,timeout=90)
  return p
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:paths=list(pool.map(one,enumerate(stamps)))
 imgs=[Image.open(p).convert('RGB') for p in paths];w=max(x.width for x in imgs);h=max(x.height for x in imgs);cols=args.columns;rows_=math.ceil(len(imgs)/cols)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
 sheet=Image.new('RGB',(cols*w,rows_*(h+32)),(25,25,25));d=ImageDraw.Draw(sheet)
 for i,(im,t) in enumerate(zip(imgs,stamps)):
  x=(i%cols)*w;y=(i//cols)*(h+32);sheet.paste(im,(x,y));d.text((x+5,y+h+4),f'{t:.3f} s (original)',font=font,fill='white')
 target=folder/'contact.jpg';sheet.save(target,quality=92)
 write(folder/'frames.json',dict(question_id=args.id,source=s['video_path'],requested_original_timestamps=stamps,crop_original_pixels=crop,frame_files=[str(p) for p in paths],contact_sheet=str(target),note='Sparse frames: not proof of continuous motion, audio content, or absence between samples. Seeking may differ by one source frame.'))
 print(json.dumps(dict(contact_sheet=str(target),frames=[str(p) for p in paths],timestamps=stamps),ensure_ascii=False))

def validate(qid):
 p=packet(qid);base=ROOT/'reviewers'/f'agent_{shard(qid)}'/'cases'/slug(qid);a=read(base/'annotation.json');s=p['sample']
 assignment_file=ROOT/'review_assignments.json'
 delegated=read(assignment_file).get(qid,{}).get('reviewer') if assignment_file.exists() else None
 assert a['question_id']==qid and a['reviewer'] in [f'agent_{shard(qid)}','root',delegated]
 assert a['review_status'] in ['reannotated','uncertain','possible_dataset_issue']
 spans=a['proposed_clue_intervals'];assert isinstance(spans,list) and 1<=len(spans)<=16
 previous=-1
 for start,end in spans:
  assert 0<=start<end<=s['media']['duration'] and start>=previous;previous=end
 assert sum(e-b for b,e in spans)<=300
 assert a['evidence'] and a['media_inspected']
 for x in a['media_inspected']:assert Path(x['artifact']).exists()
 assert a['observation_checked'] is False and a['human_verified'] is False and a['independent_verification_passed'] is False
 assert (base/'REVIEW.zh-CN.md').exists()
 print(json.dumps(dict(question_id=qid,valid=True,review_status=a['review_status'])))

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
 sub.add_parser('sync');q=sub.add_parser('pending');q.add_argument('--owner',type=int,required=True);q.add_argument('--limit',type=int,default=10)
 q=sub.add_parser('show');q.add_argument('--id',required=True)
 q=sub.add_parser('validate');q.add_argument('--id',required=True)
 q=sub.add_parser('frames');q.add_argument('--id',required=True);q.add_argument('--start',type=float,default=0);q.add_argument('--end',type=float);q.add_argument('--times');q.add_argument('--count',type=int,default=16);q.add_argument('--columns',type=int,default=4);q.add_argument('--width',type=int,default=384);q.add_argument('--crop',help='x,y,width,height in original frame pixels')
 a=p.parse_args()
 if a.action=='sync':sync()
 elif a.action=='pending':pending(a.owner,a.limit)
 elif a.action=='show':print(json.dumps(packet(a.id),ensure_ascii=False,indent=2))
 elif a.action=='frames':frames(a)
 elif a.action=='validate':validate(a.id)
if __name__=='__main__':main()
