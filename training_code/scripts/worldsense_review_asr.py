#!/usr/bin/env python3
"""CPU speech-transcript assistance for Codex reviews; never receives questions or gold."""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
import argparse,fcntl,hashlib,json,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'training_runs/worldsense_mcq_annotation_pilot100_20261003/api_deps'))
ROOT=REPO/'training_runs/worldsense_codex_reannotation_20261004/audio_support'
MODEL=ROOT/'models/whisper-small'
VIDEO=Path('/opt/huawei/dataset/hyl_ulan/ylhu/OmniFold/data/benchmarks/WorldSense/videos')
FFMPEG='/opt/huawei/dataset/hyl_ulan/ylhu/conda-envs/omni-opsd-video-client/bin/ffmpeg'

def write(p,r):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)

def prepare():
 from huggingface_hub import snapshot_download
 path=snapshot_download('openai/whisper-small',local_dir=MODEL,allow_patterns=['config.json','generation_config.json','preprocessor_config.json','tokenizer*','vocab.json','merges.txt','normalizer.json','added_tokens.json','special_tokens_map.json','model.safetensors'],max_workers=4)
 manifest={str(f.relative_to(MODEL)):hashlib.sha256(f.read_bytes()).hexdigest() for f in MODEL.glob('*') if f.is_file()}
 meta=MODEL/'.cache/huggingface/download/config.json.metadata'
 revision=meta.read_text().splitlines()[0] if meta.exists() else None
 write(ROOT/'model_manifest.json',dict(model='openai/whisper-small',revision=revision,files=manifest,downloaded_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
 print(json.dumps(dict(model_dir=str(path),revision=revision)),flush=True)

def process_request(pipe,req):
 import soundfile as sf
 vid=req['video_id'];s=float(req['start']);e=float(req['end']);assert 0<=s<e<=300
 tag=hashlib.sha256(json.dumps([vid,s,e]).encode()).hexdigest()[:12];target=ROOT/'transcripts'/f'{vid}_{tag}.json'
 target.parent.mkdir(parents=True,exist_ok=True)
 lock=open(target.with_suffix('.lock'),'a')
 try:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  if target.exists():return
  error=target.with_suffix('.error.json');previous=json.loads(error.read_text()) if error.exists() else {}
  if previous.get('attempts',0)>=3:return
  clip=ROOT/'clips'/f'{vid}_{tag}.wav';clip.parent.mkdir(parents=True,exist_ok=True)
  try:
   subprocess.run([FFMPEG,'-v','error','-threads','1','-ss',str(s),'-i',str(VIDEO/(vid+'.mp4')),'-t',str(e-s),'-vn','-ac','1','-ar','16000','-y',str(clip)],check=True,capture_output=True,timeout=90)
   audio,sr=sf.read(clip,dtype='float32');started=time.time()
   result=pipe({'array':audio,'sampling_rate':sr},return_timestamps=True,batch_size=1,generate_kwargs={'task':'transcribe','max_new_tokens':400})
   timing_issues=[]
   for i,chunk in enumerate(result.get('chunks',[])):
    chunk['original_timestamp']=[None if t is None else round(s+float(t),3) for t in chunk['timestamp']]
    left,right=chunk['original_timestamp']
    if left is None or right is None or left<s or right>e+.1 or right<left:timing_issues.append(i)
   record=dict(video_id=vid,source_interval=[s,e],clip=str(clip),model='openai/whisper-small',device='cpu',
               result=result,elapsed_seconds=round(time.time()-started,2),verified=False,timing_issue_chunk_indices=timing_issues,
               limitation='Machine ASR hypothesis, not direct human/agent hearing. No question, options, gold, caption, or expected words were supplied to ASR; timing and words can be wrong. Raw out-of-range or reversed timestamps are retained, not accepted as evidence.')
   write(target,record);print(json.dumps(dict(video_id=vid,path=str(target),elapsed_seconds=record['elapsed_seconds'],text=result.get('text','')),ensure_ascii=False),flush=True)
  except Exception as exc:
   write(error,dict(video_id=vid,source_interval=[s,e],error_type=type(exc).__name__,attempts=previous.get('attempts',0)+1,updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
   print(json.dumps(dict(video_id=vid,error_type=type(exc).__name__)),flush=True)
 finally:lock.close()

def run(requests,watch=False):
 import numpy as np
 import soundfile as sf
 import torch
 from transformers import WhisperForConditionalGeneration,WhisperProcessor,pipeline
 torch.set_num_threads(8)
 processor=WhisperProcessor.from_pretrained(MODEL,local_files_only=True)
 model=WhisperForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,use_safetensors=True,torch_dtype=torch.float32)
 pipe=pipeline('automatic-speech-recognition',model=model,tokenizer=processor.tokenizer,feature_extractor=processor.feature_extractor,device=-1,chunk_length_s=30,stride_length_s=(4,4))
 print('CPU ASR ready',flush=True)
 while True:
  reqs=json.loads(Path(requests).read_text())
  if watch:
   for p in (ROOT.parent/'reviewers').glob('agent_*/audio_requests.json'):
    try:reqs.extend(json.loads(p.read_text()))
    except json.JSONDecodeError:pass
  seen=set()
  for req in reqs:
   if req.get('non_speech'):continue
   key=(req['video_id'],float(req['start']),float(req['end']))
   if key in seen:continue
   seen.add(key);process_request(pipe,req)
  if not watch or (ROOT/'STOP_ASR').exists():break
  time.sleep(30)

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=['prepare','run'],required=True);p.add_argument('--requests',type=Path,default=ROOT/'requests.json');p.add_argument('--daemon',action='store_true');p.add_argument('--watch',action='store_true');p.add_argument('--worker-slot',type=int,default=1);a=p.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
 if a.daemon:
  log=ROOT/f'worker_{a.worker_slot}.log'
  with log.open('a') as out:
   proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),*[v for v in sys.argv[1:] if v!='--daemon']],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
  (ROOT/f'worker_{a.worker_slot}.pid').write_text(str(proc.pid)+'\n');print(json.dumps(dict(pid=proc.pid,log=str(log))));return
 slot_lock=None
 if a.watch:
  slot_lock=open(ROOT/f'worker_{a.worker_slot}.lock','a')
  try:fcntl.flock(slot_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:print('ASR worker slot already running');return
 try:
  if a.action=='prepare':prepare()
  else:run(a.requests,a.watch)
 except Exception as exc:
  print(json.dumps({'error_type':type(exc).__name__,'action':a.action,'message':'Failed; no credentials or request headers logged.'}),flush=True);raise SystemExit(1)
if __name__=='__main__':main()
