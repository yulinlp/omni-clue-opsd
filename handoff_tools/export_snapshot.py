#!/usr/bin/env python3
"""Export a sanitized, non-video handoff without mutating training artifacts."""
import argparse, concurrent.futures, gzip, hashlib, io, json, os, re, shutil, tarfile, threading, zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

VIDEO={'.mp4','.mkv','.avi','.mov','.webm','.m4v','.mpeg','.mpg','.ts'}
STATE={'.safetensors','.pt','.pth','.ckpt','.onnx'}
TEXT={'.py','.sh','.md','.txt','.json','.jsonl','.csv','.yaml','.yml','.toml','.ini','.cfg','.html','.js','.css','.xml','.log','.jinja','.conf','.patch','.diff'}
SKIP_DIR={'.git','__pycache__','.pytest_cache','.mypy_cache','node_modules','.venv'}
SECRET=re.compile(rb'(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')

def digest(data):return hashlib.sha256(data).hexdigest()
def sha_file(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def secret_name(p):
 return p.name=='.env' or p.name.startswith('.env.') or p.suffix.lower() in {'.pem','.key'} or p.name in {'id_rsa','id_ed25519','.git-credentials','.netrc','credentials.json'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--dest',type=Path,required=True);ap.add_argument('--workers',type=int,default=6);a=ap.parse_args()
 src=a.source.resolve();dst=a.dest.resolve();assert not dst.is_relative_to(src),'destination must be outside source'
 dst.mkdir(parents=True,exist_ok=True);bundle=dst/'handoff/bundles';bundle.mkdir(parents=True,exist_ok=True)
 secrets=[]
 env=src/'.env'
 if env.exists():
  for line in env.read_text().splitlines():
   if '=' in line and not line.lstrip().startswith('#'):
    key,val=line.split('=',1);val=val.strip().strip('\"\'')
    if any(k in key.upper() for k in ['KEY','TOKEN','SECRET','PASSWORD']) and len(val)>=8:secrets.append(val.encode())
 lock=threading.Lock();stats=Counter();manifest=[];excluded=[];all_files=[];errors=[]
 def clean(data):
  n=0
  for s in secrets:
   n+=data.count(s);data=data.replace(s,b'[REDACTED_SECRET]')
  data,k=SECRET.subn(b'[REDACTED_SECRET]',data);return data,n+k
 def walk(root):
  for base,dirs,files in os.walk(root,followlinks=False):
   dirs[:]=sorted(d for d in dirs if d not in SKIP_DIR and not (Path(base)/d).is_symlink())
   for name in sorted(files):yield Path(base)/name
 def classify(p):
  if p.is_symlink():return 'symlink_not_followed'
  if secret_name(p):return 'credential'
  s=p.suffix.lower()
  if s in VIDEO:return 'video'
  if s in STATE or p.name=='training_args.bin':return 'model_or_training_state'
  if s in {'.pid','.lock','.pyc','.pyo','.tmp'}:return 'ephemeral'
  return None
 def filtered_archive(p,data):
  out=io.BytesIO();removed=0;redacted=0
  if p.suffix.lower()=='.zip':
   with zipfile.ZipFile(io.BytesIO(data)) as z,zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as new:
    for m in z.infolist():
     q=Path(m.filename)
     if m.is_dir():continue
     if q.is_absolute() or '..' in q.parts or secret_name(q) or q.suffix.lower() in VIDEO|STATE or (m.external_attr>>16)&0o170000==0o120000:removed+=1;continue
     b,n=clean(z.read(m));redacted+=n;new.writestr(m.filename,b)
   return out.getvalue(),redacted,removed
  try:
   old=tarfile.open(fileobj=io.BytesIO(data),mode='r:*')
  except tarfile.ReadError:
   if p.suffix.lower()=='.gz':
    b,n=clean(gzip.decompress(data));return gzip.compress(b,mtime=0),n,0
   raise
  with old,tarfile.open(fileobj=out,mode='w:gz') as new:
   for m in old:
    q=Path(m.name)
    if not m.isfile():continue
    if q.is_absolute() or '..' in q.parts or secret_name(q) or q.suffix.lower() in VIDEO|STATE:removed+=1;continue
    b,n=clean(old.extractfile(m).read());redacted+=n;m.size=len(b);new.addfile(m,io.BytesIO(b))
  return out.getvalue(),redacted,removed
 def process(root,prefix,archive):
  local=Counter();entries=[];omitted=[];redactions=0
  tarpath=bundle/(prefix.replace('/','__')+'.tar.gz') if archive else None
  tf=tarfile.open(tarpath,'w:gz',compresslevel=6) if archive else None
  try:
   for p in walk(root):
    rel=Path(prefix)/p.relative_to(root) if prefix else p.relative_to(root)
    try:
     size=p.lstat().st_size;reason=classify(p)
     if reason:
      omitted.append({'path':str(rel),'bytes':size,'reason':reason});local['excluded_'+reason]+=1;continue
     data=p.read_bytes()
     if p.suffix.lower() in {'.zip','.tar','.gz','.tgz'}:
      data,n,removed=filtered_archive(p,data);redactions+=n;local['archive_members_removed']+=removed
     # Exact known credentials are removed even from unexpected text suffixes.
     if p.suffix.lower() in TEXT or b'\x00' not in data[:8192]:data,n=clean(data);redactions+=n
     entry={'path':str(rel),'bytes':len(data),'sha256':digest(data),'source_bytes':size}
     entries.append(entry);local['files']+=1;local['bytes']+=len(data)
     if tf:
      ti=tarfile.TarInfo(str(rel));ti.size=len(data);ti.mode=0o755 if os.access(p,os.X_OK) else 0o644;ti.mtime=int(p.stat().st_mtime);tf.addfile(ti,io.BytesIO(data))
     # Keep source, documentation and compact summaries browseable on GitHub.
     visible=(not archive or p.suffix.lower() in {'.py','.sh','.md','.pdf','.svg','.html','.csv'} or (p.suffix.lower()=='.json' and p.name in {'suite.json','protocol.json','summary.json','BATCH1_SUMMARY.json','BATCH1_COMPLETE.json','COMPLETE.json','EXPORT_SUMMARY.json'}))
     if visible:
      q=dst/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data);q.chmod(p.stat().st_mode & 0o777)
    except OSError as e:
     errors.append({'path':str(rel),'error':str(e)})
  finally:
   if tf:tf.close()
  item=None
  if tarpath:
   archive_sha=sha_file(tarpath);parts=[]
   with tarpath.open('rb') as f:
    i=0
    while True:
     data=f.read(40*1024*1024)
     if not data:break
     part=tarpath.with_name(tarpath.name+f'.part{i:03d}');part.write_bytes(data);parts.append({'file':part.name,'bytes':len(data),'sha256':digest(data)});i+=1
   item={'name':tarpath.name,'sha256':archive_sha,'files':len(entries),'uncompressed_bytes':local['bytes'],'parts':parts};tarpath.unlink()
  with lock:
   stats.update(local);stats['redactions']+=redactions;all_files.extend(entries);excluded.extend(omitted)
   if item:manifest.append(item)
   print(json.dumps({'completed':prefix or 'root','files':local['files'],'bytes':local['bytes'],'redactions':redactions}),flush=True)
 # Root files only; regular directories are separate jobs.
 root_files=dst/'handoff/root_metadata';root_files.mkdir(parents=True,exist_ok=True)
 for p in src.iterdir():
  if p.is_file() and not classify(p):
   data,n=clean(p.read_bytes());(dst/p.name).write_bytes(data);stats['redactions']+=n
 jobs=[(src/name,name,False) for name in ['training_code','screening_code','data','handoff_tools','handoff'] if (src/name).exists()]
 jobs.extend((p,'training_runs/'+p.name,True) for p in (src/'training_runs').iterdir() if p.is_dir() and not p.is_symlink())
 # Capture external source and historical runtime, excluding training weights/videos as above.
 base=src.parent
 for root,prefix,archive in [
  (base/'third_party/ms-swift','handoff/external_sources/ms-swift',True),
  (base/'omni-opsd-runtime/worldsense_sft_lora_npu24_20260928','handoff/external_runtime/worldsense_sft_lora_npu24_20260928',True),
  (base/'omni-opsd-runtime/worldsense_npu24_deps','handoff/external_sources/worldsense_npu24_deps',True)]:
  if root.exists():jobs.append((root,prefix,archive))
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
  futures=[ex.submit(process,*job) for job in jobs]
  for f in concurrent.futures.as_completed(futures):f.result()
 (bundle/'manifest.json').write_text(json.dumps(sorted(manifest,key=lambda x:x['name']),ensure_ascii=False,indent=2))
 for name,rows in [('FILES.jsonl',all_files),('EXCLUDED_FILES.jsonl',excluded)]:
  with gzip.open(dst/'handoff'/(name+'.gz'),'wt') as f:
   for row in sorted(rows,key=lambda x:x['path']):f.write(json.dumps(row,ensure_ascii=False)+'\n')
 summary={'at':datetime.now().astimezone().isoformat(),'source':str(src),'destination':str(dst),'statistics':dict(stats),'bundles':len(manifest),'bundle_bytes':sum(p['bytes'] for m in manifest for p in m['parts']),'errors':errors,'policy':'No videos/weights/credentials. Symlinks not followed. Existing archives listed for separate filtered handling. Root .env never exported. SHA256 describes sanitized snapshot, not mutable source.'}
 (dst/'handoff/EXPORT_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False),flush=True)
 if errors:raise SystemExit(2)
if __name__=='__main__':main()
