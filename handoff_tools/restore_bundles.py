#!/usr/bin/env python3
"""Verify and restore handoff bundles, never following archive links."""
import argparse,hashlib,json,tarfile,tempfile,shutil
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path('.'));p.add_argument('--verify-only',action='store_true');a=p.parse_args();root=a.repo.resolve();folder=root/'handoff/bundles';mf=folder/'manifest.json'
 if not mf.exists():raise SystemExit('No bundle manifest: full result upload is absent or incomplete.')
 for item in json.loads(mf.read_text()):
  with tempfile.TemporaryFile() as tmp:
   h=hashlib.sha256()
   for part in item['parts']:
    q=folder/part['file']
    if q.stat().st_size!=part['bytes'] or sha(q)!=part['sha256']:raise ValueError('Corrupt/missing part: '+str(q))
    with q.open('rb') as f:
     for b in iter(lambda:f.read(1024*1024),b''):tmp.write(b);h.update(b)
   if h.hexdigest()!=item['sha256']:raise ValueError('Archive checksum mismatch: '+item['name'])
   if not a.verify_only:
    tmp.seek(0)
    with tarfile.open(fileobj=tmp,mode='r:gz') as tf:
     for m in tf:
      target=(root/m.name).resolve()
      if not target.is_relative_to(root) or m.issym() or m.islnk() or not m.isfile():raise ValueError('Unsafe archive entry: '+m.name)
      target.parent.mkdir(parents=True,exist_ok=True)
      with tf.extractfile(m) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
      target.chmod(m.mode & 0o777)
  print(('Verified ' if a.verify_only else 'Restored ')+item['name'],flush=True)
if __name__=='__main__':main()
