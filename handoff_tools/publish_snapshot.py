#!/usr/bin/env python3
"""Commit sanitized export in bounded batches; optionally push without force."""
import argparse,json,os,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--push',action='store_true');a=p.parse_args();root=a.repo.resolve()
 env=os.environ.copy()
 for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','GIT_ASKPASS','SSH_ASKPASS']:env.pop(k,None)
 env['GIT_TERMINAL_PROMPT']='0'
 def git(*args,check=True):return subprocess.run(['git','-c','http.proxy=','-c','https.proxy=',*args],cwd=root,env=env,check=check,capture_output=True,text=True)
 summary=json.loads((root/'handoff/EXPORT_SUMMARY.json').read_text())
 if summary['errors']:raise SystemExit('Export errors must be resolved before publication.')
 if not (root/'handoff/bundles/manifest.json').exists():raise SystemExit('Missing bundle manifest.')
 paths=[]
 for base,dirs,files in os.walk(root):
  dirs[:]=sorted(d for d in dirs if d not in {'.git','__pycache__'})
  for name in sorted(files):
   q=Path(base)/name
   if q.is_symlink():raise SystemExit('Unexpected symlink: '+str(q))
   if q.stat().st_size>95*1024*1024:raise SystemExit('Oversized Git file: '+str(q))
   paths.append((str(q.relative_to(root)),q.stat().st_size))
 batches=[];batch=[];size=0
 for name,n in paths:
  if batch and size+n>256*1024*1024:batches.append(batch);batch=[];size=0
  batch.append(name);size+=n
 if batch:batches.append(batch)
 for i,batch in enumerate(batches):
  for start in range(0,len(batch),100):git('add','-f','--',*batch[start:start+100])
  if git('diff','--cached','--quiet',check=False).returncode:
   git('commit','-m',f'Handoff 2026-10-09: sanitized code, data and results ({i+1}/{len(batches)})')
   print(git('rev-parse','--short','HEAD').stdout.strip(),f'batch {i+1}/{len(batches)}',flush=True)
 if a.push:
  git('fetch','origin','main')
  if git('merge-base','--is-ancestor','origin/main','HEAD',check=False).returncode:raise SystemExit('Remote changed independently; reconcile before pushing. No force push performed.')
  for commit in git('rev-list','--reverse','origin/main..HEAD').stdout.splitlines():
   r=git('push','origin',commit+':refs/heads/main',check=False)
   if r.returncode:raise SystemExit('Push failed; restore GitHub write credentials or inspect repository permissions.\n'+r.stderr[-1800:])
   print('Pushed',commit,flush=True)
  remote=git('ls-remote','origin','refs/heads/main').stdout.split()[0];local=git('rev-parse','HEAD').stdout.strip()
  if remote!=local:raise SystemExit('Remote head verification failed.')
  print('Verified remote main:',remote)
 else:print('Local commits prepared. Upload not attempted. Re-run with --push after restoring credentials.')
if __name__=='__main__':main()
