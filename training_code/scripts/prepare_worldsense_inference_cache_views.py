#!/usr/bin/env python3
"""Inference-only config views: shared original weights, use_cache=True throughout."""
import argparse
import hashlib
import json
from pathlib import Path

def enable_cache(config):
    if isinstance(config,dict):
        for k,v in config.items():
            if k=='use_cache':config[k]=True
            elif isinstance(v,dict):enable_cache(v)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    tasks=json.loads((a.root/'tasks.json').read_text());mapping={};audit=[]
    for task in tasks:
        source=Path(task['model'])
        if task['label']=='base' or task['adapter']!='-':continue
        target=a.root/'inference_models'/task['label'];target.mkdir(parents=True,exist_ok=True)
        for path in source.iterdir():
            if not path.is_file() or path.name in ['config.json','generation_config.json','ema_teacher.safetensors']:continue
            if path.name.startswith('rng_state') or path.name in ['optimizer.pt','scheduler.pt']:continue
            link=target/path.name
            if not link.exists():link.symlink_to(path.resolve())
        cfg=json.loads((source/'config.json').read_text());enable_cache(cfg);cfg['use_cache']=True
        assert cfg['thinker_config']['text_config']['use_cache'] is True
        (target/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        gen=json.loads((source/'generation_config.json').read_text());gen['use_cache']=True
        (target/'generation_config.json').write_text(json.dumps(gen,indent=2)+'\n')
        mapping[str(source)]=str(target.resolve())
        shards=list(source.glob('model-*.safetensors'))
        assert len(shards)==4 and all((target/x.name).resolve()==x.resolve() for x in shards)
        audit.append(dict(label=task['label'],source=str(source),inference_view=str(target.resolve()),
                          source_config_sha256=hashlib.sha256((source/'config.json').read_bytes()).hexdigest(),
                          original_weights_unchanged=True,symlinked_weight_shards=4,use_cache=True))
    (a.root/'inference_cache_views.json').write_text(json.dumps(mapping,indent=2)+'\n')
    (a.root/'inference_cache_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('Prepared cache-enabled inference views:',len(audit))

if __name__=='__main__':main()
