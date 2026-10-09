#!/usr/bin/env python3
"""One node, disjoint single-card shards, immutable training-matched inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path('/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd')
sys.path.insert(0, str(REPO/'training_code/scripts'))
PYENV = Path('/home/ma-user/anaconda3/envs/PyTorch-2.9.0/bin')
DEPS = '/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_npu24_deps'
SWIFT_ROOT = '/opt/huawei/dataset/hyl_ulan/ylhu/third_party/ms-swift'


def read(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def write(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def env_for(devices):
    env = dict(os.environ, PYTHONPATH=':'.join([DEPS, str(REPO/'training_code/src'), SWIFT_ROOT]),
               ASCEND_RT_VISIBLE_DEVICES=devices, NPROC_PER_NODE='1', NNODES='1',
               USE_AUDIO_IN_VIDEO='1', ENABLE_AUDIO_OUTPUT='0', WORLDSENSE_DROP_TALKER='1',
               FORCE_QWENVL_VIDEO_READER='pyav_seek', MAX_NUM_WORKERS_FETCH_VIDEO='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
               PYTORCH_NPU_ALLOC_CONF='expandable_segments:True', PYTHONUNBUFFERED='1')
    for key in ['RANK_TABLE_FILE', 'RANK_TABLE_FILE_V_1_0', 'RANK', 'LOCAL_RANK', 'WORLD_SIZE']:
        env.pop(key, None)
    return env


def validate(root, mode):
    data = root/'data'/('worldsense.answer_free.jsonl' if mode == 'mcq' else 'worldsense.openqa.jsonl')
    labels = root/'data'/('worldsense.labels.jsonl' if mode == 'mcq' else 'worldsense.openqa.labels.jsonl')
    manifest = json.loads((root/'data'/('manifest.json' if mode == 'mcq' else 'openqa_manifest.json')).read_text())
    for path, key in [(data, 'answer_free_sha256'), (labels, 'labels_sha256')]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[key], str(path)
    rows = read(data)
    assert len(rows) == 518 == len({r['case_id'] for r in rows})
    assert {r['case_id'] for r in rows} == {r['sample_id'] for r in read(labels)}
    for row in rows:
        assert len(row['messages']) == 1 and row['messages'][0]['role'] == 'user'
        assert not {'answer', 'gold_answer_text', 'teacher_prompt', 'teacher_videos', 'observation'} & row.keys()
        assert row['sampling_contract']['use_audio_in_video'] is True
        assert all(Path(v['video']).is_file() for v in row['videos'])
    return data, labels, rows


def verify_predictions(path, rows):
    from score_worldsense_openqa import join
    join(read(path), rows)


def run(a):
    data, labels, rows = validate(a.root, a.mode)
    if a.validate_only:
        print(json.dumps({'valid': True, 'mode': a.mode, 'rows': len(rows)})); return
    output = a.root/a.mode/a.label
    output.mkdir(parents=True, exist_ok=True)
    result = output/'results.jsonl'
    if not a.score_only:
        mapping = json.loads((a.root/'inference_cache_views.json').read_text())
        model = mapping.get(a.model, a.model)
        cards = a.devices.split(',')
        max_tokens = 8 if a.mode == 'mcq' else 512
        specification = dict(model_source=a.model, model=model, adapter=a.adapter, mode=a.mode,
                             devices=cards, temperature=1.0, top_p=1.0, top_k=-1,
                             min_p=0.0, repetition_penalty=1.0, seed=20260904,
                             max_new_tokens=max_tokens, max_length=32768,
                             data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                             independent_single_card_shards=True, audio_in_video=True)
        (output/'generation_protocol.json').write_text(json.dumps(specification, indent=2)+'\n')
        if result.exists():
            verify_predictions(result, rows)
        else:
            children = []
            shards = []
            try:
                for i, device in enumerate(cards):
                    folder = output/'shards'/f'card_{i}'
                    folder.mkdir(parents=True, exist_ok=True)
                    shard = rows[i::len(cards)]
                    source, target = folder/'input.jsonl', folder/'results.jsonl'
                    write(source, shard); shards.append((target, shard))
                    if target.exists():
                        try:
                            verify_predictions(target, shard); continue
                        except (ValueError, AssertionError, json.JSONDecodeError):
                            target.rename(folder/f'results.partial.{time.time_ns()}.jsonl')
                    command = [str(PYENV/'swift'), 'infer', '--model', model, '--model_type', 'qwen2_5_omni',
                               '--val_dataset', str(source), '--result_path', str(target),
                               '--infer_backend', 'transformers', '--max_batch_size', '1', '--write_batch_size', '1',
                               '--max_new_tokens', str(max_tokens), '--temperature', '1', '--top_p', '1',
                               '--top_k', '-1', '--repetition_penalty', '1', '--num_beams', '1',
                               '--stream', 'false', '--torch_dtype', 'bfloat16', '--attn_impl', 'sdpa',
                               '--max_length', '32768', '--dataset_num_proc', '1', '--val_dataset_shuffle', 'false',
                               '--seed', '20260904']
                    if a.adapter != '-': command += ['--adapters', a.adapter]
                    log = (folder/'infer.log').open('w')
                    env = env_for(device)
                    env.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(29570+i))
                    proc = subprocess.Popen(command, env=env, stdout=log, stderr=log)
                    children.append((proc, log))
                    (folder/'pid').write_text(str(proc.pid))
                    (folder/'command.json').write_text(json.dumps(command, indent=2)+'\n')
                while any(proc.poll() is None for proc, _ in children):
                    if any(proc.poll() not in (None, 0) for proc, _ in children):
                        raise RuntimeError('Inference shard failed; inspect current infer.log')
                    time.sleep(2)
                if any(proc.returncode != 0 for proc, _ in children):
                    raise RuntimeError('Inference shard failed')
                merged = []
                for target, shard in shards:
                    verify_predictions(target, shard); merged.extend(read(target))
                write(result, merged); verify_predictions(result, rows)
            finally:
                for proc, log in children:
                    if proc.poll() is None: proc.terminate()
                for proc, log in children:
                    try: proc.wait(timeout=20)
                    except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                    log.close()
    verify_predictions(result, rows)
    if a.mode == 'openqa' and not a.score_only:
        print(json.dumps({'generated': a.label, 'rows': len(rows)})); return
    if a.mode == 'mcq':
        command = [str(PYENV/'python'), str(REPO/'training_code/scripts/score_worldsense_mcq_v2.py')]
    else:
        scorer = a.root/'code/score_worldsense_openqa_v2.py'
        if not scorer.exists(): scorer = REPO/'training_code/scripts/score_worldsense_openqa_v2.py'
        command = [str(PYENV/'python'), str(scorer)]
        command += json.loads((a.root/'judge_arguments.json').read_text())
        command += ['--devices', a.devices]
    command += ['--results', str(result), '--labels', str(labels), '--dataset', str(data),
                '--output', str(output/'summary.json'), '--arm', a.label, '--adapter', '' if a.adapter == '-' else a.adapter]
    with (output/'summarize.log').open('a') as log:
        subprocess.run(command, env=env_for(a.devices), stdout=log, stderr=log, check=True)
    print(json.dumps({'scored': a.label, 'mode': a.mode}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--mode', choices=['mcq', 'openqa'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--adapter', default='-')
    p.add_argument('--devices', default='0,1,2,3,4,5,6,7')
    p.add_argument('--score-only', action='store_true')
    p.add_argument('--validate-only', action='store_true')
    run(p.parse_args())
