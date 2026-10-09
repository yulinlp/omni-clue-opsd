#!/usr/bin/env python3
"""Three node queue: MCQ first, then answer-free open QA; durable task claims."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import shutil
import time
from datetime import datetime

REPO = Path(__file__).resolve().parents[2]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--worker', type=int, required=True)
    p.add_argument('--phase', choices=['both', 'openqa'], default='both',
                   help='Use openqa on an idle worker once all MCQ jobs have been assigned.')
    a = p.parse_args()
    root = a.root
    logroot = root / 'logs'
    logroot.mkdir(parents=True, exist_ok=True)
    (logroot / f'worker-{a.worker}.pid').write_text(str(os.getpid()))
    lock = (root / 'queue.lock').open('a')
    while True:
        claim = None
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            tasks = json.loads((root / 'tasks.json').read_text())
            statepath = root / 'queue_state.json'
            state = json.loads(statepath.read_text()) if statepath.exists() else {}
            for mode in (('openqa',) if a.phase == 'openqa' else ('mcq', 'openqa')):
                if mode == 'openqa':
                    if a.phase != 'openqa' and not all(state.get('mcq/' + t['label'], {}).get('status') == 'complete' for t in tasks):
                        break
                    if a.phase == 'openqa' and not all('mcq/' + t['label'] in state for t in tasks):
                        raise RuntimeError('Do not reserve an open-QA worker before every MCQ job is assigned.')
                    if not (root / 'OPENQA_READY').is_file():
                        break
                for task in tasks:
                    key = mode + '/' + task['label']
                    if key not in state:
                        state[key] = dict(status='running', worker=a.worker,
                                          pid=os.getpid(), started_at=datetime.now().astimezone().isoformat())
                        claim = (mode, task, key)
                        break
                if claim:
                    break
            complete = all(state.get(m + '/' + t['label'], {}).get('status') == 'complete'
                           for m in ('mcq', 'openqa') for t in tasks)
            tmp = root / f'queue_state.{os.getpid()}.tmp'
            tmp.write_text(json.dumps(state, indent=2) + '\n')
            tmp.replace(statepath)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
        if complete:
            # All prediction/scoring jobs are terminal before final comparison.
            with (root / 'aggregation.lock').open('a') as aggregate_lock:
                fcntl.flock(aggregate_lock, fcntl.LOCK_EX)
                if not (root / 'comparison/summary.json').exists():
                    env = dict(os.environ, PYTHONPATH=str(REPO / 'training_code/src'))
                    with (logroot / 'aggregate.log').open('w') as log:
                        rc = subprocess.call(['/home/ma-user/anaconda3/envs/PyTorch-2.9.0/bin/python',
                                              str(REPO / 'training_code/scripts/aggregate_worldsense_dual_eval.py'),
                                              '--root', str(root)], env=env, stdout=log, stderr=log)
                    if rc:
                        (logroot / f'worker-{a.worker}.exit').write_text(str(rc) + '\n')
                        raise SystemExit(rc)
            (logroot / f'worker-{a.worker}.exit').write_text('0\n')
            return
        if claim is None:
            time.sleep(15)
            continue
        mode, task, key = claim
        output = root / mode / task['label']
        output.mkdir(parents=True, exist_ok=True)
        # Bash reads later blocks during execution; never let source edits change
        # an already running invocation. Each attempt gets an immutable copy.
        runner = output / f'runner_attempt_{os.getpid()}.sh'
        shutil.copyfile(REPO / 'training_code/scripts/run_worldsense_dual_eval_npu.sh', runner)
        env = dict(os.environ, OMNI_OPSD_EVAL_MODEL=task['model'],
                   OMNI_OPSD_EVAL_MODE=mode, OMNI_OPSD_EVAL_BENCHMARK='WorldSense',
                   OMNI_OPSD_EVAL_DATASET=str(root / 'data' / ('worldsense.answer_free.jsonl' if mode == 'mcq' else 'worldsense.openqa.jsonl')),
                   OMNI_OPSD_EVAL_LABELS=str(root / 'data' / ('worldsense.labels.jsonl' if mode == 'mcq' else 'worldsense.openqa.labels.jsonl')),
                   OMNI_OPSD_EVAL_EXPECTED_ROWS='518',
                   OMNI_OPSD_EVAL_MAX_NEW_TOKENS='8' if mode == 'mcq' else '768')
        print(datetime.now().astimezone().isoformat(), 'START', key, flush=True)
        with (logroot / (mode + '_' + task['label'] + '.log')).open('a') as log:
            rc = subprocess.call(['bash', str(runner),
                                  task['arm'], task['adapter'], str(output)], env=env, stdout=log, stderr=log)
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            state = json.loads(statepath.read_text())
            state[key].update(status='complete' if rc == 0 else 'failed', exit_code=rc,
                              finished_at=datetime.now().astimezone().isoformat())
            tmp = root / f'queue_state.{os.getpid()}.tmp'
            tmp.write_text(json.dumps(state, indent=2) + '\n')
            tmp.replace(statepath)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
        print(datetime.now().astimezone().isoformat(), 'END', key, rc, flush=True)
        if rc:
            # Leave failed results and claims intact for diagnosis and explicit repair.
            (logroot / f'worker-{a.worker}.exit').write_text(str(rc) + '\n')
            raise SystemExit(rc)

if __name__ == '__main__':
    main()
