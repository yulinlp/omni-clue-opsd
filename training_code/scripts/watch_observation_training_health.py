#!/usr/bin/env python3
"""Check training jobs every minute; persist alerts and half-hour reports.

Repairs are performed by the supervising agent after diagnosis, not by blindly
restarting a failed distributed job.
"""
import datetime as dt
import argparse
import fcntl
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'training_runs/observation_training_watch_20260930'
JOBS = {
    'sft': ('worldsense_observation_sft_lora_20260930', {0: '172.16.2.181'}),
    'clue': ('worldsense_openqa_thinking_full_observation_20260930',
             {1: '172.16.12.237', 2: '172.16.12.70'}),
    'sft_full': ('worldsense_observation_sft_full_npu96_w6w7_20260930',
                 {6: 'npu96:6', 7: 'npu96:7'}),
    'clue_no_observation': ('worldsense_clue_full_no_observation_npu96_w10w11_20260930',
                            {10: 'npu96:10', 11: 'npu96:11'}),
}
FATAL = re.compile(r'Traceback \(most recent call last\)|OutOfMemoryError|'
                   r'RuntimeError:|ChildFailedError|NPU out of memory|'
                   r'\b(loss|grad_norm)[\"\x27]?\s*:\s*(nan|inf)\b', re.I)


def append(name, value):
    with (ROOT / name).open('a') as file:
        file.write(json.dumps(value, ensure_ascii=False) + '\n')


def inspect(name, directory, workers):
    root = REPO / 'training_runs' / directory
    status = json.loads((root / 'monitor_status.json').read_text())
    result = {'job': name, 'status': status, 'issues': [], 'workers': {}}
    for worker, host in workers.items():
        exit_path = root / f'logs/formal_worker-{worker}.exit'
        pid_path = root / f'logs/formal_worker-{worker}.pid'
        code = int(exit_path.read_text()) if exit_path.exists() else None
        pid = int(pid_path.read_text())
        if host.startswith('npu96:'):
            job = 'ma-job-7b525feb-b862-4352-8403-9c3a018c05e8'
            container = f'{job}-worker-{worker}.{job}'
            probe = ['ssh', '-F', '/dev/null', '-p', '31445', '-i',
                     '/home/ma-user/.ssh/ulan_31445_npu96.pem', '-o',
                     'BatchMode=yes', '-o', 'ConnectTimeout=8',
                     'ma-user@dev-modelarts-cnnorth9.huaweicloud.com',
                     f'ssh -F /dev/null -p 2222 -o BatchMode=yes '
                     f'-o ConnectTimeout=8 ma-user@{container} "kill -0 {pid}"']
        else:
            probe = ['ssh', '-F', '/dev/null', '-p', '2222', '-o',
                     'BatchMode=yes', '-o', 'ConnectTimeout=8',
                     f'ma-user@{host}', f'kill -0 {pid}']
        alive = subprocess.run(
            probe,
            capture_output=True, text=True, timeout=15)
        result['workers'][str(worker)] = {'exit_code': code,
                                         'probe_returncode': alive.returncode}
        if code not in (None, 0):
            result['issues'].append(f'worker-{worker} exited with code {code}')
        elif code is None and alive.returncode != 0:
            result['issues'].append(f'worker-{worker} process/SSH probe failed: '
                                    + alive.stderr.strip()[-300:])
        log = root / f'logs/formal_worker-{worker}.log'
        with log.open('rb') as file:
            file.seek(max(0, log.stat().st_size - 100000))
            tail = file.read().decode(errors='replace')
        errors = [line[-1000:] for line in tail.splitlines() if FATAL.search(line)]
        if errors:
            result['issues'].extend(errors[-4:])
    metrics = {key: value for key, value in status.get('latest_metrics', {}).items()
               if key != 'log_history'}
    for key in ('loss', 'grad_norm'):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not math.isfinite(value):
            result['issues'].append(f'nonfinite {key}: {value}')
    result['metrics'] = metrics
    versions = sorted((root / 'outputs/formal').glob('v*'))
    if versions:
        logging = versions[-1] / 'logging.jsonl'
        if logging.exists():
            age = time.time() - logging.stat().st_mtime
            result['seconds_since_metrics'] = round(age)
            if age > 2700 and status['status'] == 'running':
                result['issues'].append('No new training metrics for over 45 minutes; inspect before restarting')
        completions = versions[-1] / 'completions.jsonl'
        if completions.exists():
            with completions.open('rb') as file:
                file.seek(max(0, completions.stat().st_size - 1048576))
                recent = file.read().decode(errors='replace').splitlines()
            for line in reversed(recent):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                texts = record.get('completion', [])
                if not texts:
                    continue
                analyses = [re.search(r'<analysis>(.*?)</analysis>', text, re.S)
                            for text in texts]
                answers = [re.search(r'<answer>(.*?)</answer>', text, re.S)
                           for text in texts]
                words = [len(match.group(1).split()) for match in analyses if match]
                count = len(texts)
                result['quality'] = {
                    'step': record.get('step', [None])[0], 'samples': count,
                    'closed_analysis_fraction': sum(bool(x) for x in analyses) / count,
                    'closed_answer_fraction': sum(bool(x) for x in answers) / count,
                    'both_tags_fraction': sum(bool(a and b) for a, b in
                                              zip(analyses, answers)) / count,
                    'analysis_over_120_words_fraction': sum(w > 120 for w in words) / count,
                    'completed_analysis_words_mean': sum(words) / len(words) if words else None,
                    'max_analysis_words': max(words, default=0),
                }
                break
    return result


def check_jobs():
    jobs = []
    for name, (directory, workers) in JOBS.items():
        if name in ('sft_full', 'clue_no_observation') and not all(
            (REPO / 'training_runs' / directory /
             f'logs/formal_worker-{worker}.pid').exists()
            for worker in workers
        ):
            continue
        try:
            job = inspect(name, directory, workers)
        except Exception as error:
            job = {'job': name, 'issues': [f'monitor error: {error}']}
        jobs.append(job)
    return jobs


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'watch.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('A health monitor is already running.')
    (ROOT / 'watch.pid').write_text(str(os.getpid()) + '\n')
    next_report = 0
    previous_alerts = {}
    while True:
        now = dt.datetime.now().astimezone().isoformat()
        jobs = check_jobs()
        for job in jobs:
            name = job['job']
            issues = job['issues']
            if issues and issues != previous_alerts.get(name):
                append('alerts.jsonl', {'time': now, **job})
            previous_alerts[name] = issues
        snapshot = {'time': now, 'jobs': jobs}
        temp = ROOT / 'health_status.json.tmp'
        temp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
        temp.replace(ROOT / 'health_status.json')
        if time.monotonic() >= next_report:
            append('half_hour_reports.jsonl', snapshot)
            next_report = time.monotonic() + 1800
        if all(job.get('status', {}).get('status') == 'complete' for job in jobs):
            append('half_hour_reports.jsonl', snapshot)
            return
        time.sleep(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true',
                        help='Probe all workers sequentially and print a snapshot without writing files.')
    args = parser.parse_args()
    if args.once:
        jobs = check_jobs()
        print(json.dumps({'time': dt.datetime.now().astimezone().isoformat(),
                          'jobs': [{**job, 'status': job.get('status', {}).get('status')}
                                   for job in jobs]}, ensure_ascii=False))
        raise SystemExit(int(any(job['issues'] for job in jobs)))
    main()
