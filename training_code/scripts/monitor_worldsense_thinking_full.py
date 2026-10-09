#!/usr/bin/env python3
"""Persist training status until both workers have exited; CPU-only monitor."""
import datetime
import argparse
import json
import time
from pathlib import Path

ROOT = Path('/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_openqa_thinking_full_20260930')
WORKERS = (1, 2)


def snapshot():
    exit_codes = {}
    for worker in WORKERS:
        path = ROOT / f'logs/formal_worker-{worker}.exit'
        exit_codes[str(worker)] = int(path.read_text().strip()) if path.exists() else None
    status = 'running'
    if any(v is not None and v != 0 for v in exit_codes.values()):
        status = 'failed'
    elif all(v == 0 for v in exit_codes.values()):
        status = 'complete'
    versions = sorted((ROOT / 'outputs/formal').glob('v*'))
    result = {'updated_at': datetime.datetime.now().astimezone().isoformat(),
              'status': status, 'worker_exit_codes': exit_codes}
    if versions:
        version = versions[-1]
        result['output_dir'] = str(version)
        result['checkpoints'] = [str(p) for p in sorted(version.glob('checkpoint-*'))]
        logging = version / 'logging.jsonl'
        if logging.exists():
            lines = logging.read_text().splitlines()
            if lines:
                try:
                    result['latest_metrics'] = json.loads(lines[-1])
                except json.JSONDecodeError:
                    pass
    temporary = ROOT / 'monitor_status.json.tmp'
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(ROOT / 'monitor_status.json')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, default=ROOT)
    parser.add_argument('--workers', type=int, nargs='+', default=list(WORKERS))
    args = parser.parse_args()
    ROOT, WORKERS = args.run_root, args.workers
    while snapshot() == 'running':
        time.sleep(30)
