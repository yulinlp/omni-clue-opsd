#!/usr/bin/env python3
"""Prepare independent card jobs and merge them without a node-wide barrier.

The eight existing input partitions and all inference arguments stay fixed.
Physical cards on any of the three nodes may execute each partition. Legacy
runners are left alive and are harvested only after their final summary exists.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

REPO = Path('/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd')
sys.path.insert(0, str(REPO / 'training_code/scripts'))
from distributed_directory_lock import DirectoryLock
from run_worldsense_training_matched_eval import PYENV, env_for, read, validate, verify_predictions, write


def now():
    return datetime.now().astimezone().isoformat()


def atomic_json(path, value):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def load(path):
    return json.loads(path.read_text())


def generation_command(model, adapter, source, target, mode):
    command = [str(PYENV / 'swift'), 'infer', '--model', model, '--model_type', 'qwen2_5_omni',
               '--val_dataset', str(source), '--result_path', str(target),
               '--infer_backend', 'transformers', '--max_batch_size', '1', '--write_batch_size', '1',
               '--max_new_tokens', '8' if mode == 'mcq' else '512', '--temperature', '1', '--top_p', '1',
               '--top_k', '-1', '--repetition_penalty', '1', '--num_beams', '1',
               '--stream', 'false', '--torch_dtype', 'bfloat16', '--attn_impl', 'sdpa',
               '--max_length', '32768', '--dataset_num_proc', '1', '--val_dataset_shuffle', 'false',
               '--seed', '20260904']
    if adapter != '-':
        command += ['--adapters', adapter]
    return command


def judge_command(root, source, target):
    args = load(root / 'judge_arguments.json')
    model = args[args.index('--judge-model') + 1]
    model_type = args[args.index('--judge-model-type') + 1]
    return [str(PYENV / 'swift'), 'infer', '--model', model, '--model_type', model_type,
            '--val_dataset', str(source), '--result_path', str(target), '--infer_backend', 'transformers',
            '--max_batch_size', '1', '--write_batch_size', '1', '--max_new_tokens', '256',
            '--temperature', '0', '--stream', 'false', '--torch_dtype', 'bfloat16',
            '--attn_impl', 'sdpa', '--max_length', '4096', '--dataset_num_proc', '1',
            '--val_dataset_shuffle', 'false', '--seed', '20261003', '--enable_thinking', 'false']


def score_command(root, task, mode, prepare=False, judge_results=()):
    data, labels, _ = validate(root, mode)
    output = root / mode / task['label']
    scorer = (REPO / 'training_code/scripts/score_worldsense_mcq_v2.py' if mode == 'mcq'
              else root / 'code/score_worldsense_openqa_v2.py')
    command = [str(PYENV / 'python'), str(scorer), '--results', str(output / 'results.jsonl'),
               '--labels', str(labels), '--dataset', str(data), '--output', str(output / 'summary.json'),
               '--arm', task['label'], '--adapter', '' if task['adapter'] == '-' else task['adapter']]
    if mode == 'openqa':
        command += load(root / 'judge_arguments.json')
        if prepare:
            command.append('--prepare-only')
        for path in judge_results:
            command += ['--judge-results', str(path)]
    return command


def cpu_score(root, task, mode, **kwargs):
    folder = root / mode / task['label']
    with (folder / 'summarize.log').open('a') as log:
        subprocess.run(score_command(root, task, mode, **kwargs), env=env_for(''),
                       stdout=log, stderr=log, check=True)


def units_for_generation(root, task, mode, order):
    data, _, rows = validate(root, mode)
    output = root / mode / task['label']
    output.mkdir(parents=True, exist_ok=True)
    mapping = load(root / 'inference_cache_views.json')
    model = mapping.get(task['model'], task['model'])
    protocol = dict(model_source=task['model'], model=model, adapter=task['adapter'], mode=mode,
                    devices='assigned dynamically per partition; see assignment.json',
                    temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, repetition_penalty=1.0,
                    seed=20260904, max_new_tokens=8 if mode == 'mcq' else 512, max_length=32768,
                    data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                    independent_single_card_shards=True, audio_in_video=True,
                    fixed_input_partitions=8, cross_node_card_pool=True)
    atomic_json(output / 'generation_protocol.json', protocol)
    units = {}
    for i in range(8):
        shard = rows[i::8]
        folder = output / 'shards' / f'card_{i}'
        folder.mkdir(parents=True, exist_ok=True)
        source, target = folder / 'input.jsonl', folder / 'results.jsonl'
        write(source, shard)
        # Pending models must not have incomplete results: avoid quietly redoing
        # a random subset with a different RNG history.
        status = 'pending'
        if target.exists():
            verify_predictions(target, shard)
            status = 'complete'
        unit_id = f'generation/{mode}/{task["label"]}/{i}'
        units[unit_id] = dict(status=status, kind='generation', task_key=f'{mode}/{task["label"]}',
                             partition=i, folder=str(folder), result_path=str(target),
                             expected_rows=len(shard), priority=(0 if mode == 'mcq' else 1000) + order * 8 + i,
                             command=generation_command(model, task['adapter'], source, target, mode),
                             env_overrides={}, created_at=now())
    return units


def units_for_judge(root, task):
    folder = root / 'openqa' / task['label']
    cpu_score(root, task, 'openqa', prepare=True)
    inputs = read(folder / 'judge_v2_pending_inputs.jsonl')
    args = load(root / 'judge_arguments.json')
    from score_worldsense_openqa_v2 import RUBRIC_SHA
    judge_root = folder / 'judge_v2'
    judge_root.mkdir(exist_ok=True)
    config = dict(judge_model=args[args.index('--judge-model') + 1],
                  judge_model_type=args[args.index('--judge-model-type') + 1],
                  judge_rubric_sha256=RUBRIC_SHA,
                  inputs_sha256=hashlib.sha256(json.dumps(inputs, ensure_ascii=False, sort_keys=True).encode()).hexdigest())
    config_path = judge_root / 'judge_run_config.json'
    if config_path.exists() and load(config_path) != config:
        raise ValueError('Existing judge configuration differs; cached verdicts cannot be mixed')
    atomic_json(config_path, config)
    units = {}
    count = min(8, len(inputs))
    for i in range(count):
        shard = inputs[i::count]
        card = judge_root / f'card_{i}'
        card.mkdir(exist_ok=True)
        source, target = card / 'input.jsonl', card / 'results.jsonl'
        write(source, shard)
        status = 'pending'
        if target.exists():
            verify_predictions(target, shard)
            status = 'complete'
        units[f'judge/{task["label"]}/{i}'] = dict(
            status=status, kind='judge', task_key='score/' + task['label'], partition=i,
            folder=str(card), result_path=str(target), expected_rows=len(shard), priority=-100 + i,
            command=judge_command(root, source, target), env_overrides={'USE_AUDIO_IN_VIDEO': '0'}, created_at=now())
    return units


def task_units(pool, key):
    return [u for u in pool['units'].values() if u['task_key'] == key]


def merge_generation(root, task, mode, units):
    _, _, rows = validate(root, mode)
    if len(units) != 8 or {u['partition'] for u in units} != set(range(8)):
        raise ValueError('Generation requires exactly the original eight disjoint partitions')
    merged = []
    for unit in sorted(units, key=lambda u: u['partition']):
        shard = rows[unit['partition']::8]
        path = Path(unit['result_path'])
        verify_predictions(path, shard)
        merged.extend(read(path))
    result = root / mode / task['label'] / 'results.jsonl'
    tmp = result.with_name('results.merging.jsonl')
    write(tmp, merged)
    verify_predictions(tmp, rows)
    tmp.replace(result)


def publish_task(root, key, **details):
    with DirectoryLock(root / 'card_pool.lock.d'):
        state = load(root / 'queue_state.json')
        state.setdefault(key, {}).update(details)
        atomic_json(root / 'queue_state.json', state)


def add_units(root, units):
    with DirectoryLock(root / 'card_pool.lock.d'):
        pool = load(root / 'card_pool_state.json')
        overlap = pool['units'].keys() & units.keys()
        if overlap:
            raise ValueError('Attempted to replace existing work units: ' + str(overlap))
        pool['units'].update(units)
        atomic_json(root / 'card_pool_state.json', pool)


def initialize(root):
    path = root / 'card_pool_state.json'
    with DirectoryLock(root / 'card_pool.lock.d'):
        state = load(root / 'queue_state.json')
        if not path.exists():
            legacy = {key: dict(value) for key, value in state.items() if value['status'] == 'running'}
            atomic_json(path, dict(version=1, created_at=now(), units={}, legacy_tasks=legacy,
                                   scheduling='independent single-card partitions across 24 physical cards'))
    tasks = load(root / 'tasks.json')
    for mode in ('mcq', 'openqa'):
        for order, task in enumerate(tasks):
            key = mode + '/' + task['label']
            if key in state:
                continue
            # Resume initialization after a controller interruption without
            # replacing units which another card has already claimed.
            with DirectoryLock(root / 'card_pool.lock.d'):
                existing = task_units(load(path), key)
            if existing:
                if len(existing) != 8:
                    raise ValueError('Incomplete work-unit preparation: ' + key)
            else:
                units = units_for_generation(root, task, mode, order)
                add_units(root, units)
            publish_task(root, key, status='running', worker='card-pool', pid=os.getpid(),
                         devices='distributed', started_at=now(), attempt=1,
                         scheduler='card-pool', created_at=now())


def proven_unstarted_claim(root, unit, checked_epoch):
    """A later live heartbeat can prove a committed reservation never launched.

    Never recover an exited/failed inference, a missing worker, or a reservation
    with any launch/output evidence. This specifically handles an exception
    after the claim write but before claim_pending returns to its worker.
    """
    if unit.get('status') != 'running' or unit.get('allocation_stage') != 'claimed-not-started':
        return False
    if unit.get('pid') or not unit.get('claim_id'):
        return False
    claimed_epoch = unit.get('claimed_epoch')
    if not isinstance(claimed_epoch, (int, float)) or checked_epoch - claimed_epoch < 20:
        return False
    worker = unit.get('worker')
    if not isinstance(worker, str) or not worker or Path(worker).name != worker:
        return False
    try:
        heartbeat = load(root / 'card_pool_workers' / worker / 'heartbeat.json')
        heartbeat_epoch = datetime.fromisoformat(heartbeat['heartbeat_at']).timestamp()
        if (heartbeat.get('worker_pid') != unit.get('worker_pid')
                or heartbeat.get('hostname') != unit.get('hostname')
                or heartbeat.get('status') not in ('running', 'stopping')
                or heartbeat_epoch < claimed_epoch + 10
                or not 0 <= checked_epoch - heartbeat_epoch <= 30):
            return False
        if not isinstance(heartbeat.get('devices'), dict) or any(
                not isinstance(slot, dict) for slot in heartbeat['devices'].values()):
            return False
        # Matching worker/PID, later heartbeat, and no owned slot: its event
        # loop has advanced beyond the reservation without starting this unit.
        if any(slot.get('unit_id') == unit.get('_unit_id') for slot in heartbeat['devices'].values()):
            return False
        folder = Path(unit['folder'])
        if not folder.is_absolute() or not folder.resolve().is_relative_to(root.resolve()):
            return False
        for name in ('pid', 'assigned_device.json', 'infer.log', 'results.jsonl'):
            path = folder / name
            if path.exists() and (not path.is_file() or path.stat().st_size):
                return False
    except (KeyError, TypeError, ValueError, OSError):
        return False
    return True


def recover_unstarted_claims(root):
    """Return only positively proven never-launched claims to the same queue."""
    snapshot = load(root / 'card_pool_state.json')
    epoch = time.time()
    candidates = [(key, unit['claim_id']) for key, unit in snapshot['units'].items()
                  if proven_unstarted_claim(root, dict(unit, _unit_id=key), epoch)]
    if not candidates:
        return []
    recovered = []
    with DirectoryLock(root / 'card_pool.lock.d'):
        pool = load(root / 'card_pool_state.json')
        for key, claim_id in candidates:
            unit = pool['units'].get(key, {})
            if unit.get('claim_id') != claim_id or not proven_unstarted_claim(
                    root, dict(unit, _unit_id=key), time.time()):
                continue
            reservation = {name: unit.get(name) for name in
                           ('claim_id', 'worker', 'worker_pid', 'hostname', 'assigned_device', 'claimed_at')}
            reservation.update(recovered_at=now(), reason='Later live worker heartbeat proves no launch; no launch/output evidence')
            unit.setdefault('recovered_unstarted_claims', []).append(reservation)
            for name in ('claim_id', 'worker', 'worker_pid', 'hostname', 'assigned_device',
                         'claimed_at', 'claimed_epoch', 'pid', 'pid_start_ticks', 'started_at',
                         'command_sha256', 'master_port'):
                unit.pop(name, None)
            unit.update(status='pending', allocation_stage='requeued-unstarted-orphan')
            recovered.append(dict(unit_id=key, **reservation))
        if recovered:
            atomic_json(root / 'card_pool_state.json', pool)
    for item in recovered:
        print(json.dumps(dict(event='unstarted_claim_recovered', **item), ensure_ascii=False), flush=True)
    return recovered


def tick(root):
    recover_unstarted_claims(root)
    pool = load(root / 'card_pool_state.json')
    state = load(root / 'queue_state.json')
    tasks = load(root / 'tasks.json')
    # Finish preserved legacy jobs only after the last file is published. A
    # brief stability delay avoids overlapping with the legacy finalizer.
    for key in pool['legacy_tasks']:
        if state[key]['status'] != 'running':
            continue
        mode, label = key.split('/', 1)
        output = root / ('openqa' if mode == 'score' else mode) / label
        summary = output / 'summary.json'
        if not summary.exists() or time.time() - summary.stat().st_mtime < 30:
            continue
        value = load(summary)
        if value.get('total') != 518:
            raise ValueError('Legacy summary has an unexpected sample count: ' + key)
        if mode == 'score' and not value.get('scoring_pipeline_reliable_on_calibration'):
            continue
        if mode == 'openqa':
            continue  # Legacy generation is harvested from its generated state below.
        task = next(t for t in tasks if t['label'] == label)
        _, _, rows = validate(root, 'openqa' if mode == 'score' else mode)
        verify_predictions(output / 'results.jsonl', rows)
        if mode == 'mcq':
            cpu_score(root, task, mode)
        publish_task(root, key, status='complete', exit_code=0, finished_at=now(),
                     harvested_by_card_pool=True)
        if mode == 'score':
            publish_task(root, 'openqa/' + label, status='complete', scored_at=now())
    # Normalize already completed legacy MC grades with the same v2 parser.
    for task in tasks:
        key = 'mcq/' + task['label']
        if state.get(key, {}).get('status') == 'complete':
            summary = root / 'mcq' / task['label'] / 'summary.json'
            if not load(summary).get('scoring', '').startswith('mcq-v2:'):
                cpu_score(root, task, 'mcq')
    for mode in ('mcq', 'openqa'):
        for task in tasks:
            key = mode + '/' + task['label']
            units = task_units(pool, key)
            if not units or state[key]['status'] in ('complete', 'generated', 'failed'):
                continue
            if any(u['status'] == 'failed' for u in units):
                publish_task(root, key, status='failed', failure='card-pool unit failed; manual repair required')
                continue
            if all(u['status'] == 'complete' for u in units):
                merge_generation(root, task, mode, units)
                if mode == 'mcq':
                    cpu_score(root, task, mode)
                publish_task(root, key, status='complete' if mode == 'mcq' else 'generated',
                             exit_code=0, finished_at=now())
    state = load(root / 'queue_state.json')
    for task in tasks:
        key = 'score/' + task['label']
        generated = state.get('openqa/' + task['label'], {}).get('status') == 'generated'
        if state.get(key, {}).get('status') == 'complete' and generated:
            summary = load(root / 'openqa' / task['label'] / 'summary.json')
            if summary.get('total') != 518 or not summary.get('scoring_pipeline_reliable_on_calibration'):
                raise ValueError('Completed score has no final calibrated summary: ' + key)
            publish_task(root, 'openqa/' + task['label'], status='complete', scored_at=now())
            continue
        if generated and key not in state:
            existing = task_units(pool, key)
            if not existing:
                units = units_for_judge(root, task)
                add_units(root, units)
            publish_task(root, key, status='running', worker='card-pool', pid=os.getpid(),
                         devices='distributed', started_at=now(), scheduler='card-pool')
        units = task_units(pool, key)
        if state.get(key, {}).get('status') in ('complete', 'failed'):
            continue
        if not units:
            # An interrupted empty-judge finalizer must resume on its next
            # tick, while an existing legacy score stays with its live runner.
            current = load(root / 'queue_state.json').get(key, {})
            pending = root / 'openqa' / task['label'] / 'judge_v2_pending_inputs.jsonl'
            if current.get('scheduler') == 'card-pool' and pending.exists() and not read(pending):
                empty = pending.parent / 'judge_v2' / 'empty_results.jsonl'
                write(empty, [])
                cpu_score(root, task, 'openqa', judge_results=[empty])
                publish_task(root, key, status='complete', exit_code=0, finished_at=now())
                publish_task(root, 'openqa/' + task['label'], status='complete', scored_at=now())
            continue
        if any(u['status'] == 'failed' for u in units):
            publish_task(root, key, status='failed', failure='judge card unit failed; manual repair required')
        elif all(u['status'] == 'complete' for u in units):
            for unit in units:
                verify_predictions(unit['result_path'], read(Path(unit['folder']) / 'input.jsonl'))
            cpu_score(root, task, 'openqa', judge_results=[u['result_path'] for u in units])
            publish_task(root, key, status='complete', exit_code=0, finished_at=now())
            publish_task(root, 'openqa/' + task['label'], status='complete', scored_at=now())
    state = load(root / 'queue_state.json')
    return all(state.get(mode + '/' + t['label'], {}).get('status') == 'complete'
               for t in tasks for mode in ('mcq', 'openqa'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--initialize-only', action='store_true')
    p.add_argument('--interval', type=float, default=5)
    a = p.parse_args()
    a.root = a.root.resolve()
    with DirectoryLock(a.root / 'card_pool_controller.lease.lock.d', timeout=1):
        initialize(a.root)
        if a.initialize_only:
            print(json.dumps({'initialized': str(a.root)})); return
        try:
            while True:
                complete = tick(a.root)
                atomic_json(a.root / 'card_pool_controller.json', dict(
                    pid=os.getpid(), hostname=socket.gethostname(), heartbeat_at=now(),
                    complete=complete, status='complete' if complete else 'running'))
                if complete:
                    with DirectoryLock(a.root / 'card_pool.lock.d'):
                        pool = load(a.root / 'card_pool_state.json')
                        pool.update(controller_complete=True, completed_at=now())
                        atomic_json(a.root / 'card_pool_state.json', pool)
                    break
                time.sleep(a.interval)
        except BaseException as exc:
            atomic_json(a.root / 'card_pool_controller.json', dict(
                pid=os.getpid(), hostname=socket.gethostname(), heartbeat_at=now(),
                status='failed', error=repr(exc)))
            raise


if __name__ == '__main__':
    main()
