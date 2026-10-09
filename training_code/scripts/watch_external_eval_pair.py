#!/usr/bin/env python3
"""Persistent, conservative supervision of the open-QA -> MCQ evaluation pair."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
import traceback

REPO = Path('/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd')
sys.path.insert(0, str(REPO / 'training_code/scripts'))
import generalization_openqa_fleet as fleet

OPEN = REPO / 'training_runs/generalization_openqa_npu120_20261003'
MCQ = REPO / 'training_runs/generalization_mcq_strict_answer_npu120_20261003'
TRAIN = REPO / 'training_runs/worldsense_mcq_full_3arms_20261004'
MON = REPO / 'training_runs/external_eval_pair_monitor_20261004'
SCRIPT = Path(__file__).resolve()


def cmdline(pid):
    try:
        return [x.decode() for x in (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0') if x]
    except (OSError, UnicodeError):
        return []


def matching_processes(script, root, action=None, worker=None):
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        args = cmdline(p.name)
        if str(script) not in args or str(root) not in args: continue
        if action and not ('--action' in args and args[args.index('--action') + 1] == action): continue
        if worker and not ('--worker' in args and args[args.index('--worker') + 1] == worker): continue
        found.append(int(p.name))
    return found


def spawn(root, script, action=None, worker=None, reporter=False):
    label = 'reporter' if reporter else (worker or action)
    args = [str(fleet.PYENV / 'python'), str(script), '--root', str(root)]
    if reporter: args += ['--watch']
    else: args += ['--action', action]
    if worker: args += ['--worker', worker]
    with (root / 'logs' / (label + '.log')).open('a') as log:
        child = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 start_new_session=True, cwd=REPO, env=fleet.env_for(''))
    (root / 'logs' / (label + '.pid')).write_text(str(child.pid) + '\n')
    receipt = dict(at=fleet.now(), hostname=socket.gethostname(), action='restarted', component=label,
                   pid=child.pid, command=args)
    fleet.atomic_save(root / 'logs' / (label + '.watchdog_restart.json'), receipt)
    return receipt


def ensure_local(root, component, worker=None):
    # This entry point is invoked on the component's actual owner host.
    if (root/'USER_STOP_REQUEST.json').exists():
        return dict(action='stopped_by_user',component=component)
    if root == TRAIN:
        assert component == 'controller'
        import worldsense_mcq_training_suite as suite
        if (root/'COMPLETE.json').exists():return dict(action='already_complete', component='training_suite')
        return suite.launch_controller(root)
    mcq = root == MCQ
    script = root / 'code' / ('generalization_mcq_strict_fleet.py' if mcq else 'generalization_openqa_fleet.py')
    if component == 'reporter': script = root / 'code/report_generalization_openqa.py'
    existing = matching_processes(script, root, None if component == 'reporter' else component, worker)
    if existing: return dict(action='already_alive', component=worker or component, pids=existing)
    if component in ('controller', 'chain') and (root / 'COMPLETE.json').exists():
        return dict(action='already_complete', component=component)
    if mcq and component != 'chain':
        import generalization_mcq_strict_fleet as strict
        if not strict.ready(OPEN)[0]: return dict(action='prerequisite_pending', component=worker or component)
    if component == 'worker':
        wr = root / 'worker_pools' / worker
        hostname = re.sub(r'[^A-Za-z0-9_.-]', '_', socket.gethostname())
        # A live worker holds this local lease. Never edit its state concurrently.
        with fleet.LocalLock(wr / 'card_pool_worker_leases' / (hostname + '.lock.d'), timeout=0):
            state = fleet.load(wr / 'card_pool_state.json')
            live_children = []
            for p in Path('/proc').iterdir():
                if not p.name.isdigit(): continue
                args = cmdline(p.name)
                if '--result_path' in args and str(wr / 'units') in args[args.index('--result_path') + 1]:
                    live_children.append(int(p.name))
            if live_children:
                return dict(action='waiting_for_orphan_children', worker=worker, pids=live_children)
            recovered = []
            with fleet.LocalLock(wr / 'card_pool.lock.d'):
                state = fleet.load(wr / 'card_pool_state.json')
                for uid, unit in state['units'].items():
                    if unit['status'] != 'running': continue
                    try:
                        result = fleet.read(Path(unit['result_path']))
                        source = fleet.read(Path(unit['folder']) / 'input.jsonl')
                        assert len(result) == unit['expected_rows']
                        fleet.join(result, source)
                        unit.update(status='complete', exit_code=None, finished_at=fleet.now(),
                                    allocation_stage='recovered-validated-output',
                                    recovery_reason='Owner and inference processes exited; complete output IDs independently validated; original exit status unavailable.')
                    except Exception as error:
                        unit.update(status='failed', exit_code=None, finished_at=fleet.now(),
                                    allocation_stage='recovered-incomplete-output',
                                    failure_reason='Owner exited and no inference processes remain: ' + repr(error))
                    recovered.append(dict(unit=uid, status=unit['status']))
                fleet.atomic_save(wr / 'card_pool_state.json', state)
            hp = wr / 'card_pool_workers' / worker / 'heartbeat.json'
            if hp.exists() and fleet.load(hp).get('status') == 'complete' and state.get('stop_workers'):
                return dict(action='already_complete', component=worker)
        result = spawn(root, script, 'worker', worker)
        result['recovered_units'] = recovered
        return result
    if component == 'reporter':
        p = root / 'report/summary.json'
        if p.exists() and fleet.load(p)['completed_tasks'] == 26:
            return dict(action='already_complete', component='reporter')
    return spawn(root, script, component, reporter=component == 'reporter')


def remote(root, component, node, worker=None):
    args = [str(fleet.PYENV / 'python'), str(SCRIPT), '--action', 'ensure', '--root', str(root), '--component', component]
    if worker: args += ['--worker', worker]
    command = shlex.join(args)
    opts = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8']
    number = int(node.rsplit('-', 1)[1])
    if node.startswith('npu24'):
        address = ['172.16.2.181', '172.16.12.237', '172.16.12.70'][number]
        ssh = opts + ['-p', '2222', 'ma-user@' + address, command]
    else:
        hostname = f'ma-job-7b525feb-b862-4352-8403-9c3a018c05e8-worker-{number}.ma-job-7b525feb-b862-4352-8403-9c3a018c05e8'
        nested = shlex.join(opts + ['-p', '2222', 'ma-user@' + hostname, command])
        ssh = opts + ['-p', '31445', '-i', '/home/ma-user/.ssh/ulan_31445_npu96.pem',
                      'ma-user@dev-modelarts-cnnorth9.huaweicloud.com', nested]
    p = subprocess.run(ssh, capture_output=True, text=True, timeout=60)
    if p.returncode: raise RuntimeError(f'{node}: {p.stderr[-3000:]} {p.stdout[-1000:]}')
    return dict(node=node, component=worker or component, output=p.stdout[-6000:])


def event(kind, **kwargs):
    record = dict(at=fleet.now(), kind=kind, **kwargs)
    with (MON / 'events.jsonl').open('a') as f: f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps(record, ensure_ascii=False), flush=True)


def age(path):
    return time.time() - path.stat().st_mtime if path.exists() else float('inf')


def unit_idle_seconds(unit):
    """A newly claimed unit has made progress even before its log is created."""
    timestamps = []
    for field in ['started_at', 'claimed_at']:
        if unit.get(field):
            timestamps.append(datetime.fromisoformat(unit[field]).timestamp())
    if unit.get('claimed_epoch'):
        timestamps.append(float(unit['claimed_epoch']))
    for p in [Path(unit['result_path']), Path(unit['folder']) / 'infer.log']:
        try: timestamps.append(p.stat().st_mtime)
        except FileNotFoundError: pass
    # Without a timestamp there is no evidence that 30 minutes have elapsed.
    return max(0., time.time() - max(timestamps)) if timestamps else None


def inspect(root):
    hp = root / 'health.json'
    if not hp.exists(): return dict(status='not_started'), []
    health = fleet.load(hp)
    summary = {k: v for k, v in health.items() if k != 'workers'}
    summary['health_age_seconds'] = round(age(hp), 1)
    problems = list(health.get('issues', []))
    if age(hp) > 180 and not (root / 'COMPLETE.json').exists(): problems.append('controller_health_stale')
    for p in (root / 'worker_pools').glob('*/card_pool_state.json'):
        for uid, unit in fleet.load(p)['units'].items():
            if unit['status'] == 'failed': problems.append('failed_unit:' + uid + ':attempt=' + str(unit.get('attempt')))
            if unit['status'] == 'running':
                idle = unit_idle_seconds(unit)
                if idle is not None and idle > 1800: problems.append('no_output_or_log_progress_30min:' + uid)
    return summary, sorted(set(problems))


def watch():
    MON.mkdir(parents=True, exist_ok=True)
    with fleet.LocalLock(MON / 'watcher.lease', timeout=0):
        seen = set(); last_repair = {}; cycle = 0
        while True:
            try:
                summaries = {}; issues = {}; repairs = []
                open_done = (OPEN / 'COMPLETE.json').exists()
                mcq_done = (MCQ / 'COMPLETE.json').exists()
                monitored = [OPEN, MCQ] + ([TRAIN] if (TRAIN/'suite.json').exists() else [])
                for root in monitored:
                    summaries[root.name], issues[root.name] = inspect(root)
                    if root == TRAIN:
                        for arm, state in summaries[root.name].get('arms', {}).items():
                            if state['status']=='failed':issues[root.name].append('training_arm_failed:'+arm)
                    for problem in issues[root.name]:
                        key = (root.name, problem)
                        if key not in seen: event('alert', run=root.name, problem=problem); seen.add(key)
                if not open_done and age(OPEN / 'health.json') > 180:
                    repairs.append((OPEN, 'controller', 'npu24-worker-2', None))
                if cycle % 10 == 0:
                    repairs.append((OPEN, 'reporter', 'npu24-worker-2', None))
                if not mcq_done and age(MCQ / 'chain_status.json') > 180:
                    repairs.append((MCQ, 'chain', 'npu24-worker-0', None))
                if (MCQ / 'logs/controller.pid').exists() and not mcq_done and age(MCQ / 'health.json') > 180:
                    repairs.append((MCQ, 'controller', 'npu24-worker-0', None))
                if (TRAIN/'controller.pid').exists() and not (TRAIN/'COMPLETE.json').exists() and age(TRAIN/'health.json')>180:
                    repairs.append((TRAIN,'controller','npu24-worker-0',None))
                for root in [OPEN, MCQ]:
                    # MCQ queues exist in advance; do not start them prematurely.
                    if root == MCQ and not (root / 'logs/controller.pid').exists(): continue
                    for name in fleet.WORKERS:
                        hp = root / 'worker_pools' / name / 'card_pool_workers' / name / 'heartbeat.json'
                        if hp.exists() and fleet.load(hp).get('status') == 'complete': continue
                        if age(hp) > 180: repairs.append((root, 'worker', name, name))
                pending = []
                for spec in repairs:
                    key = tuple(map(str, spec))
                    if time.time() - last_repair.get(key, 0) < 120: continue
                    last_repair[key] = time.time(); pending.append(spec)
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [(spec, executor.submit(remote, *spec)) for spec in pending]
                    for spec, future in futures:
                        try: event('component_check_or_recovery', run=spec[0].name, result=future.result())
                        except Exception: event('recovery_error', run=spec[0].name, error=traceback.format_exc())
                all_drained = all((root / 'worker_pools' / name / 'card_pool_workers' / name / 'heartbeat.json').exists()
                    and fleet.load(root / 'worker_pools' / name / 'card_pool_workers' / name / 'heartbeat.json')['status'] == 'complete'
                    for root in [OPEN, MCQ] for name in fleet.WORKERS)
                training_done = TRAIN not in monitored or (TRAIN/'COMPLETE.json').exists()
                status = dict(at=fleet.now(), pid=os.getpid(), poll_seconds=30,
                              status='complete' if open_done and mcq_done and all_drained and training_done else 'monitoring',
                              runs=summaries, issues=issues)
                fleet.atomic_save(MON / 'status.json', status)
                with (MON / 'history.jsonl').open('a') as f: f.write(json.dumps(status, ensure_ascii=False) + '\n')
                if status['status'] == 'complete': event('evaluations_and_training_complete'); return
                cycle += 1
            except Exception: event('watch_cycle_error', error=traceback.format_exc())
            time.sleep(30)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--action', choices=['watch', 'ensure', 'launch', 'restart'], required=True)
    p.add_argument('--root', type=Path, choices=[OPEN, MCQ, TRAIN]); p.add_argument('--component', choices=['controller', 'chain', 'worker', 'reporter'])
    p.add_argument('--worker', choices=fleet.WORKERS); a = p.parse_args()
    if a.action == 'watch': watch()
    elif a.action == 'ensure': print(json.dumps(ensure_local(a.root, a.component, a.worker)))
    else:
        MON.mkdir(parents=True, exist_ok=True)
        pidfile = MON / 'watcher.pid'
        if pidfile.exists():
            args = cmdline(pidfile.read_text().strip())
            if str(SCRIPT) in args and '--action' in args and args[args.index('--action') + 1] == 'watch':
                if a.action!='restart':
                    print(json.dumps(dict(already_running=True, pid=int(pidfile.read_text())))); return
                os.kill(int(pidfile.read_text()), 15)
                time.sleep(1)
        with (MON / 'watcher.log').open('a') as log:
            child = subprocess.Popen([str(fleet.PYENV / 'python'), str(SCRIPT), '--action', 'watch'],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, cwd=REPO)
        pidfile.write_text(str(child.pid) + '\n'); print(json.dumps(dict(pid=child.pid, monitor=str(MON))))


if __name__ == '__main__': main()
