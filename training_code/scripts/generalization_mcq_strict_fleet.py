#!/usr/bin/env python3
"""Run format-explicit MCQ on the fleet only after open-QA has fully drained."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import traceback

REPO = Path('/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd')
sys.path.insert(0, str(REPO / 'training_code/src'))
sys.path.insert(0, str(REPO / 'training_code/scripts'))
import generalization_openqa_fleet as fleet
from score_generalization_analysis_mcq_v3 import score_rows, VERSION

WORKERS = fleet.WORKERS
PARTS = 8  # Keep original external MCQ partitions and their random seed streams.
OLD = REPO / 'training_runs/worldsense_generalization_mcq500_npu96_20261003'
PREREQUISITE = REPO / 'training_runs/generalization_openqa_npu120_20261003'
OLD_INSTRUCTION = ('Briefly analyze the video and audio evidence in English using at most 120 words. '
                   'Write your analysis inside <analysis>...</analysis>, then select exactly one option '
                   'and write only its letter (A, B, C, or D) inside <answer>...</answer>.')
INSTRUCTION = ('Briefly analyze the video and audio evidence in English using at most 120 words.\n'
               'Put ALL analysis inside <analysis>...</analysis>.\n'
               'Then select exactly ONE option.\n'
               'Between <answer> and </answer>, write ONLY ONE uppercase option letter: A, B, C, or D.\n'
               'Do NOT put analysis, explanations, option text, punctuation, or any other text '
               'inside <answer>...</answer>.\n'
               'Close the answer with </answer> and do not write any text after it.')
load, read, save, now = fleet.load, fleet.read, fleet.atomic_save, fleet.now


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    (root / 'code').mkdir(exist_ok=True)
    if (root / 'experiment_manifest.json').exists():
        validate(root)
        return
    for filename in ['tasks.json', 'inference_cache_views.json']:
        shutil.copy2(PREREQUISITE / filename, root / filename)
    manifests = {}
    for bench in ['omnivideobench', 'dailyomni']:
        source = OLD / 'data' / bench
        target = root / 'data' / bench
        rows = read(source / 'inputs.jsonl')
        changed = copy.deepcopy(rows)
        for previous, row in zip(rows, changed):
            assert len(row['messages']) == 1 and row['messages'][0]['role'] == 'user'
            text = row['messages'][0]['content']
            assert text.endswith(OLD_INSTRUCTION)
            row['messages'][0]['content'] = text[:-len(OLD_INSTRUCTION)] + INSTRUCTION
            assert {k: v for k, v in row.items() if k != 'messages'} == {k: v for k, v in previous.items() if k != 'messages'}
            assert all(0 < v['video_end'] <= 300 for v in row['videos'])
            assert row['sampling_contract']['use_audio_in_video'] is True
        fleet.jsonl(target / 'inputs.jsonl', changed)
        shutil.copy2(source / 'labels.jsonl', target / 'labels.jsonl')
        labels = read(target / 'labels.jsonl')
        assert len(changed) == len(labels) == 500
        assert {r['case_id'] for r in changed} == {r['sample_id'] for r in labels}
        manifests[bench] = dict(count=500, input_sha256=fleet.sha(target / 'inputs.jsonl'),
                               labels_sha256=fleet.sha(target / 'labels.jsonl'),
                               original_input_sha256=fleet.sha(source / 'inputs.jsonl'),
                               media_and_question_options_unchanged=True)
        save(target / 'manifest.json', manifests[bench])
    code_files = [Path(__file__), REPO / 'training_code/scripts/generalization_openqa_fleet.py',
                  REPO / 'training_code/scripts/worker_worldsense_card_pool.py',
                  REPO / 'training_code/scripts/controller_worldsense_card_pool.py',
                  REPO / 'training_code/scripts/run_worldsense_training_matched_eval.py']
    code_files += list((REPO / 'training_code/scripts').glob('score_generalization_analysis_mcq*.py'))
    for p in code_files:
        shutil.copy2(p, root / 'code' / p.name)
    save(root / 'experiment_manifest.json', dict(
        created_at=now(), prerequisite=str(PREREQUISITE), previous_mcq=str(OLD), datasets=manifests,
        tasks_sha256=fleet.sha(root / 'tasks.json'), models=13, tasks=26, responses=13000,
        instruction=INSTRUCTION, partitions=PARTS, workers=WORKERS, max_new_tokens=512,
        temperature=1., top_p=1., requested_top_k=-1, effective_top_k=50,
        repetition_penalty=1., seed=20260904, max_length=32768,
        scorer=VERSION, audio_in_video=True, original_results_preserved=True,
        code_sha256={p.name: fleet.sha(root / 'code' / p.name) for p in code_files},
        comparison_note='Same original MCQ samples, partitions, media and inference settings; only final prompt instructions change. Hardware scheduling differs; sampled answers are not guaranteed bitwise repeatable.'))
    init(root)


def validate(root):
    m = load(root / 'experiment_manifest.json')
    assert fleet.sha(root / 'tasks.json') == m['tasks_sha256']
    for bench, expected in m['datasets'].items():
        for filename, key in [('inputs.jsonl', 'input_sha256'), ('labels.jsonl', 'labels_sha256')]:
            assert fleet.sha(root / 'data' / bench / filename) == expected[key]
    for name, expected in m['code_sha256'].items():
        assert fleet.sha(root / 'code' / name) == expected


def init(root):
    if (root / 'controller_state.json').exists():
        return
    state = dict(created_at=now(), units={}, tasks={})
    mapping = load(root / 'inference_cache_views.json')
    for name in WORKERS:
        wr = root / 'worker_pools' / name
        (wr / 'inbox').mkdir(parents=True, exist_ok=True)
        save(wr / 'card_pool_state.json', dict(units={}))
    for bench in ['omnivideobench', 'dailyomni']:
        rows = read(root / 'data' / bench / 'inputs.jsonl')
        for order, task in enumerate(load(root / 'tasks.json')):
            key = bench + '/' + task['label']
            ids = []
            for i in range(PARTS):
                uid = f'generation/{key}/{i}'
                source = root / 'staging' / uid / 'input.jsonl'
                fleet.jsonl(source, rows[i::PARTS])
                state['units'][uid] = dict(kind='generation', task_key=key, partition=i,
                    source=str(source), expected_rows=len(rows[i::PARTS]), priority=order*100+i,
                    status='unassigned', model=mapping.get(task['model'], task['model']), adapter=task['adapter'])
                ids.append(uid)
            state['tasks'][key] = dict(stage='generating', benchmark=bench, model=task['label'], generation_units=ids)
            out = root / 'eval' / key
            out.mkdir(parents=True, exist_ok=True)
            save(out / 'generation_protocol.json', dict(task, actual_model=mapping.get(task['model'], task['model']),
                 partitions=PARTS, seed=20260904, temperature=1., top_p=1., effective_top_k=50,
                 max_new_tokens=512, max_length=32768, audio_in_video=True,
                 input_sha256=fleet.sha(root / 'data' / bench / 'inputs.jsonl')))
    save(root / 'controller_state.json', state)


def score_task(root, key, state):
    task = state['tasks'][key]
    out = root / 'eval' / key
    fleet.merge(root, task['generation_units'], state, out / 'results.jsonl')
    data = root / 'data' / task['benchmark']
    rows = score_rows(read(out / 'results.jsonl'), read(data / 'labels.jsonl'), read(data / 'inputs.jsonl'))
    for row in rows:
        bodies = re.findall(r'<answer>(.*?)</answer>', row['response'], re.S)
        row['strict_answer_letter'] = len(bodies) == 1 and bool(re.fullmatch('[ABCD]', bodies[0].strip()))
        row['strict_complete_format'] = bool(re.fullmatch(r'\s*<analysis>.*?</analysis>\s*<answer>\s*[ABCD]\s*</answer>\s*', row['response'], re.S)) and row['strict_answer_letter']
    fleet.jsonl(out / 'scored.jsonl', rows)
    total = len(rows)
    correct = sum(r['correct'] for r in rows)
    previous = next(r for r in load(OLD / 'rescore_explicit_formats_v3/comparison.json')['rows']
                    if r['benchmark'] == task['benchmark'] and r['model'] == task['model'])
    summary = dict(benchmark=task['benchmark'], model=task['model'], total=total, correct=correct,
        accuracy_percent=correct*100/total, unparsed=sum(not r['parsed'] for r in rows),
        strict_answer_letter=sum(r['strict_answer_letter'] for r in rows),
        strict_complete_format=sum(r['strict_complete_format'] for r in rows),
        strict_answer_accuracy_percent=sum(r['correct'] and r['strict_answer_letter'] for r in rows)*100/total,
        previous_accuracy_percent=previous['accuracy_percent'], previous_unparsed=previous['unparsed'],
        delta_previous_pp=correct*100/total-previous['accuracy_percent'], scorer=VERSION,
        result_sha256=fleet.sha(out / 'results.jsonl'))
    save(out / 'summary.json', summary)
    task.update(stage='complete', completed_at=now())


def report(root, state):
    rows = [load(root / 'eval' / key / 'summary.json') for key, t in state['tasks'].items() if t['stage'] == 'complete']
    save(root / 'comparison.json', dict(at=now(), completed_tasks=len(rows), total_tasks=26, rows=rows))
    lines = ['# 明确最终选项格式的选择题复评', '', f'已完成 {len(rows)}/26 组。每组 500 题；原模型及四项实验第 1/2/3 轮。', '',
             '主准确率使用同一 v3 解析器；无法解析计错误。另报严格答案格式，避免把理解能力与格式遵循混为一谈。', '',
             '| 数据集 | 模型 | 准确率 | 旧提示准确率 | 变化（百分点） | 无法解析 | answer 内仅字母 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['benchmark']} | {r['model']} | {r['accuracy_percent']:.1f}% | {r['previous_accuracy_percent']:.1f}% | {r['delta_previous_pp']:+.1f} | {r['unparsed']}/500 | {r['strict_answer_letter']}/500 |")
    lines += ['', '随机采样、同一组题目、同一组分片及种子；硬件调度不同，结果不保证逐位一致。旧回答保留，新提示重新生成。', '']
    temp = root / 'RESULTS.zh-CN.md.tmp'
    temp.write_text('\n'.join(lines)); temp.replace(root / 'RESULTS.zh-CN.md')
    if rows:
        with (root / 'scores.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def ready(prerequisite):
    if not (prerequisite / 'COMPLETE.json').exists():
        return False, 'waiting_for_openqa_generation_and_semantic_scoring'
    if load(prerequisite / 'comparison.json')['completed_tasks'] != 26:
        return False, 'waiting_for_all_26_scored_tasks'
    for name in WORKERS:
        wr = prerequisite / 'worker_pools' / name
        p = wr / 'card_pool_workers' / name / 'heartbeat.json'
        if not p.exists() or load(p)['status'] != 'complete':
            return False, 'waiting_for_worker_drain:' + name
        if any(u['status'] != 'complete' for u in load(wr / 'card_pool_state.json')['units'].values()):
            return False, 'waiting_for_worker_units:' + name
    return True, 'openqa_scored_and_workers_drained'


def controller(root):
    with fleet.LocalLock(root / 'controller.lease', timeout=0):
        validate(root)
        assert ready(PREREQUISITE)[0], 'Open-QA prerequisite not complete'
        while True:
            state = load(root / 'controller_state.json'); issues = []; loads = {}; workers = {}
            for name in WORKERS:
                wr = root / 'worker_pools' / name
                local = load(wr / 'card_pool_state.json')['units']
                hp = wr / 'card_pool_workers' / name / 'heartbeat.json'
                if hp.exists():
                    h = load(hp); age = time.time() - hp.stat().st_mtime
                    workers[name] = dict(pid=h['pid'], status=h['status'], age=age)
                    if age > 180 and h['status'] != 'complete': issues.append(name + ':stale-heartbeat')
                else: workers[name] = dict(status='starting')
                for uid, u in state['units'].items():
                    if u.get('worker') != name: continue
                    if uid not in local:
                        fleet.publish_unit(root, uid, u); continue
                    lu = local[uid]
                    u.update(status=lu['status'], result_path=lu['result_path'], attempt=lu.get('attempt', 0))
                    if lu['status'] == 'failed':
                        issues.append('failed:' + uid)
                        if lu.get('attempt', 0) < 3:
                            fleet.inbox(root, name, f"retry_{hashlib.sha256(uid.encode()).hexdigest()}_{lu.get('attempt', 0)}.json",
                                        dict(action='retry', key=uid, attempt=lu.get('attempt', 0)))
                loads[name] = sum(u.get('worker') == name and u['status'] in ['queued', 'pending', 'running'] for u in state['units'].values())
            for key, task in state['tasks'].items():
                if task['stage'] != 'complete' and all(state['units'][uid]['status'] == 'complete' for uid in task['generation_units']):
                    try: score_task(root, key, state)
                    except Exception:
                        issues.append('scoring:' + key)
                        with (root / 'errors.jsonl').open('a') as f: f.write(json.dumps(dict(at=now(), task=key, error=traceback.format_exc()))+'\n')
            for uid, u in sorted(state['units'].items(), key=lambda kv: (kv[1]['priority'], kv[0])):
                if u['status'] != 'unassigned': continue
                name = min(WORKERS, key=lambda n: loads[n])
                if loads[name] >= 9: break
                u.update(worker=name, status='queued', assigned_at=now()); loads[name] += 1
                save(root / 'controller_state.json', state); fleet.publish_unit(root, uid, u)
            counts = Counter(u['status'] for u in state['units'].values())
            complete = sum(t['stage'] == 'complete' for t in state['tasks'].values())
            written = sum(sum(bool(x.strip()) for x in Path(u['result_path']).open()) for u in state['units'].values()
                          if u.get('result_path') and Path(u['result_path']).exists())
            health = dict(at=now(), controller_pid=os.getpid(), completed_tasks=complete, total_tasks=26,
                          generation_rows=written, total_generation_rows=13000, active_cards=counts['running'],
                          unit_statuses=dict(counts), workers=workers, issues=issues, healthy=not issues)
            save(root / 'controller_state.json', state); save(root / 'health.json', health); report(root, state)
            print(json.dumps({k: v for k, v in health.items() if k != 'workers'}), flush=True)
            if complete == 26:
                for name in WORKERS: fleet.inbox(root, name, 'zz_stop.json', dict(action='stop'))
                save(root / 'COMPLETE.json', dict(at=now(), tasks=26)); return
            time.sleep(15)


def launch(root, action, name=None):
    label = name or action
    pidfile = root / 'logs' / (label + '.pid')
    script = root / 'code' / Path(__file__).name
    if pidfile.exists():
        proc = Path('/proc') / pidfile.read_text().strip()
        try:
            command = (proc / 'cmdline').read_bytes().decode().replace('\0', ' ')
            if str(script) in command and f'--action {action}' in command:
                print(json.dumps(dict(component=label, pid=int(pidfile.read_text()), already_running=True))); return
        except (FileNotFoundError, ProcessLookupError): pass
    cmd = [str(fleet.PYENV / 'python'), str(script), '--root', str(root), '--action', action]
    if name: cmd += ['--worker', name]
    with (root / 'logs' / (label + '.log')).open('a') as log:
        child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 start_new_session=True, cwd=REPO, env=fleet.env_for(''))
    pidfile.write_text(str(child.pid) + '\n')
    save(root / 'logs' / (label + '.launch.json'), dict(at=now(), pid=child.pid, hostname=socket.gethostname(), command=cmd))
    print(json.dumps(dict(component=label, pid=child.pid)))


def remote_launch(root, name):
    cmd = shlex.join([str(fleet.PYENV / 'python'), str(root / 'code' / Path(__file__).name),
                      '--root', str(root), '--action', 'launch-worker', '--worker', name])
    opts = ['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8']
    i = int(name.rsplit('-', 1)[1])
    if name.startswith('npu24'):
        host = ['172.16.2.181', '172.16.12.237', '172.16.12.70'][i]
        command = opts + ['-p', '2222', 'ma-user@' + host, cmd]
    else:
        host = f'ma-job-7b525feb-b862-4352-8403-9c3a018c05e8-worker-{i}.ma-job-7b525feb-b862-4352-8403-9c3a018c05e8'
        nested = shlex.join(opts + ['-p', '2222', 'ma-user@' + host, cmd])
        command = opts + ['-p', '31445', '-i', '/home/ma-user/.ssh/ulan_31445_npu96.pem',
                          'ma-user@dev-modelarts-cnnorth9.huaweicloud.com', nested]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    return dict(worker=name, at=now(), returncode=result.returncode, stdout=result.stdout[-3000:], stderr=result.stderr[-3000:])


def chain(root):
    with fleet.LocalLock(root / 'chain.lease', timeout=0):
        validate(root)
        while True:
            ok, reason = ready(PREREQUISITE)
            status = dict(at=now(), pid=os.getpid(), status='ready' if ok else 'waiting', reason=reason, prerequisite=str(PREREQUISITE))
            save(root / 'chain_status.json', status)
            print(json.dumps(status), flush=True)
            if ok: break
            time.sleep(30)
        launch(root, 'controller')
        remaining = set(WORKERS)
        while remaining:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {name: executor.submit(remote_launch, root, name) for name in sorted(remaining)}
                for name, future in futures.items():
                    try: receipt = future.result()
                    except Exception as e: receipt = dict(worker=name, at=now(), returncode=-1, error=repr(e))
                    with (root / 'logs/chain_launches.jsonl').open('a') as f: f.write(json.dumps(receipt)+'\n')
                    if receipt['returncode'] == 0: remaining.remove(name)
            save(root / 'chain_status.json', dict(at=now(), pid=os.getpid(), status='launching' if remaining else 'running', workers_pending=sorted(remaining)))
            if remaining: time.sleep(30)
        while not (root / 'COMPLETE.json').exists():
            # Restart a crashed CPU controller. Workers remain owners of their children.
            launch(root, 'controller')
            health = load(root / 'health.json') if (root / 'health.json').exists() else {}
            save(root / 'chain_status.json', dict(at=now(), pid=os.getpid(), status='running',
                 completed_tasks=health.get('completed_tasks', 0), issues=health.get('issues', [])))
            time.sleep(30)
        save(root / 'chain_status.json', dict(at=now(), pid=os.getpid(), status='complete', completed_tasks=26))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--action', choices=['prepare', 'chain', 'controller', 'worker', 'launch-chain', 'launch-controller', 'launch-worker'], required=True)
    p.add_argument('--worker', choices=WORKERS)
    a = p.parse_args(); root = a.root.resolve()
    if a.action == 'prepare': prepare(root)
    elif a.action.startswith('launch-'): launch(root, a.action[7:], a.worker)
    elif a.action == 'chain': chain(root)
    elif a.action == 'controller': controller(root)
    else:
        assert ready(PREREQUISITE)[0], 'Open-QA prerequisite not complete'
        fleet.worker(root, a.worker)


if __name__ == '__main__':
    main()
