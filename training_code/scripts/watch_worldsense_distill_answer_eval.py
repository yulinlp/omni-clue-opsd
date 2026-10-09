#!/usr/bin/env python3
"""Read-only monitoring and a live Markdown result table for the MCQ fleet."""
import argparse,json,os,re,time
from collections import Counter
from pathlib import Path
from datetime import datetime

def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def main(root):
    seen=set()
    while True:
        h=json.loads((root/'health.json').read_text());issues=list(h['issues']);events=[]
        for p in (root/'worker_pools').rglob('infer.log'):
            stat=p.stat();stamp=(str(p),stat.st_size,stat.st_mtime_ns)
            if stamp in seen:continue
            seen.add(stamp)
            # Include failures even after the controller schedules a retry.
            text=p.read_text(errors='replace')
            if re.search(r'Traceback|OutOfMemoryError|RuntimeError|\[ERROR\]',text):
                events.append(dict(log=str(p),tail=text[-2000:]))
        if events:
            with (root/'monitor_alerts.jsonl').open('a') as f:
                f.write(json.dumps(dict(at=datetime.now().astimezone().isoformat(),errors=events),ensure_ascii=False)+'\n')
        if time.time()-(root/'health.json').stat().st_mtime>180:issues.append('controller heartbeat stale')
        workers={n:dict(Counter(d['status'] for d in w.get('devices',{}).values())) for n,w in h['workers'].items()}
        save(root/'monitor_status.json',dict(at=datetime.now().astimezone().isoformat(),pid=os.getpid(),generation_rows=h['generation_rows'],completed_tasks=h['completed_tasks'],issues=issues,workers=workers))
        scores=json.loads((root/'comparison.json').read_text())
        lines=['# 评测实时进度','',f"更新时间：{h['at']}",'',f"已保存回答：{h['generation_rows']:,} / 44,000。已完成整组：{h['completed_tasks']} / 88。运行卡数：{h['active_cards']}。",'', '每组固定500题，全部完成后才列入下表。无法解析的回答计为错误。', '', '| 模型检查点 | 数据集 | 正确/500 | 正确率 | 无法解析 | 严格格式合规 |','|---|---|---:|---:|---:|---:|']
        for s in sorted(scores,key=lambda s:(s['arm'],s['step'],s['prompt'],s['benchmark'])):
            lines.append(f"| {s['model']} | {s['benchmark']} | {s['correct']} | {s['accuracy_percent']:.1f}% | {s['unparsed']} | {s['format_compliant']} |")
        lines+=['','## 节点状态','', '| 节点 | 本评测占用 | 其他进程占用 | 空闲 |','|---|---:|---:|---:|']
        for n,w in workers.items():lines.append(f"| {n} | {w.get('owned_unit',0)} | {w.get('busy_legacy_or_external',0)} | {w.get('idle',0)} |")
        lines+=['','当前调度告警：'+('；'.join(issues) if issues else '无。')]
        tmp=root/'PROGRESS.zh-CN.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(root/'PROGRESS.zh-CN.md')
        print(json.dumps(dict(at=h['at'],rows=h['generation_rows'],completed=h['completed_tasks'],issues=issues)),flush=True)
        if (root/'COMPLETE.json').exists():return
        time.sleep(30)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root.resolve())
