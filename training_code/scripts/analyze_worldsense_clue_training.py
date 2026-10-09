#!/usr/bin/env python3
"""Plot recorded CLUE loss; reconstruct source branches from the trainer seed rule.

The source-specific curves remain TOTAL loss. The original run did not record
separate JSD/CE values, rollout correctness, completion lengths or truncation rate.
"""
import csv
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / 'training_runs/worldsense_openqa_20260929'
RUN = RUN_ROOT / 'outputs/clue_formal/v0-20260929-174734'
OUT = RUN_ROOT / 'analysis_clue_20260930'


def moving(values, window):
    return np.array([np.mean(values[max(0, i-window+1):i+1]) for i in range(len(values))])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    args = json.loads((RUN / 'args.json').read_text())
    records = [json.loads(line) for line in (RUN / 'logging.jsonl').open() if line.strip()]
    metrics = [r for r in records if 'loss' in r and 'global_step/max_steps' in r]
    history = json.loads((RUN / 'checkpoint-135/trainer_state.json').read_text())['log_history']
    history = {r['step']: r for r in history if 'loss' in r}
    rows = []
    for r in metrics:
        step = int(r['global_step/max_steps'].split('/')[0])
        # _get_random_num uses state.global_step before this optimizer update.
        draw = random.Random(args['seed'] + step - 1).random()
        source = 'student_rollout' if draw <= args['lmbda'] else 'gold_dataset'
        assert abs(r['loss'] - history[step]['loss']) < 1e-6
        rows.append(dict(step=step, epoch=r['epoch'], loss=r['loss'],
                         grad_norm=r['grad_norm'], learning_rate=r['learning_rate'],
                         source_reconstructed=source, branch_draw=draw))
    assert [r['step'] for r in rows] == list(range(1, 136))
    with (OUT / 'clue_step_metrics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    summary = {'source_log': str(RUN / 'logging.jsonl'), 'steps': len(rows),
               'branch_method': 'Random(seed + logged_step - 1), draw <= lmbda => student',
               'branch_is_reconstructed_not_logged': True,
               'components_not_logged': ['jsd_loss', 'sft_ce_loss', 'rollout_correctness',
                                         'completion_token_lengths', 'generation_cap_hit_rate'],
               'finite_logged_loss_and_gradients': all(math.isfinite(r[k]) for r in rows for k in ('loss', 'grad_norm')),
               'max_grad_norm_logged': max(r['grad_norm'] for r in rows),
               'steps_preclip_grad_norm_gt_1': sum(r['grad_norm'] > 1 for r in rows),
               'epochs': []}
    for epoch in (1, 2, 3):
        subset = [r for r in rows if (epoch-1)*45 < r['step'] <= epoch*45]
        item = {'epoch': epoch, 'mean_total_loss': float(np.mean([r['loss'] for r in subset])),
                'mean_grad_norm': float(np.mean([r['grad_norm'] for r in subset]))}
        means = []
        for source in ('student_rollout', 'gold_dataset'):
            group = [r for r in subset if r['source_reconstructed'] == source]
            mean = float(np.mean([r['loss'] for r in group])); means.append(mean)
            item[source] = {'steps': len(group), 'mean_total_loss': mean}
        item['equal_branch_weight_mean'] = float(np.mean(means))
        summary['epochs'].append(item)
    summary['branch_counts'] = {source: sum(r['source_reconstructed'] == source for r in rows)
                                for source in ('student_rollout', 'gold_dataset')}

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'student_rollout': '#2475b0', 'gold_dataset': '#d46c20'}
    names = {'student_rollout': 'Student rollout: JSD only',
             'gold_dataset': 'Gold dataset: JSD + 0.25 x CE'}
    x = np.array([r['step'] for r in rows]); y = np.array([r['loss'] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(x, y, color='#999999', alpha=.35, lw=.8)
    for source in colors:
        group = [r for r in rows if r['source_reconstructed'] == source]
        ax.scatter([r['step'] for r in group], [r['loss'] for r in group], s=18,
                   color=colors[source], label=names[source], zorder=3)
    ax.plot(x, moving(y, 9), color='#222222', lw=1.7, label='9-step trailing mean (mixed branches)')
    ax.set(title='A. Recorded total loss: branch switching causes jumps', ylabel='Logged total loss')
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    for source in colors:
        group = [r for r in rows if r['source_reconstructed'] == source]
        sx = [r['step'] for r in group]; sy = [r['loss'] for r in group]
        ax.scatter(sx, sy, s=15, alpha=.3, color=colors[source])
        ax.plot(sx, moving(sy, 7), color=colors[source], lw=2, label=names[source])
    ax.set(title='B. Source-specific total loss (7 observations trailing mean)', ylabel='Loss; NOT isolated CE/JSD components')
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    grads = np.array([r['grad_norm'] for r in rows])
    ax.plot(x, grads, color='#6e54a3', lw=1, alpha=.75)
    ax.axhline(1, color='#555555', ls='--', label='Gradient clipping threshold = 1')
    ax.set(title='C. Reported gradient norm before clipping', ylabel='Gradient norm')
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.plot(x, [r['learning_rate']*1e6 for r in rows], color='#267e54', lw=2)
    ax.set(title='D. Learning rate: peak 2e-6, cosine decay to zero', ylabel='Learning rate (x 1e-6)')
    for ax in axes.flat:
        for boundary in (45.5, 90.5): ax.axvline(boundary, color='#aaaaaa', ls=':', lw=1)
        ax.set(xlabel='Optimizer step', xlim=(1, 135)); ax.grid(alpha=.15)
    fig.suptitle('WorldSense open-QA CLUE-OPSD: 135 updates / 3 epochs\nBranch labels reconstructed from trainer code; no loss components or rollouts were logged', fontsize=14)
    fig.savefig(OUT / 'clue_loss_curve.png', dpi=160)
    fig.savefig(OUT / 'clue_loss_curve.pdf')
    plt.close(fig)

    report = json.loads((RUN_ROOT / 'eval_worldsense_tier_abd518/comparison_all/training_eval_summary.json').read_text())
    acc = [report['arms'][f'clue_epoch{i}']['accuracy']*100 for i in (1, 2, 3)]
    summary['worldsense518_accuracy_percent'] = {'base': report['arms']['base']['accuracy']*100,
                                                'clue_epoch1': acc[0], 'clue_epoch2': acc[1], 'clue_epoch3': acc[2]}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for source in colors:
        axes[0].plot([1, 2, 3], [e[source]['mean_total_loss'] for e in summary['epochs']],
                     marker='o', color=colors[source], label=names[source])
    axes[0].plot([1, 2, 3], [e['mean_total_loss'] for e in summary['epochs']],
                 marker='o', color='#333333', ls='--', label='Mixed total loss')
    axes[0].set(title='Training objective improves', ylabel='Epoch mean total loss'); axes[0].legend(fontsize=9)
    axes[1].plot([1, 2, 3], acc, marker='o', color='#2475b0', label='CLUE checkpoint')
    axes[1].axhline(report['arms']['base']['accuracy']*100, ls='--', color='#555555', label='Base model')
    axes[1].set(title='WorldSense 518 accuracy does not improve', ylabel='MCQ accuracy (%)', ylim=(44.5, 46.5))
    axes[1].legend(fontsize=9)
    for ax in axes: ax.set(xlabel='Epoch', xticks=[1, 2, 3]); ax.grid(alpha=.2)
    fig.savefig(OUT / 'clue_loss_vs_accuracy.png', dpi=160)
    plt.close(fig)
    (OUT / 'analysis_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
