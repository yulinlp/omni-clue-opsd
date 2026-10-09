#!/usr/bin/env python3
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((ROOT/'training_runs/worldsense_clue_repair_ablation_20261007/evaluation/BATCH1_ALL_RESULTS.csv').open()))
fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharex=True)
for ax,bench,base in zip(axes,['omnivideobench','dailyomni'],[36,57.6]):
 for arm in ['clue_reference','clue_clean_ce','clue_token_jsd','clue_p2_only','clue_p3_only']:
  rs=sorted([r for r in rows if r['arm']==arm and r['benchmark']==bench],key=lambda r:int(r['step']))
  ax.plot([int(r['step']) for r in rs],[float(r['accuracy_percent']) for r in rs],marker='o',markersize=3,label=arm.removeprefix('clue_'))
 ax.axhline(base,color='black',linestyle='--',linewidth=1,label='Historical base (sampling caveat)')
 ax.set(title=bench+' (500 questions)',xlabel='Training update',ylabel='Accuracy (%)');ax.grid(alpha=.25);ax.set_xticks([1,3,5,7,10,15,20])
axes[1].legend(fontsize=8,loc='lower left');fig.suptitle('Batch 1: saved-checkpoint MCQ evaluation, 2026-10-09');fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(ROOT/'handoff'/('BATCH1_ACCURACY.'+ext),dpi=180)
