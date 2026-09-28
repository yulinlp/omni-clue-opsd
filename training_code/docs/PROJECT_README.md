# Omni-OPSD

Independent research repository for **golden temporal evidence conditioned
Omni video understanding** and, after the inference evidence is established,
multi-view on-policy self-distillation (M-OPSD).

The primary training backbone is **Qwen2.5-Omni-7B**; a
**Qwen3-Omni-30B-A3B-Instruct** vLLM path is retained as a robustness track.
Light-Omni is a provenance dependency only:
its streaming baseline, Qwen2.5 runner, CG-Bench reproduction, and early
Nemotron pilots are retained under `archive/` but are not imported by the
default package.

The approved research roadmap is maintained in
[docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md). Actual run status and
claim boundaries are recorded in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md),
with compact versioned results under
[results/native_qwen3_video_odyssey_584/](results/native_qwen3_video_odyssey_584/).

## Current stage gate

Formal OPSD/LoRA training is paused until the repository has:

1. frozen train/dev/test IDs and baseline contracts;
2. a measured full-timeline A/V setting with no silent frame/audio truncation;
3. a paired OmniVideo-100K atomic study showing overall and per-task evidence
   gains against both full and matched-budget controls.

Stage 1 is now complete for Qwen2.5-Omni: the frozen full setting is continuous
audio plus complete-timeline 2 FPS video at typically 224x112. It covers the
300-second OmniVideo-Test maximum and passes a 179.7-second LoRA step. The
higher-resolution 1 FPS candidate was rejected after a measured training OOM.
Formal training remains paused for the Stage 2 A0--A5 atomic-view study.

Primary data roles are:

- OmniVideo-100K evidence-bearing MCQ: training, internal dev, and the
  pre-training atomic study;
- OmniVideo-Test: primary locked human-verified test;
- OmniVideoBench and VideoOdyssey-AV: frozen external Omni evaluation;
- Video-MME-v2: broad video/audio comparison.

CG-Bench is a legacy result source only. It is not the new main benchmark.

## Reproduce on a new server

```bash
git clone https://github.com/yulinlp/Omni-OPSD.git
cd Omni-OPSD

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[media,analysis,test]'

# Download datasets without inheriting proxy variables.
python scripts/download_omni_datasets.py \
  --root /data/omni_benchmarks --endpoint https://hf-mirror.com

# Build the video-disjoint evidence split and isolated evaluation labels.
PYTHONPATH=src python scripts/prepare_omnivideo_100k.py \
  --annotation /data/omni_benchmarks/OmniVideo-100K/train_mcq_30k.jsonl \
  --video-root /data/omni_benchmarks/OmniVideo-100K \
  --output-dir /data/omni_benchmarks/manifests/omnivideo_100k

pytest
```

Copy `.env.example` to `.env` and set machine-local paths. Model weights,
raw videos, gated benchmark annotations, caches, and raw predictions are not
stored in Git. `docs/REPRODUCIBILITY.md` records the exact server contract and
artifact verification procedure.

## Repository map

- `src/omni_opsd/`: benchmark-independent evidence, data and statistics code;
- `experiments/qwen3_vllm/`: current Qwen3-Omni vLLM inference/calibration;
- `configs/`: experiment definitions, never machine paths;
- `scripts/`: data download, manifest conversion, environment capture;
- `results/`: compact aggregate results safe to version;
- `archive/`: useful validated historical code, excluded from current API;
- `docs/`: protocol, data licensing, migration and reproducibility notes.

The original Light-Omni checkout remains untouched at its existing commit.
