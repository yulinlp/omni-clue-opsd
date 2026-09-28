#!/usr/bin/env python3
"""Read-only CUDA training preflight. A pass is necessary, not a training smoke test."""
from __future__ import annotations
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


ARMS = ('sft', 'opsd', 'clue_opsd', 'grpo')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ms-swift-root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--matrix-dir', type=Path, required=True)
    parser.add_argument(
        '--arm', choices=ARMS, action='append', dest='arms',
        help='check only this training arm; repeat for multiple arms (default: all arms)')
    parser.add_argument(
        '--allow-non-ema-clue', action='store_true',
        help='allow clue_opsd preflight without the unavailable LoRA-shadow EMA implementation')
    parser.add_argument('--static-only', action='store_true', help='skip dependency imports and CUDA checks')
    args = parser.parse_args()
    arms = tuple(dict.fromkeys(args.arms or ARMS))
    errors = []
    required = {
        'swift/infer_engine/protocol.py': ['videos: List[Any]'],
        'swift/template/templates/qwen.py': ['process_audio_info'],
    }
    if any(arm in ('opsd', 'clue_opsd') for arm in arms):
        required['swift/rl_core/data.py'] = ['teacher_videos']
    if 'clue_opsd' in arms and not args.allow_non_ema_clue:
        required['swift/rlhf_trainers/gkd_trainer.py'] = ['clue_ema_alpha']
        required['swift/arguments/rlhf_args.py'] = ['clue_ema_alpha']
    for name, tokens in required.items():
        path = args.ms_swift_root / name
        text = path.read_text() if path.is_file() else ''
        for token in tokens:
            if token not in text:
                errors.append(f'{path}: missing {token}')
    try:
        commit = subprocess.check_output(['git', '-C', str(args.ms_swift_root), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        if commit != '960c5bf2cb070d1e3483ed93965f2e338d3ae93a':
            print(
                f'INFO: ms-swift commit {commit} differs from the unavailable '
                'reference commit 960c5bf2cb070d1e3483ed93965f2e338d3ae93a',
                file=sys.stderr)
    except (OSError, subprocess.CalledProcessError):
        errors.append('cannot identify ms-swift git commit')
    index = args.model / 'model.safetensors.index.json'
    if not (args.model / 'config.json').is_file() or not index.is_file():
        errors.append('missing model config or weight index')
    else:
        for name in set(json.loads(index.read_text())['weight_map'].values()):
            if not (args.model / name).is_file():
                errors.append(f'missing model shard: {name}')
    reference = None
    for arm in arms:
        path = args.matrix_dir / f'omnivideo_100k_train.{arm}.jsonl'
        if not path.is_file():
            errors.append(f'missing {path}')
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        ids = [r['case_id'] for r in rows]
        if len(ids) != 5000 or len(set(ids)) != 5000:
            errors.append(f'{arm}: expected 5000 unique rows')
        inputs = [(r['case_id'], r['messages'][0], r['videos']) for r in rows]
        if reference is not None and reference != inputs:
            errors.append(f'{arm}: student input/order mismatch')
        reference = inputs
        for row in rows:
            for spec in row.get('videos', []) + row.get('teacher_videos', []):
                media_path = spec.get('video') if isinstance(spec, dict) else spec
                if not media_path or not Path(media_path).is_file():
                    errors.append(f'missing video: {media_path}')
    if not args.static_only:
        sys.path.insert(0, str(args.ms_swift_root.resolve()))
        for name in ['torch', 'torchvision', 'transformers', 'trl', 'peft', 'accelerate', 'swift', 'qwen_omni_utils', 'msgspec', 'av']:
            try:
                module = importlib.import_module(name)
                print(f'{name}: {getattr(module, "__version__", "import OK")}')
                if name == 'torch' and not module.cuda.is_available():
                    errors.append('CUDA unavailable; run this check inside the allocated GPU job')
                if name == 'qwen_omni_utils':
                    if importlib.metadata.version('qwen-omni-utils') != '0.0.9':
                        errors.append('qwen-omni-utils must be 0.0.9')
                    vision = Path(module.__file__).parent / 'v2_5/vision_process.py'
                    text = vision.read_text()
                    if 'source_total_frames' not in text or 'os.environ.get("MAX_NUM_WORKERS_FETCH_VIDEO"' not in text:
                        errors.append('qwen-omni-utils training/edge patches missing')
            except Exception as exc:
                errors.append(f'{name}: {exc}')
    for error in sorted(set(errors)):
        print(f'FAIL: {error}', file=sys.stderr)
    checked = ','.join(arms)
    print('NOT READY' if errors else (
        f'STATIC CHECKS PASS for {checked} (runtime not checked)' if args.static_only else
        f'PREFLIGHT PASS for {checked}; next run one-step smoke tests'))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
