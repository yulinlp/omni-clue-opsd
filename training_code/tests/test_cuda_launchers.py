"""Exercise CUDA wrapper and queue without training or GPUs."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def test_cuda_queue_preserves_devices_and_skips_failed_smoke(tmp_path):
    scripts = tmp_path/'scripts'
    scripts.mkdir()
    shutil.copyfile(SCRIPTS/'run_training_arm_a100.sh', scripts/'run_training_arm_a100.sh')
    (scripts/'run_training_arm_a100.sh').chmod(0o755)
    (scripts/'run_video_odyssey_training_arm.sh').write_text('''printf '%s %s %s %s %s\\n' "$OMNI_OPSD_ARM" "$OMNI_OPSD_NPROC_PER_NODE" "$ASCEND_RT_VISIBLE_DEVICES" "$OMNI_OPSD_GKD_SAFE_MODE" "$OMNI_OPSD_MAX_GRAD_NORM" >> "$TEST_RECORD"
if [ "$OMNI_OPSD_ARM" = opsd ]; then exit 9; fi
''')
    root = tmp_path/'swift'
    (root/'swift/rl_core').mkdir(parents=True)
    (root/'swift/rl_core/data.py').touch()
    dataset = tmp_path/'dataset'
    dataset.mkdir()
    for arm in ['sft','opsd']:
        (dataset/f'omnivideo_100k_train.{arm}.jsonl').write_text('{}\n')
    bindir = tmp_path/'bin'
    bindir.mkdir()
    (bindir/'sleep').write_text('#!/bin/sh\n/bin/sleep 0.01\n')
    (bindir/'sleep').chmod(0o755)
    env = {k:v for k,v in os.environ.items() if not k.startswith('OMNI_OPSD_')}
    env.update(OMNI_OPSD_PROJECT_ROOT=str(tmp_path), OMNI_OPSD_QUEUE_ROOT=str(tmp_path/'queue'), OMNI_OPSD_MATRIX_DATASET_ROOT=str(dataset), OMNI_OPSD_MODEL=str(tmp_path), OMNI_OPSD_MS_SWIFT_ROOT=str(root), OMNI_OPSD_PYTHON_DEPS=str(tmp_path), OMNI_OPSD_PYTHON_BIN=sys.executable, OMNI_OPSD_SWIFT_BIN=sys.executable, OMNI_OPSD_QUEUE_ARMS='sft,opsd', CUDA_VISIBLE_DEVICES='2,5', ASCEND_RT_VISIBLE_DEVICES='0,1,2,3', TEST_RECORD=str(tmp_path/'records'), PATH=str(bindir)+':'+os.environ['PATH'])
    result = subprocess.run(['bash', str(SCRIPTS/'run_omnivideo_baseline_queue.sh')], env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 1, result.stderr
    records = (tmp_path/'records').read_text().splitlines()
    assert records == ['sft 2 2,5 0 1.0','opsd 2 2,5 0 1.0','sft 2 2,5 0 1.0']
    assert (tmp_path/'queue/full/opsd/SKIPPED').is_file()
    assert (tmp_path/'queue/full/sft/SUCCESS').is_file()
