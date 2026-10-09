#!/usr/bin/env python3
"""Existing full-parameter CLUE implementation plus MCQ output diagnostics."""
import os
import re
import torch
from worldsense_full_gkd_entry import install


def install_mcq_audit():
    from swift.rlhf_trainers.gkd_trainer import GKDTrainer
    from transformers import TrainerCallback
    original_setup = GKDTrainer._setup_clue_ema_teacher
    original_jsd = GKDTrainer._compute_jsd_loss
    def finite_jsd(self, *args, **kwargs):
        value = original_jsd(self, *args, **kwargs)
        if not torch.isfinite(value.detach()).item():
            raise FloatingPointError('MCQ GKD loss is not finite; stop before optimizer update')
        return value
    GKDTrainer._compute_jsd_loss = finite_jsd
    def setup(self):
        original_setup(self)
        resume = os.environ.get('MCQ_RESUME_PATH')
        if resume:
            trainer = self
            class RestoreEMA(TrainerCallback):
                def on_train_begin(self, args, state, control, **kwargs):
                    # DeepSpeed restores student/optimizer separately; ensure
                    # the resumed teacher shadow matches this same checkpoint.
                    trainer._load_clue_ema_teacher(resume)
            self.add_callback(RestoreEMA())
    GKDTrainer._setup_clue_ema_teacher = setup
    generate = GKDTrainer._generate_completions
    log = GKDTrainer.log
    def audited_generate(self, samples):
        rows = generate(self, samples)
        counts = getattr(self, '_mcq_counts', [0., 0.])
        for row in rows:
            text = str(row.messages[-1]['content'])
            bodies = re.findall(r'<answer>(.*?)</answer>', text, re.S)
            counts[0] += len(bodies) == 1 and bool(re.fullmatch('[ABCD]', bodies[0].strip()))
            counts[1] += 1
        self._mcq_counts = counts
        return rows
    def audited_log(self, logs, start_time=None):
        counts = torch.tensor([getattr(self, '_mcq_counts', [0., 0.])], device=self.accelerator.device)
        total = self.accelerator.gather(counts).sum(0).cpu().tolist()
        if total[1]: logs['mcq_answer_format_fraction'] = total[0] / total[1]
        self._mcq_counts = [0., 0.]
        return log(self, logs, start_time)
    GKDTrainer._generate_completions = audited_generate
    GKDTrainer.log = audited_log
    from worldsense_mcq_checkpointing import install_checkpoint_markers
    install_checkpoint_markers(GKDTrainer)


if __name__ == '__main__':
    torch.set_num_threads(int(os.environ.get('OMNI_FULL_CPU_THREADS', '4')))
    install()
    install_mcq_audit()
    from swift.cli.utils import try_use_single_device_mode
    from swift.pipelines import rlhf_main
    try_use_single_device_mode()
    rlhf_main()
