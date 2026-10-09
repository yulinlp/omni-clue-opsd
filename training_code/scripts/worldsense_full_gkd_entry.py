#!/usr/bin/env python3
"""Run-local full-parameter EMA and exact full-vocabulary GKD for Ascend.

ZeRO-2 leaves BF16 parameters replicated. EMA and swap buffers live on CPU;
optimizer states may be sharded/offloaded independently. No shared Swift edits.
"""
import os
import re
import json
from contextlib import contextmanager

import torch
from torch.utils.checkpoint import checkpoint


def install():
    from swift.rlhf_trainers.gkd_trainer import GKDTrainer, CLUEEMATeacherCallback, logger
    from swift.rlhf_trainers.gkd_loss import extract_active, jsd_loss
    original_generate = GKDTrainer._generate_completions
    original_log = GKDTrainer.log
    original_compute = GKDTrainer.compute_loss

    def compute(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        from swift.rlhf_trainers.gkd_loss import DataSource
        if inputs['gkd_batch'].data_source != DataSource.STUDENT:
            raise RuntimeError('This run requires student-generated sequences exclusively.')
        if not getattr(self, '_worldsense_input_audited', False):
            audit = {'rank': self.accelerator.process_index, 'source': 'student'}
            for view, key in [('student', 'model_inputs'), ('teacher', 'teacher_model_inputs')]:
                fields = inputs[key]
                features = fields.get('input_features')
                if features is None or features.numel() == 0:
                    raise RuntimeError(f'{view} audio input_features are missing.')
                audit[view] = {k: list(v.shape) for k, v in fields.items()
                               if torch.is_tensor(v)}
            logger.info('MULTIMODAL_INPUT_AUDIT ' + json.dumps(audit))
            self._worldsense_input_audited = True
        return original_compute(self, model, inputs, return_outputs, num_items_in_batch)

    def generate(self, samples):
        generated = original_generate(self, samples)
        audit = getattr(self, '_worldsense_rollout_audit', [0.0] * 5)
        for sample in generated:
            ids = sample.response_token_ids
            length = sum(len(part) for part in ids) if ids and isinstance(ids[0], list) else len(ids)
            text = sample.messages[-1]['content']
            match = re.search(r'<analysis>(.*?)</analysis>', str(text), re.S)
            audit[0] += length
            audit[1] += 1
            audit[2] += sample.finish_reason == 'length'
            audit[3] += bool(match)
            audit[4] += len(match.group(1).split()) if match else 0
        self._worldsense_rollout_audit = audit
        return generated

    def log(self, logs, start_time=None):
        audit = getattr(self, '_worldsense_rollout_audit', [0.0] * 5)
        values = torch.tensor([audit], dtype=torch.float32, device=self.accelerator.device)
        values = self.accelerator.gather(values).sum(0).cpu().tolist()
        if values[1]:
            logs.update(rollout_tokens_mean=values[0] / values[1],
                        rollout_cap_fraction=values[2] / values[1],
                        analysis_format_fraction=values[3] / values[1],
                        analysis_words_mean=values[4] / values[1])
        self._worldsense_rollout_audit = [0.0] * 5
        return original_log(self, logs, start_time)

    def parameters(self, model=None):
        model = self.accelerator.unwrap_model(model or self.model)
        result = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if not result or any('lora_' in n for n in result):
            raise ValueError('This entry requires a full-parameter student, without LoRA.')
        return result

    def setup(self):
        self._use_clue_ema_teacher = True
        self._clue_ema_alpha = self.args.clue_ema_alpha
        self._clue_ema_last_step = 0
        if self._clue_ema_alpha is None:
            raise ValueError('Full EMA requires clue_ema_alpha.')
        if self.use_teacher_api or not self._is_self_distillation or self._teacher_use_disable_adapter:
            raise ValueError('Full EMA requires local self-distillation.')
        if self.is_fsdp_enabled:
            raise ValueError('This full EMA implementation requires replicated parameters.')
        if self.is_deepspeed_enabled:
            cfg = self.accelerator.state.deepspeed_plugin.deepspeed_config
            if cfg['zero_optimization']['stage'] > 2:
                raise ValueError('Full CPU EMA supports ZeRO <= 2 only.')
        params = parameters(self)
        self._clue_ema_shadow = {}
        self._clue_ema_swap = {}
        with torch.no_grad():
            for name, param in params.items():
                self._clue_ema_shadow[name] = param.detach().to(device='cpu', dtype=torch.float32, copy=True)
                self._clue_ema_swap[name] = torch.empty(param.shape, dtype=param.dtype, device='cpu')
        self.add_callback(CLUEEMATeacherCallback(self))
        count = sum(p.numel() for p in params.values())
        logger.info(f'FULL_EMA_CPU initialized: alpha={self._clue_ema_alpha}, '
                    f'tensors={len(params)}, trainable_parameters={count}; FULL_VOCAB_JSD.')

    @contextmanager
    def teacher_context(self, model):
        unwrapped = self.accelerator.unwrap_model(model)
        params = parameters(self, unwrapped)
        if params.keys() != self._clue_ema_shadow.keys():
            raise RuntimeError('Full EMA parameter set changed.')
        was_training = unwrapped.training
        swapped = []
        try:
            with torch.no_grad():
                for name, param in params.items():
                    self._clue_ema_swap[name].copy_(param.detach())
                    swapped.append(name)
                    param.copy_(self._clue_ema_shadow[name])
            unwrapped.eval()
            yield
        finally:
            with torch.no_grad():
                for name in swapped:
                    params[name].copy_(self._clue_ema_swap[name])
            unwrapped.train(was_training)

    def update(self, global_step):
        if global_step <= self._clue_ema_last_step:
            return False
        with torch.no_grad():
            for name, param in parameters(self).items():
                value = param.detach().to(device='cpu', dtype=torch.float32)
                self._clue_ema_shadow[name].lerp_(value, self._clue_ema_alpha)
        self._clue_ema_last_step = global_step
        return True

    def full_jsd(self, student_logits, teacher_output, labels):
        teacher_output.labels = torch.roll(teacher_output.labels, -1, 1)
        shifted = torch.roll(labels, -1, 1)
        student, teacher, count = extract_active(student_logits, teacher_output, shifted)
        if int(count) == 0:
            return student_logits.sum() * 0
        target = teacher.full_logits.detach()
        if student.shape != target.shape:
            raise ValueError('Student/teacher active logits must have the same shape.')
        def chunk_loss(s, t):
            return jsd_loss(s.float() / self.temperature, t.float() / self.temperature,
                            self.beta, chunk_size=32)
        total = student.new_zeros((), dtype=torch.float32)
        for start in range(0, len(student), 32):
            s, t = student[start:start + 32], target[start:start + 32]
            total = total + (checkpoint(chunk_loss, s, t, use_reentrant=False)
                             if s.requires_grad else chunk_loss(s, t))
        self._metrics['train']['distill_response_tokens'].append(float(count))
        return total / count

    GKDTrainer._trainable_lora_parameters = parameters
    GKDTrainer._setup_clue_ema_teacher = setup
    GKDTrainer._clue_ema_teacher_context = teacher_context
    GKDTrainer._update_clue_ema_teacher = update
    GKDTrainer._compute_jsd_loss = full_jsd
    GKDTrainer._generate_completions = generate
    GKDTrainer.log = log
    GKDTrainer.compute_loss = compute


if __name__ == '__main__':
    torch.set_num_threads(int(os.environ.get('OMNI_FULL_CPU_THREADS', '4')))
    install()
    from swift.cli.utils import try_use_single_device_mode
    from swift.pipelines import rlhf_main
    try_use_single_device_mode()
    rlhf_main()
