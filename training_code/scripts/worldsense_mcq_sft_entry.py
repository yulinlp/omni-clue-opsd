#!/usr/bin/env python3
"""Full SFT with a per-example, region-normalized MCQ answer objective."""
import json
import os
import re

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def response_regions(tokenizer, target_ids):
    """Map the literal answer block to original label tokens, never prompt tokens."""
    text = tokenizer.decode(target_ids, skip_special_tokens=False)
    matches = list(re.finditer(r'<answer>\s*([ABCD])\s*</answer>', text))
    if len(matches) != 1 or len(re.findall(r'<answer>', text)) != 1:
        raise ValueError('Expected exactly one well-formed gold option: ' + repr(text[:800]))
    answer = matches[0]
    # Use the actual label stream, including im_end. Prefix decoding avoids
    # assumptions about BPE boundaries at a preceding newline / XML tag.
    ends = [len(tokenizer.decode(target_ids[:i+1], skip_special_tokens=False)) for i in range(len(target_ids))]
    answer_mask = [end > answer.start() for end in ends]
    option_mask = [end > answer.start(1) and (ends[i-1] if i else 0) < answer.end(1)
                   for i, end in enumerate(ends)]
    if not any(answer_mask) or not any(option_mask): raise ValueError('Answer token alignment failed')
    has_analysis = '<analysis>' in text
    if has_analysis:
        if not re.fullmatch(r'<analysis>.*?</analysis>\s*<answer>\s*[ABCD]\s*</answer>.*', text, re.S):
            raise ValueError('Malformed observation target')
    elif not text.startswith('<answer>'):
        raise ValueError('Answer-only target contains unexpected preceding text')
    return answer_mask, option_mask, has_analysis


def regional_loss(logits, labels, tokenizer, answer_weight):
    """(mean observation CE + w * mean answer-region CE) / (1+w).

    XML delimiters and EOS belong to their adjacent region. Answer-only rows
    use mean CE over the entire answer response. Prompt labels are ignored.
    """
    shifted = F.pad(labels, (0, 1), value=-100)[:, 1:]
    losses = []; metrics = []
    for logit, target in zip(logits, shifted):
        positions = (target != -100).nonzero(as_tuple=True)[0]
        ids = target.index_select(0, positions).detach().cpu().tolist()
        answer_mask, option_mask, has_analysis = response_regions(tokenizer, ids)
        active = logit.index_select(0, positions)
        y = target.index_select(0, positions)
        def ce(x, t): return F.cross_entropy(x.float(), t, reduction='none')
        pieces = []
        for start in range(0, len(ids), 32):
            x, t = active[start:start+32], y[start:start+32]
            pieces.append(checkpoint(ce, x, t, use_reentrant=False) if x.requires_grad else ce(x, t))
        token_ce = torch.cat(pieces)
        answer_positions = torch.tensor([i for i, yes in enumerate(answer_mask) if yes], device=logit.device)
        option_positions = torch.tensor([i for i, yes in enumerate(option_mask) if yes], device=logit.device)
        answer = token_ce.index_select(0, answer_positions).mean()
        option = token_ce.index_select(0, option_positions).mean()
        if has_analysis:
            obs_positions = torch.tensor([i for i, yes in enumerate(answer_mask) if not yes], device=logit.device)
            observation = token_ce.index_select(0, obs_positions).mean()
            loss = (observation + answer_weight * answer) / (1 + answer_weight)
        else:
            observation = answer.detach() * 0
            loss = answer
        losses.append(loss)
        metrics.append(dict(answer_ce=answer.detach(), option_ce=option.detach(), observation_ce=observation.detach(),
                            answer_tokens=len(answer_positions), observation_tokens=len(ids)-len(answer_positions)))
    return torch.stack(losses).mean(), metrics


def install():
    from swift.loss import BaseLoss, loss_map
    from swift.trainers.seq2seq_trainer import Seq2SeqTrainer
    from swift.utils import get_logger
    logger = get_logger()
    weight = float(os.environ.get('MCQ_ANSWER_WEIGHT', '3'))

    class MCQRegionalLoss(BaseLoss):
        def __call__(self, outputs, labels, *, num_items_in_batch=None, **kwargs):
            loss, metrics = regional_loss(outputs.logits, labels, self.trainer.template.tokenizer, weight)
            if not torch.isfinite(loss.detach()).item():
                raise FloatingPointError('MCQ SFT loss is not finite; stop before optimizer update')
            for m in metrics:
                for key, value in m.items():
                    self.trainer.custom_metrics['train']['mcq_' + key].update(value)
            # HF does not divide a custom compute_loss_func by accumulation.
            # Normalize by actual microbatches, including a short final group.
            divisor = getattr(self.trainer, 'current_gradient_accumulation_steps', 1) if self.trainer.model.training else 1
            return loss / divisor

    loss_map['mcq_region_weighted'] = MCQRegionalLoss
    original = Seq2SeqTrainer.compute_loss

    def audited(self, model, inputs, *args, **kwargs):
        if not getattr(self, '_mcq_audited', False):
            audio = inputs.get('input_features')
            if audio is None or audio.numel() == 0: raise RuntimeError('SFT audio features are absent')
            params = {n: p for n, p in model.named_parameters() if p.requires_grad}
            if not params or any('lora_' in n for n in params): raise RuntimeError('Expected full-parameter training')
            groups = {'audio': 0, 'vision': 0, 'other_language_and_alignment': 0}
            for name, p in params.items():
                group = 'audio' if 'audio_tower' in name else ('vision' if 'visual' in name else 'other_language_and_alignment')
                groups[group] += p.numel()
            if not groups['audio'] or not groups['vision']: raise RuntimeError('Audio or vision encoder is frozen')
            logger.info('MCQ_SFT_AUDIT ' + json.dumps(dict(rank=self.accelerator.process_index,
                input_shapes={k: list(v.shape) for k,v in inputs.items() if torch.is_tensor(v)},
                trainable_parameters=groups, answer_weight=weight,
                loss='region_means_then_weighted_average', accumulation_normalization='actual_microbatch_count')))
            self._mcq_audited = True
        return original(self, model, inputs, *args, **kwargs)
    Seq2SeqTrainer.compute_loss = audited
    from worldsense_mcq_checkpointing import install_checkpoint_markers
    install_checkpoint_markers(Seq2SeqTrainer)


if __name__ == '__main__':
    torch.set_num_threads(int(os.environ.get('OMNI_FULL_CPU_THREADS', '4')))
    install()
    from swift.pipelines import sft_main
    sft_main()
