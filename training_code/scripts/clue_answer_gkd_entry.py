#!/usr/bin/env python3
"""WorldSense GKD entry with an auxiliary gold-answer loss on each rollout.

This wrapper keeps the shared ms-swift checkout untouched. The gold answer is
passed as dataset metadata; it is never inserted into the student's prompt.
"""

from __future__ import annotations

import os
import json

import torch
import torch.nn.functional as F

from swift.rlhf_trainers.gkd_trainer import GKDTrainer
from swift.trainers.trainer_factory import TrainerFactory
from swift.utils import get_logger


logger = get_logger()


def gold_letter_ce(first_logits: torch.Tensor, token_ids: list[list[int]]) -> torch.Tensor:
    """Negative log probability of the correct letter, with optional leading space."""
    if first_logits.ndim != 2 or first_logits.shape[0] != len(token_ids):
        raise ValueError("one first-response logit row is required per gold answer")
    log_probs = F.log_softmax(first_logits.float(), dim=-1)
    return torch.stack([
        -torch.logsumexp(log_probs[i, ids], dim=0)
        for i, ids in enumerate(token_ids)
    ]).mean()


class AnswerGKDTrainer(GKDTrainer):
    """GKD/JSD plus gold-letter CE evaluated at the student's first response token."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gold_ce_alpha = float(os.environ.get("WORLDSENSE_GOLD_CE_ALPHA", "0.25"))
        if self.gold_ce_alpha <= 0:
            raise ValueError("WORLDSENSE_GOLD_CE_ALPHA must be positive")
        tokenizer = self.template.tokenizer
        self.gold_token_ids = {}
        for letter in "ABCD":
            forms = [tokenizer.encode(s, add_special_tokens=False) for s in (letter, " " + letter)]
            if any(len(form) != 1 for form in forms):
                raise ValueError(f"gold option {letter} is not one token in this tokenizer: {forms}")
            self.gold_token_ids[letter] = sorted({form[0] for form in forms})
        dataset = os.environ.get("WORLDSENSE_GOLD_DATASET")
        if not dataset:
            raise ValueError("WORLDSENSE_GOLD_DATASET must identify the answer-conditioned JSONL")
        self.gold_by_prompt: dict[str, str] = {}
        with open(dataset, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                prompt = row["messages"][0]["content"]
                answer = row["gold_answer"]
                if prompt in self.gold_by_prompt or answer not in self.gold_token_ids:
                    raise ValueError(f"duplicate prompt or invalid gold answer: {row.get('case_id')}")
                self.gold_by_prompt[prompt] = answer
        self._active_gold_answers: list[str] | None = None
        logger.info(f"WorldSense answer CE enabled: alpha={self.gold_ce_alpha}, "
                    f"rows={len(self.gold_by_prompt)}, token_ids={self.gold_token_ids}")

    def _prepare_batch_inputs(self, samples):
        microbatches = super()._prepare_batch_inputs(samples)
        chunks = self.split_by_mini_batches(samples)
        if len(microbatches) != len(chunks):
            raise RuntimeError("GKD microbatch/gold-answer alignment failed")
        for batch, chunk in zip(microbatches, chunks):
            answers = []
            for sample in chunk:
                user_messages = [message["content"] for message in sample.messages if message["role"] == "user"]
                if len(user_messages) != 1:
                    raise ValueError("expected exactly one student user prompt")
                answers.append(self.gold_by_prompt.get(user_messages[0], ""))
            if any(answer not in self.gold_token_ids for answer in answers):
                raise ValueError(f"missing or invalid gold answer in training batch: {answers}")
            batch["gold_answers"] = answers
        return microbatches

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        answers = inputs.get("gold_answers")
        if not answers:
            raise ValueError("gold_answers missing from GKD microbatch")
        self._active_gold_answers = answers
        try:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        finally:
            self._active_gold_answers = None

    def _compute_jsd_loss(self, student_logits, teacher_output, labels):
        jsd = super()._compute_jsd_loss(student_logits, teacher_output, labels)
        answers = self._active_gold_answers
        if not answers:
            raise ValueError("gold answers unavailable while computing GKD loss")
        if student_logits.ndim != 3 or labels.ndim != 2 or student_logits.shape[:2] != labels.shape:
            raise ValueError(f"unexpected student logits/labels shapes: {student_logits.shape}, {labels.shape}")
        # The labels mask marks response tokens. After shifting, the first active
        # logit predicts the first generated token from the prompt alone.
        active = torch.roll(labels, shifts=-1, dims=1) != -100
        if active.shape[0] != len(answers):
            raise ValueError("gold-answer count differs from model batch")
        first_logits = []
        for row in range(active.shape[0]):
            indices = torch.nonzero(active[row], as_tuple=True)[0]
            if indices.numel() == 0:
                raise ValueError("no response token available for gold-answer supervision")
            first_logits.append(student_logits[row, indices[0]])
        ids = [self.gold_token_ids[answer] for answer in answers]
        ce = gold_letter_ce(torch.stack(first_logits), ids)
        if self.accelerator.is_main_process:
            logger.info(f"WorldSense GKD components: jsd={jsd.detach().item():.5f} "
                        f"gold_ce={ce.detach().item():.5f} alpha={self.gold_ce_alpha}")
        return jsd + self.gold_ce_alpha * ce


def main() -> None:
    TrainerFactory.TRAINER_MAPPING["gkd"] = "__main__.AnswerGKDTrainer"
    from swift.cli.utils import try_use_single_device_mode
    try_use_single_device_mode()
    from swift.pipelines import rlhf_main
    rlhf_main()


if __name__ == "__main__":
    main()
