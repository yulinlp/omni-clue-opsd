"""Training-ready pieces for a GT-conditioned multi-view OPSD teacher.

The model backend owns multimodal preprocessing and prefix scoring.  This
module only combines the resulting logits, so the teacher design can be
unit-tested without loading a checkpoint or inventing a visual-token rule.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch


@dataclass
class PrefixTeacherTarget:
    """A confidence-gated target for one student's on-policy prefix."""

    logits: torch.Tensor
    token_mask: torch.Tensor
    token_confidence: torch.Tensor
    token_agreement: torch.Tensor
    view_weights: torch.Tensor
    stats: dict[str, Any]


def _as_batch(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    if logits.ndim != 3:
        raise ValueError("prefix logits must have shape [tokens, vocab] or [batch, tokens, vocab]")
    return logits


def _view_confidence(logits: torch.Tensor) -> torch.Tensor:
    """Return per-token confidence without looking at the answer label."""
    probabilities = torch.softmax(logits.float(), dim=-1)
    top_values = torch.topk(probabilities, k=min(2, probabilities.shape[-1]), dim=-1).values
    top1 = top_values[..., 0]
    margin = top_values[..., 0] - top_values[..., 1] if top_values.shape[-1] > 1 else top1
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    entropy = entropy / max(1.0, math.log(probabilities.shape[-1]))
    return (0.5 * top1 + 0.3 * (1.0 - entropy) + 0.2 * margin).clamp(0.0, 1.0)


def aggregate_prefix_logits(
    view_logits: Sequence[torch.Tensor],
    *,
    view_weights: Sequence[float] | None = None,
    min_confidence: float = 0.15,
    min_agreement: float = 0.5,
    temperature: float = 1.0,
) -> PrefixTeacherTarget:
    """Confidence-weight complementary GT views in probability space.

    All tensors must score the same student-generated token IDs.  The method
    intentionally never receives or derives the ground-truth answer.  A view
    can therefore contribute useful soft targets even when its hard answer is
    not selected, while low-agreement tokens are excluded from distillation.
    """
    if not view_logits:
        raise ValueError("at least one teacher view is required")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    tensors = [_as_batch(item) for item in view_logits]
    shape = tensors[0].shape
    if any(item.shape != shape for item in tensors[1:]):
        raise ValueError("all teacher views must have identical prefix-logit shapes")
    count = len(tensors)
    if view_weights is None:
        base_weights = torch.ones(count, dtype=torch.float32, device=tensors[0].device)
    else:
        if len(view_weights) != count:
            raise ValueError("view_weights must match view_logits")
        base_weights = torch.tensor(view_weights, dtype=torch.float32, device=tensors[0].device)
    if bool((base_weights < 0).any()) or float(base_weights.sum()) <= 0:
        raise ValueError("view_weights must be non-negative with positive sum")
    base_weights = base_weights / base_weights.sum()

    log_probabilities = torch.stack([
        torch.log_softmax(item.float() / temperature, dim=-1) for item in tensors
    ], dim=0)
    weighted_log_probabilities = log_probabilities + base_weights.view(count, 1, 1, 1).clamp_min(1e-12).log()
    aggregate_log_probs = torch.logsumexp(weighted_log_probabilities, dim=0)
    aggregate_logits = aggregate_log_probs * temperature

    view_top = log_probabilities.argmax(dim=-1)
    aggregate_top = aggregate_log_probs.argmax(dim=-1)
    agreement = (view_top == aggregate_top.unsqueeze(0)).float().mean(dim=0)
    confidence = _view_confidence(aggregate_logits)
    token_mask = (agreement >= float(min_agreement)) & (confidence >= float(min_confidence))

    entropy = -(aggregate_log_probs.exp() * aggregate_log_probs).sum(dim=-1)
    stats = {
        "views": count,
        "tokens": int(shape[1]),
        "vocab": int(shape[2]),
        "masked_tokens": int(token_mask.sum().item()),
        "mask_fraction": float(token_mask.float().mean().item()),
        "mean_confidence": float(confidence.mean().item()),
        "mean_agreement": float(agreement.mean().item()),
        "mean_entropy": float(entropy.mean().item()),
        "min_confidence": float(min_confidence),
        "min_agreement": float(min_agreement),
        "temperature": float(temperature),
    }
    return PrefixTeacherTarget(
        logits=aggregate_logits,
        token_mask=token_mask,
        token_confidence=confidence,
        token_agreement=agreement,
        view_weights=base_weights,
        stats=stats,
    )


def gem_generalized_jsd(
    student_logits: torch.Tensor,
    teacher_target: PrefixTeacherTarget,
    *,
    top_k: int = 100,
    beta: float = 0.5,
    temperature: float = 1.0,
    support_strategy: str = "student",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply the existing top-K generalized JSD on the gated target."""
    from omni_opsd.losses import masked_generalized_jsd

    loss, stats = masked_generalized_jsd(
        _as_batch(student_logits),
        teacher_target.logits,
        teacher_target.token_mask,
        top_k=top_k,
        beta=beta,
        temperature=temperature,
        support_strategy=support_strategy,
    )
    stats = {**stats, "teacher_mask_fraction": teacher_target.stats["mask_fraction"]}
    return loss, stats
