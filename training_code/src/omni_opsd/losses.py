"""Pure-torch pieces of the Clue-OPSD objective.

Keeping the loss independent of a particular VLM makes it unit-testable on
CPU and prevents accidental changes to the multimodal data path while we
iterate on the Qwen3.5 adapter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F


@dataclass
class DistillationStats:
    loss: float
    tokens: int
    student_entropy: float
    teacher_entropy: float
    teacher_topk_mass: float
    student_topk_mass: float


def _topk_union_row(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
    support_strategy: str = "student",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if student_logits.ndim != 1 or teacher_logits.ndim != 1:
        raise ValueError("one row of logits is required")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    vocab = int(student_logits.shape[-1])
    k = min(int(top_k), vocab)
    if k < 1:
        raise ValueError("top_k must be positive")

    student_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.float() / temperature, dim=-1).detach()
    _, student_ids = torch.topk(student_logp, k=k, dim=-1)
    _, teacher_ids = torch.topk(teacher_logp, k=k, dim=-1)
    if support_strategy == "student":
        # This is the paper's stated compression: select top-K tokens from
        # the Student distribution, then gather the Teacher logits at those
        # same token IDs plus one residual tail bucket.
        support = torch.sort(student_ids).values
    elif support_strategy == "union":
        # Kept as an explicit diagnostic alternative, not the default paper
        # reproduction protocol.
        support = torch.unique(torch.cat((student_ids, teacher_ids), dim=0), sorted=True)
    else:
        raise ValueError("support_strategy must be 'student' or 'union'")

    student_values = student_logp[support].exp()
    teacher_values = teacher_logp[support].exp()
    # The residual bucket represents all vocabulary items outside the union.
    # It retains a gradient through the student mass and is detached on the
    # teacher side, matching the top-K compressed distribution used in OPSD.
    student_tail = (1.0 - student_values.sum()).clamp_min(0.0)
    teacher_tail = (1.0 - teacher_values.sum()).clamp_min(0.0)
    student_distribution = torch.cat((student_values, student_tail[None]))
    teacher_distribution = torch.cat((teacher_values, teacher_tail[None])).detach()
    return student_distribution, teacher_distribution, student_logp, teacher_logp


def generalized_jsd_topk(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    top_k: int = 100,
    beta: float = 0.5,
    temperature: float = 1.0,
    support_strategy: str = "student",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute generalized JSD over a top-K compressed vocabulary support.

    ``beta`` weights the teacher-to-mixture term; beta=.5 is the paper
    default.  The paper selects top-K tokens from the Student distribution;
    ``support_strategy='union'`` is available only for an explicit diagnostic.
    The final residual bucket makes the compressed distributions sum to one
    instead of silently renormalizing away tail probability.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student_logits.ndim == 1:
        student_logits = student_logits.unsqueeze(0)
        teacher_logits = teacher_logits.unsqueeze(0)
    if student_logits.ndim != 2 or student_logits.shape != teacher_logits.shape:
        raise ValueError("logits must be [tokens, vocab] with matching shapes")

    losses = []
    student_entropies = []
    teacher_entropies = []
    student_masses = []
    teacher_masses = []
    for student_row, teacher_row in zip(student_logits, teacher_logits):
        student, teacher, student_logp, teacher_logp = _topk_union_row(
            student_row,
            teacher_row,
            top_k=top_k,
            temperature=temperature,
            support_strategy=support_strategy,
        )
        mixture = beta * teacher + (1.0 - beta) * student
        eps = torch.finfo(mixture.dtype).eps
        student_safe = student.clamp_min(eps)
        teacher_safe = teacher.clamp_min(eps)
        mixture_safe = mixture.clamp_min(eps)
        teacher_kl = (teacher_safe * (teacher_safe.log() - mixture_safe.log())).sum()
        student_kl = (student_safe * (student_safe.log() - mixture_safe.log())).sum()
        losses.append(beta * teacher_kl + (1.0 - beta) * student_kl)
        student_entropies.append(-(student_safe * student_safe.log()).sum())
        teacher_entropies.append(-(teacher_safe * teacher_safe.log()).sum())
        student_masses.append(float(student[:-1].sum().detach()))
        teacher_masses.append(float(teacher[:-1].sum().detach()))

    loss = torch.stack(losses).mean()
    stats = {
        "loss": float(loss.detach()),
        "tokens": int(student_logits.shape[0]),
        "student_entropy": float(torch.stack(student_entropies).mean().detach()),
        "teacher_entropy": float(torch.stack(teacher_entropies).mean().detach()),
        "student_topk_mass": sum(student_masses) / len(student_masses),
        "teacher_topk_mass": sum(teacher_masses) / len(teacher_masses),
    }
    return loss, stats


def masked_generalized_jsd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    top_k: int = 100,
    beta: float = 0.5,
    temperature: float = 1.0,
    support_strategy: str = "student",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Token-mask wrapper for batched ``[batch, tokens, vocab]`` logits."""
    if student_logits.ndim == 2:
        student_logits = student_logits.unsqueeze(0)
        teacher_logits = teacher_logits.unsqueeze(0)
    if student_logits.ndim != 3 or student_logits.shape != teacher_logits.shape:
        raise ValueError("logits must be [batch, tokens, vocab] with matching shapes")
    if mask is None:
        mask = torch.ones(
            student_logits.shape[:2], dtype=torch.bool, device=student_logits.device
        )
    if mask.shape != student_logits.shape[:2]:
        raise ValueError("mask must have shape [batch, tokens]")
    selected_student = student_logits[mask]
    selected_teacher = teacher_logits[mask]
    if selected_student.shape[0] == 0:
        raise ValueError("distillation mask selects zero tokens")
    return generalized_jsd_topk(
        selected_student,
        selected_teacher,
        top_k=top_k,
        beta=beta,
        temperature=temperature,
        support_strategy=support_strategy,
    )


def ema_update_(teacher: torch.nn.Module, student: torch.nn.Module, alpha: float) -> None:
    """In-place EMA update ``teacher = (1-alpha)*teacher + alpha*student``."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("EMA alpha must be in (0, 1]")
    teacher_parameters = dict(teacher.named_parameters())
    student_parameters = dict(student.named_parameters())
    if set(teacher_parameters) != set(student_parameters):
        missing = sorted(set(teacher_parameters) ^ set(student_parameters))[:5]
        raise ValueError(f"teacher/student parameter names differ: {missing}")
    with torch.no_grad():
        for name, teacher_parameter in teacher_parameters.items():
            teacher_parameter.mul_(1.0 - alpha).add_(student_parameters[name].detach(), alpha=alpha)


def dataclass_dict(value: DistillationStats) -> dict[str, Any]:
    return asdict(value)
