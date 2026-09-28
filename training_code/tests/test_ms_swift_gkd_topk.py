"""CPU checks for the paper-style OPSD top-K + tail-mass path."""

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ms-swift"))

from swift.rlhf_trainers.gkd_loss import TeacherOutput, gkd_loss
from swift.rlhf_trainers.gkd_diagnostics import compute_distribution_diagnostics


def test_topk_selects_student_support_and_keeps_both_tail_masses():
    student_logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    teacher_logits = torch.tensor([[1.0, 0.0, 4.0, 3.0]])

    compressed = TeacherOutput(full_logits=teacher_logits).to_topk(
        2, student_logits=student_logits, temperature=1.0
    )

    # The support is selected by the student (tokens 0 and 1), while the
    # teacher contributes probabilities at those same IDs.
    assert compressed.topk_indices.tolist() == [[0, 1]]
    expected_teacher = torch.log_softmax(teacher_logits, dim=-1)[..., :2]
    torch.testing.assert_close(compressed.topk_logprobs, expected_teacher)

    student_distribution = torch.cat(
        [compressed.student_topk_logprobs, compressed.student_tail_logprob[:, None]], dim=-1
    ).exp()
    teacher_distribution = torch.cat(
        [compressed.topk_logprobs, compressed.teacher_tail_logprob[:, None]], dim=-1
    ).exp()
    torch.testing.assert_close(student_distribution.sum(dim=-1), torch.ones(1))
    torch.testing.assert_close(teacher_distribution.sum(dim=-1), torch.ones(1))

    # Teacher top-K mass is gathered at the student's IDs; it is not the
    # teacher's own top-K support (tokens 2 and 3 in this example).
    expected_teacher_tail = 1.0 - torch.softmax(teacher_logits, dim=-1)[..., :2].sum(dim=-1)
    torch.testing.assert_close(teacher_distribution[..., -1], expected_teacher_tail)


def test_gkd_loss_masks_before_topk_when_teacher_prompt_is_longer():
    torch.manual_seed(7)
    student_logits = torch.randn(1, 3, 6, requires_grad=True)
    teacher_logits = torch.randn(1, 5, 6)
    student_labels = torch.tensor([[-100, 2, -100]])
    teacher_labels = torch.tensor([[-100, -100, -100, 2, -100]])
    teacher_output = TeacherOutput(full_logits=teacher_logits, labels=teacher_labels)

    loss, num_valid = gkd_loss(
        student_logits,
        teacher_output,
        student_labels,
        beta=0.5,
        temperature=1.0,
        topk=100,
    )

    assert num_valid.item() == 1
    assert torch.isfinite(loss)
    loss.backward()
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()


def test_read_only_diagnostics_leave_gkd_loss_and_gradient_unchanged():
    torch.manual_seed(11)
    student_a = torch.randn(1, 4, 7, requires_grad=True)
    student_b = student_a.detach().clone().requires_grad_(True)
    teacher = torch.randn(1, 4, 7)
    labels = torch.tensor([[-100, 2, 3, -100]])
    teacher_output_a = TeacherOutput(full_logits=teacher, labels=labels)
    teacher_output_b = TeacherOutput(full_logits=teacher, labels=labels)

    loss_a, _ = gkd_loss(student_a, teacher_output_a, labels, beta=0.5, temperature=1.0, topk=3)
    loss_a.backward()
    _ = compute_distribution_diagnostics(student_b[:, 1:3].reshape(-1, 7),
                                         teacher[:, 1:3].reshape(-1, 7), top_k=2)
    loss_b, _ = gkd_loss(student_b, teacher_output_b, labels, beta=0.5, temperature=1.0, topk=3)
    loss_b.backward()

    torch.testing.assert_close(loss_a, loss_b)
    torch.testing.assert_close(student_a.grad, student_b.grad)
