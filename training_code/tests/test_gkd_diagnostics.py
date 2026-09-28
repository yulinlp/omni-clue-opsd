"""Small CPU tests for the paper-aligned OPD diagnostics."""

import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "ms-swift"))

from swift.rlhf_trainers.gkd_diagnostics import (  # noqa: E402
    aggregate_distribution_diagnostics,
    completion_segment_ids,
    compute_answer_distribution,
    compute_distribution_diagnostics,
)


def test_same_distribution_has_full_overlap_and_zero_entropy_gap():
    logits = torch.tensor([[4.0, 2.0, 1.0, -1.0]])
    values = compute_distribution_diagnostics(logits, logits.clone(), top_k=2)

    assert values["overlap_ratio"].item() == pytest.approx(1.0)
    assert values["empty_overlap_rate"].item() == pytest.approx(0.0)
    assert values["student_overlap_mass"].item() == pytest.approx(
        torch.softmax(logits, dim=-1)[0, :2].sum().item()
    )
    assert values["teacher_overlap_mass"].item() == pytest.approx(
        torch.softmax(logits, dim=-1)[0, :2].sum().item()
    )
    assert values["overlap_adv_paper"].item() == pytest.approx(0.0, abs=1e-7)
    assert values["entropy_gap_abs"].item() == pytest.approx(0.0, abs=1e-7)


def test_partial_and_empty_intersections_use_full_vocab_mass():
    student = torch.tensor([[5.0, 4.0, 0.0, -1.0], [5.0, 4.0, 0.0, -1.0]])
    teacher = torch.tensor([[0.0, 5.0, 4.0, -1.0], [-1.0, 0.0, 4.0, 5.0]])
    values = compute_distribution_diagnostics(student, teacher, top_k=2)

    assert values["overlap_ratio"].tolist() == pytest.approx([0.5, 0.0])
    assert values["empty_overlap_rate"].tolist() == pytest.approx([0.0, 1.0])
    assert values["overlap_nonempty"].tolist() == [True, False]
    # The first row shares token 1; the mass is taken from the complete
    # softmax, not a softmax renormalized over the top-k support.
    expected_student_mass = torch.softmax(student[0], dim=-1)[1].item()
    expected_teacher_mass = torch.softmax(teacher[0], dim=-1)[1].item()
    assert values["student_overlap_mass"][0].item() == pytest.approx(expected_student_mass)
    assert values["teacher_overlap_mass"][0].item() == pytest.approx(expected_teacher_mass)
    assert values["student_overlap_mass"][1].item() == pytest.approx(0.0)
    assert values["teacher_overlap_mass"][1].item() == pytest.approx(0.0)
    assert values["overlap_adv_paper"][1].item() == pytest.approx(0.0)
    assert (values["overlap_adv_paper"] <= 1e-7).all()

    aggregated = aggregate_distribution_diagnostics(values)
    assert aggregated["all/valid_count"] == (2.0, 1.0, True)
    assert aggregated["all/overlap_effective_count"] == (1.0, 1.0, True)
    # Overlap KL is averaged only over the non-empty row.
    assert aggregated["all/overlap_kl"][1] == 1.0


def test_diagnostics_accept_active_positions_from_different_prompt_lengths():
    # The caller masks prompt positions before invoking this function.  The
    # resulting active frames have equal completion length despite different
    # original prompt lengths.
    student = torch.tensor([[3.0, 1.0, 0.0], [1.0, 2.0, 0.0]])
    teacher = torch.tensor([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0]])
    values = compute_distribution_diagnostics(student, teacher, top_k=2, chunk_size=1)
    assert values["overlap_ratio"].shape == (2,)
    assert torch.isfinite(values["student_entropy"]).all()
    assert torch.isfinite(values["teacher_entropy"]).all()


def test_diagnostics_broadcasts_overlap_mass_across_multiple_positions():
    # A real training microbatch has many active completion positions.  The
    # per-position overlap mass must broadcast over K, rather than aligning
    # the K dimension with the batch dimension.
    torch.manual_seed(19)
    student = torch.randn(7, 64)
    teacher = torch.randn(7, 64)
    values = compute_distribution_diagnostics(student, teacher, top_k=20)
    assert values["overlap_ratio"].shape == (7,)
    assert torch.isfinite(values["overlap_kl"]).all()


def test_diagnostics_are_detached_and_do_not_change_student_gradient():
    baseline_logits = torch.tensor([[2.0, 1.0, -1.0]], requires_grad=True)
    baseline_logits.sum().backward()
    expected_grad = baseline_logits.grad.detach().clone()

    student = torch.tensor([[2.0, 1.0, -1.0]], requires_grad=True)
    teacher = torch.tensor([[1.0, 2.0, -1.0]])
    values = compute_distribution_diagnostics(student, teacher, top_k=2)
    assert all(not value.requires_grad for value in values.values())
    student.sum().backward()
    assert torch.equal(student.grad, expected_grad)


class _MarkerTokenizer:
    _ids = {'<think>': [10], '</think>': [11], '<answer>': [12], '</answer>': [13]}
    _pieces = {10: '<think>', 11: '</think>', 12: '<answer>', 13: '</answer>', 21: 'reason', 22: 'A'}

    def encode(self, text, add_special_tokens=False):
        return self._ids.get(text, [])

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(self._pieces.get(int(token_id), '') for token_id in ids)


def test_completion_segments_ignore_padding_and_mark_answer_region():
    labels = torch.tensor([[-100, 10, 21, 11, 12, 22, 13, -100]])
    segments = completion_segment_ids(labels, _MarkerTokenizer())
    assert segments.tolist() == [[0, 0, 1, 0, 0, 2, 0, 0]]


class _ContextualTokenizer:
    def encode(self, text, add_special_tokens=False):
        if text == "<answer>":
            return [10, 11]
        letters = {"A": 20, "B": 21, "C": 22, "D": 23}
        if text.startswith("<answer>") and text[-1] in letters:
            return [10, 11, letters[text[-1]]]
        return []

    def decode(self, ids, skip_special_tokens=False):
        return {20: ">A", 21: ">B", 22: ">C", 23: ">D"}.get(ids[0], "")


def test_answer_diagnostics_uses_contextual_option_tokens_and_four_way_normalization():
    tokenizer = _ContextualTokenizer()
    student = torch.zeros(1, 5, 30)
    teacher = torch.zeros(1, 5, 30)
    # The answer letter token is at input position 3, so causal logits at 2
    # are the answer slot.  Both distributions agree on A vs. the other
    # options, but the teacher assigns it more probability.
    student[0, 2, [20, 21, 22, 23]] = torch.tensor([2.0, 1.0, 0.0, -1.0])
    teacher[0, 2, [20, 21, 22, 23]] = torch.tensor([4.0, 1.0, 0.0, -1.0])

    def find_slot(row, tok, logits_len):
        ids = {"A": 20, "B": 21, "C": 22, "D": 23}
        if 20 not in row:
            return None
        return 2, ids

    stats = compute_answer_distribution(
        student,
        teacher,
        torch.tensor([[1, 10, 11, 20, 30]]),
        torch.tensor([[1, 10, 11, 20, 30]]),
        [{"gold_answer": "A"}],
        tokenizer=tokenizer,
        find_answer_slot=find_slot,
    )
    assert stats["answer_valid_count"] == 1
    assert stats["answer_total_count"] == 1
    assert stats["answer_unavailable_count"] == 0
    assert stats["answer_gt_probability_gap_sum"] > 0
    assert stats["student_answer_accuracy_sum"] == 1
    assert stats["teacher_answer_accuracy_sum"] == 1
