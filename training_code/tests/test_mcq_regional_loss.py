import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import DefaultDeviceType

torch.set_num_threads(2)
DefaultDeviceType.set_device_type('cpu')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from worldsense_mcq_sft_entry import regional_loss, response_regions


class Characters:
    def decode(self, ids, **kwargs): return ''.join(chr(i) for i in ids)


def make(text):
    ids = [ord(c) for c in text]
    labels = torch.tensor([[-100, -100] + ids])
    logits = torch.zeros(1, len(ids)+2, 128, requires_grad=True)
    return logits, labels, ids


@pytest.mark.parametrize('observation', ['Short.', 'Much longer evidence. ' * 30])
def test_total_answer_region_gradient_is_three_times_observation(observation):
    text = '<analysis>' + observation + '</analysis>\n<answer>C</answer>EOS'
    logits, labels, ids = make(text)
    loss, _ = regional_loss(logits, labels, Characters(), 3.)
    loss.backward()
    shifted = F.pad(labels, (0,1), value=-100)[:,1:]
    indices = (shifted[0] != -100).nonzero(as_tuple=True)[0]
    answer, _, _ = response_regions(Characters(), ids)
    target_grad = logits.grad[0,indices,torch.tensor(ids)].abs()
    a = sum(target_grad[i] for i, yes in enumerate(answer) if yes)
    o = sum(target_grad[i] for i, yes in enumerate(answer) if not yes)
    assert torch.allclose(a/o, torch.tensor(3.), atol=1e-5)
    assert logits.grad[0,0].abs().sum() == 0  # ignored prompt position


def test_answer_only_matches_regular_ce():
    logits, labels, _ = make('<answer>B</answer>EOS')
    torch.manual_seed(19)
    logits = torch.randn_like(logits, requires_grad=True)
    loss, _ = regional_loss(logits, labels, Characters(), 3.)
    shifted = F.pad(labels, (0,1), value=-100)[:,1:]
    expected = F.cross_entropy(logits.reshape(-1,128), shifted.reshape(-1), ignore_index=-100)
    assert torch.allclose(loss,expected)


@pytest.mark.parametrize('body', ['B because it is blue', 'A/B', 'Blue', ''])
def test_malformed_training_answer_fails(body):
    logits, labels, _ = make('<answer>' + body + '</answer>')
    with pytest.raises(ValueError): regional_loss(logits, labels, Characters(), 3.)
