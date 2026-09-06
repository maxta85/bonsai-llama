"""tests/test_sft_shift.py

Godmode BLOCKER regression tests for the SFT causal shift:

Old behaviour compared logits[t] with labels[t] at the SAME position --
teaching the model to repeat its input instead of predicting the next
token. The fix scores logits[:, :-1] against labels[:, 1:]. These tests
pin that contract for BOTH dataset paths:

  * SFTDataset (training/sft.py): copy-labels aligned at same position ->
    MUST be scored through sft_loss(), which applies the shift internally.

  * PackedTextDataset (training/data.py): labels already next-token
    targets -> MUST NOT be double-shifted; plain CE / sft_loss_aligned.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.distill import sft_loss, sft_loss_aligned


class TinyLM(torch.nn.Module):
    """Single-layer causal LM producing (B, T, V) logits."""

    def __init__(self, vocab: int = 32, hidden: int = 16):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab, bias=False)

    def forward(self, ids):
        h = self.emb(ids)
        h = h + h.mean(dim=1, keepdim=True)   # cheap context mixing
        h = h * torch.tril(torch.ones(h.size(1), h.size(1))).unsqueeze(-1)[..., 0].clamp(min=0.9).unsqueeze(-1) \
            if False else h                    # keep it simple & deterministic
        return self.head(h)


def manual_shifted_ce(logits, labels):
    """Reference implementation: mean CE of logits[:-1] vs labels[1:]."""
    pred = logits[:, :-1, :]
    tgt = labels[:, 1:]
    return F.cross_entropy(pred.reshape(-1, pred.size(-1)),
                           tgt.reshape(-1), ignore_index=-100)


def test_sft_loss_equals_manual_shifted_cross_entropy():
    torch.manual_seed(0)
    model = TinyLM()
    B, T, V = 3, 12, 32
    ids = torch.randint(0, V, (B, T))
    labels = torch.randint(0, V, (B, T))     # copy-style labels (SFTDataset)
    labels[:, 0] = -100                       # mask some positions incl. edge
    labels[:, -1] = -100

    logits = model(ids)
    got = sft_loss(logits, labels)
    want = manual_shifted_ce(logits, labels)
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


def test_sft_loss_is_no_longer_position_aligned_ce():
    """The old bug returned CE(logits[t], labels[t]); ensure we now get a
    different (shifted) number -- and specifically the shifted one."""
    torch.manual_seed(3)
    model = TinyLM()
    B, T, V = 2, 10, 32
    ids = torch.randint(0, V, (B, T))
    labels = torch.randint(0, V, (B, T))
    logits = model(ids)

    old_style = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1),
                                ignore_index=-100)
    got = sft_loss(logits, labels)
    assert not torch.allclose(got, old_style, rtol=1e-5), \
        "sft_loss still computes position-aligned CE"
    torch.testing.assert_close(
        got, manual_shifted_ce(logits, labels), rtol=1e-6, atol=1e-6)


def test_ignore_index_survives_the_shift():
    torch.manual_seed(5)
    model = TinyLM()
    B, T, V = 2, 8, 32
    ids = torch.randint(0, V, (B, T))
    labels = torch.full((B, T), -100, dtype=torch.long)
    labels[:, 3:6] = torch.randint(0, V, (B, 3))   # sparse assistant span

    logits = model(ids)
    got = sft_loss(logits, labels)
    want = manual_shifted_ce(logits, labels)
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)

    # Fully-masked row contributes nothing (mean over remaining pairs).
    labels2 = torch.full_like(labels, -100)
    labels2[:, 6] = 11
    got2 = sft_loss(logits, labels2)
    want2 = manual_shifted_ce(logits, labels2)
    torch.testing.assert_close(got2, want2, rtol=1e-6, atol=1e-6)


def test_tiny_model_learns_next_token_with_new_loss():
    """Train on copy-labels with the fixed loss: loss must DECREASE toward
    next-token predictability (the old loss plateaued teaching echo)."""
    torch.manual_seed(11)
    V, T = 24, 16
    model = TinyLM(V)
    data_seq = torch.arange(0, T * 40, device=None) % V
    inputs = []
    targets = []                      # copy-labels like SFTDataset emits
    for i in range(0, len(data_seq) - T - 1, T):
        inputs.append(data_seq[i:i + T])
        targets.append(data_seq[i:i + T].clone())
    x = torch.stack(inputs[:20])
    y = torch.stack(targets[:20])

    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    first = last = None
    for epoch in range(150):
        opt.zero_grad()
        loss = sft_loss(model(x), y.clone())
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
        last = loss.item()
    assert last < first * 0.85, (first, last)


def test_double_shift_guard_for_packed_dataset_labels():
    """PackedTextDataset-style labels are already targets: the ALIGNED loss
    equals manual CE; feeding them through sft_loss would double-shift."""
    torch.manual_seed(7)
    model = TinyLM()
    B, T, V = 2, 9, 32
    ids = torch.randint(0, V, (B, T))
    packed_labels = torch.roll(ids, -1, dims=1)   # chunk[1:] semantics
    packed_labels[:, -1] = -100                   # final target masked
    logits = model(ids)

    aligned = sft_loss_aligned(logits, packed_labels)
    plain = F.cross_entropy(logits.reshape(-1, V), packed_labels.reshape(-1),
                            ignore_index=-100)
    torch.testing.assert_close(aligned, plain, rtol=0, atol=0)

    # Feeding pre-shifted labels through sft_loss scores prediction at
    # position t against packed_labels[t+1] == ids[t+2] -- TWO positions
    # ahead: a distinct, wrong number. Reference built manually: pair
    # logits[:, :-1] with ids shifted by two (last two targets masked).
    shifted_again = sft_loss(logits, packed_labels)      # WRONG for path
    two_ahead = torch.full_like(packed_labels, -100)
    two_ahead[:, :-2] = ids[:, 2:]                       # target[t] = ids[t+2]
    want_two_ahead = F.cross_entropy(
        logits[:, :-2, :].reshape(-1, V), two_ahead[:, :-2].reshape(-1),
        ignore_index=-100)
    torch.testing.assert_close(shifted_again, want_two_ahead,
                               rtol=1e-6, atol=1e-6)
    assert not torch.allclose(aligned, shifted_again, rtol=1e-5)


@pytest.mark.parametrize("ignore_index_arg", [-100, 255])
def test_label_shapes_validated(ignore_index_arg):
    logits = torch.randn(2, 6, 9)
    bad = torch.randint(0, 9, (2, 5))
    with pytest.raises(AssertionError):
        sft_loss(logits, bad)
