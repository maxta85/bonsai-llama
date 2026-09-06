"""Test: streamed training must UPDATE canonical weights (Godmode BLOCKER fix).

The bridge must make streamed training produce the same canonical-weight
updates as full-load training on the same model/batches/optimizer.
"""
import copy
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))

from test_layer_stream import _build_tiny_llama


def _train_pair(seed_data=True):
    """Build (reference full-load model+opt, streamed model+bridge) over same weights."""
    from training.layer_stream import StreamingModel
    from training.stream_bridge import install_streaming_optimizer_bridge

    torch.manual_seed(42)
    base = _build_tiny_llama()
    base.eval()

    ref = copy.deepcopy(base)
    for p in ref.parameters():
        p.requires_grad_(True)
    ref_opt = torch.optim.AdamW(ref.parameters(), lr=1e-3)

    sm = StreamingModel(base, compute_device="cpu")
    # unfreeze canonical params: bridge owns them now
    for p in sm.inner.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(sm.inner.parameters(), lr=1e-3)
    bridge = install_streaming_optimizer_bridge(sm, opt)
    return ref, ref_opt, sm, bridge


def test_streamed_training_updates_canonical_weights():
    ref, ref_opt, sm, bridge = _train_pair()
    torch.manual_seed(7)
    ids = torch.randint(0, ref.config.vocab_size, (2, 8))

    before = torch.cat([p.detach().flatten() for p in sm.inner.parameters()])

    # one streamed step: forward + backward (populates grad_fields) + bridge.step
    out = sm.forward_with_streaming(ids)
    out.logits.sum().backward()
    assert sm.streamed_grads(), "no streamed grads produced"
    updated = bridge.step()
    assert updated, "bridge.step() reported no update"

    after = torch.cat([p.detach().flatten() for p in sm.inner.parameters()])
    assert not torch.equal(before, after), "canonical weights unchanged after streamed step"


def test_streamed_matches_fullload_two_steps():
    ref, ref_opt, sm, bridge = _train_pair()
    torch.manual_seed(7)
    ids = torch.randint(0, ref.config.vocab_size, (2, 8))

    ref_losses, sm_losses = [], []
    for _ in range(2):
        # reference full-load (eval mode — deterministic, matches streamed path)
        ref.eval()
        ro = ref(input_ids=ids).logits
        ref_opt.zero_grad()
        ro.sum().backward()
        torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0)
        ref_opt.step()
        ref_losses.append(ro.sum().item())

        # streamed
        out = sm.forward_with_streaming(ids)
        out.logits.sum().backward()
        bridge.step()
        sm_losses.append(out.logits.sum().item())

    # identical starting weights + identical math => losses should match closely
    assert abs(ref_losses[0] - sm_losses[0]) < 1e-3, (ref_losses[0], sm_losses[0])

    # after 2 AdamW updates on matched gradients, weights must match within tol
    max_diff = max(
        (rp - sp).abs().max().item()
        for rp, sp in zip(ref.parameters(), sm.inner.parameters())
    )
    assert max_diff < 1e-4, f"canonical weights diverged after 2 steps: {max_diff:.2e}"


def test_snapshot_refresh_after_step():
    """Next streamed forward must use UPDATED weights (copy-plan invalidated)."""
    ref, ref_opt, sm, bridge = _train_pair()
    torch.manual_seed(7)
    ids = torch.randint(0, ref.config.vocab_size, (2, 8))

    out1 = sm.forward_with_streaming(ids)
    l1 = out1.logits.sum().item()
    out1.logits.sum().backward()
    bridge.step()

    out2 = sm.forward_with_streaming(ids)
    l2 = out2.logits.sum().item()
    # If snapshot refresh failed, second forward replays OLD weights => same loss
    assert abs(l1 - l2) > 1e-6, "second forward identical -> snapshots not refreshed"
