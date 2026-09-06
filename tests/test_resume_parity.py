"""tests/test_resume_parity.py

End-to-end resume parity: a tiny LLaMA trained with a QLoRA-style LoRA
configuration on deterministic data. An uninterrupted run must produce
bit-near-identical final adapter weights AND identical per-step losses vs
run interrupted mid-way (save -> simulated session death -> fresh process
-> load -> continue).

Runs on CPU with no network access; reuses ``_build_tiny_llama`` from
tests/test_layer_stream.py.

NOTE ON LOSS SHAPING: batches here use PackedTextDataset-style semantics --
labels are pre-shifted next-token targets -- so the aligned loss
(sft_loss_aligned / plain CE) is the correct choice and deliberately avoids
training.distill.sft_loss()'s internal shift.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_layer_stream import _build_tiny_llama   # noqa: E402

TOTAL_STEPS = 6
SAVE_AFTER_STEP = 2           # uninterrupted: 6 steps; resumed: 2 + 4
TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# Test doubles & harness
# ---------------------------------------------------------------------------

class DeterministicData:
    """Pre-generated micro-batch groups (identical across processes).

    ``micro_groups[g]`` holds exactly ``micro_per_step`` tensors consumed
    before optimizer step g completes -- the unit resumption slices on.
    """

    def __init__(self, vocab_size: int, total_micro: int,
                 micro_per_step: int = 2, seq_len: int = 16):
        g = torch.Generator().manual_seed(20260906)
        self.micro_groups = [
            [torch.randint(0, vocab_size, (2, seq_len), generator=g)
             for _ in range(micro_per_step)]
            for _ in range(total_micro)
        ]


def build_lora_config(model, r: int = 8):
    """QLoRA-style LoRA injection on attention projections (q,v like QLoRA's
    typical setup): W + alpha/r * B(A(x)); A kaiming-init, B zero-init."""
    lora_params: list[torch.nn.Parameter] = []

    class LoraLinear(torch.nn.Module):
        def __init__(self, base: torch.nn.Linear):
            super().__init__()
            self.base = base
            hidden = base.in_features
            out = base.out_features
            g = torch.Generator().manual_seed(1234 + abs(hash(hidden)) % 97)
            scale_alpha = 32.0
            self.lora_A = torch.nn.Parameter(
                torch.randn(r, hidden, generator=g) *
                (1.0 / hidden ** 0.5))
            self.lora_B = torch.nn.Parameter(torch.zeros(out, r))
            self.scaling = scale_alpha / r
            self.base.weight.requires_grad_(False)
            if self.base.bias is not None:
                self.base.bias.requires_grad_(False)
            lora_params.extend([self.lora_A, self.lora_B])

        def forward(self, x):
            delta = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
            return torch.nn.functional.linear(x, self.base.weight,
                                              self.base.bias) + delta

    for block in model.model.layers:
        block.self_attn.q_proj = LoraLinear(block.self_attn.q_proj)
        block.self_attn.v_proj = LoraLinear(block.self_attn.v_proj)
    return lora_params


def extract_adapter_state(model) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone()
            for name, param in model.named_parameters()
            if "lora_" in name}


def train_steps(student, params, micro_groups, optimizer, scheduler, n,
                grad_accum: int = 2) -> list[float]:
    """Run ``n`` optimizer steps over ``micro_groups`` (each group has
    ``grad_accum`` micro-batches, mirroring distill.train's accumulation).
    Returns per-OPTIMIZER-STEP mean losses."""
    from training.distill import sft_loss_aligned
    losses = []
    student.train()
    for group in micro_groups:
        assert len(group) == grad_accum
        accum = 0.0
        optimizer.zero_grad()
        for batch in group:
            out = student(input_ids=batch).logits
            labels = torch.roll(batch, -1, dims=1)   # next-token targets
            loss = sft_loss_aligned(out, labels)
            (loss / grad_accum).backward()
            accum += float(loss.item())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(accum / grad_accum)
        if len(losses) >= n:
            break
    return losses[:n]


def new_experiment(seed: int = 5501):
    """Fresh 'process': model + LoRA + optimizer + cosine scheduler."""
    torch.manual_seed(seed)
    model = _build_tiny_llama()
    params = build_lora_config(model)
    optimizer = torch.optim.AdamW(params, lr=3e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: 0.5 * (1 + __import__("math").cos(
            __import__("math").pi * s / TOTAL_STEPS)))
    return model, params, optimizer, scheduler


# ---------------------------------------------------------------------------
# The parity test
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env():
    vocab = _build_tiny_llama().config.vocab_size
    data = DeterministicData(vocab, total_micro=TOTAL_STEPS + 4)
    return data


def test_uninterrupted_vs_saved_and_resumed(env, tmp_path):
    from training.checkpoint import CheckpointManager, build_state

    groups_all = env.micro_groups

    # ---------------- Reference: uninterrupted 6 steps ------------------
    m_ref, p_ref, o_ref, sch_ref = new_experiment()
    ref_losses = train_steps(m_ref, p_ref, groups_all, o_ref, sch_ref,
                             TOTAL_STEPS)
    assert len(ref_losses) == TOTAL_STEPS

    # ------------- Interrupted: 2 steps, save, "session dies" -----------
    m_a, p_a, o_a, sch_a = new_experiment()          # same seed as ref
    a_losses = train_steps(m_a, p_a, groups_all, o_a, sch_a, SAVE_AFTER_STEP)

    saved_adapter_fp = extract_adapter_state(m_a)
    saved_exp_avg_fp = {pid: st["exp_avg"].detach().clone()
                        for pid, st in o_a.state.items()}

    ckpt_mgr = CheckpointManager(local_dir=str(tmp_path / "ckpt"))
    info = ckpt_mgr.save(
        build_state(model=m_a, optimizer=o_a, scheduler=sch_a,
                    tokens_consumed=SAVE_AFTER_STEP * 64,
                    cursor={"micro_batches_done": SAVE_AFTER_STEP * 2}),
        SAVE_AFTER_STEP)
    assert info["backend"] == "local"

    # ---------------- Fresh process resumes for 4 more -------------------
    # Different init seed on purpose: every piece of warm state must come
    # from the checkpoint, not from shared initialization.
    m_b, p_b, o_b, sch_b = new_experiment(seed=999_999)
    rest_groups = groups_all[SAVE_AFTER_STEP:]

    loaded = ckpt_mgr.load_latest()
    assert loaded is not None
    step, cursor = ckpt_mgr.apply(loaded, model=m_b, optimizer=o_b,
                                  scheduler=sch_b, restore_rng=False)
    assert step == SAVE_AFTER_STEP
    assert cursor == {"micro_batches_done": SAVE_AFTER_STEP * 2}

    # The exact pre-death LoRA weights were restored byte-for-byte...
    got_adapter = extract_adapter_state(m_b)
    assert set(got_adapter) == set(saved_adapter_fp)
    for name, t in saved_adapter_fp.items():
        torch.testing.assert_close(t, got_adapter[name], rtol=0, atol=0)
    # ...as were AdamW first moments. Optimizer state is keyed by PARAM OBJECT,
    # and m_b's parameters are different objects — map by positional index.
    saved_params = list(p_a)                       # o_a param order
    resumed_params = list(p_b)                     # o_b param order
    assert len(saved_params) == len(resumed_params)
    for i, pid in enumerate(saved_params):
        exp_avg = saved_exp_avg_fp[pid]
        st = o_b.state[resumed_params[i]]
        torch.testing.assert_close(exp_avg, st["exp_avg"],
                                   rtol=0, atol=1e-8)

    b_losses = train_steps(m_b, p_b, rest_groups, o_b, sch_b,
                           TOTAL_STEPS - SAVE_AFTER_STEP)
    resumed = a_losses + b_losses

    # --------------------------- Parity ---------------------------------
    assert len(resumed) == TOTAL_STEPS
    for i, (r, f) in enumerate(zip(ref_losses, resumed)):
        assert abs(r - f) <= TOLERANCE, (
            f"loss diverged at step {i}: reference {r:.10f} vs "
            f"resumed {f:.10f}")

    ref_adapter = extract_adapter_state(m_ref)
    final_adapter = extract_adapter_state(m_b)
    assert set(ref_adapter) == set(final_adapter)
    max_diff = 0.0
    for name in ref_adapter:
        diff = (ref_adapter[name] - final_adapter[name]).abs().max().item()
        max_diff = max(max_diff, diff)
        assert diff < TOLERANCE, f"adapter {name} differs by {diff}"


def test_checkpoint_roundtrip_matches_pre_save_state(env, tmp_path):
    """Sanity anchor: what we save is EXACTLY what was live at save time."""
    from training.checkpoint import CheckpointManager, build_state
    groups = env.micro_groups
    m, p, o, sch = new_experiment()
    train_steps(m, p, groups, o, sch, SAVE_AFTER_STEP)

    mgr = CheckpointManager(local_dir=str(tmp_path / "c"))
    mgr.save(build_state(model=m, optimizer=o, scheduler=sch,
                         tokens_consumed=128, cursor={"b": 4}),
             SAVE_AFTER_STEP)
    before_adapter = extract_adapter_state(m)
    before_sch_last_epoch = sch.last_epoch

    m2, p2, o2, sch2 = new_experiment(seed=777)
    got = mgr.load_latest()
    assert got is not None and got["step"] == SAVE_AFTER_STEP
    mgr.apply(got, model=m2, optimizer=o2, scheduler=sch2, restore_rng=False)

    for n, pr in m2.named_parameters():
        if "lora_" in n:
            torch.testing.assert_close(pr.detach(), before_adapter[n],
                                       rtol=0, atol=0)
    # optimizer state is keyed by param OBJECT; map o2's params to o's params
    # positionally (different objects, same insertion order).
    o_params = list(p)
    o2_params = list(p2)
    assert len(o_params) == len(o2_params)
    for i, pr in enumerate(o2_params):
        st = o2.state[pr]
        ref_st = o.state[o_params[i]]
        torch.testing.assert_close(st["exp_avg"], ref_st["exp_avg"],
                                   rtol=0, atol=0)
        torch.testing.assert_close(st["exp_avg_sq"], ref_st["exp_avg_sq"],
                                   rtol=0, atol=0)
    assert sch2.last_epoch == before_sch_last_epoch
