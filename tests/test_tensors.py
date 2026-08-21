"""Round-trip tests for the Python bonsai_tensor library."""

import numpy as np
import pytest

from bonsai_tensor import quantize_q1_0, quantize_q2_0, GROUP_SIZE


def test_q1_0_sign_preserved():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 256)).astype(np.float32)
    t = quantize_q1_0(w)
    dq = t.dequantize()
    assert np.all(np.sign(dq) == np.sign(w))
    assert dq.shape == w.shape
    assert 1.0 < t.bits_per_weight < 1.2


def test_q1_0_block_size():
    w = np.random.randn(2, GROUP_SIZE).astype(np.float32)
    t = quantize_q1_0(w)
    assert t.n_groups == 2
    assert t.packed.shape == (2, GROUP_SIZE // 8)


def test_q2_0_ternary_values():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((4, 256)).astype(np.float32)
    t = quantize_q2_0(w)
    dq = t.dequantize()
    flat = dq.reshape(-1, GROUP_SIZE)
    scales = t.scales.astype(np.float32)
    for g in range(t.n_groups):
        s = scales[g]
        uniq = np.unique(flat[g])
        for v in uniq:
            assert any(np.isclose(v, cand, atol=1e-4) for cand in (-s, 0.0, s)), (
                f"group {g}: value {v} not in {{-{s}, 0, +{s}}}"
            )
    assert dq.shape == w.shape
    assert 2.0 < t.bits_per_weight < 2.2


def test_q2_0_block_size():
    w = np.random.randn(2, GROUP_SIZE).astype(np.float32)
    t = quantize_q2_0(w)
    assert t.n_groups == 2
    assert t.packed.shape == (2, GROUP_SIZE // 4)


def test_matvec():
    rng = np.random.default_rng(2)
    w = rng.standard_normal((8, 256)).astype(np.float32)
    x = rng.standard_normal(256).astype(np.float32)
    for q in (quantize_q1_0(w), quantize_q2_0(w)):
        y = q.matvec(x)
        assert y.shape == (8,)
        assert np.all(np.isfinite(y))


def test_zero_block():
    w = np.zeros((1, GROUP_SIZE), dtype=np.float32)
    t1 = quantize_q1_0(w)
    assert np.allclose(t1.dequantize(), 0)
    t2 = quantize_q2_0(w)
    assert np.allclose(t2.dequantize(), 0)
