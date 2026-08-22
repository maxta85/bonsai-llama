"""BitLinear: the BitNet b1.58 quantization-aware Linear layer.

A drop-in replacement for ``nn.Linear`` that fake-quantizes its weights to
ternary {-1, 0, +1} (mode="1.58b") or binary {-1, +1} (mode="1b") with
abs-mean group-wise scaling (group size 128, matching Bonsai's Q1_0/Q2_0
g128 layout). A straight-through estimator (STE) passes gradients through
the non-differentiable rounding.

Master weights are stored in FP32 (this module's ``.weight``); only the
forward pass uses the quantized values, so the optimizer updates the FP32
master copy as in BitNet.

Key details from the BitNet b1.58 paper:
  * Input is LayerNorm'd and clipped to [-clip, +clip] before matmul.
  * Weight scale per group: ``scale = mean(|W_group|)``.
  * Ternary: ``W_q = round(W / scale)`` then clip to {-1,0,+1}.
  * Binary:  ``W_q = sign(W)``.
  * STE: the rounding/sign op's derivative is treated as 1 in backward.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP_SIZE = 128


class STEQuantize(torch.autograd.Function):
    """Straight-through estimator for (round + clip)."""

    @staticmethod
    def forward(ctx, x, lo, hi):
        return torch.clamp(torch.round(x), lo, hi)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None


def _grouped_abs_mean_scale(w, group_size=GROUP_SIZE):
    """Compute abs-mean scale per group of `group_size` along the last dim."""
    *lead, n = w.shape
    assert n % group_size == 0, f"last dim {n} not divisible by group {group_size}"
    w_g = w.reshape(-1, group_size)
    scale = w_g.abs().mean(dim=-1, keepdim=True)
    scale = scale.repeat_interleave(group_size, dim=-1)
    return scale.reshape(*lead, n)


class BitLinear(nn.Module):
    """Quantization-aware Linear with ternary (1.58b) or binary (1b) weights."""

    def __init__(self, in_features, out_features, bias=True, mode="1.58b",
                 group_size=GROUP_SIZE, input_clip=30.0):
        super().__init__()
        assert mode in ("1.58b", "1b"), mode
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.group_size = group_size
        self.input_clip = input_clip
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def _quantize_weight(self):
        w = self.weight
        scale = _grouped_abs_mean_scale(w, self.group_size)
        scale = scale.clamp(min=1e-8)
        w_norm = w / scale
        if self.mode == "1.58b":
            w_q = STEQuantize.apply(w_norm, -1.0, 1.0)
        else:
            w_q = torch.sign(w_norm)
            w_q = torch.where(w_q == 0, torch.ones_like(w_q), w_q)
        return w_q * scale

    def forward(self, x):
        # BitNet: LayerNorm the input, clip, then quantized matmul.
        x = F.layer_norm(x, (x.shape[-1],))
        x = torch.clamp(x, -self.input_clip, self.input_clip)
        w_q = self._quantize_weight()
        # Cast quantized weight to input dtype for the matmul.
        # Master weight stays FP32 (for optimizer/STE), but the actual
        # matmul runs in the input dtype (BF16 on T4, FP32 on 6000).
        # This halves activation memory on BF16 hardware.
        w_q = w_q.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w_q, bias)

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, mode={self.mode}, "
                f"group_size={self.group_size}")


def replace_linears_with_bitlinear(module, mode="1.58b", group_size=GROUP_SIZE,
                                   skip=("lm_head",)):
    """Recursively replace every ``nn.Linear`` with a :class:`BitLinear`.

    Layers whose name contains any of `skip` are left as FP. Returns the
    number of replaced layers.
    """
    count = 0
    for name, child in module.named_children():
        full = name
        if isinstance(child, nn.Linear) and not any(s in full for s in skip):
            bl = BitLinear(child.in_features, child.out_features,
                           bias=child.bias is not None, mode=mode,
                           group_size=group_size)
            with torch.no_grad():
                bl.weight.copy_(child.weight)
                if child.bias is not None and bl.bias is not None:
                    bl.bias.copy_(child.bias)
            setattr(module, name, bl)
            count += 1
        else:
            count += replace_linears_with_bitlinear(child, mode, group_size, skip)
    return count
