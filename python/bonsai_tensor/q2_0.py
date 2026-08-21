"""Ternary 1.58-bit Q2_0 (g128) packed weight tensor.

Each weight is one of ``{-1, 0, +1}``, encoded as a 2-bit code
``q in {0, 1, 2, 3}`` with ``w = (q - 1) * scale``. One FP16 scale per group
of 128 weights. Code ``3`` (reconstructing ``+2*scale``) is reserved for future
extensions and unused for ternary weights.

This is the format used by the Ternary-Bonsai models
(``prism-ml/Ternary-Bonsai-*-gguf``) and matches ggml's ``Q2_0_g128`` block
layout: 2 bytes FP16 scale + 32 bytes of packed 2-bit codes = 34 bytes per
128 weights (~2.125 bpw).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GROUP_SIZE = 128
# 128 weights -> 32 bytes of packed 2-bit codes + 2 bytes fp16 scale = 34 bytes/block.
CODES_BYTES = GROUP_SIZE // 4  # 32
BLOCK_BYTES = CODES_BYTES + 2  # 34


def _abs_mean_scale(w: np.ndarray) -> np.float16:
    """Per-group scale = mean of absolute values (BitNet b1.58 abs-mean)."""
    return np.float16(np.mean(np.abs(w)))


def _round_to_ternary(w_norm: np.ndarray) -> np.ndarray:
    """Round normalized weights to {-1, 0, +1} via thresholding at +-0.5.

    ``w_norm = w / scale``. Following BitNet b1.58, values with
    ``|w_norm| > 0.5`` take the sign; otherwise 0.
    """
    return np.where(w_norm > 0.5, 1, np.where(w_norm < -0.5, -1, 0)).astype(np.int8)


def quantize_q2_0(w: np.ndarray) -> "Q2_0Tensor":
    """Quantize an FP weight matrix to ternary Q2_0 (g128).

    Ternarization with abs-mean scaling, grouped per 128 weights along the
    last axis. Returns a :class:`Q2_0Tensor` holding packed 2-bit codes + fp16
    scales.
    """
    w = np.ascontiguousarray(w, dtype=np.float32)
    orig_shape = w.shape
    flat = w.reshape(-1, GROUP_SIZE)
    n_groups = flat.shape[0]

    scales = np.empty(n_groups, dtype=np.float16)
    packed = np.zeros((n_groups, CODES_BYTES), dtype=np.uint8)

    for g in range(n_groups):
        block = flat[g]
        s = _abs_mean_scale(block)
        scales[g] = s
        if float(s) == 0.0:
            ternary = np.zeros(GROUP_SIZE, dtype=np.int8)
        else:
            ternary = _round_to_ternary(block / float(s))
        # code q = ternary + 1  -> {-1,0,+1} maps to {0,1,2}; 3 reserved.
        codes = (ternary + 1).astype(np.uint8)  # values in {0,1,2}
        # pack 4 codes per byte, little-endian (first code in low 2 bits).
        packed[g] = _pack_2bit(codes)

    return Q2_0Tensor(packed=packed, scales=scales, shape=orig_shape)


def _pack_2bit(codes: np.ndarray) -> np.ndarray:
    """Pack 128 2-bit codes into 32 bytes (low bits first)."""
    assert codes.shape[0] == GROUP_SIZE
    out = np.zeros(CODES_BYTES, dtype=np.uint8)
    for i in range(GROUP_SIZE):
        out[i // 4] |= (codes[i] & 0x3) << ((i % 4) * 2)
    return out


def _unpack_2bit(packed: np.ndarray) -> np.ndarray:
    """Unpack 32 bytes into 128 2-bit codes."""
    codes = np.empty(GROUP_SIZE, dtype=np.uint8)
    for i in range(GROUP_SIZE):
        codes[i] = (packed[i // 4] >> ((i % 4) * 2)) & 0x3
    return codes


def dequantize_q2_0(t: "Q2_0Tensor") -> np.ndarray:
    """Dequantize a :class:`Q2_0Tensor` to FP32: ``w = (q - 1) * scale``."""
    n_groups = t.packed.shape[0]
    out = np.empty((n_groups, GROUP_SIZE), dtype=np.float32)
    for g in range(n_groups):
        codes = _unpack_2bit(t.packed[g])
        out[g] = (codes.astype(np.float32) - 1.0) * float(t.scales[g])
    return out.reshape(t.shape)


@dataclass
class Q2_0Tensor:
    """A packed ternary 1.58-bit Q2_0 (g128) tensor.

    Attributes
    ----------
    packed : np.ndarray[uint8]
        2-bit-packed codes, shape ``(n_groups, CODES_BYTES)``.
    scales : np.ndarray[float16]
        Per-group FP16 scales, shape ``(n_groups,)``.
    shape : tuple[int, ...]
        Original (logical) weight shape.
    """

    packed: np.ndarray
    scales: np.ndarray
    shape: tuple[int, ...]

    @property
    def n_groups(self) -> int:
        return self.packed.shape[0]

    @property
    def nbytes(self) -> int:
        return self.packed.nbytes + self.scales.nbytes

    @property
    def bits_per_weight(self) -> float:
        n = int(np.prod(self.shape))
        return (self.nbytes * 8) / n if n else 0.0

    def dequantize(self) -> np.ndarray:
        return dequantize_q2_0(self)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Reference dequantize-then-matvec: ``y = dequant(W) @ x``."""
        W = self.dequantize()
        return W @ x

    def __repr__(self) -> str:
        return (
            f"Q2_0Tensor(shape={self.shape}, n_groups={self.n_groups}, "
            f"nbytes={self.nbytes}, bpw={self.bits_per_weight:.3f})"
        )
