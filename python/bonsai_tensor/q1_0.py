"""1-bit Q1_0 (g128) packed weight tensor.

Each weight is a single bit: ``0 -> -scale``, ``1 -> +scale``. One FP16 scale
per group of 128 weights. This is the format used by the 1-bit Bonsai models
(``prism-ml/Bonsai-*-gguf``) and matches ggml's ``Q1_0_g128`` block layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GROUP_SIZE = 128
# 128 weights -> 16 bytes of packed bits + 2 bytes fp16 scale = 18 bytes/block.
BLOCK_BYTES = GROUP_SIZE // 8 + 2  # 18


def _abs_mean_scale(w: np.ndarray) -> np.float16:
    """Per-group scale = mean of absolute values (BitNet-style abs-mean)."""
    return np.float16(np.mean(np.abs(w)))


def quantize_q1_0(w: np.ndarray) -> "Q1_0Tensor":
    """Quantize an FP weight matrix to 1-bit Q1_0 (g128).

    Sign-binarization with abs-mean scaling, grouped per 128 weights along the
    last axis. Returns a :class:`Q1_0Tensor` holding packed bits + fp16 scales.
    """
    w = np.ascontiguousarray(w, dtype=np.float32)
    orig_shape = w.shape
    flat = w.reshape(-1, GROUP_SIZE)
    n_groups = flat.shape[0]

    scales = np.empty(n_groups, dtype=np.float16)
    packed = np.zeros((n_groups, GROUP_SIZE // 8), dtype=np.uint8)

    for g in range(n_groups):
        block = flat[g]
        s = _abs_mean_scale(block)
        scales[g] = s
        # sign: 1 if w >= 0 else 0  (0 -> -scale, 1 -> +scale)
        signs = (block >= 0).astype(np.uint8)
        # pack 8 bits per byte, little-endian within byte (bit 0 = first weight)
        packed[g] = np.packbits(signs, bitorder="little")

    return Q1_0Tensor(packed=packed, scales=scales, shape=orig_shape)


def dequantize_q1_0(t: "Q1_0Tensor") -> np.ndarray:
    """Dequantize a :class:`Q1_0Tensor` back to FP32: ``w = (2*bit - 1) * scale``."""
    n_groups = t.packed.shape[0]
    out = np.empty((n_groups, GROUP_SIZE), dtype=np.float32)
    for g in range(n_groups):
        bits = np.unpackbits(t.packed[g], bitorder="little")[:GROUP_SIZE]
        out[g] = (2.0 * bits.astype(np.float32) - 1.0) * float(t.scales[g])
    return out.reshape(t.shape)


@dataclass
class Q1_0Tensor:
    """A packed 1-bit Q1_0 (g128) tensor.

    Attributes
    ----------
    packed : np.ndarray[uint8]
        Bit-packed weights, shape ``(n_groups, GROUP_SIZE // 8)``.
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
        return dequantize_q1_0(self)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Reference dequantize-then-matvec: ``y = dequant(W) @ x``.

        ``x`` is 1-D (shape ``(in_features,)``) or 2-D ``(in_features, k)``.
        """
        W = self.dequantize()
        return W @ x

    def __repr__(self) -> str:
        return (
            f"Q1_0Tensor(shape={self.shape}, n_groups={self.n_groups}, "
            f"nbytes={self.nbytes}, bpw={self.bits_per_weight:.3f})"
        )
