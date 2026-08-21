"""bonsai_tensor: packed 1-bit (Q1_0) and ternary 1.58-bit (Q2_0) weight tensors.

These are the two packed weight formats used by the Prism ML Bonsai model family
and understood by the PrismML-Eng/llama.cpp fork (and, increasingly, mainline
llama.cpp).

Formats
-------
Q1_0 (g128)
    Each weight is a single bit. ``0 -> -scale``, ``1 -> +scale``. One FP16
    scale is shared per group of 128 weights. Effective ~1.125 bits/weight
    (1 sign bit + 16-bit scale amortized over 128 weights).

    Block layout (matches ggml Q1_0_g128): ``[d (fp16 scale)][16 bytes of
    packed bits, 128 weights]`` -> 18 bytes per 128 weights.

Q2_0 (g128)
    Each weight is ternary {-1, 0, +1}, encoded as a 2-bit code
    ``q in {0,1,2,3}`` with ``w = (q - 1) * scale``. One FP16 scale per 128
    weights. Code 3 (reconstructing +2*scale) is reserved for future use.
    Effective ~2.125 bits/weight (2 bits + 16-bit scale over 128 weights).

    Block layout (matches ggml Q2_0_g128): ``[d (fp16 scale)][32 bytes of
    packed 2-bit codes, 128 weights]`` -> 34 bytes per 128 weights.

References
----------
* Prism ML, "1-bit Bonsai 8B" whitepaper
  (https://github.com/PrismML-Eng/Bonsai-demo/blob/main/1-bit-bonsai-8b-whitepaper.pdf)
* Prism ML, "Ternary-Bonsai 8B" model card
  (https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf)
* Wang et al., "BitNet: Scaling 1-bit Transformers for Large Language Models",
  arXiv:2310.11453 (2023)
* Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58
  Bits", arXiv:2402.17764 (2024)
"""

from .q1_0 import Q1_0Tensor, quantize_q1_0, dequantize_q1_0
from .q2_0 import Q2_0Tensor, quantize_q2_0, dequantize_q2_0
from .gguf_io import write_gguf_tensor, read_gguf_header

GROUP_SIZE = 128

__all__ = [
    "GROUP_SIZE",
    "Q1_0Tensor",
    "quantize_q1_0",
    "dequantize_q1_0",
    "Q2_0Tensor",
    "quantize_q2_0",
    "dequantize_q2_0",
    "write_gguf_tensor",
    "read_gguf_header",
]

__version__ = "0.1.0"
