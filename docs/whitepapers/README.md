# Whitepapers & References

Everything needed to understand and reproduce training of models with the
**1-bit (`Q1_0`)** and **ternary 1.58-bit (`Q2_0`)** tensor types used by
Bonsai.

## Primary: Bonsai / Prism ML

| What | Link |
| --- | --- |
| 1-bit Bonsai 8B whitepaper (PDF) | https://github.com/PrismML-Eng/Bonsai-demo/blob/main/1-bit-bonsai-8b-whitepaper.pdf |
| Ternary-Bonsai 8B model card (format spec) | https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf |
| Bonsai-8B GGUF (1-bit Q1_0) model card | https://huggingface.co/prism-ml/Bonsai-8B-gguf |
| Prism ML announcement (Bonsai 8B) | https://prismml.com/news/bonsai-8b |
| Bonsai-demo repo (run + benchmark) | https://github.com/PrismML-Eng/Bonsai-demo |
| Prism ML docs (formats & runtime support) | https://docs.prismml.com/download/formats |
| Prism ML llama.cpp fork (Q1_0 + Q2_0 kernels) | https://github.com/PrismML-Eng/llama.cpp (branch `prism`) |

**Quantization format specs** (from the model cards):

- **Q1_0 (g128)**: each weight is 1 bit. `0 -> -scale`, `1 -> +scale`. One
  FP16 scale per 128 weights. ~1.125 bpw. Block = 18 bytes
  (`[scale(2)][bits(16)]`).
- **Q2_0 (g128)**: each weight is ternary `{-1, 0, +1}`, encoded as 2-bit
  code `q in {0,1,2,3}` with `w = (q-1)*scale`. One FP16 scale per 128
  weights. ~2.125 bpw. Block = 34 bytes (`[scale(2)][codes(32)]`). Code `3`
  (`+2*scale`) is reserved.

## Foundational: BitNet (Microsoft)

| What | Link |
| --- | --- |
| BitNet: Scaling 1-bit Transformers for LLMs (2023) | https://arxiv.org/abs/2310.11453 |
| The Era of 1-bit LLMs: All LLMs are in 1.58 Bits (BitNet b1.58, 2024) | https://arxiv.org/abs/2402.17764 |
| BitNet: 1-bit Pre-training for LLMs (JMLR v26, 2025) | https://jmlr.org/papers/v26/24-2050.html |
| BitNet official code + training tips / FAQ | https://github.com/microsoft/bitnet |
| Training Tips, Code, FAQ (PDF) | https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf |
| BitNet b1.58-2B-4T (official 2B model, 4T tokens) | https://huggingface.co/microsoft/BitNet-b1.58-2B-4T |
| bitnet.cpp (official 1.58-bit inference engine) | https://github.com/microsoft/bitnet (bitnet.cpp) |

**Key training recipe (BitNet b1.58):**

1. Replace every `nn.Linear` with `BitLinear` (ternary weights + abs-mean
   group scaling + STE).
2. Pre-train from scratch, OR distill from an FP teacher (see below).
3. Master weights stay FP32 in the optimizer; only the forward fake-quantizes.
4. LayerNorm + clip the input before the quantized matmul.
5. RLVR / instruction-tune the quantized model as a final phase.

## Distillation into 1.58-bit

| What | Link |
| --- | --- |
| Training 1.58bit LLMs via Distillation (mini-paper + code) | https://github.com/leszkolukasz/training-1.58bit-llms-via-distillation |
| BitNet-RWKV: 1.58-bit QAT on a single 8GB GPU (engineering log) | https://github.com/hafizradzi8901/bitnet-rwkv-lm |
| Hinton et al., Distilling the Knowledge in a Neural Network (2015) | https://arxiv.org/abs/1503.02531 |

The distillation loss used in this repo's `training/distill.py`:

```
L = alpha * CE(student, labels) + (1 - alpha) * T^2 * KL(student || teacher)
```

## Background: quantization & packing

| What | Link |
| --- | --- |
| GGUF format spec | https://github.com/ggerganov/llama.cpp/blob/master/docs/gguf.md |
| ggml quantization types (ggml.h) | https://github.com/ggerganov/llama.cpp/blob/master/ggml/include/ggml.h |
| GPTQ (post-training quantization) | https://arxiv.org/abs/2210.17323 |
| AWQ (activation-aware weight quantization) | https://arxiv.org/abs/2306.00978 |
| LLM.int8() | https://arxiv.org/abs/2208.07339 |
| SmoothQuant | https://arxiv.org/abs/2211.10438 |
| Wanda (pruning by weights & activations) | https://arxiv.org/abs/2310.10394 |
| SparseGPT | https://arxiv.org/abs/2301.00774 |

## How to get the whitepaper PDFs locally

```bash
# 1-bit Bonsai 8B whitepaper
curl -L -o docs/whitepapers/1-bit-bonsai-8b-whitepaper.pdf \
  https://github.com/PrismML-Eng/Bonsai-demo/raw/main/1-bit-bonsai-8b-whitepaper.pdf

# BitNet training tips / FAQ
curl -L -o docs/whitepapers/bitnet-training-tips-faq.pdf \
  https://github.com/microsoft/unilm/raw/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
```
