# Reproduction Guide

How to reproduce a Bonsai-style 1-bit / ternary model from an open-weights
base, end to end, using this repo.

## 0. What you are reproducing

A model whose every weight (embeddings, attention projections, MLP
projections, LM head) is quantized to either:

- **1-bit** (`Q1_0`, g128): `{-1, +1}` with abs-mean group scale, ~1.125 bpw, or
- **1.58-bit ternary** (`Q2_0`, g128): `{-1, 0, +1}` with abs-mean group scale,
  ~2.125 bpw.

The reference target is Prism ML's Bonsai-8B (Qwen3-8B base, 1-bit) and
Ternary-Bonsai-8B (Qwen3-8B base, ternary). See
[`whitepapers/`](whitepapers/README.md) for the full provenance.

## 1. Environment

```bash
pip install -e ".[train]"
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
pytest -q          # Python tensor round-trip tests
ctest --test-dir build/cpp --output-on-failure
```

## 2. Verify the tensor formats

```python
import numpy as np
from bonsai_tensor import quantize_q1_0, quantize_q2_0

w = np.random.randn(256, 512).astype(np.float32)
t1 = quantize_q1_0(w)   # 1-bit
t2 = quantize_q2_0(w)   # ternary
print(t1, t2)
print(t1.dequantize().shape, t2.dequantize().shape)
```

`tests/test_tensors.py` checks sign preservation (Q1_0) and ternary value
coverage (Q2_0).

## 3. Distill an open-weights model into 1-bit / ternary

```bash
# Ternary (1.58-bit), the BitNet b1.58 recipe
bonsai-distill --teacher Qwen/Qwen3-8B --mode 1.58b --out bonsai-ternary

# 1-bit
bonsai-distill --teacher Qwen/Qwen3-8B --mode 1b --out bonsai-1bit
```

This:

1. Loads the teacher (FP32, frozen) and a student (same arch, all `nn.Linear`
   swapped for `BitLinear`).
2. Trains with `alpha*CE + (1-alpha)*T^2*KL` against the teacher's logits.
3. Saves the student (FP32 master weights) to `--out`.

> The bundled data pipeline is a placeholder (a constant string) so the
> script runs without a dataset download. To actually reproduce Bonsai,
> plug in a large pre-training / instruction corpus (see the Prism ML
> whitepaper for the data mix). The `training/distill.py` `train_step`
> function is the integration point.

## 4. Export to GGUF and run in llama.cpp

```bash
# Export the trained weights to Q2_0 GGUF
python -m training.export_gguf --model bonsai-ternary --format q2_0 --out bonsai-Q2_0.gguf

# Build the Prism fork (has Q1_0 + Q2_0 g128 kernels)
./scripts/setup_llamacpp.sh
cmake --build build/llamacpp -j

# Run
./build/llamacpp/bin/llama-cli -m bonsai-Q2_0.gguf -p "Hello, Bonsai!"
```

For a full model (tokenizer + all metadata), use llama.cpp's
`convert_hf_to_gguf.py` on the FP16 export of the student, then re-quantize
the weight tensors with `bonsai-quantize`.

## 5. Standalone C++ kernel sanity check

```bash
# Make a random (4,256) FP32 weight file
python -c "import numpy as np; np.random.randn(4,256).astype('float32').tofile('w.f32')"

./build/cpp/bonsai_infer q2_0 w.f32 4 256 1
./build/cpp/bonsai_infer q1_0 w.f32 4 256 1
```

## 6. What's needed for a faithful Bonsai reproduction

Per the Prism ML whitepaper, the full recipe (not all public) includes:

1. **Base model**: Qwen3-8B (dense, GQA 32/8, SwiGLU, RoPE, RMSNorm).
2. **Quantization-aware training** with ternary/1-bit targets across *all*
   layers including embeddings and LM head (this repo's `BitLinear` covers
   the projections; embeddings/LM head need the same STE quantizer applied).
3. **Distillation** from the FP16 base (this repo's `distill.py`).
4. **RLVR / instruction tuning** phase on the quantized model (not in this
   repo; use your preferred RL framework, e.g. TRL).
5. **Data mix**: the whitepaper describes the corpus; a reproduction
   typically uses a SlimPajama/StarCoder-style mix + instruction data.
6. **Compute**: significant (the 8B model was trained on a large cluster).
   For a research reproduction, start with a 1.7B or 4B base.

The pieces in this repo give you the tensor types, the QAT layer, the
distillation loop, the GGUF export, and the inference kernels — the
scaffolding to experiment. The data and compute are the parts you bring.
