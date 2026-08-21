# bonsai-llama

A research prototype for **1-bit (`Q1_0`) and ternary 1.58-bit (`Q2_0`)** language models,
built around the [Prism ML Bonsai](https://huggingface.co/prism-ml) model family and a
[llama.cpp](https://github.com/ggerganov/llama.cpp) fork that runs those packed tensor types.

This repo contains three things:

1. **`python/bonsai_tensor`** — a pure-Python (+ NumPy) reference implementation of the two
   packed weight formats used by Bonsai:
   - **1-bit `Q1_0` g128**: each weight is 1 bit (`0 -> -scale`, `1 -> +scale`), one FP16
     scale per 128 weights. ~1.125 bits/weight.
   - **Ternary `Q2_0` g128**: each weight is one of `{-1, 0, +1}`, packed as a 2-bit code
     `q in {0,1,2,3}` with `w = (q - 1) * scale`, one FP16 scale per 128 weights.
     ~2.125 bits/weight. Code `3` is reserved (reconstructs `+2*scale`).
2. **`cpp/`** — a small C++ inference-kernel library (CMake) that dequantizes both formats
   and does a reference matmul, plus a thin CLI. It is the minimal standalone engine; for
   full-featured inference you build the bundled llama.cpp fork (see below).
3. **`training/`** — a PyTorch **quantization-aware training (QAT) + distillation** pipeline
   that takes an open-weights FP16/BF16 model (e.g. Qwen3-8B) and distills it into either
   the 1-bit or ternary packed format, following the BitNet b1.58 recipe. It can export
   directly to GGUF `Q1_0` / `Q2_0` for the llama.cpp engine.

The `docs/` directory indexes every whitepaper and reference needed to reproduce the
training of models with these tensor types from scratch.

## Quick start

```bash
# 1. Python env (tensor lib + training)
pip install -e ".[train]"

# 2. C++ inference kernels
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 3. llama.cpp fork (full inference engine, optional)
./scripts/setup_llamacpp.sh        # clones PrismML-Eng/llama.cpp @ prism
cmake --build build/llamacpp -j
```

See [`docs/`](docs/) for the whitepaper index, the reproduction guide, and the
architecture overview.

## Status

Research prototype. The Python tensor library and C++ kernels are self-contained and
tested. The training pipeline is a working QAT scaffold that reproduces the BitNet b1.58
`BitLinear` + straight-through estimator; full reproduction of Bonsai's proprietary
training recipe requires the data and compute described in the Prism ML whitepaper (see
`docs/whitepapers/`).

## License

MIT for the code in this repo. The Bonsai models and the Prism ML whitepapers are
Apache-2.0 / property of Prism ML — see the links in `docs/whitepapers/`.
