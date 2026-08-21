# Architecture

```
bonsai-llama/
├── python/bonsai_tensor/      # Pure-Python packed tensor types (the spec)
│   ├── __init__.py
│   ├── q1_0.py                # 1-bit Q1_0 (g128): pack/unpack/quantize/dequant
│   ├── q2_0.py                # Ternary Q2_0 (g128): pack/unpack/quantize/dequant
│   ├── gguf_io.py             # Minimal GGUF writer/reader for Q1_0/Q2_0
│   └── cli.py                 # `bonsai-quantize` entry point
├── cpp/                       # C++ inference kernels (CMake)
│   ├── kernels/
│   │   ├── bonsai_kernels.h   # Public API
│   │   ├── q1_0.cpp           # dequant + matmul + quantize + fp16 helpers
│   │   └── q2_0.cpp           # dequant + matmul + quantize
│   ├── cli/bonsai_infer.cpp   # Standalone CLI: quantize + matmul sanity
│   └── tests/test_kernels.cpp # Round-trip + matmul tests
├── training/                  # QAT + distillation (PyTorch)
│   ├── bit_linear.py          # BitLinear (BitNet b1.58 STE layer)
│   ├── distill.py             # `bonsai-distill` trainer (CE + KL)
│   └── export_gguf.py         # Export trained weights -> GGUF Q1_0/Q2_0
├── docs/
│   ├── whitepapers/README.md  # All whitepapers + references
│   ├── bitnet-2b.md           # Microsoft BitNet b1.58 2B4T reference model
│   ├── REPRODUCTION.md        # Step-by-step reproduction guide
│   └── ARCHITECTURE.md        # This file
├── scripts/
│   ├── setup_llamacpp.sh      # Clone + build the PrismML-Eng/llama.cpp fork
│   └── setup_bitnet_cpp.sh    # Clone + build microsoft/bitnet (bitnet.cpp)
├── tests/test_tensors.py      # Python tensor round-trip tests
├── pyproject.toml
├── CMakeLists.txt (root)      # Top-level: builds cpp/ and wires llamacpp/
└── README.md
```

## Data flow

```
            (open weights, FP16)
                     │
                     ▼
        ┌────────────────────────┐
        │ training/distill.py    │   BitLinear (STE) + CE/KL distillation
        │   teacher (frozen FP)  │   master weights stay FP32
        │   student (ternary/1b) │
        └────────────┬───────────┘
                     │  trained FP32 master weights
                     ▼
        ┌────────────────────────┐
        │ training/export_gguf.py│   quantize with bonsai_tensor -> GGUF
        └────────────┬───────────┘
                     │  *.gguf (Q1_0 or Q2_0)
                     ▼
        ┌────────────────────────┐
        │ llama.cpp (Prism fork) │   full inference engine
        │  or cpp/bonsai_infer   │   minimal standalone kernel
        └────────────────────────┘
```

## Why three inference paths?

- **`cpp/`** is a tiny, dependency-free reference implementation of the
  Q1_0/Q2_0 dequant + matmul. It exists to make the format concrete and
  testable without pulling in all of llama.cpp. It is *not* fast.
- **The PrismML-Eng/llama.cpp fork** is the real inference engine for the
  Bonsai / Ternary-Bonsai GGUF models: optimized NEON/AVX/Metal/CUDA/Vulkan
  kernels for Q1_0 and Q2_0, full transformer, tokenizer, server.
  `scripts/setup_llamacpp.sh` wires it in.
- **microsoft/bitnet (bitnet.cpp)** is the official engine for the BitNet
  b1.58 2B model (`microsoft/bitnet-b1.58-2B-4T`): optimized I2_S / TL1
  CPU kernels that deliver the actual speed/energy wins from the paper.
  `scripts/setup_bitnet_cpp.sh` wires it in. See `docs/bitnet-2b.md`.

## Tensor type contract

`python/bonsai_tensor` and `cpp/kernels` implement the *same* format and are
cross-checked by the tests. The block layout matches ggml's
`Q1_0_g128` / `Q2_0_g128`:

```
Q1_0 block (18 bytes):  [ fp16 scale ][ 16 bytes packed bits  ]   128 weights
Q2_0 block (34 bytes):  [ fp16 scale ][ 32 bytes packed 2-bit ]   128 weights
```

The training pipeline's `BitLinear` produces weights that, when exported via
`export_gguf.py`, are byte-compatible with what the llama.cpp fork loads.
