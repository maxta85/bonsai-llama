# AGENTS.md — bonsai-llama

Research prototype for 1-bit (Q1_0) and ternary 1.58-bit (Q2_0) LLMs, built
around the Prism ML Bonsai model family + a llama.cpp fork.

## Build & test

```bash
# Python (tensor lib + training). Use a venv — system pip is PEP-668 blocked.
python3 -m venv .venv && .venv/bin/pip install -e ".[train,dev]"
.venv/bin/python -m pytest tests/ -q

# C++ kernels + CLI
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build/cpp --output-on-failure
```

## Layout

- `python/bonsai_tensor/` — pure-Python packed tensor types (Q1_0, Q2_0) + GGUF IO. THE spec.
- `cpp/` — C++ reference dequant/matmul kernels + `bonsai_infer` CLI. Mirrors the Python lib.
- `training/` — PyTorch QAT (`BitLinear`, BitNet b1.58 STE) + distillation + GGUF export.
- `docs/` — whitepapers index, reproduction guide, architecture.
- `scripts/setup_llamacpp.sh` — fetch + configure the PrismML-Eng/llama.cpp fork (real engine).

## Key facts

- Q1_0 (g128): 1 bit/weight, `0->-scale, 1->+scale`, 1 FP16 scale per 128 weights. **1.125 bpw**, 18 bytes/block.
- Q2_0 (g128): ternary `{-1,0,+1}` as 2-bit code `q`, `w=(q-1)*scale`, 1 FP16 scale per 128 weights. **2.125 bpw**, 34 bytes/block. Code 3 reserved.
- Block layout (ggml): `[fp16 scale(2)][packed weights]` per group, interleaved.
- Both formats verified to produce exactly these bpw values in Python AND C++.

## Environment notes

- System `pip` is externally-managed (PEP 668). Always use `.venv`.
- The `write` tool does not persist files at the repo root in this env; use shell heredocs if re-creating root files (CMakeLists.txt, README.md, pyproject.toml, .gitignore).
- Toolchain: CMake 4.2, GCC 15.2, Python 3.14, 32 cores.

## References

See `docs/whitepapers/README.md` for the full paper list (BitNet, BitNet b1.58,
Prism ML Bonsai whitepapers, distillation refs, GGUF spec).
