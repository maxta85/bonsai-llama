"""CLI: quantize an FP16 safetensors weight file to Q1_0 or Q2_0 GGUF.

Usage:
    bonsai-quantize --format q1_0 --in weights.safetensors --out model-Q1_0.gguf
    bonsai-quantize --format q2_0 --in weights.safetensors --out model-Q2_0.gguf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .q1_0 import quantize_q1_0
from .q2_0 import quantize_q2_0
from .gguf_io import write_gguf_tensors, GGML_TYPE_Q1_0, GGML_TYPE_Q2_0


def _load_safetensors(path):
    from safetensors import safe_open

    out = {}
    with safe_open(path, framework="numpy") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def _interleaved_blocks(t):
    """Pack as [scale(2)][packed(N)] per group, concatenated (ggml layout)."""
    out = bytearray()
    for g in range(t.n_groups):
        out += t.scales[g].tobytes()  # 2 bytes fp16
        out += t.packed[g].tobytes()  # 16 (q1_0) or 32 (q2_0) bytes
    return bytes(out)


def quantize_main():
    ap = argparse.ArgumentParser(description="Quantize weights to Q1_0 / Q2_0 GGUF")
    ap.add_argument("--format", choices=["q1_0", "q2_0"], required=True)
    ap.add_argument("--in", dest="inp", required=True, help="input safetensors file")
    ap.add_argument("--out", required=True, help="output .gguf path")
    args = ap.parse_args()

    weights = _load_safetensors(Path(args.inp))
    tensors = []
    for name, w in weights.items():
        w32 = w.astype(np.float32)
        if args.format == "q1_0":
            t = quantize_q1_0(w32)
            ggml_type = GGML_TYPE_Q1_0
        else:
            t = quantize_q2_0(w32)
            ggml_type = GGML_TYPE_Q2_0
        raw = _interleaved_blocks(t)
        tensors.append(
            {
                "name": name,
                "ggml_type": ggml_type,
                "shape": tuple(w.shape),
                "raw_bytes": raw,
            }
        )
        print(f"  {name}: {t}")

    kv = {
        "general.architecture": (8, "bonsai"),
        "general.quantization": (8, args.format),
    }
    write_gguf_tensors(args.out, kv, tensors)
    print(f"Wrote {args.out} ({Path(args.out).stat().st_size} bytes)")
