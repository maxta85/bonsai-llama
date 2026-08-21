"""Export a trained BitLinear student model to GGUF Q1_0 / Q2_0.

Reads the FP32 master weights from each BitLinear, quantizes them with the
bonsai_tensor library (matching the g128 layout), and writes a GGUF file with
the correct ggml type ids so the PrismML-Eng/llama.cpp fork (or recent
mainline) can load it.

For a full model (tokenizer, architecture metadata, all tensors) prefer
llama.cpp's ``convert_hf_to_gguf.py`` on an FP16 export; this module handles
the quantized weight tensors specifically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from bonsai_tensor.q1_0 import quantize_q1_0
from bonsai_tensor.q2_0 import quantize_q2_0
from bonsai_tensor.gguf_io import write_gguf_tensors, GGML_TYPE_Q1_0, GGML_TYPE_Q2_0


def _interleaved_blocks(t):
    """Pack as [scale(2)][packed(N)] per group, concatenated (ggml layout)."""
    out = bytearray()
    for g in range(t.n_groups):
        out += t.scales[g].tobytes()
        out += t.packed[g].tobytes()
    return bytes(out)


def export_model(model_path, fmt, out_path):
    """Load a saved BitLinear model and export its weights to GGUF."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model.eval()

    tensors = []
    for name, param in model.named_parameters():
        if not name.endswith("weight"):
            continue
        w = param.detach().cpu().numpy().astype(np.float32)
        if fmt == "q1_0":
            t = quantize_q1_0(w)
            ggml_type = GGML_TYPE_Q1_0
        else:
            t = quantize_q2_0(w)
            ggml_type = GGML_TYPE_Q2_0
        raw = _interleaved_blocks(t)
        tensors.append({"name": name, "ggml_type": ggml_type,
                        "shape": tuple(w.shape), "raw_bytes": raw})
        print(f"  {name}: {t}")

    kv = {"general.architecture": (8, "bonsai"),
          "general.quantization": (8, fmt)}
    write_gguf_tensors(out_path, kv, tensors)
    print(f"Wrote {out_path} ({Path(out_path).stat().st_size} bytes)")


def main():
    ap = argparse.ArgumentParser(description="Export BitLinear model to GGUF")
    ap.add_argument("--model", required=True, help="path to saved student model")
    ap.add_argument("--format", choices=["q1_0", "q2_0"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    export_model(args.model, args.format, args.out)


if __name__ == "__main__":
    main()
