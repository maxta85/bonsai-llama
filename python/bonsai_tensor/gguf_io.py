"""Minimal GGUF writer/reader for Q1_0 and Q2_0 tensors.

This is a *tiny* GGUF implementation sufficient to write a quantized model's
weight tensors with the right ggml type ids so the PrismML-Eng/llama.cpp fork
(or recent mainline) can load them. For a full model conversion (tokenizer,
architecture metadata, etc.) use the bundled llama.cpp's
``convert_hf_to_gguf.py`` after exporting FP16 weights, or extend this module.

ggml type ids (from ggml.h):
    GGML_TYPE_Q1_0  = 8   (block 32, ~1.5 bpw)        -- NOT used here
    GGML_TYPE_Q2_0  = 10  (block 16, ~2.56 bpw)       -- mainline default
    We rely on the Prism fork's Q1_0_g128 / Q2_0_g128 block-128 variants,
    which reuse the same type ids but with group size 128. The fork detects
    the block size from the tensor's byte layout.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# GGUF magic + version (v3).
GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
GGUF_VERSION = 3

# GGUF value types (subset).
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_F16 = 6
GGUF_TYPE_F32 = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_F64 = 12

# ggml type ids we care about.
GGML_TYPE_F16 = 1
GGML_TYPE_F32 = 0
GGML_TYPE_Q1_0 = 8
GGML_TYPE_Q2_0 = 10


def _pack_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _pack_scalar(t: int, v) -> bytes:
    if t == GGUF_TYPE_UINT32:
        return struct.pack("<I", v)
    if t == GGUF_TYPE_INT32:
        return struct.pack("<i", v)
    if t == GGUF_TYPE_F32:
        return struct.pack("<f", v)
    if t == GGUF_TYPE_UINT64:
        return struct.pack("<Q", v)
    raise ValueError(f"unsupported scalar type {t}")


def write_gguf_header(f, kv, n_tensors):
    """Write the GGUF header: magic, version, n_kv, n_tensors, then KV pairs."""
    f.write(struct.pack("<I", GGUF_MAGIC))
    f.write(struct.pack("<I", GGUF_VERSION))
    f.write(struct.pack("<Q", len(kv)))
    f.write(struct.pack("<Q", n_tensors))
    for name, (t, v) in kv.items():
        f.write(_pack_str(name))
        f.write(struct.pack("<I", t))
        if t == GGUF_TYPE_STRING:
            f.write(_pack_str(v))
        else:
            f.write(_pack_scalar(t, v))


def write_gguf_tensor(f, name, ggml_type, shape, raw_bytes, offset):
    """Write one tensor info + return the data offset (aligned to 32)."""
    f.write(_pack_str(name))
    f.write(struct.pack("<I", len(shape)))
    for d in reversed(shape):
        f.write(struct.pack("<Q", d))
    f.write(struct.pack("<I", ggml_type))
    f.write(struct.pack("<Q", offset))
    return offset + len(raw_bytes)


def write_gguf_tensors(path, kv, tensors):
    """Write a complete GGUF file with the given KV metadata and tensors.

    Each tensor dict has: name, ggml_type, shape, raw_bytes.
    """
    path = Path(path)
    import io

    with open(path, "wb") as f:
        # First, write header (we need n_tensors known up front).
        buf = io.BytesIO()
        write_gguf_header(buf, kv, len(tensors))
        # Write tensor infos with placeholder offsets to measure size.
        info_buf = io.BytesIO()
        for t in tensors:
            info_buf.write(_pack_str(t["name"]))
            info_buf.write(struct.pack("<I", len(t["shape"])))
            for d in reversed(t["shape"]):
                info_buf.write(struct.pack("<Q", d))
            info_buf.write(struct.pack("<I", t["ggml_type"]))
            info_buf.write(struct.pack("<Q", 0))  # placeholder
        info_bytes = info_buf.getvalue()
        header_bytes = buf.getvalue()
        data_base = len(header_bytes) + len(info_bytes)
        data_base = (data_base + 31) & ~31
        # Write header.
        f.write(header_bytes)
        # Write real infos with correct offsets.
        offset = data_base
        for t in tensors:
            f.write(_pack_str(t["name"]))
            f.write(struct.pack("<I", len(t["shape"])))
            for d in reversed(t["shape"]):
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", t["ggml_type"]))
            f.write(struct.pack("<Q", offset))
            offset += len(t["raw_bytes"])
        # Pad to data_base.
        pad = data_base - f.tell()
        if pad > 0:
            f.write(b"\x00" * pad)
        # Write tensor data.
        for t in tensors:
            f.write(t["raw_bytes"])


def read_gguf_header(path):
    """Read and return the GGUF header (magic, version, kv, tensor infos)."""
    path = Path(path)
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        version = struct.unpack("<I", f.read(4))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        return {
            "magic": magic,
            "version": version,
            "n_kv": n_kv,
            "n_tensors": n_tensors,
            "path": str(path),
        }
