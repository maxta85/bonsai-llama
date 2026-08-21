"""Distillation trainer: student (BitLinear) learns from a frozen FP teacher.

Loss = alpha * CE(student, labels) + (1 - alpha) * KL(student || teacher)
with temperature `T` on the KL term, following Hinton et al. (2015) and the
"Training 1.58bit LLMs via Distillation" recipe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def distillation_loss(student_logits, teacher_logits, labels, *,
                      alpha=0.5, temperature=2.0):
    """CE + temperature-scaled KL."""
    ce = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        labels.reshape(-1), ignore_index=-100)
    sl = student_logits / temperature
    tl = teacher_logits / temperature
    kl = F.kl_div(F.log_softmax(sl, dim=-1), F.softmax(tl, dim=-1),
                  reduction="batchmean") * (temperature * temperature)
    return alpha * ce + (1.0 - alpha) * kl


@torch.no_grad()
def teacher_logits(teacher, input_ids):
    teacher.eval()
    out = teacher(input_ids)
    if hasattr(out, "logits"):
        return out.logits
    return out[0]


def train_step(student, teacher, input_ids, labels, optimizer, *,
               alpha=0.5, temperature=2.0, grad_accum=1):
    """One gradient-accumulated training step. Returns the loss value."""
    student.train()
    with torch.no_grad():
        t_logits = teacher_logits(teacher, input_ids)
    s_logits = student(input_ids)
    if hasattr(s_logits, "logits"):
        s_logits = s_logits.logits
    loss = distillation_loss(s_logits, t_logits, labels,
                             alpha=alpha, temperature=temperature)
    (loss / grad_accum).backward()
    if grad_accum <= 1:
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()


def main():
    ap = argparse.ArgumentParser(description="Distill an FP model into 1-bit/ternary")
    ap.add_argument("--teacher", required=True, help="HF model id or path for the teacher")
    ap.add_argument("--mode", choices=["1b", "1.58b"], default="1.58b")
    ap.add_argument("--dataset", default="wikitext", help="dataset name")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--out", default="bonsai-distilled", help="output dir")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .bit_linear import replace_linears_with_bitlinear

    print(f"Loading teacher: {args.teacher}")
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.float32)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.float32)
    n = replace_linears_with_bitlinear(student, mode=args.mode)
    print(f"Replaced {n} Linear layers with BitLinear (mode={args.mode})")

    tok = AutoTokenizer.from_pretrained(args.teacher)

    # Minimal placeholder data pipeline: a tiny constant batch so the script
    # runs end to end without a dataset download. Swap for a real corpus.
    text = "The quick brown fox jumps over the lazy dog. " * 64
    ids = tok(text, return_tensors="pt").input_ids
    seq = ids.size(1)
    if seq % 2:
        ids = ids[:, :-1]
        seq = ids.size(1)
    input_ids = ids[:, : seq // 2]
    labels = ids[:, seq // 2 : seq]
    m = min(input_ids.size(1), labels.size(1))
    input_ids = input_ids[:, :m]
    labels = labels[:, :m]

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        loss = train_step(student, teacher, input_ids, labels, optimizer,
                          alpha=args.alpha, temperature=args.temperature)
        print(f"epoch {ep}: loss={loss:.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"Saved student to {out}")
    print("Next: export to GGUF with python -m training.export_gguf")


if __name__ == "__main__":
    main()
