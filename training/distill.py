"""Distillation trainer: student (BitLinear) learns from a frozen FP teacher.

Loss = alpha * CE(student, labels) + (1 - alpha) * KL(student || teacher)
with temperature `T` on the KL term, following Hinton et al. (2015) and the
"Training 1.58bit LLMs via Distillation" recipe.

Two GPU modes
-------------
Single-GPU (default, faster for small models):
  Both teacher and student on the same GPU. Uses full-vocab KL (exact, no
  approximation). Best when both models fit in VRAM — e.g. Qwen3-1.7B
  teacher + Qwen3-0.6B student on a 96GB card.

Multi-GPU (for when the student won't fit alongside the teacher):
  Teacher on a large GPU (e.g. 96GB Blackwell), student on a smaller one
  (e.g. 12GB 3060). Uses top-k logit distillation: transfers only the top-k
  teacher probabilities per position (~50x less PCIe traffic than full vocab).

Reference teacher
-----------------
`microsoft/bitnet-b1.58-2B-4T-bf16` can be used as a ternary-aware teacher.
See docs/bitnet-2b.md.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def distillation_loss_full(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float = 0.5,
    temperature: float = 2.0,
) -> torch.Tensor:
    """CE + full-vocab temperature-scaled KL (exact, no approximation).

    Use when teacher and student are on the same device.
    """
    ce = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )
    sl = student_logits / temperature
    tl = teacher_logits / temperature
    kl = F.kl_div(
        F.log_softmax(sl, dim=-1),
        F.softmax(tl, dim=-1),
        reduction="batchmean",
    ) * (temperature * temperature)
    return alpha * ce + (1.0 - alpha) * kl


def distillation_loss_topk(
    student_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    teacher_topk_probs: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float = 0.5,
    temperature: float = 2.0,
) -> torch.Tensor:
    """CE + top-k temperature-scaled KL (approximation for multi-GPU).

    Only the teacher's top-k logits are used, renormalized to sum to 1.
    """
    ce = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )
    sl = student_logits / temperature
    sl_topk = torch.gather(sl, dim=-1, index=teacher_topk_indices)
    log_sl_topk = F.log_softmax(sl_topk, dim=-1)
    kl = F.kl_div(log_sl_topk, teacher_topk_probs, reduction="batchmean")
    kl = kl * (temperature * temperature)
    return alpha * ce + (1.0 - alpha) * kl


# ---------------------------------------------------------------------------
# Teacher forwards
# ---------------------------------------------------------------------------

@torch.no_grad()
def teacher_forward_full(teacher: nn.Module, input_ids: torch.Tensor):
    """Run teacher, return full logits (B, T, V). Same-device only."""
    teacher.eval()
    out = teacher(input_ids)
    return out.logits if hasattr(out, "logits") else out[0]


@torch.no_grad()
def teacher_forward_topk(teacher: nn.Module, input_ids: torch.Tensor,
                         topk: int = 50, temperature: float = 2.0):
    """Run teacher, return top-k indices + renormalized probs. Multi-GPU."""
    teacher.eval()
    out = teacher(input_ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, k=topk, dim=-1)
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return topk_indices, topk_probs


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def get_lr(step, base_lr, warmup_steps, total_steps):
    """Cosine LR schedule with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(
    student: nn.Module,
    teacher: nn.Module,
    dataloader,
    optimizer,
    *,
    teacher_device: torch.device,
    student_device: torch.device,
    topk: int = 50,
    alpha: float = 0.5,
    temperature: float = 2.0,
    grad_accum: int = 4,
    max_steps: int = 1000,
    warmup_steps: int = 50,
    base_lr: float = 1e-4,
    log_every: int = 10,
    save_every: int = 200,
    out_dir: str = "bonsai-distilled",
    tokenizer=None,
):
    """Run the distillation training loop.

    Single-GPU mode (teacher_device == student_device):
        Full-vocab KL, no cross-device transfer. Fastest for small models.
    Multi-GPU mode (different devices):
        Top-k KL to minimize PCIe transfer.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    student.train()
    student.to(student_device)
    teacher.eval()
    teacher.to(teacher_device)
    for p in teacher.parameters():
        p.requires_grad = False

    single_gpu = (teacher_device == student_device)
    if single_gpu:
        print("Single-GPU mode: full-vocab KL (exact distillation, no transfer)")
    else:
        print(f"Multi-GPU mode: top-{topk} KL (teacher={teacher_device}, "
              f"student={student_device})")

    step = 0
    accum_loss = 0.0
    start_time = time.time()
    lr = base_lr

    for batch in dataloader:
        if step >= max_steps:
            break

        input_ids = batch["input_ids"]  # (B, T)
        labels = batch["labels"]        # (B, T)

        if single_gpu:
            # --- Both on same device: full-vocab KL ---
            ids = input_ids.to(student_device)
            lbls = labels.to(student_device)
            with torch.no_grad():
                t_logits = teacher_forward_full(teacher, ids)
            s_out = student(ids)
            s_logits = s_out.logits if hasattr(s_out, "logits") else s_out[0]
            loss = distillation_loss_full(
                s_logits, t_logits, lbls,
                alpha=alpha, temperature=temperature,
            )
        else:
            # --- Multi-GPU: top-k KL to minimize PCIe transfer ---
            teacher_input = input_ids.to(teacher_device)
            topk_indices, topk_probs = teacher_forward_topk(
                teacher, teacher_input, topk=topk, temperature=temperature
            )
            topk_indices = topk_indices.to(student_device)
            topk_probs = topk_probs.to(student_device)

            student_input = input_ids.to(student_device)
            student_labels = labels.to(student_device)
            s_out = student(student_input)
            s_logits = s_out.logits if hasattr(s_out, "logits") else s_out[0]
            loss = distillation_loss_topk(
                s_logits, topk_indices, topk_probs, student_labels,
                alpha=alpha, temperature=temperature,
            )

        (loss / grad_accum).backward()
        accum_loss += loss.item()

        # --- Optimizer step (every grad_accum batches) ---
        if (step + 1) % grad_accum == 0:
            lr = get_lr(step, base_lr, warmup_steps, max_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # --- Logging ---
        if (step + 1) % log_every == 0:
            elapsed = time.time() - start_time
            avg_loss = accum_loss / log_every
            tokens = (step + 1) * input_ids.numel()
            tps = tokens / elapsed if elapsed > 0 else 0
            print(f"step {step+1:>5}/{max_steps} | loss {avg_loss:.4f} | "
                  f"lr {lr:.2e} | {tps:.0f} tok/s | {elapsed:.0f}s")
            accum_loss = 0.0

        # --- Checkpoint ---
        if (step + 1) % save_every == 0:
            ckpt = out_dir / f"checkpoint-{step+1}"
            ckpt.mkdir(parents=True, exist_ok=True)
            student.save_pretrained(ckpt)
            if tokenizer is not None:
                tokenizer.save_pretrained(ckpt)
            print(f"  saved checkpoint to {ckpt}")

        step += 1

    # Final save.
    student.save_pretrained(out_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)
    print(f"\nDone. Saved final model to {out_dir} ({step} steps)")
    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Distill an FP model into 1-bit/ternary via QAT + KD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Models
    ap.add_argument("--teacher", required=True,
                    help="HF model id or path for the teacher (e.g. Qwen/Qwen3-1.7B)")
    ap.add_argument("--student", default=None,
                    help="HF model id for the student base (defaults to --teacher)")
    ap.add_argument("--mode", choices=["1b", "1.58b"], default="1.58b",
                    help="1.58b = ternary {-1,0,+1}, 1b = binary {-1,+1}")
    # Data
    ap.add_argument("--dataset", default="wikitext",
                    help="HF dataset name (default: wikitext-103-raw-v1)")
    ap.add_argument("--text-key", default="text",
                    help="column name for text in the dataset")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="cap on number of sequences (for quick tests)")
    # Training
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="CE weight (1-alpha = KL weight)")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--topk", type=int, default=50,
                    help="top-k teacher logits for multi-GPU mode")
    # Devices
    ap.add_argument("--teacher-device", default="cuda:0",
                    help="device for the teacher")
    ap.add_argument("--student-device", default="cuda:0",
                    help="device for the student. If same as teacher, uses "
                         "single-GPU mode with full-vocab KL (exact, faster). "
                         "Set to cuda:1 for multi-GPU (top-k transfer).")
    ap.add_argument("--teacher-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="teacher precision (bf16 saves VRAM)")
    # Output
    ap.add_argument("--out", default="bonsai-distilled")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .bit_linear import replace_linears_with_bitlinear
    from .data import load_wikitext, load_dataset_by_name

    teacher_device = torch.device(args.teacher_device)
    student_device = torch.device(args.student_device)

    print(f"Teacher device: {teacher_device}")
    print(f"Student device: {student_device}")

    # --- Load tokenizer ---
    tok = AutoTokenizer.from_pretrained(args.teacher)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- Load teacher (frozen) ---
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}
    t_dtype = dtype_map[args.teacher_dtype]
    print(f"Loading teacher: {args.teacher} ({args.teacher_dtype})")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=t_dtype)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.to(teacher_device)
    if teacher_device.type == "cuda":
        t_mem = torch.cuda.memory_allocated(teacher_device) / 1e9
        print(f"  teacher on {teacher_device}, {t_mem:.1f} GB allocated")

    # --- Load student (BitLinear replacement) ---
    student_id = args.student or args.teacher
    print(f"Loading student: {student_id} (fp32, BitLinear mode={args.mode})")
    student = AutoModelForCausalLM.from_pretrained(
        student_id, torch_dtype=torch.float32)
    n = replace_linears_with_bitlinear(student, mode=args.mode)
    print(f"  replaced {n} Linear layers with BitLinear")
    student.to(student_device)
    if student_device.type == "cuda":
        s_mem = torch.cuda.memory_allocated(student_device) / 1e9
        print(f"  student on {student_device}, {s_mem:.1f} GB allocated")

    # --- Data ---
    print(f"Loading dataset: {args.dataset}")
    if args.dataset == "wikitext":
        dl = load_wikitext(tok, batch_size=args.batch_size,
                           seq_len=args.seq_len, max_tokens=args.max_tokens)
    else:
        dl = load_dataset_by_name(args.dataset, tok,
                                  batch_size=args.batch_size,
                                  seq_len=args.seq_len,
                                  text_key=args.text_key,
                                  max_tokens=args.max_tokens)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr,
                                  weight_decay=0.01)

    # --- Train ---
    print(f"\nStarting distillation: {args.max_steps} steps, "
          f"batch={args.batch_size}, seq={args.seq_len}, "
          f"grad_accum={args.grad_accum}")
    if teacher_device != student_device:
        print(f"  topk={args.topk}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}\n")

    train(
        student, teacher, dl, optimizer,
        teacher_device=teacher_device,
        student_device=student_device,
        topk=args.topk,
        alpha=args.alpha,
        temperature=args.temperature,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        base_lr=args.lr,
        log_every=args.log_every,
        save_every=args.save_every,
        out_dir=args.out,
        tokenizer=tok,
    )

    fmt = "q1_0" if args.mode == "1b" else "q2_0"
    print(f"\nNext: export to GGUF with:")
    print(f"  python -m training.export_gguf --model {args.out} "
          f"--format {fmt} --out {args.out}.gguf")


if __name__ == "__main__":
    main()
