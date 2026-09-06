"""Distillation + SFT trainer for ternary/bitnet models.

Two modes:
  1. Pre-training distillation (default): student learns from teacher's
     logits on raw text. Loss = alpha*CE + (1-alpha)*KL.
  2. SFT mode (--sft): student learns from instruction-response pairs.
     Loss = CE on assistant tokens only (labels masked on user/system).

Single-GPU mode (default): both models on same GPU, full-vocab KL.
Multi-GPU mode: teacher on big GPU, student on small GPU, top-k KL.
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
    """CE + full-vocab temperature-scaled KL (exact, no approximation)."""
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
    """CE + top-k temperature-scaled KL (approximation for multi-GPU)."""
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


def sft_loss(student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Pure cross-entropy on assistant tokens (labels masked with -100)."""
    return F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )


# ---------------------------------------------------------------------------
# Teacher forwards
# ---------------------------------------------------------------------------

@torch.no_grad()
def teacher_forward_full(teacher: nn.Module, input_ids: torch.Tensor,
                         attention_mask: torch.Tensor | None = None):
    """Run teacher, return full logits (B, T, V). Same-device only."""
    teacher.eval()
    out = teacher(input_ids, attention_mask=attention_mask)
    return out.logits if hasattr(out, "logits") else out[0]


@torch.no_grad()
def teacher_forward_topk(teacher: nn.Module, input_ids: torch.Tensor,
                         topk: int = 50, temperature: float = 2.0,
                         attention_mask: torch.Tensor | None = None):
    """Run teacher, return top-k indices + renormalized probs. Multi-GPU."""
    teacher.eval()
    out = teacher(input_ids, attention_mask=attention_mask)
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
    teacher: nn.Module | None,
    dataloader,
    optimizer,
    *,
    teacher_device: torch.device,
    student_device: torch.device,
    sft_mode: bool = False,
    topk: int = 50,
    topk_kl: bool = False,
    chunked_loss: bool = False,
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
    stream_model=None,
):
    """Run the training loop.

    SFT mode: pure CE on assistant tokens, no teacher needed.
    Distillation mode: CE + KL from teacher logits.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    student.train()
    if stream_model is None:
        student.to(student_device)  # streamed models never move wholesale
    else:
        print("Layer streaming enabled: base weights stay in CPU RAM "
              "(pinned), only embed/norm/lm_head resident on GPU.")

    if teacher is not None:
        teacher.eval()
        teacher.to(teacher_device)
        for p in teacher.parameters():
            p.requires_grad = False

    single_gpu = (teacher is not None and teacher_device == student_device)
    use_topk_kl = (teacher is not None) and ((not single_gpu) or topk_kl)
    use_chunked = chunked_loss and use_topk_kl and not sft_mode

    if sft_mode:
        print("SFT mode: pure CE on assistant tokens (loss-masked)")
    elif use_chunked:
        print(f"Single-GPU mode: chunked CE + top-{topk} KL "
              f"(no full vocab logits, max memory savings)")
    elif single_gpu and topk_kl:
        print(f"Single-GPU mode: top-{topk} KL (memory-efficient, "
              f"~14GB less than full-vocab)")
    elif single_gpu:
        print("Single-GPU mode: full-vocab KL (exact distillation)")
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
        attention_mask = batch.get("attention_mask")  # (B, T) or None

        if sft_mode:
            # --- SFT: pure CE on assistant tokens ---
            ids = input_ids.to(student_device)
            lbls = labels.to(student_device)
            am = attention_mask.to(student_device) if attention_mask is not None else None
            s_out = student(ids, attention_mask=am)
            s_logits = s_out.logits if hasattr(s_out, "logits") else s_out[0]
            loss = sft_loss(s_logits, lbls)

        elif use_chunked:
            # --- Chunked CE + top-k KL (no full vocab logits at all) ---
            # Student base model produces hidden states (B, T, H), then the
            # loss is computed by iterating over vocab chunks. Never
            # materializes the (B, T, 151936) logits tensor.
            from .chunked_loss import chunked_ce_and_topk_kl, get_hidden_states
            teacher_input = input_ids.to(teacher_device)
            am_t = attention_mask.to(teacher_device) if attention_mask is not None else None
            with torch.no_grad():
                topk_indices, topk_probs = teacher_forward_topk(
                    teacher, teacher_input, topk=topk, temperature=temperature,
                    attention_mask=am_t,
                )
            if not single_gpu:
                topk_indices = topk_indices.to(student_device)
                topk_probs = topk_probs.to(student_device)

            student_input = input_ids.to(student_device)
            student_labels = labels.to(student_device)
            am_s = attention_mask.to(student_device) if attention_mask is not None else None
            hidden = get_hidden_states(student, student_input, attention_mask=am_s)
            loss = chunked_ce_and_topk_kl(
                hidden, student.lm_head.weight, student_labels,
                topk_indices, topk_probs,
                alpha=alpha, temperature=temperature,
            )
        elif use_topk_kl:
            # --- Top-k KL (single-GPU memory-efficient or multi-GPU) ---
            # Teacher produces top-k indices + probs, then we free its full
            # logits immediately. Student only gathers k logits per position
            # instead of materializing the full 151K vocab.
            teacher_input = input_ids.to(teacher_device)
            am_t = attention_mask.to(teacher_device) if attention_mask is not None else None
            with torch.no_grad():
                topk_indices, topk_probs = teacher_forward_topk(
                    teacher, teacher_input, topk=topk, temperature=temperature,
                    attention_mask=am_t,
                )
            if not single_gpu:
                topk_indices = topk_indices.to(student_device)
                topk_probs = topk_probs.to(student_device)

            student_input = input_ids.to(student_device)
            student_labels = labels.to(student_device)
            am_s = attention_mask.to(student_device) if attention_mask is not None else None
            s_out = student(student_input, attention_mask=am_s)
            s_logits = s_out.logits if hasattr(s_out, "logits") else s_out[0]
            loss = distillation_loss_topk(
                s_logits, topk_indices, topk_probs, student_labels,
                alpha=alpha, temperature=temperature,
            )
        else:
            # --- Single-GPU full-vocab KL (exact, no approximation) ---
            ids = input_ids.to(student_device)
            lbls = labels.to(student_device)
            am = attention_mask.to(student_device) if attention_mask is not None else None
            with torch.no_grad():
                t_logits = teacher_forward_full(teacher, ids, attention_mask=am)
            s_out = student(ids, attention_mask=am)
            s_logits = s_out.logits if hasattr(s_out, "logits") else s_out[0]
            loss = distillation_loss_full(
                s_logits, t_logits, lbls,
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
        description="Train ternary/bitnet models via distillation or SFT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Models
    ap.add_argument("--teacher", default=None,
                    help="HF model id for teacher (required for distillation, "
                         "not needed for --sft mode)")
    ap.add_argument("--student", required=True,
                    help="HF model id or path for the student base")
    ap.add_argument("--mode", choices=["1b", "1.58b"], default="1.58b",
                    help="1.58b = ternary {-1,0,+1}, 1b = binary {-1,+1}")
    ap.add_argument("--init-from", default=None,
                    help="Path to a checkpoint to initialize student from "
                         "(e.g. a pre-trained ternary model to fine-tune)")

    # SFT mode
    ap.add_argument("--sft", action="store_true",
                    help="Supervised fine-tuning mode (instruction-response pairs)")
    ap.add_argument("--sft-data", default=None,
                    help="Path to local JSONL file with SFT data "
                         "(ShareGPT/Alpaca/messages format)")
    ap.add_argument("--sft-dataset", default=None,
                    choices=["alpaca", "dolly", "openhermes"],
                    help="Use a built-in SFT dataset instead of --sft-data")
    ap.add_argument("--system-prompt", default="You are a helpful assistant.",
                    help="System prompt for SFT mode")

    # Pre-training data
    ap.add_argument("--dataset", default="wikitext",
                    help="HF dataset name for pre-training (default: wikitext)")
    ap.add_argument("--text-key", default="text",
                    help="column name for text in the dataset")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="cap on number of sequences (for quick tests)")

    # Training
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="CE weight (1-alpha = KL weight) [distillation only]")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--topk", type=int, default=50,
                    help="top-k teacher logits for multi-GPU mode")
    ap.add_argument("--topk-kl", action="store_true",
                    help="use top-k KL even on single GPU (saves ~14GB of "
                         "vocab logits memory, no quality loss for k>=50)")
    ap.add_argument("--chunked-loss", action="store_true",
                    help="chunked CE+KL loss (avoids materializing full "
                         "151K vocab logits, saves ~1.5GB, requires --topk-kl)")

    # Devices
    ap.add_argument("--teacher-device", default="cuda:0",
                    help="device for the teacher")
    ap.add_argument("--student-device", default="cuda:0",
                    help="device for the student")
    ap.add_argument("--teacher-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])

    # Output
    ap.add_argument("--out", default="bonsai-distilled")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=10)
    # Memory optimization
    ap.add_argument("--fp8", action="store_true",
                    help="use FP8 tensor cores for matmul (Blackwell/Hopper, "
                         "30%% less memory, ~1.1x speed)")
    ap.add_argument("--8bit-adam", action="store_true",
                    help="use 8-bit AdamW (75%% less optimizer memory)")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="gradient checkpointing (90%% less activation memory, "
                         "~20%% slower)")
    ap.add_argument("--stream", action="store_true",
                    help="layer-stream the student from CPU RAM (Soup v0.74): "
                         "only embedding/final-norm/lm_head (+LoRA) live on "
                         "the GPU; decoder layers stream via double-buffered "
                         "non-blocking H2D copies each step. Implies no full "
                         "model .to(device).")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.fp8:
        from .mxfp4_linear import replace_linears_with_mxfp4 as replace_fn
        print("  Using FP8 tensor core acceleration (Blackwell)")
    else:
        from .bit_linear import replace_linears_with_bitlinear as replace_fn

    teacher_device = torch.device(args.teacher_device)
    student_device = torch.device(args.student_device)

    print(f"Student device: {student_device}")

    # --- Load tokenizer ---
    tok = AutoTokenizer.from_pretrained(args.student)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- Load teacher (frozen) if not SFT-only ---
    teacher = None
    if not args.sft and args.teacher:
        print(f"Teacher device: {teacher_device}")
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

    # --- Load student ---
    student_id = args.init_from or args.student
    print(f"Loading student: {student_id} (fp32, BitLinear mode={args.mode})")
    student = AutoModelForCausalLM.from_pretrained(
        student_id, torch_dtype=torch.float32)
    n = replace_fn(student, mode=args.mode)
    print(f"  replaced {n} Linear layers with BitLinear")

    stream_student = None
    if args.stream:
        from .layer_stream import StreamingModel
        # Keep the base model on CPU as pinned streaming source; residents
        # (embed/final-norm/lm_head, where LoRA adapters would be added) go
        # to the compute device inside StreamingModel.
        stream_student = StreamingModel(
            student.to("cpu"), compute_device=student_device)
        student = stream_student
        print(f"  streaming: {stream_student._num_layers} decoder layers "
              f"stay in CPU RAM; residents on {student_device}")
    else:
        student.to(student_device)
        if student_device.type == "cuda":
            s_mem = torch.cuda.memory_allocated(student_device) / 1e9
            print(f"  student on {student_device}, {s_mem:.1f} GB allocated")
    if args.grad_checkpoint:
        student.gradient_checkpointing_enable()
        print("  gradient checkpointing enabled (saves activation memory)")

    # --- Data ---
    if args.sft:
        from .sft import (load_sft_dataset, load_alpaca, load_dolly,
                          load_openhermes)
        if args.sft_data:
            print(f"Loading SFT data: {args.sft_data}")
            dl = load_sft_dataset(args.sft_data, tok,
                                  batch_size=args.batch_size,
                                  seq_len=args.seq_len,
                                  system=args.system_prompt,
                                  max_examples=args.max_tokens)
        elif args.sft_dataset == "alpaca":
            print("Loading Alpaca dataset...")
            dl = load_alpaca(tok, batch_size=args.batch_size,
                             seq_len=args.seq_len, max_examples=args.max_tokens)
        elif args.sft_dataset == "dolly":
            print("Loading Dolly 15K dataset...")
            dl = load_dolly(tok, batch_size=args.batch_size,
                            seq_len=args.seq_len, max_examples=args.max_tokens)
        elif args.sft_dataset == "openhermes":
            print("Loading OpenHermes 2.5 dataset...")
            dl = load_openhermes(tok, batch_size=args.batch_size,
                                 seq_len=args.seq_len,
                                 max_examples=args.max_tokens)
        else:
            ap.error("--sft requires --sft-data or --sft-dataset")
    else:
        from .data import load_wikitext, load_dataset_by_name
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
    if getattr(args, "8bit_adam", False):
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(student.parameters(), lr=args.lr,
                                         weight_decay=0.01)
        print("  using 8-bit AdamW (75% less optimizer memory)")
    else:
        optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr,
                                      weight_decay=0.01)

    # --- Train ---
    mode_str = "SFT" if args.sft else "distillation"
    print(f"\nStarting {mode_str}: {args.max_steps} steps, "
          f"batch={args.batch_size}, seq={args.seq_len}, "
          f"grad_accum={args.grad_accum}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}\n")

    train(
        student, teacher, dl, optimizer,
        teacher_device=teacher_device,
        student_device=student_device,
        stream_model=stream_student,
        sft_mode=args.sft,
        topk=args.topk,
        topk_kl=args.topk_kl,
        chunked_loss=args.chunked_loss,
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
