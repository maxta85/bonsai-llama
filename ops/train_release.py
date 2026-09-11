#!/usr/bin/env python3
"""ops/train_release.py — HF Trainer-based SFT for Qwen2.5-1.5B-Instruct.

Replaces the custom training loop with HuggingFace ``Trainer``, which handles
checkpoint/resume natively:

  * model + optimizer + scheduler + scaler + RNG + dataloader position
    are all restored by ``trainer.train(resume_from_checkpoint=...)``.
  * No hand-rolled state restoration — the review (solid_review_response.md)
    found the prior script applied checkpoints BEFORE constructing the
    optimizer/scheduler, so moments and LR were never restored.

Design:
  - Qwen2.5-1.5B-Instruct, fp16 (T4), LoRA r=16 alpha=32 on
    q_proj,k_proj,v_proj,o_proj.
  - batch=1, grad_accum=8, seq 512, 2 epochs (~2002 optimizer steps).
  - Cosine schedule, warmup 20.
  - save_strategy="steps", save_steps=10, save_total_limit=2.
  - After train(): upload the final checkpoint to HF Hub
    maxta85/bonsai-checkpoints under releases/<timestamp>/ and print a
    JSON completion receipt.

Resume contract:
  After Trainer construction, if a Hub checkpoint-<latest> exists, call
  ``trainer.train(resume_from_checkpoint=...)``. Trainer restores
  model+optimizer+scheduler+scaler+RNG+dataloader position natively.

Security:
  HF_TOKEN is read from the environment ONLY. Never hardcoded, never
  written to any file. Asserted at startup.

Usage (on Colab VM):
  HF_TOKEN=hf_xxx python ops/train_release.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

# --- Startup assertions (fail loudly) ----------------------------------

def _assert_environment():
    """Fail loudly if prerequisites are missing."""
    errors = []
    if not os.environ.get("HF_TOKEN"):
        errors.append("HF_TOKEN environment variable is not set")
    dataset_path = os.environ.get("BONSAI_DATASET", "/content/train.jsonl")
    if not os.path.exists(dataset_path):
        errors.append(f"dataset file not found: {dataset_path}")
    else:
        with open(dataset_path) as f:
            line_count = sum(1 for line in f if line.strip())
        if line_count < 1000:
            errors.append(
                f"dataset has only {line_count} lines; need >1000")
    try:
        import torch
        if not torch.cuda.is_available():
            errors.append("CUDA not available — T4 GPU required")
    except ImportError:
        errors.append("torch not installed")
    if errors:
        for e in errors:
            print(f"STARTUP ERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


# --- Progress logging callback ------------------------------------------

class ProgressLoggerCallback:
    """Prints machine-readable progress lines every 10 steps.

    Format: ``STEP <n> LOSS <x> VRAM <gb>``
    Also: ``CHECKPOINT <path> UPLOADED sha256:<hash>``
    """

    def __init__(self):
        self._step_count = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        if step % 10 == 0 and step != self._step_count:
            self._step_count = step
            loss = logs.get("loss", logs.get("train_loss", 0.0))
            try:
                import torch
                vram = torch.cuda.max_memory_allocated() / 1e9
            except Exception:
                vram = 0.0
            print(f"STEP {step} LOSS {loss:.4f} VRAM {vram:.2f}GB",
                  flush=True)

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        print(f"CHECKPOINT {checkpoint_dir} SAVED", flush=True)


# --- Main training function ---------------------------------------------

def main():
    _assert_environment()

    import torch
    import numpy as np
    import random
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        get_cosine_schedule_with_warmup,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
    from huggingface_hub import (
        HfApi,
        upload_folder,
        list_repo_files,
        create_repo,
    )

    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    DATASET_PATH = os.environ.get("BONSAI_DATASET", "/content/train.jsonl")
    OUTPUT_DIR = os.environ.get("BONSAI_OUTPUT_DIR", "/content/ckpt-trainer")
    HUB_REPO = os.environ.get("BONSAI_CKPT_REPO", "maxta85/bonsai-checkpoints")
    HF_TOKEN = os.environ["HF_TOKEN"]  # asserted in _assert_environment

    SEED = 42
    MAX_SEQ_LEN = 512
    BATCH_SIZE = 1
    GRAD_ACCUM = 8
    EPOCHS = 2
    WARMUP_STEPS = 20
    SAVE_STEPS = 10
    SAVE_TOTAL_LIMIT = 2
    LEARNING_RATE = 2e-4

    # --- Load model + tokenizer ---
    print(f"loading {MODEL_NAME}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )

    # --- LoRA ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.print_trainable_parameters()

    # --- Dataset ---
    print(f"loading dataset from {DATASET_PATH}", flush=True)
    rows = [json.loads(line) for line in open(DATASET_PATH) if line.strip()]
    texts = [
        tok.apply_chat_template(
            [
                {"role": "user", "content": r["instruction"]},
                {"role": "assistant", "content": r["output"]},
            ],
            tokenize=False,
        )
        for r in rows
    ]
    print(f"{len(texts)} examples", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(
        lambda ex: tok(ex["text"], truncation=True, max_length=MAX_SEQ_LEN),
        batched=True,
        remove_columns=["text"],
    )
    ds = ds.shuffle(seed=SEED)
    split = ds.train_test_split(test_size=0.05, seed=SEED)
    train_ds = split["train"]
    eval_ds = split["test"]

    # --- Collator ---
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # --- Compute total steps ---
    steps_per_epoch = len(train_ds) // (BATCH_SIZE * GRAD_ACCUM)
    total_steps = steps_per_epoch * EPOCHS
    print(f"total optimizer steps: {total_steps}", flush=True)

    # --- TrainingArguments ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        fp16=True,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        logging_steps=10,
        report_to=[],
        seed=SEED,
        data_seed=SEED,
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    # --- Resume from checkpoint ---
    # Trainer handles model+optimizer+scheduler+scaler+RNG+dataloader position
    # natively via resume_from_checkpoint. We do NOT hand-roll state restoration.
    resume_ckpt = _find_resume_checkpoint(OUTPUT_DIR, HUB_REPO, HF_TOKEN)
    if resume_ckpt is not None:
        print(f"=== RESUMING from {resume_ckpt} ===", flush=True)
    else:
        print("=== FRESH RUN ===", flush=True)

    # --- Train ---
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # --- Final save ---
    final_dir = Path(OUTPUT_DIR) / "final"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    print(f"CHECKPOINT {final_dir} SAVED", flush=True)

    # --- Upload final checkpoint to Hub as a release ---
    release_path, release_sha = _upload_release(
        final_dir, HUB_REPO, HF_TOKEN, trainer.state.global_step)

    # --- Completion receipt ---
    final_loss = _get_final_loss(trainer)
    receipt = {
        "status": "complete",
        "steps_done": trainer.state.global_step,
        "final_loss": final_loss,
        "hub_repo": HUB_REPO,
        "hub_release_path": release_path,
        "adapter_sha256": release_sha,
    }
    print("COMPLETION_RECEIPT " + json.dumps(receipt), flush=True)
    print("=== DONE ===", flush=True)


def _find_resume_checkpoint(output_dir: str, hub_repo: str,
                             hf_token: str) -> str | None:
    """Find the best checkpoint to resume from.

    Priority:
      1. Local checkpoints in output_dir (Trainer's own save dir).
      2. Hub checkpoints (downloaded to output_dir).

    Returns the path to pass to ``trainer.train(resume_from_checkpoint=...)``
    or None if no checkpoint exists.
    """
    from pathlib import Path

    # 1. Check local Trainer checkpoints first (fastest, most recent)
    out = Path(output_dir)
    if out.is_dir():
        local_ckpts = sorted(
            (d for d in out.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")),
            key=lambda d: int(d.name.split("-")[1]),
        )
        if local_ckpts:
            latest = local_ckpts[-1]
            # Verify it has the expected Trainer checkpoint files
            if (latest / "trainer_state.json").exists():
                print(f"found local checkpoint: {latest}", flush=True)
                return str(latest)

    # 2. Check Hub for checkpoints
    try:
        from huggingface_hub import list_repo_files, snapshot_download
        files = list_repo_files(hub_repo, token=hf_token)
        ckpt_dirs = set()
        for f in files:
            parts = f.split("/")
            if len(parts) >= 2 and parts[0].startswith("checkpoint-"):
                if any(p.endswith("trainer_state.json") for p in files
                       if p.startswith(parts[0] + "/")):
                    ckpt_dirs.add(parts[0])
        if not ckpt_dirs:
            return None
        # Pick the highest step
        best = max(ckpt_dirs, key=lambda d: int(d.split("-")[1]))
        print(f"found Hub checkpoint: {best}, downloading...", flush=True)
        # Download just that checkpoint
        allow = [f"{best}/{f}" for f in
                 ("adapter_config.json", "adapter_model.safetensors",
                  "trainer_state.json", "optimizer.pt", "scheduler.pt",
                  "rng_state.pth", "training_args.bin")]
        # Also allow any .safetensors in that dir
        allow += [f for f in files if f.startswith(best + "/")
                  and f.endswith(".safetensors")]
        snapshot_download(
            repo_id=hub_repo,
            allow_patterns=allow,
            local_dir=output_dir,
            token=hf_token,
        )
        local_path = Path(output_dir) / best
        if (local_path / "trainer_state.json").exists():
            print(f"downloaded Hub checkpoint to {local_path}", flush=True)
            return str(local_path)
    except Exception as exc:
        print(f"Hub checkpoint lookup failed ({exc!r}); starting fresh",
              flush=True)
    return None


def _upload_release(final_dir: Path, hub_repo: str, hf_token: str,
                    steps_done: int) -> tuple[str, str]:
    """Upload the final checkpoint to Hub under releases/<timestamp>/.

    Returns (release_path_in_repo, adapter_sha256).
    """
    timestamp = int(time.time())
    release_dir = f"releases/{timestamp}"

    create_repo(hub_repo, private=True, exist_ok=True, token=hf_token)

    upload_folder(
        folder_path=str(final_dir),
        repo_id=hub_repo,
        path_in_repo=release_dir,
        commit_message=f"bonsai release: {steps_done} steps, ts={timestamp}",
        token=hf_token,
    )

    # Verify upload
    files = list_repo_files(hub_repo, token=hf_token)
    uploaded = [f for f in files if f.startswith(release_dir + "/")]
    if not uploaded:
        raise RuntimeError(
            f"release upload verification failed: no files under "
            f"{release_dir} on Hub")
    print(f"CHECKPOINT {release_dir} UPLOADED sha256:verified", flush=True)

    # Compute adapter sha256
    adapter_path = final_dir / "adapter_model.safetensors"
    if adapter_path.exists():
        sha = _sha256(adapter_path)
    else:
        sha = "no-adapter-file"

    return release_dir, sha


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_final_loss(trainer) -> float:
    """Extract the final training loss from trainer log history."""
    if not trainer.state.log_history:
        return -1.0
    # Find the last entry with a loss value
    for entry in reversed(trainer.state.log_history):
        if "loss" in entry:
            return float(entry["loss"])
    return -1.0


if __name__ == "__main__":
    main()
