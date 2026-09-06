# Layer-Streaming Integration Spec

## Goal
Add CPU-RAM layer streaming to training/distill.py so a 27B teacher/student pair
can be distilled on a single 16GB GPU (T4). Validated first at 1.6B, then 7B.

## Architecture (Soup v0.74 pattern, proven bit-exact)

1. Load student model with device_map="cpu", dtype=bfloat16 (or ternary-quantized per-layer).
2. Keep ONLY: embedding, final norm, lm_head, LoRA adapters, optimizer states on GPU.
3. Per training step:
   a. Stream decoder layers to GPU one at a time (double-buffer: prefetch layer i+1 while computing layer i).
   b. Forward: hidden = layer(to_gpu(hidden)); offload layer back to CPU (pinned memory).
   c. BitLinear QAT: student layers use BitLinear (training/bit_linear.py) — ternary weights simulated via STE.
   d. Chunked KL loss vs teacher logits (training/chunked_loss.py already handles vocab chunking).
   e. Backward: stream layers in REVERSE; gradients flow through LoRA adapters only (base frozen).
4. Teacher: either (a) pre-compute teacher logits for the dataset offline and store as .pt shards, or (b) stream teacher layers similarly. Prefer (a) for T4 — halves VRAM.

## Files to create/modify
- training/layer_stream.py (NEW): StreamingModel class — wraps a HF model, provides
  forward_with_streaming(input_ids, bitlinear_fn) and the double-buffer logic.
- training/distill.py (MODIFY): add --stream flag; when set, use StreamingModel and
  skip full-model .to(device).
- Test: tests/test_layer_stream.py (NEW) — validate streaming output == full-load output
  on a tiny model (e.g. Qwen2.5-0.5B, cpu→gpu roundtrip must be bit-exact in eval mode).

## Constraints
- PyTorch 2.x, no new deps beyond torch/transformers/accelerate.
- Pinned memory for CPU side: torch.empty(..., pin_memory=True).
- Non-blocking H2D/D2H copies (copy_ with non_blocking=True) + explicit torch.cuda.synchronize only at buffer boundaries.
- Base weights frozen: requires_grad=False except LoRA params.
- Must work with training/bit_linear.py BitLinear layers already patched into the student.

## Validation ladder
1. tests/test_layer_stream.py passes (0.5B, output equivalence).
2. 1.6B distill run on T4 via colab-cli: loss decreases, peak VRAM < 8GB (print torch.cuda.max_memory_allocated).
3. Later: 7B, then 27B.
