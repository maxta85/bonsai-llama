## Review: Layer Streaming Implementation

---

### BLOCKER

**[layer_stream.py ~L230–260] Backward through streamed layers is broken.**
The copy plan writes weights into `scratch[s_idx].parameters()` via `p.data.copy_(...)` — modifying `.data` detaches from autograd. When backward runs, the scratch params have been overwritten with the *next* layer's weights before their gradient is ever computed. There is no reverse-order streaming loop; the forward just overwrites buffers in sequence, and `out.logits.sum().backward()` will backprop through the *last* layer's weights sitting in the scratch buffers, not each layer's correct weights. The test `test_trainable_addon_receives_grad` only exercises LoRA on `lm_head` (a resident module), so it never catches this — decoder-layer gradients are never verified.

**Fix required:** Implement a true reverse streaming pass saving per-layer activations (or use gradient checkpointing with per-layer recompute), and load each layer's weights back before its backward step.

---

**[layer_stream.py ~L242] Autograd graph references freed tensors.**
`h` is computed through `self.scratch[s_idx](h, ...)` but `scratch[s_idx]`'s weights are unconditionally overwritten on the *next* iteration via `_prefetch`. The saved activation graph holds references to the layer's intermediate tensors, but the weight storage they depend on has been clobbered. On CUDA this is a silent data corruption; backward reads wrong values without error.

---

### MAJOR

**[layer_stream.py ~L247] `torch.cuda.current_stream().synchronize()` inside the loop destroys double-buffer benefit and is placed wrong.**
The sync fires on the *compute* stream before prefetching i+1 — this blocks compute waiting for nothing (the copy is on the same stream). The intended pattern needs a *separate* `torch.cuda.Stream` for copies; without it there is no actual overlap, just sequential H2D + compute + H2D...

**[layer_stream.py ~L220] `_prefetch` records event on compute stream, not a copy stream.**
`ev.record(torch.cuda.current_stream(...))` records on the stream doing the copy, but since no separate stream is created, `p.data.copy_(snap_t, non_blocking=True)` runs on the default stream — non-blocking is a no-op on the same stream. Events are meaningless here.

**[tests/test_layer_stream.py ~L196] `test_trainable_addon_receives_grad` does not test streaming backward.**
The LoRA adapter is on `lm_head` (resident, never streamed). This test proves nothing about gradient flow through decoder layers. A BLOCKER-class bug in reverse streaming would not be caught.

---

### MINOR

**[layer_stream.py ~L270] `_causal_4d_mask` diverges from HF's implementation for models using `_update_causal_mask`.**
Llama 3 / Qwen2.5 >= 7B use `AttentionMaskConverter` with sliding-window and chunked attention support. This hand-rolled mask may produce different results for long sequences or architectures with non-standard attention, breaking bit-exactness at 27B scale.

**[distill.py ~L456] `student.gradient_checkpointing_enable()` called on `StreamingModel`.**
`StreamingModel` delegates `.train()` to `self.inner` but does not implement `gradient_checkpointing_enable`. This will raise `AttributeError` when `--stream --grad-checkpoint` are combined.

**[layer_stream.py ~L195] `_copy_plan_for` skips buffers (e.g. running_mean) not in `named_parameters`.**
`state_dict()` includes buffers; `named_parameters()` does not. Layers with BatchNorm or RMS stats will silently run with wrong buffer values in scratch.