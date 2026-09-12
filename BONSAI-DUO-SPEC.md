# Bonsai-Duo — Single-Package Architecture Spec
Logged 2026-09-12. This is the Phase E target architecture reference.

## One-line summary
A single model package: ternary MoE backbone for reasoning + a dense coder
head grafted at the output — two precision tiers in one forward pass, one
KV cache, one deployment.

## Origin
Converged from Max's insight during the Bonsai-27B community testing:
- Bonsai 27B outpaced all other ternary models on a full-shot website build,
  but produced 80+ errors / 32K tokens (mechanical code errors: broken links,
  missing assets — second-pass failures, not intelligence failures)
- A reasoning block transplanted from Qwen3.8-Flash-Next let Bonsai attempt
  a far more ambitious build (it could EXECUTE hard plans it couldn't
  formulate) — the ternary network has compute headroom; planning depth is
  the gap
- Qwen3.8-Flash-Next (Aug 26) validated Engram-style n-gram conditional
  memory at frontier scale (51B n-gram table, NVMe-streamed, `qwen4exp` in
  llama.cpp)
- Key insight: ternary compression is a FIT for reasoning (robust/redundant
  — a near-miss word in a plan is survivable) but adversarial for code
  execution (one wrong token = broken build). Compress the reasoning, keep
  the execution surgical.

## Architecture

```
[120B ternary MoE backbone — streamed, offloaded experts, ~31GB @ 2.06bpw]
        ↓ (hidden states — already encode the plan/reasoning)
[~3B dense coder head — fp8, ALWAYS resident in VRAM, ~3GB]
        ↓
[LM head + n-gram exactness lookup (system RAM)]
```

- Backbone: ternary MoE, QAT-native, distilled from frontier reasoner CoT
  traces. ~6B active per token, experts offloaded to 32GB DDR4, hot cache
  in VRAM (LRU, ~70% hit rate).
- Coder head: dense transformer block (3B) trained specifically on
  self-correction code trajectories. Precision comes from fp8 weights
  (exact tokens), not from scale. Grafted onto the FROZEN backbone
  (mmproj-style — precedent: Bonsai's own vision tower pattern).
- Shared: ONE context, ONE KV cache, ONE n-gram table, ONE llama-server.
  No handoff latency, no duplicate embeddings, no state transfer.

## VRAM budget (12GB card)

| Component | GB |
|---|---|
| Backbone active layers (MoE, streamed) | ~1.5 |
| Dense coder head (always resident) | 3.0 |
| Shared KV cache | 1.5 |
| Hot expert cache (LRU) | 3-4 |
| Overhead/scratch | 1.0 |
| **Total** | **~10GB of 12GB** ✓ |

Backbone weights (~31GB) stream from 32GB DDR4-3200 (51GB/s); active
experts load per token (2-4GB/token → 40-80ms → 15-30 tok/s with hot cache).

## Training plan (3 stages)

1. **Backbone pre-train**: ternary MoE on reasoning traces distilled from
   a frontier reasoner (CoT traces transfer reasoning better than standard
   instruction distillation per 2026 research)
2. **Graft + train the coder head** on self-correction code trajectories
   (generate-with-errors → reread → repaired) — backbone FROZEN, it is the
   feature extractor. Seed data: Bonsai's own error logs from community
   testing (80 mechanical errors + fixes)
3. **Optional light joint QAT**: brief backbone+head fine-tune so backbone
   representations adapt to the head

## Why this beats the alternatives

| Approach | Reasoning quality | Code precision | VRAM | Deployment |
|---|---|---|---|---|
| Two separate models (Duo-v1) | Frontier-ish (large planner) | High | 12+12GB, 2 servers | Handoff latency, duplicated state |
| Single dense 3B | Weak (too small for planning) | High | 8GB | One process |
| **Bonsai-Duo (this)** | **Frontier-scale (ternary MoE backbone)** | **Surgical (fp8 head)** | **~10GB, one process** | One file, one server |

## Competitive context
- Prism ML Ternary-Bonsai-27B: general model ternarized; agentic coding
  explicitly deferred to roadmap (community testing confirms the gap)
- Qwen3.8-Flash-Next: Engram n-gram table = MEMORY, not reasoning weights
- NOBODY has shipped: ternary reasoning-specialist MoE + dense coder head
  as one package, QAT-native. Open territory.

## Implementation path
- llama.cpp supports the arch family (`qwen4exp`): hybrid attention, MoE
  layers, per-layer embeddings (PLE), MTP heads
- mmproj grafting pattern already exists in llama.cpp (vision towers) —
  the coder head graft is the same mechanism, applied to output
- Serve via llama-server with `--jinja` (reasoning_content separation
  provides the routing signal for free)
- Training: Soup/custom QAT pipeline (proven in this repo), bitsandbytes
  for the dense head, Hub checkpointing via CheckpointManager

## Prerequisites (gates)
- Phase A complete (single-model ternary QAT pipeline proven) — in progress
- Phase B: 1.5B ternary QAT + export + llama.cpp coherence gate
- Phase C: streaming equivalence validated
- Gate 0: confirm llama.cpp qwen4exp/TQ support for the target arch
- THEN: Bonsai-Duo becomes the Phase E deliverable
