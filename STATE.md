# STATE — bonsai-llama (canonical project state; compression-proof)

Updated: 2026-09-06. Rule: updated at milestones by the agent; a fresh session
resumes from THIS FILE + TASKS.json, never from chat history.

## Goal
Replicate Prism-ML Ternary-Bonsai-27B (base Qwen3.6-27B) one generation forward:
Bonsai-27B-v2 on Qwen3.8-27B, ternary, TQ GGUF, coherent in llama.cpp on
consumer GPUs. Prove method at 1.5B (coherent chat) first. Free Colab T4s
(3 accounts) + minimal paid budget.

## Current phase
Phase A ops work (per Godmode review-2). NOT authorized: Phase D/E.

## Decisions (durable)
- Teacher-free SFT+QAT route (KD plumbing deferred; matches Prism-ML precedent)
- LoRA merged into fp base BEFORE QAT init; no floating adapters in ternary export
- Export semantics (llama.cpp compat) decided BEFORE ternary training spend
- Prefer llama.cpp upstream converter/quantizer over custom packing
- 5-min atomic checkpoints to private HF Hub; resume-not-restart
- Streaming is Phase C; must not block non-streamed Phase B
- fp16 not bf16 on T4; gradient checkpointing AFTER peft wrap; use_cache off

## Blockers
- T4 availability flaky (503s); rotation over 3 accounts implemented
- Astra: full-load 1.5B QAT may NOT fit T4 (fp32 masters+moments ~24GB) —
  oracle comparison may need 0.5B

## Reviews on file
- Sonnet x4: /home/coder/projects/bonsai-llama/SONNET-REVIEW.md (shippable verdict, pre-Godmode)
- Godmode r1: /tmp/godmode_review/GODMODE-REVIEW.md (prototype verdict, 4 BLOCKERs)
- Godmode r2: /tmp/godmode_review/GODMODE-REVIEW-2.md (gated approval of v2 plan)

## Next action
Gate 0: pin HF revisions (Qwen2.5-1.5B-Instruct, Qwen3.8-27B), verify Prism-ML
recipe details, verify llama.cpp TQ1_0/TQ2_0 support at pinned revision.
