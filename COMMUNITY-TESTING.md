# Community Testing Log — Ternary-Bonsai-27B (Max, 2026-09-12)

## Test: Full-shot travel website build (agentic coding workload)

**Setup:** Ternary Bonsai 27B (Q2_g64 GGUF, llama.cpp) vs a competitor ternary
model built on Qwen3.8-27B with 55 hours of H200 training. Both tasked with
building a complete travel website.

## Results

- **Bonsai vs competitor:** Bonsai clearly outpaced the Qwen3.8-based ternary
  in coherence, structure, and reasoning across the build.
- **Bonsai's weakness confirmed:** unable to produce bug-free code —
  **over 80 errors accumulated across ~32K output tokens**.
- **Verdict:** Bonsai is the strongest ternary model tested, but agentic
  multi-file code generation remains its hard failure mode. Matches the
  vendor's own "Limitations" note (agentic coding not a strong target;
  coding-tuned variant deferred to future roadmap).

## Analysis

The failure mode is precision-driven, not capability-driven: 80+ errors over
32K tokens ≈ error accumulation in long-horizon code synthesis, where each
token must be exact and errors compound through the build. Prompting tricks
(thinking mode, retries) get close but do not close the gap.

## Implication for Bonsai-27B-v2

1. Agentic coding QAT (multi-file, run-test-repair trajectories) is the
   open differentiator — Prism ML has NOT shipped this yet.
2. Recovery training data must include long-horizon code tasks, not just
   general chat — otherwise Bonsai-v2 will inherit the same weakness.
3. Compile/test feedback loops (error signals in training data) are the
   missing ingredient: the model must learn from its own error patterns.

## Relevance to our pipeline
Phase B recovery data plan must weight coding trajectories heavily.
The Engram/n-gram conditional-memory idea (see Astra review + Qwen3.8-Flash-
Next) is complementary: lookup tables handle exactness (syntax, API names,
imports) that ternary weights struggle to encode at 1.7 bpw.
