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

## Follow-up test (same session): reasoning-block transplant

Round 1: Bonsai alone produced a vanilla site — main pages only, small errors
(broken image links, no working navigation links). Coherent but shallow.

Round 2: pasted the REASONING BLOCK from Qwen3.8-Flash-Next (same prompt) into
Bonsais

## Follow-up test (same session): reasoning-block transplant

Round 1: Bonsai alone produced a vanilla site — main pages only, small errors
(broken image links, no working navigation links). Coherent but shallow.

Round 2: pasted the REASONING BLOCK from Qwen3.8-Flash-Next (same prompt) into
Bonsai's context. Result: a substantially more ambitious, impressive-but-broken
website — bigger scope, more features, ~80 errors accumulated across ~32K
output tokens. After hand-correcting the 80 errors the site looked genuinely
good.

## Two takeaways

1. **The reasoning block transfer worked.** Bonsai's weakness is planning/
   scope, not just precision — given Flash-Next's reasoning as scaffolding, it
   attempted a far more ambitious build. The ternary network can EXECUTE a
   hard plan it couldn't formulate itself.

2. **Self-review pass would catch most of the 80 errors.** The errors were
   mechanical (broken links, missing assets) — a harness loop that makes the
   model reread its own output (or run a linter/screenshot check) before
   finalizing would likely fix the majority. The model has the capability to
   write good code; it lacks the second-pass discipline without a harness.

## Pipeline implication

A harness loop (generate -> reread -> fix -> repeat) + reasoning-scaffolding
transplant is the practical recipe for agentic coding with Bonsai TODAY,
before any retraining. For Bonsai-v2 QAT: training data should include
self-correction trajectories (generate with errors -> reread -> repaired), not
just clean code — teaching the second pass as a behavior, not hoping it
emerges.
