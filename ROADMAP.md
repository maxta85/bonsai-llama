# Bonsai Roadmap — Plan to Delivered Product
Written 2026-09-12. Maps the full path from current state to a deliverable,
including compute grants. Supersedes all prior roadmaps as the master plan.

## Stage 1 — PLAN (DONE)
- Architecture designed: Bonsai-Duo (BONSAI-DUO-SPEC.md)
- Method validated by prior art: Prism ML (ternary QAT works, 94.6% retention)
  + DeepSeek Engram paper (arXiv 2601.07372) + Qwen3.8-Flash-Next (51B n-gram
  table, qwen4exp in llama.cpp)
- Gaps identified: agentic coding QAT is open territory (community testing)
- Failure modes mapped: 80+ errors/32K tokens = token-exactness gap that
  ternary weights can't close alone → dense coder head + n-gram exactness layer

## Stage 2 — PROOF OF CONCEPT (IN PROGRESS)
Goal: prove the training pipeline produces a coherent small ternary model.

### 2a. Phase A (NOW): 1.5B QLoRA coherent chat — the pipeline proof
- Detached training running on Colab T4 via colab-train-watch daemon
- Checkpoint/resume PROVEN (cross-session, cross-account, HF Hub)
- Deliverable: 1.5B model, loss < 2.0, coherent on eval prompts
- Status: step 180/2002, running, watcher-managed

### 2b. Phase B: ternary QAT at 1.5B
- BitLinear QAT on 1.5B (already built: bit_linear.py)
- Recovery data: general chat + CODING TRAJECTORIES (per community testing)
- Deliverable: ternary 1.5B GGUF in llama.cpp, coherent
- GATE B = METHOD PROVEN

### 2c. Phase C: streaming equivalence (infrastructure for scale)
- Streamed == full-load (bridge already built, tests green)
- Gate C: A/B comparison on 0.5B oracle

## Stage 3 — TRAINING PLATFORM
Goal: a repeatable training infrastructure, not one-off Colab hacks.

### 3a. Self-hosted (already have)
- apiserver: 2×3060 + 512GB RAM + NVMe — Phase B/C experiments, n-gram table
  (currently offline, hardware repair in progress: channel fault + new CPU)
- myserver: 4070 Ti (FastContext co-located)

### 3b. Cloud burst (paid, small)
- Vast.ai / RunPod: ~$0.20-0.40/hr for A100/H100 instances
- Used for: Phase D resource gate (7-8B streaming validation), Phase E
  training runs that exceed consumer VRAM
- Budget: ~$50-200 total depending on measured requirements

### 3c. Platform decision criteria
- Reliability: sessions survive ≥24h, no arbitrary reclaims
- Throughput: tokens/sec at ternary quantization
- Checkpoint-friendly: fast durable storage, pre-emptible OK with resume
- Cost per completed training run (measured, not estimated)

## Stage 4 — TRAINING DATA SOURCE
The differentiator. Community testing proved general chat isn't enough —
Phase B needs error-inclusive coding trajectories.

### 4a. Seed data (have)
- Bonsai 27B community testing error logs (80+ mechanical errors + fixes)
- 1,054 domain examples (persona + agent-skills from Training-Data repo)

### 4b. Self-correction trajectory generation
- Run current Bonsai/models on coding tasks → collect (task, errors, fix) pairs
- Compile/test feedback: actual error messages + repaired code
- Scale via the 10-account Colab pool (inference only — cheap)
- Target: 5-10K coding trajectories with error-repair patterns

### 4c. General reasoning distillation
- CoT traces from frontier reasoner (Flash-Next / Astra) via litellm/OpenRouter
- Per research: CoT distillation transfers reasoning better than standard SFT
- Target: 10-20K reasoning traces for backbone pre-train

### 4d. Public data
- Ultrachat-200k subset (MIT) for general chat base
- Filtered, pinned revision, provenance recorded

## Stage 4 — COMPUTE GRANT (apply)
Target: grant programs that fund open-weight model training.

### Candidates (research needed)
- **HF + Google TPU Research Cloud (TRC)**: free TPU pods for researchers,
  open-weight projects eligible — strong fit
- **AMD AI Accelerator program**: MI300X hours for open research
- **Stability AI compute grants**: for open model releases
- **EleutherAI / LAION partnerships**: existing open-source training collectives
- **Hugging Face community grants**: infra credits for open projects
- **University partnerships**: Cairns QLD — James Cook University has HPC;
  academic collaboration route
- **CoreWeave/lambda research credits**: case-by-case

### Application package (build when Stage 2 completes)
- Bonsai-Duo SPEC + whitepaper draft
- Phase A/B results as proof of method (the 1.5B ternary model + benchmarks)
- Community testing data (the 80-errors finding = the problem statement)
- Training plan with token budget, GPU-hours estimate
- Open-source commitment (Apache 2.0 like Bonsai)
- Track record: this repo (pipeline, tests, community findings)

### Realistic ask
- 500-2000 H100-hours for a 120B ternary MoE QAT run (estimate from
  measured throughput on smaller scales)
- OR: Vast.ai/RunPod self-funded ~$500-2000 if grants don't land

## Stage 5 — DELIVER
- Bonsai-27B-v2 (ternary, Engram-enhanced, agentic-coding-tuned)
- Bonsai-Duo (ternary reasoner + dense coder head, single package)
- GGUF for llama.cpp (consumer hardware, phones via 1-bit companion)
- Apache 2.0 open release + whitepaper
- Differentiator: first ternary model good at agentic coding

## Success metrics
| Metric | Target |
|---|---|
| Phase A: 1.5B coherent chat | loss < 2.0, 10 prompts coherent |
| Phase B: ternary 1.5B in llama.cpp | perplexity within 15% of QLoRA |
| Agentic coding: error rate | < 10 errors per 32K tokens (vs current 80+) |
| Bonsai-Duo VRAM | ≤ 12GB single card (consumer accessible) |
| Community | open release + whitepaper + benchmarks published |

## Current status summary
| Stage | Status |
|---|---|
| 1. Plan | ✅ DONE |
| 2. Proof of concept | 🔄 Phase A in progress (step 180/2002) |
| 3. Training platform | ⏳ Self-hosted partially ready; cloud burst TBD |
| 4. Training data | 🔄 Seed data built (1,054); coding trajectories needed |
| 4b. Compute grant | 📋 Package when Stage 2 completes |
| 5. Deliver | Blocked by 4b |
