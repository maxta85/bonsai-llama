# STATE — bonsai-llama (canonical project state; compression-proof)

Updated: 2026-09-10. Rule: updated at milestones by the agent; a fresh session
resumes from THIS FILE + TASKS.json, never from chat history.

## Goal
Replicate Prism-ML Ternary-Bonsai-27B (base Qwen3.6-27B) one generation forward:
Bonsai-27B-v2 on Qwen3.8-27B, ternary, TQ GGUF, coherent in llama.cpp on
consumer GPUs. Prove method at 1.5B (coherent chat) first.

## Current phase
**Phase A RUN IN PROGRESS** — detached training on Colab T4, auto-managed by
the colab-train-watch daemon (supervisord). Resumes from Hub checkpoint on
any session death. Gate A: run completes (2,002 steps) + 10 frozen prompts
judged coherent by Max.

## Verdict recorded (2026-09-08): FREE COLAB = NOT VIABLE as primary substrate
- Sessions died ~200 steps in; 3-account rotation exhausted; 11/12 loop cycles
  got "no T4"
- BUT: timeline analysis later proved some "reclaims" were misdiagnosed — the
  CLI viewer websocket dropped while VMs kept training server-side. Lesson:
  treat connection drops as VIEWER loss; run training DETACHED on the VM and
  check the log/Hub, never assume death.
- Colab remains usable for opportunistic runs via the watcher; NOT the
  reliability substrate. The Astra capacity gate answered: free tier fails.

## What the Colab period produced (durable assets)
1. Pipeline complete + tested: BitLinear QAT, chunked KL, ternary packing,
   GGUF export, layer streaming + bridge (streamed == full-load, 33/33 tests)
2. Checkpoint/resume machinery PROVEN — cross-session, cross-account resume
   from HF Hub (maxta85/bonsai-checkpoints, private)
3. colab-train-watch daemon (supervisord): keeps detached training alive
   across session deaths; auto-rotates accounts; exits on completion
4. 10-account Colab rotation, all verified (see ACCOUNTS below)
5. Proven training config: Qwen2.5-1.5B QLoRA fp16, 4.30GB peak VRAM,
   loss 3.05->2.62 in 40 steps, batch=1 seq=512 grad-ckpt after peft wrap
6. Dataset: 1,054 examples (bonsai persona + agent-skills), deduped

## Account pool (auth tokens)
SAVED IN THE DEVCONTAINER ONLY — deliberately NOT in this repo (secrets).
Location: devcontainer /home/coder/.config/colab-cli/accounts/
- 10 accounts, all verified via Google userinfo API, manifest.json rebuilt
- cyno5guy, mazta85, evetoono2, proboardtest111, max.telford85, ultraevetoon,
  toonholder, telfordautomotive, maxta85catchall, maxxcomputerscns
- Rotation: try-t4.sh v3 auto-discovers accounts/*.json
- HF token (write, maxta85) also in devcontainer .cache/huggingface/token —
  required for Hub checkpoint uploads

## Decisions (durable)
- Teacher-free SFT+QAT route (KD plumbing deferred; matches Prism-ML precedent)
- LoRA merged into fp base BEFORE QAT init; no floating adapters in ternary export
- Export semantics (llama.cpp compat) decided BEFORE ternary training spend
- Prefer llama.cpp upstream converter/quantizer over custom packing
- Checkpoints every 10 steps to HF Hub, atomic, keep 2; resume-not-restart
- fp16 not bf16 on T4; gradient checkpointing AFTER peft wrap; use_cache off
- Training runs DETACHED on VM (setsid nohup, log at /content/train.log);
  viewer connections are disposable — never interpret stream drop as VM death

## Blockers
- None currently. Training detached + watcher + 10-account pool.

## Reviews on file
- Sonnet x4: SONNET-REVIEW.md (pre-Godmode, "shippable" — later superseded)
- Godmode r1: /tmp/godmode_review/GODMODE-REVIEW.md (prototype verdict, 4 BLOCKERs)
- Godmode r2: /tmp/godmode_review/GODMODE-REVIEW-2.md (gated approval of v2 plan)
- All Godmode BLOCKERs fixed: stream bridge (a8467d6), checkpoint/resume +
  causal shift (d3f88fb)

## Next action
Monitor detached run to completion (watcher handles it). On completion:
capture coherence samples, Gate A judgment by Max, then Phase B ternary work.

## Ops quick-reference (devcontainer)
- Watch daemon: supervisorctl status colab-train-watch; log /tmp/train_watch.log
- Rotation: /home/coder/.config/colab-cli/try-t4.sh (v3, auto-discovers accounts)
- Colab VM paths: /content/train_detached.py (script), /content/train.jsonl
  (dataset), /content/train.log (live log), /content/ckpt (local fallback)
- Add account: ~/.config/colab-cli/add-account.sh (browser must be signed OUT
  or incognito — otherwise it re-auths the current account)