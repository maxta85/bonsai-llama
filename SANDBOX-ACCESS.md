# Bonsai GPU Sandbox — Access Guide

Created 2026-09-13. Local GPU training sandbox on apiserver, replacing
flaky Colab T4s for Phase A/B training.

## Connection

```bash
# From this devcontainer:
ssh bonsai-sandbox
```

- **Host**: 100.113.148.14 (apiserver Tailscale IP)
- **Port**: 2224
- **User**: train (non-root)
- **Key**: ~/.ssh/id_ed25519_bonsai_sandbox
- **SSH config**: already in ~/.ssh/config (Host bonsai-sandbox)

## Hardware

- 1x RTX 3060 (12GB VRAM)
- 16GB RAM, 8 CPUs, 4GB shared memory
- CUDA 12.4, PyTorch 2.14+cu130

## Network access (restricted)

The sandbox can ONLY reach (via whitelist proxy):
- huggingface.co (HF Hub — model downloads, checkpoint uploads)
- pypi.org / files.pythonhosted.org (pip install)
- github.com (git clone)
- download.pytorch.org (PyTorch wheels)

Everything else is blocked: Tailscale, LAN, apiserver services, other
containers. This is a security boundary — the sandbox is untrusted.

## Workflow (Colab-like)

```bash
# 1. Upload training script + dataset
scp ops/train_release.py bonsai-sandbox:~/
scp /tmp/bonsai_sft_merged.jsonl bonsai-sandbox:~/train.jsonl

# 2. SSH in
ssh bonsai-sandbox

# 3. Set your HF token (bring your own — NOT apiserver's)
export HF_TOKEN=hf_xxx

# 4. Run training
python train_release.py

# 5. Checkpoints go to HF Hub (maxta85/bonsai-checkpoints)
#    Workspace persists in /workspace (named volume, survives restart)

# 6. When done, just exit. Container stays running.
#    To reset: ask Max to run recreate.sh on apiserver.
```

## File locations in sandbox

- /home/train/ — home directory (scripts, datasets land here)
- /workspace/ — persistent named volume (survives container restart)
- /workspace/.hf_cache/ — HF model cache

## What the sandbox CANNOT do

- Reach any Tailscale node (myserver, gaming-system, main-rig, etc.)
- Reach apiserver services (SSH, LLM inference, etc.)
- Access apiserver filesystem (no bind mounts)
- Use the PRO 6000 or the second 3060 (only GPU 1 passed through)
- Install packages from non-whitelisted repos

## Ops (apiserver side, for Max)

- Recreate after reboot: `sudo bash /home/ai/bonsai-sandbox/recreate.sh`
- Destroy: `docker rm -f bonsai-sandbox bonsai-proxy bonsai-ssh-relay`
- Logs: `docker logs bonsai-sandbox`
- Proxy logs: `docker logs bonsai-proxy`
- Full docs: ~/.agent-memory/tooling/bonsai-sandbox.md
