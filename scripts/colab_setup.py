"""Self-contained setup for Colab — no GitHub clone needed.

Upload this file + the bonsai-llama folder to Colab, then run:
    !python colab_setup.py
"""

import subprocess
import sys
import os

def run(cmd, check=True):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout[:2000])
    if result.returncode != 0 and check:
        print(f"ERROR: {result.stderr[:1000]}")
        sys.exit(1)
    return result

print("=== Bonsai Colab Setup ===")
print()

# 1. Check GPU
try:
    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        cap = torch.cuda.get_device_capability(0)
        print(f"Compute capability: {cap}")
        if cap[0] >= 9:
            print("FP8 supported (Hopper+) — can use --fp8")
        else:
            print("FP8 NOT supported — will use FP32 BitLinear")
    else:
        print("WARNING: No GPU detected! Enable GPU in Runtime settings.")
except ImportError:
    print("PyTorch not installed yet")

print()

# 2. Install dependencies
print("Installing dependencies...")
run("pip install -q torch transformers datasets bitsandbytes sentencepiece huggingface_hub")

# 3. Check if bonsai-llama is available
if os.path.exists("bonsai-llama"):
    print("\nFound bonsai-llama/ directory")
    os.chdir("bonsai-llama")
    run("pip install -q -e \".[train,dev]\"")
    print("\n✓ bonsai-llama installed from local directory")
elif os.path.exists("training"):
    print("\nAlready in bonsai-llama directory")
    run("pip install -q -e \".[train,dev]\"")
    print("\n✓ bonsai-llama installed")
else:
    print("\nERROR: bonsai-llama/ not found!")
    print("Upload the bonsai-llama folder to Colab:")
    print("  1. Click the folder icon in the left sidebar")
    print("  2. Click 'Upload to session storage'")
    print("  3. Upload the bonsai-llama folder (or a zip of it)")
    print("  Then re-run this script")
    sys.exit(1)

# 4. Quick smoke test
print("\n=== Smoke test ===")
try:
    from training.bit_linear import BitLinear
    import torch
    layer = BitLinear(128, 128, mode="1.58b")
    x = torch.randn(2, 10, 128)
    out = layer(x)
    print(f"✓ BitLinear works: input {x.shape} → output {out.shape}")
except Exception as e:
    print(f"✗ Smoke test failed: {e}")

print("\n=== Setup complete ===")
print("Next: run training with:")
print("  !python -m training.distill --teacher Qwen/Qwen3-0.6B \\")
print("    --student Qwen/Qwen3-0.6B --mode 1.58b \\")
print("    --batch-size 4 --seq-len 1024 --max-steps 500 \\")
print("    --topk 100 --topk-kl --chunked-loss \\")
print("    --8bit-adam --grad-checkpoint \\")
print("    --out bonsai-checkpoint")
