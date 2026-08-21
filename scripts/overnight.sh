#!/bin/bash
# Overnight training: pre-train then SFT, then test the chatbot.
# Run with: nohup bash scripts/overnight.sh > overnight.log 2>&1 &

set -e
cd /home/ai/bonsai-llama
export HF_HOME=/home/ai/bonsai-llama/.hf_cache
export PYTHONUNBUFFERED=1
PY=/home/ai/bonsai-llama/.venv/bin/python
DISTILL=/home/ai/bonsai-llama/.venv/bin/bonsai-distill

echo "============================================"
echo "  OVERNIGHT TRAINING PIPELINE"
echo "  Start: $(date)"
echo "============================================"

# Stage 1: Pre-training (10000 steps, ~80 min)
echo ""
echo "[Stage 1] Pre-training distillation (10000 steps)..."
$DISTILL \
  --teacher Qwen/Qwen3-0.6B \
  --student Qwen/Qwen3-0.6B \
  --mode 1.58b \
  --teacher-device cuda:0 \
  --student-device cuda:0 \
  --teacher-dtype bfloat16 \
  --dataset wikitext \
  --batch-size 2 \
  --seq-len 1024 \
  --grad-accum 4 \
  --max-steps 10000 \
  --warmup-steps 200 \
  --lr 3e-4 \
  --log-every 100 \
  --save-every 2000 \
  --out bonsai-overnight-pretrain

echo "[Stage 1] Complete: $(date)"

# Stage 2: SFT on Dolly 15K (3000 steps, ~30 min)
echo ""
echo "[Stage 2] SFT on Dolly 15K (3000 steps)..."
$DISTILL \
  --student Qwen/Qwen3-0.6B \
  --init-from bonsai-overnight-pretrain \
  --mode 1.58b \
  --sft \
  --sft-dataset dolly \
  --system-prompt "You are a helpful assistant. Answer questions clearly and concisely." \
  --student-device cuda:0 \
  --batch-size 2 \
  --seq-len 1024 \
  --grad-accum 4 \
  --max-steps 3000 \
  --warmup-steps 100 \
  --lr 1e-4 \
  --log-every 50 \
  --save-every 1000 \
  --out bonsai-overnight-chat

echo "[Stage 2] Complete: $(date)"

# Stage 3: Test the chatbot
echo ""
echo "[Stage 3] Testing chatbot..."
$PY -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained('bonsai-overnight-chat')
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained('bonsai-overnight-chat', torch_dtype=torch.float32).to('cuda:0')
model.eval()

tests = [
    'Hello! How are you?',
    'What is the capital of France?',
    'Write a short poem about the ocean.',
    'How do I make a cup of tea?',
]

for prompt in tests:
    full = '<|im_start|>system\nYou are a helpful assistant. Answer questions clearly and concisely.<|im_end|>\n<|im_start|>user\n' + prompt + '<|im_end|>\n<|im_start|>assistant\n'
    inputs = tok(full, return_tensors='pt').to('cuda:0')
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, temperature=0.7,
                             do_sample=True, top_p=0.9, top_k=50,
                             pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id,
                             repetition_penalty=1.15)
    new = out[0][inputs['input_ids'].shape[1]:]
    resp = tok.decode(new, skip_special_tokens=True).strip()
    for tag in ['<|im_start|>', '<|im_end|>']:
        resp = resp.replace(tag, '')
    resp = resp.strip()
    print(f'You: {prompt}')
    print(f'Assistant: {resp}')
    print('---')
"

echo ""
echo "============================================"
echo "  OVERNIGHT PIPELINE COMPLETE"
echo "  End: $(date)"
echo "============================================"
echo ""
echo "To chat with the model:"
echo "  .venv/bin/python scripts/chat.py --model bonsai-overnight-chat"
