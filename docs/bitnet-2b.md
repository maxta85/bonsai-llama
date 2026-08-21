# BitNet b1.58 2B4T — the reference ternary model

`microsoft/bitnet-b1.58-2B-4T` is the first open-source, **native 1-bit** LLM
at the 2B scale, trained from scratch by Microsoft Research on 4 trillion
tokens. It is the canonical reference for the ternary `{-1, 0, +1}` weight
scheme this repo implements, and a useful teacher / sanity-check target for
the distillation pipeline.

- Model: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T
- Technical report: https://arxiv.org/abs/2504.12285
- Official inference engine: https://github.com/microsoft/bitnet (bitnet.cpp)
- License: MIT

## Model variants

| Repo | Format | Use for |
| --- | --- | --- |
| `microsoft/bitnet-b1.58-2B-4T` | packed 1.58-bit (HF) | **deployment** |
| `microsoft/bitnet-b1.58-2B-4T-bf16` | BF16 master weights | **training / fine-tuning** |
| `microsoft/bitnet-b1.58-2B-4T-gguf` | GGUF | **bitnet.cpp CPU inference** |

## Architecture

- Transformer with **`BitLinear`** layers (BitNet framework) — every Linear
  fake-quantizes weights to ternary `{-1,0,+1}` via absmean, with a
  straight-through estimator. This is exactly what this repo's
  `training/bit_linear.py` implements.
- RoPE positional embeddings.
- **ReLU²** activation in FFN (squared ReLU).
- `subln` normalization.
- **No bias terms** in linear or normalization layers.
- **W1.58A8**: ternary weights + 8-bit activations (per-token absmax).
- ~2B parameters, 4096 context, LLaMA 3 tokenizer (vocab 128,256).

## Training stages (from the technical report)

1. **Pre-training** — large-scale public text/code + synthetic math, two-stage
   LR and weight-decay schedule. **Trained from scratch with the quantization
   active** (not post-training quantized).
2. **SFT** — instruction-following + conversational data, sum loss aggregation.
3. **DPO** — human-preference alignment.

This is the recipe to reproduce for a from-scratch ternary model. For
distillation from an existing FP base (the Bonsai approach), see
[`REPRODUCTION.md`](REPRODUCTION.md).

## How to run it

### bitnet.cpp (recommended — gives the real speed/energy wins)

```bash
./scripts/setup_bitnet_cpp.sh

# Download the GGUF variant
hf download microsoft/bitnet-b1.58-2B-4T-gguf --local-dir models/bitnet-2b

# Run
./build/bitnet_cpp/bin/bitnet -m models/bitnet-2b/bitnet-b1.58-2b-4t.gguf -p "Hello"
```

> **Do not use `transformers` for efficiency.** The model card explicitly
> warns that the transformers path gives no speed/latency/energy benefit —
> only `bitnet.cpp` has the optimized I2_S / TL1 kernels. transformers is
> fine for correctness checks / fine-tuning.

### transformers (correctness / fine-tuning only)

Requires a specific transformers fork:
```bash
pip install git+https://github.com/huggingface/transformers.git@096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/bitnet-b1.58-2B-4T"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

messages = [{"role": "user", "content": "How are you?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

## Using it as a reference / teacher in this repo

The BF16 master-weights variant (`bitnet-b1.58-2B-4T-bf16`) can serve as a
**ternary-aware teacher** in `training/distill.py`: it already produces
ternary logits, so distilling a fresh student into it transfers the
quantized representation directly.

```bash
bonsai-distill --teacher microsoft/bitnet-b1.58-2B-4T-bf16 --mode 1.58b --out bonsai-from-bitnet
```

It is also the best **sanity-check checkpoint**: load it, run our
`bonsai_tensor.quantize_q2_0` on its weights, and confirm the round-trip
matches the packed variant's outputs.

## Evaluation highlights (vs similar-size FP models)

| Metric | BitNet b1.58 2B | LLaMA 3.2 1B | Qwen2.5 1.5B |
| --- | --- | --- | --- |
| Memory (non-emb) | **0.4 GB** | 2 GB | 2.6 GB |
| CPU decode latency | **29 ms** | 48 ms | 65 ms |
| Energy/token | **0.028 J** | 0.258 J | 0.347 J |
| Average benchmark | 54.19 | 44.90 | **55.23** |

Comparable quality at ~6x less memory, ~1.6x lower latency, ~9x less energy.
