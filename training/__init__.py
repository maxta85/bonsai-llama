"""bonsai training: quantization-aware training (QAT) + distillation.

Distills an open-weights FP16/BF16 model (e.g. Qwen3-8B) into either the
1-bit (Q1_0) or ternary 1.58-bit (Q2_0) packed format, following the
BitNet b1.58 recipe:

  * Replace every ``nn.Linear`` with a :class:`BitLinear` that fake-quantizes
    weights to {-1,0,+1} (ternary) or {-1,+1} (1-bit) with abs-mean group
    scaling, using a straight-through estimator (STE) so gradients flow.
  * Train against a teacher's logits (KL) + the ground-truth CE
    (knowledge distillation).
  * Master weights stay FP32 in the optimizer; only the forward pass uses
    the quantized weights.
  * Export the trained ternary/1-bit weights to GGUF Q1_0 / Q2_0.

References
----------
* Wang et al., "BitNet: Scaling 1-bit Transformers for LLMs", arXiv:2310.11453
* Ma et al., "The Era of 1-bit LLMs ... 1.58 Bits", arXiv:2402.17764
* "Training 1.58bit LLMs via Distillation"
  (https://github.com/leszkolukasz/training-1.58bit-llms-via-distillation)
* Prism ML Bonsai whitepapers (see docs/whitepapers/)
"""
