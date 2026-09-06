"""tests/test_layer_stream.py

Validate that StreamingModel's layer-by-layer streaming produces output
identical to a plain full-load forward.

Per LAYER-STREAMING-SPEC.md the gold standard is Qwen2.5-0.5B with a
bit-exact cpu->gpu roundtrip in eval mode. This box may have no CUDA, so:

  * With TEST_USE_QWEN=1 and network access we run Qwen/Qwen2.5-0.5B.
  * Default fallback: a tiny random-weight LlamaConfig model — identical
    math paths, still exercising pinning, snapshots, double-buffer copies,
    freezing and equivalence.
"""

import os

import pytest
import torch


def _build_tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(42)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=172,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    return LlamaForCausalLM(cfg)


def _maybe_load_qwen():
    """Return Qwen/Qwen2.5-0.5B if obtainable quickly enough, else None."""
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download("Qwen/Qwen2.5-0.5B", allow_patterns=[
            "*.json", "*.safetensors"])
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(path)
    except Exception:
        return None


@pytest.fixture(scope="module")
def model_pair(request):
    qwen = None
    if os.environ.get("TEST_USE_QWEN", "0") == "1":
        qwen = _maybe_load_qwen()
    model = qwen if qwen is not None else _build_tiny_llama()
    model.eval()

    # Pristine fingerprint: verify streaming never mutated the wrapped model.
    fp = {k: v.detach().clone() for k, v in model.state_dict().items()}

    from training.layer_stream import StreamingModel
    sm = StreamingModel(model, compute_device="cpu")
    sm.eval()
    return model, sm, fp


@torch.no_grad()
def test_output_equivalence_eval_mode(model_pair):
    """Streaming forward == full-load forward, bit-for-bit (CPU no-op copies)."""
    base, sm, _ = model_pair
    ids = torch.randint(0, base.config.vocab_size, (2, 16))
    ref = base(input_ids=ids).logits
    out = sm.forward_with_streaming(ids)
    assert out.logits.shape == ref.shape
    torch.testing.assert_close(out.logits, ref, rtol=0, atol=0)


@torch.no_grad()
def test_output_equivalence_with_padding_mask(model_pair):
    """Equivalence holds when an attention mask excludes padding tokens."""
    base, sm, _ = model_pair
    ids = torch.randint(0, base.config.vocab_size, (3, 12))
    am = torch.ones_like(ids)
    am[:, -4:] = 0  # pad tail
    ref = base(input_ids=ids, attention_mask=am).logits
    out = sm.forward_with_streaming(ids, attention_mask=am)
    torch.testing.assert_close(out.logits, ref, rtol=0, atol=0)


@torch.no_grad()
def test_double_buffer_weights_restored_each_layer(model_pair):
    """Scratch buffers must carry each layer's exact weights at call time:
    two passes rotate buffer<->layer pairing; equality both times catches
    buffer-reuse clobbering."""
    base, sm, _ = model_pair
    ids = torch.randint(0, base.config.vocab_size, (1, 8))
    for _ in range(2):
        out = sm.forward_with_streaming(ids)
        ref = base(input_ids=ids).logits
        torch.testing.assert_close(out.logits, ref, rtol=0, atol=0)


@torch.no_grad()
def test_base_model_untouched_by_streaming(model_pair):
    """After several streamed forwards the wrapped base weights are exactly
    as they started (streaming reads snapshots, never writes back)."""
    base, sm, fp = model_pair
    ids = torch.randint(0, base.config.vocab_size, (1, 8))
    for _ in range(3):
        sm.forward_with_streaming(ids)
    cur = base.state_dict()
    assert set(cur) == set(fp)
    for k, v in fp.items():
        torch.testing.assert_close(cur[k], v, rtol=0, atol=0)


@torch.no_grad()
def test_bitlinear_hook_applied(model_pair):
    """bitlinear_fn transforms every hidden state (embed + post-layer).

    Reference is built by monkeypatching the ORIGINAL model's embed/norm/
    lm_head and every decoder layer with hook-wrapped wrappers, then doing
    a normal full-model forward — the safest way to replicate the pipeline.
    """
    base, sm, _ = model_pair
    scale = 0.5
    hook = lambda t: t * scale  # noqa: E731
    ids = torch.randint(0, base.config.vocab_size, (1, 8))

    core = base.model
    saved = {"emb": core.embed_tokens, "norm": core.norm}
    saved_layers = list(core.layers)

    class WrapOut(torch.nn.Module):
        """Hook the module OUTPUT (embed/norm operate on hidden states)."""

        def __init__(self, inner, use_hook):
            super().__init__()
            self.inner = inner
            self.use_hook = use_hook

        def forward(self, x, *a, **kw):
            out = self.inner(x, *a, **kw)
            if self.use_hook and isinstance(out, torch.Tensor):
                out = hook(out)
            return out

    class WrapLayer(torch.nn.Module):
        """Hook ONLY the layer output — matches StreamingModel semantics
        (previous layer's hooked output flows straight in as the next
        input; no second application)."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x, *a, **kw):
            out = self.inner(x, *a, **kw)
            return hook(out[0]) if isinstance(out, tuple) else hook(out)

    try:
        core.embed_tokens = WrapOut(saved["emb"], True)
        core.norm = WrapOut(saved["norm"], False)
        new_layers = torch.nn.ModuleList([WrapLayer(l) for l in saved_layers])
        core.layers = new_layers
        ref = base(input_ids=ids).logits
    finally:
        core.embed_tokens = saved["emb"]
        core.norm = saved["norm"]
        core.layers = torch.nn.ModuleList(saved_layers)

    out = sm.forward_with_streaming(ids, bitlinear_fn=hook)
    # Streamed-with-hook matches hooked reference exactly.
    torch.testing.assert_close(out.logits, ref, rtol=0, atol=0)


def test_base_frozen_and_residents_on_device(model_pair):
    """Base requires_grad=False; embed/norm/lm_head live on compute device."""
    _, sm, _ = model_pair
    for p in sm.inner.parameters():
        assert p.requires_grad is False
    dev = sm.compute_device
    for m in sm.resident_modules:
        for p in m.parameters():
            assert p.device.type == dev.type
    snap = sm.cpu_layers[0]
    first = next(iter(snap.values()))
    assert first.device.type == "cpu"


def test_trainable_addon_receives_grad(model_pair):
    """A trainable resident add-on (LoRA-style side adapter on lm_head) gets
    gradients while base stays frozen."""
    base, sm, _ = model_pair
    dev = sm.compute_device
    lora_down = torch.nn.Linear(base.config.hidden_size, 8, bias=False,
                                device=dev)
    lora_up = torch.nn.Linear(8, base.config.vocab_size, bias=False,
                              device=dev)
    # NOTE: deliberately NOT zero-initialising lora_up — with W_up = 0 the
    # down-projection's gradient is identically zero (W_down.grad =
    # W_up^T @ g), which would make the "gradients flow" check vacuous.

    orig_lm_head = sm.lm_head

    class AdapterHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.down = lora_down
            self.up = lora_up
            self.base_weight = orig_lm_head.weight

        def forward(self, h):
            delta = self.up(self.down(h))
            return torch.nn.functional.linear(h, self.base_weight) + delta

    try:
        sm.lm_head = AdapterHead()
        ids = torch.randint(0, base.config.vocab_size, (1, 8))
        out = sm.forward_with_streaming(ids)
        out.logits.sum().backward()
        # LoRA params got grads...
        assert lora_down.weight.grad is not None
        assert lora_down.weight.grad.abs().sum() > 0
        assert lora_up.weight.grad is not None
        # ...base stayed frozen (never received grad).
        for name, p in base.named_parameters():
            assert not p.requires_grad or p.grad is None, name
    finally:
        sm.lm_head = orig_lm_head

def test_streamed_decoder_grads_match_full_load(model_pair):
    """The core value proposition: gradients that reach streamed decoder-layer
    weights must equal what a full-load (all-on-device) reference produces.
    Compares per-parameter grads from streamed_grads() against a plain
    autograd pass on the unstreamed base model with all params trainable."""
    import torch as _t
    from training.layer_stream import StreamingModel
    torch.manual_seed(42)
    base = _build_tiny_llama()
    base.eval()
    dev = "cpu"
    sm = StreamingModel(base, compute_device=dev)

    # Unfreeze the reference model entirely and put it on the compute device.
    ref = base
    for p in ref.parameters():
        p.requires_grad_(True)
        p.grad = None
    ref = ref.to(dev)

    _t.manual_seed(1234)
    ids = _t.randint(0, ref.config.vocab_size, (2, 8))

    # Reference: full-load forward/backward.
    ref_out = ref(input_ids=ids.to(dev)).logits
    ref_out.sum().backward()

    # Streamed: same input, forward_with_streaming + tail-driven reverse pass.
    out = sm.forward_with_streaming(ids)
    out.logits.sum().backward()

    grads = sm.streamed_grads()
    assert grads, "streamed_grads() returned nothing"

    for li, fields in grads.items():
        layer_ref = ref.model.layers[li]
        for name, g in fields.items():
            # names are like "self_attn.q_proj.weight"
            ref_param = layer_ref
            for part in name.split("."):
                ref_param = getattr(ref_param, part)
            ref_grad = ref_param.weight.grad if isinstance(ref_param, _t.nn.Linear) else ref_param.grad
            assert ref_grad is not None, f"no ref grad for layer {li} {name}"
            g_dev = g.to(dev)
            # CPU float32 vs device — tolerance for reduction-order noise
            assert _t.allclose(g_dev, ref_grad, atol=1e-4, rtol=1e-3), (
                f"layer {li} {name}: max diff "
                f"{(g_dev - ref_grad).abs().max().item():.2e}"
            )
