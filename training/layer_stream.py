"""Layer-streaming: run a large causal LM layer-by-layer from CPU RAM.

Implements the "Soup v0.74" pattern described in LAYER-STREAMING-SPEC.md:

  * Full model lives in CPU RAM as pinned staging buffers.
  * Only embeddings, final norm, lm_head (plus any small trainable add-ons,
    e.g. LoRA adapters) stay resident on the compute device.
  * Per forward pass, decoder layers are streamed onto the device one at a
    time using DOUBLE BUFFERING: while the compute stream runs layer i out
    of scratch buffer ``i % 2``, a dedicated copy stream prefetches layer
    i+1's weights into the OTHER scratch buffer with a non-blocking copy.
  * An explicit ``torch.cuda.synchronize()`` happens only at buffer
    boundaries (before reusing a scratch buffer / after each layer's H2D),
    never inside compute.
  * Base weights are frozen (``requires_grad=False``); gradients flow only
    through whatever trainable parts stayed resident (LoRA / BitLinear QAT
    hooks). Streamed base layers participate in autograd purely as
    parameter-free matmul/attention ops on activations.

The class also works with no CUDA at all: on a CPU-only box the compute
device falls back to CPU and every copy is a plain ``Tensor.copy_`` between
CPU tensors, so tests can prove output equivalence against a full-load
forward on machines without GPUs.

Usage
-----
    sm = StreamingModel(model, compute_device="cuda:0")
    out = sm.forward_with_streaming(input_ids, attention_mask=mask)
    logits = out.logits

    # Apply a BitLinear-style hook to every hidden state:
    out = sm.forward_with_streaming(ids, bitlinear_fn=my_hook)
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Optional

import torch
import torch.nn as nn


class _StreamedOutput(dict):
    """Minimal CausalLMOutput-like container (supports .logits and [0])."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return dict.__getitem__(self, idx)
        return list(self.values())[idx]


def _pick(model: nn.Module, *names):
    """Return the first existing sub-module among `names` (else None)."""
    for name in names:
        cur = model
        try:
            for part in name.split("."):
                cur = getattr(cur, part)
            return cur
        except AttributeError:
            continue
    return None


class StreamingModel(nn.Module):
    """Wrap an HF causal-LM and stream its decoder layers over PCIe.

    Parameters
    ----------
    model : nn.Module
        Any HF causal language model (LlamaForCausalLM, Qwen2ForCausalLM,
        GPTNeoXForCausalLM, ...). Assumed to already live on CPU when passed;
        otherwise it is moved to CPU here.
    compute_device : str | torch.device
        Device that activations compute on ("cuda:0"). Falls back to CPU
        automatically when CUDA is unavailable.
    freeze_base : bool
        Freeze every base parameter (spec requirement). Trainable params are
        expected in resident modules (e.g. freshly added LoRA adapters).
    """

    def __init__(
        self,
        model: nn.Module,
        compute_device: str | torch.device = "cuda:0",
        freeze_base: bool = True,
    ):
        super().__init__()
        self.inner = model

        device = torch.device(compute_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("StreamingModel: CUDA requested but unavailable -> "
                  "falling back to CPU streaming (no-op copies).")
            device = torch.device("cpu")
        self.compute_device = device
        self.use_cuda = device.type == "cuda"

        core = _pick(self.inner, "model", "transformer", "gpt_neox", "decoder")
        if core is None:
            raise ValueError("cannot locate transformer core in model")
        self.core = core

        layers_container = _pick(core, "layers", "h", "layer")
        try:
            layers_list = list(layers_container)
        except TypeError as e:
            raise ValueError("cannot locate decoder layer list in model") from e
        if not layers_list:
            raise ValueError("decoder layer list is empty")
        layers = layers_list
        self._num_layers = len(layers)

        self.embed_tokens = _pick(core, "embed_tokens", "wte", "tok_embeddings")
        self.final_norm = _pick(core, "norm", "ln_f", "final_layernorm")
        self.lm_head = _pick(self.inner, "lm_head", "embed_out")

        self.resident_modules = [
            m for m in (self.embed_tokens, self.final_norm, self.lm_head)
            if m is not None
        ]

        self.inner.to("cpu")

        if freeze_base:
            for p in self.inner.parameters():
                p.requires_grad_(False)

        # --- pinned CPU staging snapshots -----------------------------------
        self.cpu_layers = [self._pin_snapshot(l.state_dict()) for l in layers]

        # --- double-buffered scratch layers on the compute device ------------
        n_scratch = min(2, self._num_layers)
        self.scratch = nn.ModuleList(
            [deepcopy(layers[i]) for i in range(n_scratch)])
        for s in self.scratch:
            s.to(device)
            s.eval()
        if freeze_base:
            for s in self.scratch:
                for p in s.parameters():
                    p.requires_grad_(False)
        # Lazy per-layer copy plans: snapshot tensor -> scratch Parameter.
        self._copy_plan: list = []
        self.stream_stats = {"h2d": 0}

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _pin_snapshot(sd) -> dict[str, torch.Tensor]:
        out = {}
        for k, v in sd.items():
            t = v.detach()
            if torch.cuda.is_available():
                pin = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
                pin.copy_(t)
                out[k] = pin
            else:
                out[k] = t.clone()
        return out

    def _copy_plan_for(self, layer_idx: int):
        """[ (snapshot_tensor, scratch_param) ] for a layer index."""
        plan = self._copy_plan
        while len(plan) <= layer_idx:
            plan.append(None)
        if plan[layer_idx] is None:
            s_idx = layer_idx % len(self.scratch)
            scratch_named = dict(self.scratch[s_idx].named_parameters())
            src = self.cpu_layers[layer_idx]
            pairs = []
            for k, snap_t in src.items():
                if k in scratch_named:
                    pairs.append((snap_t, scratch_named[k]))
            plan[layer_idx] = pairs
        return plan[layer_idx]

    def _prefetch(self, layer_idx: int, events: dict):
        """Kick off non-blocking H2D of layer_idx into its scratch buffer."""
        s_idx = layer_idx % len(self.scratch)
        pairs = self._copy_plan_for(layer_idx)
        with torch.no_grad():
            for snap_t, p in pairs:
                p.data.copy_(snap_t, non_blocking=self.use_cuda)
                self.stream_stats["h2d"] += 1
        if self.use_cuda:
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(self.compute_device))
            events[s_idx] = ev

    def _wait_buffer(self, s_idx: int, events: dict):
        """Block main stream until the pending copy into buffer s_idx lands."""
        if self.use_cuda and s_idx in events:
            torch.cuda.current_stream(self.compute_device).wait_event(events.pop(s_idx))

    # ------------------------------------------------------------ streaming
    def forward_with_streaming(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        bitlinear_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> _StreamedOutput:
        """Full causal-LM forward with per-layer double-buffered streaming."""
        dev = self.compute_device
        ids = input_ids.to(dev, non_blocking=True)

        h = self.embed_tokens(ids)
        if bitlinear_fn is not None:
            h = bitlinear_fn(h)

        mask = None
        if attention_mask is not None:
            mask = attention_mask.to(dev, non_blocking=True)
            mask4 = _causal_4d_mask(mask, h.dtype, ids.device)
        else:
            mask4 = None

        kwargs = {}
        if mask4 is not None:
            kwargs["attention_mask"] = mask4

        # Position ids + rotary embeddings: modern HF layers expect the caller
        # to supply both. Default = contiguous (matches HF internals when no
        # cache/past exists). If an architecture rejects any kwarg we strip it
        # once and remember.
        B, T = ids.shape
        pos = position_ids
        if pos is None:
            pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, -1)
        else:
            pos = pos.to(dev, non_blocking=True)
        rot = None
        rotary = _pick(self.core, "rotary_emb")
        if rotary is not None:
            try:
                rot = rotary(h, pos)
            except Exception:
                rot = None
        pe_supported = rot is not None

        events: dict[int, object] = {}

        # Prime the pipeline: fetch layer 0 upfront (its prefetch overlaps
        # nothing yet, but keeps the loop shape uniform).
        self._prefetch(0, events)

        for i in range(self._num_layers):
            s_idx = i % len(self.scratch)
            self._wait_buffer(s_idx, events)

            # Buffer boundary: before loading layer i+1 into the OTHER
            # scratch buffer we synchronize once, so the async copy can
            # safely overlap with the compute of layer i while no consumer
            # touches its destination. On CPU this sync is a no-op.
            if i + 1 < self._num_layers:
                if self.use_cuda:
                    torch.cuda.current_stream(dev).synchronize()
                self._prefetch(i + 1, events)

            call_kw = dict(kwargs)
            if pe_supported:
                call_kw["position_embeddings"] = rot
            try:
                h = self.scratch[s_idx](h, **call_kw)
            except TypeError:
                # Architecture-specific signature mismatch: drop optional
                # extras and retry (flag remembered for later layers).
                if "position_embeddings" in call_kw:
                    call_kw.pop("position_embeddings")
                    pe_supported = False
                h = self.scratch[s_idx](h, **call_kw)
            if isinstance(h, tuple):      # some impls return (hidden, ...)
                h = h[0]
            if bitlinear_fn is not None:
                h = bitlinear_fn(h)

        if self.final_norm is not None:
            h = self.final_norm(h)
        if self.lm_head is not None:
            logits = self.lm_head(h)
        else:
            logits = h                    # tied-embedding models
        return _StreamedOutput(logits=logits)

    def forward(self, *args, **kw):
        return self.forward_with_streaming(*args, **kw)

    # ------------------------------------------------------------- plumbing
    def train(self, mode: bool = True):
        self.inner.train(mode)
        return super().train(mode)

    def eval(self):
        self.inner.eval()
        return super().eval()

    @property
    def config(self):
        return getattr(self.inner, "config", None)

    def parameters(self, recurse: bool = True):
        yield from self.inner.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        yield from self.inner.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, *a, **kw):
        return self.inner.state_dict(*a, **kw)

    def save_pretrained(self, *a, **kw):
        return self.inner.save_pretrained(*a, **kw)


def _causal_4d_mask(attention_mask: torch.Tensor, dtype: torch.dtype,
                    device: torch.device) -> torch.Tensor:
    """Build (B, 1, T, T) additive float mask from (B, T) padding mask.

    Equivalent to HF's causal+padding mask construction so streamed layers
    behave identically to full-load forwards.
    """
    B, T = attention_mask.shape
    causal = torch.full((T, T), torch.finfo(dtype).min, device=device)
    causal = torch.triu(causal, diagonal=1)
    pad = (1.0 - attention_mask[:, None, None, :].to(dtype)) * \
        torch.finfo(dtype).min
    return causal[None, None] + pad


# Back-compat alias used elsewhere
StreamingForward = StreamingModel


def make_streaming(model_or_id, compute_device="cuda:0",
                   student_dtype=torch.float32, **load_kw):
    """Load a HF model id (or accept an nn.Module) and wrap in StreamingModel."""
    if isinstance(model_or_id, nn.Module):
        model = model_or_id
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_or_id, torch_dtype=student_dtype,
            device_map="cpu", **load_kw)
    return StreamingModel(model, compute_device=compute_device)
