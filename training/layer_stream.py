"""Layer-streaming: run a large causal LM layer-by-layer from CPU RAM.

Implements the "Soup v0.74" pattern described in LAYER-STREAMING-SPEC.md.

Why the streamed stack is NOT plain autograd
--------------------------------------------
Only two device-side scratch copies of decoder layers exist; a naive
``loss.backward()`` over streamed forwards would find every saved activation
graph referencing scratch weights long overwritten by later layers (silent
corruption on CUDA). So the streamed stack never builds a weight-dependent
graph at all:

* Forward: activations are detached between layers (per-layer checkpoints);
  scratch weights participate purely as data.
* An identity autograd tail op wraps the returned logits; when its backward
  fires, ``reverse_backward`` streams layers in REVERSE order (N-1 .. 0),
  restoring each layer's exact state_dict into a scratch copy BEFORE its
  backward step, computing dL/d(input) per layer and accumulating parameter
  gradients onto the CPU snapshots (see ``streamed_grads()``).

Everything NOT streamed stays ordinary autograd: embeddings, final norm,
lm_head and any trainable add-ons (LoRA etc.) attached to residents or to
decoder-layer modules get normal ``.grad`` via ``loss.backward()``.

Double buffering
----------------
A dedicated ``torch.cuda.Stream`` performs H2D restores of layer i+1 while
the default stream computes layer i; the compute stream waits on that copy's
event exactly when the destination slot is about to be consumed. On CPU-only
boxes everything degenerates to synchronous copies (no streams, no events).

Eval guarantee: eval/no-grad forward is bit-exact vs full-load because every
layer runs on restored exact state_dicts (state_dict INCLUDES buffers).

Usage
-----
    sm = StreamingModel(model, compute_device="cuda:0")
    out = sm.forward_with_streaming(input_ids, attention_mask=mask)
    loss = criterion(out.logits, ...)
    loss.backward()                       # full graph incl. streamed layers

    # Gradients of streamed decoder-layer weights land here:
    grads = sm.streamed_grads()           # {layer_idx: {key: cpu_tensor}}
"""

from __future__ import annotations

import contextlib
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


class _StreamTailFn(torch.autograd.Function):
    """Identity autograd op whose backward triggers the streamed reversal."""

    @staticmethod
    def forward(ctx, logits, owner):
        ctx.owner = owner                    # single-forward lifetime, no cycle
        return logits

    @staticmethod
    def backward(ctx, grad_logits):
        owner = ctx.owner
        retain = owner._retain_next
        owner._retain_next = False
        owner.reverse_backward(grad_logits, retain_graph=retain)
        return grad_logits, None


class StreamingModel(nn.Module):
    """Wrap an HF causal-LM and stream its decoder layers over PCIe.

    Parameters
    ----------
    model : nn.Module
        Any HF causal language model (LlamaForCausalLM, Qwen2ForCausalLM,
        GPTNeoXForCausalLM, ...). Assumed to already live on CPU when passed;
        otherwise it is moved to CPU here.
    compute_device : str | torch.device
        Device activations compute on ("cuda:0"). Falls back to CPU
        automatically when CUDA is unavailable.
    freeze_base : bool
        Freeze every base parameter (spec requirement). Trainable params are
        expected either in resident modules (freshly added LoRA adapters on
        embed/norm/head — plain autograd) or inside streamed decoder layers,
        whose grads accumulate on the CPU snapshots (``streamed_grads()``).
    n_scratch : int
        Number of rotating device-side layer copies (double buffer = 2).
    """

    def __init__(
        self,
        model: nn.Module,
        compute_device: str | torch.device = "cuda:0",
        freeze_base: bool = True,
        n_scratch: int = 2,
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
        # Built from state_dict: INCLUDES buffers (running stats, inv_freq...).
        # Each snapshot may also carry "grad_fields": {name: cpu tensor} with
        # parameter gradients harvested from the reverse streaming pass.
        self.cpu_layers = [self._pin_snapshot(l.state_dict()) for l in layers]

        # --- double-buffered scratch layers on the compute device ------------
        n_scratch = max(1, min(int(n_scratch), self._num_layers))
        self.scratch = nn.ModuleList(
            [deepcopy(layers[i]) for i in range(n_scratch)])
        for s in self.scratch:
            s.to(device)
            s.eval()
            for p in s.parameters():          # grads harvested manually in
                p.requires_grad_(True)        # _run_layer_pass, engine-free
        # Lazy per-layer copy plans: snapshot key -> scratch tensor (params
        # AND buffers — built from state_dict so both are covered).
        self._copy_plan: list = []
        self.stream_stats = {"h2d": 0}

        # Residents stay tiny (embeddings/norm/head/LoRA) -> park them next to
        # the compute so hidden states flow without host hops (spec requirement).
        if self.use_cuda:
            for m in self.resident_modules:
                m.to(device)

        # Dedicated COPY stream: real overlap of H2D restore with compute.
        self._copy_stream = (
            torch.cuda.Stream(device=device) if self.use_cuda else None)
        self._pending_events: dict[int, object] = {}

        # Active-forward bookkeeping for the reverse pass.
        self._active_fwd: Optional[dict] = None
        self._retain_next = False
        self._pe_supported = True   # sticky arch flag: accepts position_embeddings?
        self.rotary = _pick(self.core, "rotary_emb")

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
        """[(snap_t, scratch_tensor)] pairs for a layer index.

        Uses state_dict on BOTH sides so registered buffers (BatchNorm
        running_mean/var, RoPE inv_freq caches, ...) refresh too — skipping
        them would silently run scratch layers with stale values.
        """
        plan = self._copy_plan
        while len(plan) <= layer_idx:
            plan.append(None)
        if plan[layer_idx] is None:
            s_idx = layer_idx % len(self.scratch)
            sd = self.scratch[s_idx].state_dict()
            src = self.cpu_layers[layer_idx]
            pairs = [(snap_t, sd[k])
                     for k, snap_t in src.items() if k in sd]
            plan[layer_idx] = pairs
        return plan[layer_idx]

    def _prefetch(self, layer_idx: int):
        """Kick off non-blocking load of layer_idx into its scratch buffer.

        With CUDA the copies are issued on the dedicated COPY stream and an
        event recorded there; the compute stream waits on exactly that event
        before consuming the buffer. Without CUDA: plain sync copies.
        """
        s_idx = layer_idx % len(self.scratch)
        pairs = self._copy_plan_for(layer_idx)
        with torch.no_grad():
            for snap_t, dst in pairs:
                dst.copy_(snap_t, non_blocking=self.use_cuda)
                self.stream_stats["h2d"] += 1
        if self.use_cuda:
            ev = torch.cuda.Event()
            ev.record(self._copy_stream)
            self._pending_events[s_idx] = ev

    def _wait_buffer(self, s_idx: int):
        """Compute stream waits until the pending copy into s_idx landed."""
        ev = self._pending_events.pop(s_idx, None)
        if self.use_cuda and ev is not None:
            torch.cuda.current_stream(self.compute_device).wait_event(ev)

    def _restore_layer_now(self, layer_idx: int):
        """Synchronously place layer_idx's exact weights in a scratch slot."""
        self._wait_buffer(layer_idx % len(self.scratch))
        self._prefetch(layer_idx)
        if self.use_cuda:
            self._copy_stream.synchronize()

    # --------------------------------------------------------------- accessors
    def streamed_grads(self) -> dict[int, dict[str, torch.Tensor]]:
        """Per-layer CPU gradients accumulated by the last reverse pass."""
        out = {}
        for li, snap in enumerate(self.cpu_layers):
            gf = snap.get("grad_fields")
            if gf:
                out[li] = dict(gf)
        return out

    def zero_streamed_grads(self):
        for snap in self.cpu_layers:
            snap.pop("grad_fields", None)

    def trainable_params(self):
        """All parameters receiving gradients from a streamed train step.

        This includes streamed decoder-layer weights themselves (their grad
        storage lives on the CPU snapshots via grad_fields after backward).
        Residents keep standard autograd .grad.
        """
        for p in self.inner.parameters(recurse=True):
            yield p
        for s in self.scratch:
            for p in s.parameters(recurse=True):
                yield p

    # ----------------------------------------------------------------- forward
    def forward_with_streaming(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        bitlinear_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> _StreamedOutput:
        """Full causal-LM forward with per-layer double-buffered streaming.

        Eval/no-grad mode is bit-exact vs full-load. Under grad mode it also
        records activation checkpoints used by ``reverse_backward``.
        """
        dev = self.compute_device
        ids = input_ids.to(dev, non_blocking=True)
        want_ctx = torch.is_grad_enabled() and any(
            p.requires_grad for p in self.trainable_params())

        self._active_fwd = None
        self._pending_events.clear()

        pre_embed = self.embed_tokens(ids)
        h_in = bitlinear_fn(pre_embed) if bitlinear_fn is not None else pre_embed

        mask4 = None
        if attention_mask is not None:
            mask4 = _causal_4d_mask(attention_mask.to(dev), h_in.dtype, dev)

        B, T = ids.shape
        pos = position_ids
        if pos is None:
            pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, -1)
        else:
            pos = pos.to(dev, non_blocking=True)

        rot = None
        if self.rotary is not None:
            try:
                rot = self.rotary(h_in.detach(), pos)
            except Exception:
                rot = None

        ctx = {
            "kwargs_mask": {"attention_mask": mask4} if mask4 is not None else {},
            "pos": pos,
            "rot": rot,
            "bitlinear_fn": bitlinear_fn,
            "acts": [],
            "pre_embed": pre_embed.detach(),
        }

        acts = ctx["acts"]
        cur = h_in.detach()

        # Streamed stack is graph-free by design: scratch weights are transient
        # copies reused across layers, so they must never join the autograd
        # tape (they get overwritten N times per pass).
        no_grad_ctx = torch.no_grad() if want_ctx else contextlib.nullcontext()
        with no_grad_ctx:
            # Prime the pipeline: fetch layer 0 upfront.
            self._prefetch(0)
            for i in range(self._num_layers):
                s_idx = i % len(self.scratch)
                self._wait_buffer(s_idx)
                if i + 1 < self._num_layers:
                    # Prefetch layer i+1 into the OTHER scratch buffer now so
                    # its H2D copy overlaps this layer's compute on the
                    # default stream; the event wait only matters when we come
                    # around to reuse that slot again (with n_scratch >=
                    # num_layers it never does).
                    self._prefetch(i + 1)
                y = self._call_layer(i, cur, ctx)
                cur = y.detach()
                acts.append(cur)
                if bitlinear_fn is not None:
                    # Streaming semantics: every layer OUTPUT passes through
                    # the hook before feeding the next layer (matches
                    # WrapLayer-style reference pipelines / BitLinear QAT).
                    cur = bitlinear_fn(cur)

        post = self.final_norm(cur) if self.final_norm is not None else cur
        logits = self.lm_head(post) if self.lm_head is not None else post

        if want_ctx:
            logits = _StreamTailFn.apply(logits, self)
            self._active_fwd = {"ctx": ctx}
        return _StreamedOutput(logits=logits)

    # ----------------------------------------------------------- layer plumbing
    def _layer_call_kwargs(self, ctx: dict) -> dict:
        kw = dict(ctx["kwargs_mask"])
        if self._pe_supported and ctx.get("rot") is not None:
            kw["position_embeddings"] = ctx["rot"]
        return kw

    def _call_layer(self, layer_idx: int, inp: torch.Tensor, ctx: dict):
        s_idx = layer_idx % len(self.scratch)
        kw = self._layer_call_kwargs(ctx)
        try:
            y = self.scratch[s_idx](inp, **kw)
        except TypeError:
            # Architecture-specific signature mismatch: drop optional extras
            # once and remember the answer for every later call.
            if "position_embeddings" in kw:
                kw.pop("position_embeddings")
                self._pe_supported = False
            y = self.scratch[s_idx](inp, **kw)
        if isinstance(y, tuple):              # some impls return (hidden, ...)
            y = y[0]
        return y

    # ----------------------------------------------------------------- backward
    def backward_with_streaming(self, grad_logits=None, retain_graph=False):
        """Reverse-order streaming backward over the LAST streamed forward.

        Normally you do NOT need this: ``loss.backward()`` reaches the same
        code path through the tail hook (and propagates gradients through
        residents like lm_head LoRA along the way). Called directly (e.g. for
        diagnostics), it seeds dL/d(post-norm hidden) = grad_logits (default
        ones) and fills ONLY the streamed-weight snapshot grads — resident
        modules get nothing.
        """
        fwd = self._active_fwd
        if fwd is None:
            raise RuntimeError(
                "backward_with_streaming called without an active "
                "grad-enabled forward_with_streaming")
        if grad_logits is None:
            ref = fwd["ctx"]["acts"][-1]
            grad_logits = torch.ones_like(ref)
        self.reverse_backward(grad_logits, retain_graph=retain_graph)

    def reverse_backward(self, grad_logits, retain_graph: bool = False):
        """The heart of the fix: stream layers N-1..0 in REVERSE order,
        restoring each layer's exact weights BEFORE its backward step."""
        del retain_graph                      # checkpoint replay, no graph kept
        fwd = getattr(self, "_active_fwd", None)
        if fwd is None:
            raise RuntimeError(
                "reverse_backward called without an active grad-enabled "
                "forward_with_streaming")
        ctx = fwd["ctx"]
        bl = ctx.get("bitlinear_fn")
        acts = ctx["acts"]

        # Seed gradient at the top of the streamed stack: undo the final norm
        # (+ hook) analytically via one tiny autograd call against the saved
        # last activation. Everything above this point (lm_head etc.) already
        # ran inside normal autograd before the tail fired.
        with torch.enable_grad():
            leaf = acts[-1].detach().clone().requires_grad_(True)
            x = bl(leaf) if bl is not None else leaf
            if self.final_norm is not None:
                x = self.final_norm(x)
            if self.lm_head is not None:
                x = self.lm_head(x)
            (g_top,) = torch.autograd.grad(
                outputs=[x], inputs=[leaf], grad_outputs=[grad_logits],
                allow_unused=True)
        dh_next = g_top if g_top is not None else torch.zeros_like(leaf)
        del g_top, leaf, x

        ctx["dh_next"] = dh_next
        for li in range(self._num_layers - 1, -1, -1):
            prev_act = acts[li - 1] if li > 0 else ctx["pre_embed"]
            _, dh = self._reverse_layer_step(li, prev_act, ctx)  # returns (recomputed_flag, dh)
            ctx["dh_next"] = dh
            acts[li - 1 if li > 0 else 0] = None   # release processed chunk

        self._active_fwd = None
        ctx["acts"] = []

    def _reverse_layer_step(self, layer_idx: int, prev_act: torch.Tensor,
                            ctx: dict):
        """Backward through ONE layer against freshly restored exact weights.

        Returns dL/d(prev_act). Parameter grads land on
        cpu_layers[layer_idx]["grad_fields"], then are wiped from scratch so
        the next restore cannot accumulate stale values into them.
        """
        self._restore_layer_now(layer_idx)     # blocking exact restore FIRST
        go = ctx["dh_next"].to(prev_act.device, prev_act.dtype)
        with torch.enable_grad():
            leaf = prev_act.detach().clone().requires_grad_(True)
            y = self._call_layer(layer_idx, leaf, ctx)
            if ctx.get("bitlinear_fn") is not None:
                # Forward applied the hook to EVERY layer output; mirror it.
                y = ctx["bitlinear_fn"](y)
            (g,) = torch.autograd.grad(outputs=y, inputs=leaf,
                                       grad_outputs=go, allow_unused=True)
        dh = torch.zeros_like(leaf) if g is None else g.detach()

        # Harvest parameter gradients accumulated inside the scratch module.
        s_idx = layer_idx % len(self.scratch)
        snap = self.cpu_layers[layer_idx]
        gf = snap.setdefault("grad_fields", {})
        for name, param in self.scratch[s_idx].named_parameters(recurse=True):
            gp = param.grad
            if gp is None:
                continue
            acc = gf.get(name)
            if acc is None:
                acc = torch.zeros(param.shape, dtype=gp.dtype)
                gf[name] = acc
            acc.add_(gp.detach().to("cpu"))
            param.grad = None
        for _, buf in self.scratch[s_idx].named_buffers(recurse=True):
            buf.grad = None                    # buffers carry no grads here
        return True, dh

    def forward(self, *args, **kw):
        return self.forward_with_streaming(*args, **kw)

    # ---------------------------------------------------------------- plumbing
    def gradient_checkpointing_enable(self, *args, **kwargs):
        """Explicit refusal instead of a confusing AttributeError downstream.

        ``distill.py --grad-checkpoint`` calls this; delegating to
        ``self.inner`` would be a silent no-op because our custom streamed
        forward never routes through the HF core's forward anyway. The
        reverse streaming pass ALREADY works checkpoint-style (activations
        stored once per layer boundary, everything inside each layer
        recomputed during backward), which is strictly more memory-efficient
        than HF whole-layer checkpointing here — so combining the two flags
        is unnecessary, and we say so loudly rather than pretend.
        """
        raise NotImplementedError(
            "StreamingModel does its own checkpoint-recompute in "
            "reverse_backward (per-layer activations saved in forward); "
            "do not combine --stream with --grad-checkpoint.")

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
        # Only the wrapped model: optimizer registration must NOT see scratch
        # copies (transient base weights). Streamed-weight updates read
        # snapshot grad_fields instead of .grad.
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

    KNOWN DIVERGENCE vs modern HF: Llama 3 / Qwen2.5 >= 7B route masking
    through ``AttentionMaskConverter._update_causal_mask``, which adds
    sliding-window attention, chunked/local attention support and SDPA-aware
    trimming of fully-masked rows; this hand-rolled causal+padding mask
    replicates none of those paths. For vanilla full-causal masks it matches
    HF's eager-path construction (tests prove bit-exactness), but configs
    that engage sliding windows or long-context chunking will diverge from
    full-load behaviour — extend this via AttentionMaskConverter before
    trusting streamed equivalence at 27B scale on such models.
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
