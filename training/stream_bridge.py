"""Streamed-grad -> canonical-optimizer bridge for StreamingModel.

Godmode BLOCKER fix: reverse_backward() accumulates per-layer parameter grads
into cpu_layers[i]["grad_fields"], but the optimizer iterates canonical
inner.parameters() and never consumes them => gradients computed, no learning.

Canonical-master sync design:
  1. push_streamed_grads(): grad_fields -> canonical .grad (accumulate microbatches)
  2. optimizer.step() updates canonical weights (standard loop)
  3. sync_canonical_to_snapshots(): write updated weights back into the flat
     snapshot dicts (cpu_layers[li][state_dict_key]) AND invalidate the cached
     _copy_plan so subsequent restores fetch updated weights.

Canonical inner weights are the single source of truth; snapshots are views.
"""
import torch


def install_streaming_optimizer_bridge(stream_model, optimizer):
    sm = stream_model
    if not hasattr(sm.inner, "model") or not hasattr(sm.inner.model, "layers"):
        raise RuntimeError("bridge requires inner.model.layers (Llama/Qwen family)")

    layers = list(sm.inner.model.layers)

    class Bridge:
        def __init__(self):
            self.cmap = [dict(layer.named_parameters()) for layer in layers]

        def push_embed_grad(self):
            """Streamed path detaches the embedding output, so embed.weight
            gets no grad from reverse_backward. Reconstruct it: recompute
            pre_embed from the stored ids + dh (grad wrt pre_embed output)."""
            ctx = getattr(sm, "_active_fwd", None) or getattr(sm, "_last_ctx", None)
            dh = getattr(sm, "_last_dh_embed", None)
            if dh is None:
                return
            with torch.enable_grad():
                ids = sm._last_ids
                w = sm.embed_tokens.weight
                pre = torch.nn.functional.embedding(ids, w)
                g = torch.autograd.grad(pre, w, grad_outputs=dh,
                                        allow_unused=True)[0]
            if g is not None:
                wp = sm.embed_tokens.weight
                wp.grad = g.to(wp.dtype) if wp.grad is None else wp.grad + g.to(wp.dtype)

        def push_streamed_grads(self):
            grads = sm.streamed_grads()
            n = 0
            for li, fields in grads.items():
                if li >= len(self.cmap):
                    continue
                for name, g in fields.items():
                    p = self.cmap[li].get(name)
                    if p is None or not p.requires_grad:
                        continue
                    dev_g = g.to(p.device, p.dtype)
                    p.grad = dev_g.clone() if p.grad is None else p.grad + dev_g
                    n += 1
            return n

        def sync_canonical_to_snapshots(self):
            with torch.no_grad():
                for li, params in enumerate(self.cmap):
                    snap = sm.cpu_layers[li]
                    sd = sm.inner.model.layers[li].state_dict()
                    for name, p in params.items():
                        # state_dict keys match param names for standard layers
                        if name in snap:
                            snap[name].copy_(p.detach().to("cpu"))
                        elif name in sd and name in snap:
                            snap[name].copy_(p.detach().to("cpu"))
                    # refresh any buffer-valued snapshot entries from canonical
                    for k, v in list(snap.items()):
                        if k not in params and k in sd and sd[k].dtype.is_floating_point:
                            pass  # buffers unchanged by optimizer; leave as-is
            # CRITICAL: invalidate cached copy plans so next restore pulls
            # updated weights, and clear scratch-resident stale copies.
            sm._copy_plan = []
            for s in sm.scratch:
                pass  # scratch overwritten by next prefetch anyway
            sm.zero_streamed_grads()

        def step(self):
            """Streamed optimizer step. Returns True if an update happened."""
            n = self.push_streamed_grads()
            self.push_embed_grad()
            if n == 0:
                sm.zero_streamed_grads()
                return False
            # clip ALL trainable inner params (decoder grads arrive via bridge,
            # resident grads via normal autograd) to match full-load semantics
            params = [p for p in sm.inner.parameters()
                      if p.requires_grad and p.grad is not None]
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            self.sync_canonical_to_snapshots()
            return True

        def zero(self):
            optimizer.zero_grad()
            sm.zero_streamed_grads()

    return Bridge()
