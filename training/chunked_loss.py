"""Memory-efficient loss computation for large-vocabulary models.

Avoids materializing the full (B, T, V) logits tensor by:
  1. Computing CE via chunked logsumexp over the vocab (no full logits)
  2. Computing top-k KL via direct weight gather (only k logits per position)

For a 151K vocab with batch=2, seq=1024:
  Full logits:  2 × 1024 × 151936 × 4 = 1.2 GB (avoided)
  Chunked CE:   16384 × 2048 × 4 = 134 MB per chunk (iterates)
  Top-k gather: 2048 × 100 × 1024 × 4 = 838 MB (one pass, k=100)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

CHUNK_SIZE = 16384


def chunked_cross_entropy(
    hidden_flat: torch.Tensor,
    weight: torch.Tensor,
    labels_flat: torch.Tensor,
    chunk_size: int = CHUNK_SIZE,
) -> torch.Tensor:
    """Compute cross-entropy without materializing full logits.

    Iterates over vocab chunks, accumulating a numerically stable logsumexp
    and gathering the label logits. Never creates the (N, V) tensor.

    Parameters
    ----------
    hidden_flat : (N, H) — flattened hidden states
    weight : (V, H) — lm_head weight
    labels_flat : (N,) — label ids (-100 = ignore)
    chunk_size : int — vocab chunk size
    """
    N, H = hidden_flat.shape
    V = weight.shape[0]

    label_logits = torch.zeros(N, device=hidden_flat.device, dtype=torch.float32)
    running_max = torch.full((N,), float('-inf'), device=hidden_flat.device,
                             dtype=torch.float32)
    running_sumexp = torch.zeros(N, device=hidden_flat.device, dtype=torch.float32)

    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        # (N, chunk) — only this chunk exists in memory
        logits_chunk = (hidden_flat @ weight[start:end].t()).to(torch.float32)

        # Gather label logits
        in_chunk = (labels_flat >= start) & (labels_flat < end)
        if in_chunk.any():
            local_labels = labels_flat[in_chunk] - start
            label_logits[in_chunk] = logits_chunk[in_chunk, local_labels]

        # Numerically stable logsumexp accumulation
        chunk_max = logits_chunk.max(dim=-1).values
        chunk_sumexp = torch.exp(logits_chunk - chunk_max.unsqueeze(-1)).sum(dim=-1)
        new_max = torch.maximum(running_max, chunk_max)
        running_sumexp = (running_sumexp * torch.exp(running_max - new_max)
                          + chunk_sumexp * torch.exp(chunk_max - new_max))
        running_max = new_max

        del logits_chunk

    logsumexp = running_max + torch.log(running_sumexp.clamp(min=1e-20))
    valid = labels_flat != -100
    return -(label_logits - logsumexp)[valid].mean()


def gather_topk_logits(
    hidden_flat: torch.Tensor,
    weight: torch.Tensor,
    topk_indices_flat: torch.Tensor,
    n_chunk: int = 512,
) -> torch.Tensor:
    """Gather logits at specific indices without full logits.

    Instead of computing hidden @ weight.t() (N, V) and gathering,
    we gather the weight rows first and compute only the needed logits.

    weight[topk_indices] gives (N, k, H) which is large for big N.
    We chunk over N to keep memory bounded.

    Parameters
    ----------
    hidden_flat : (N, H)
    weight : (V, H)
    topk_indices_flat : (N, k)
    n_chunk : int — chunk size for the N dimension
    """
    N, H = hidden_flat.shape
    k = topk_indices_flat.size(-1)
    out = torch.empty(N, k, device=hidden_flat.device, dtype=hidden_flat.dtype)

    for start in range(0, N, n_chunk):
        end = min(start + n_chunk, N)
        # Gather weight rows: (chunk, k, H)
        w_gathered = weight[topk_indices_flat[start:end]]
        # Compute logits: (chunk, k) = (chunk, 1, H) * (chunk, k, H) summed over H
        h_chunk = hidden_flat[start:end].unsqueeze(1)  # (chunk, 1, H)
        out[start:end] = (h_chunk * w_gathered).sum(dim=-1)

    return out


def chunked_ce_and_topk_kl(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    teacher_topk_indices: torch.Tensor | None = None,
    teacher_topk_probs: torch.Tensor | None = None,
    *,
    alpha: float = 0.5,
    temperature: float = 2.0,
    chunk_size: int = CHUNK_SIZE,
) -> torch.Tensor:
    """Compute CE + top-k KL in a single chunked pass over the vocab.

    Never materializes the full (B, T, V) logits tensor. Iterates over vocab
    chunks ONCE, computing:
      - CE: gather label logits + accumulate logsumexp
      - KL: gather student logits at teacher's top-k indices

    This avoids both the 1.2 GB full logits tensor AND the 0.8 GB
    weight[topk_indices] gather tensor from the two-pass approach.

    Parameters
    ----------
    hidden : (B, T, H) — final hidden states from the base model
    lm_head_weight : (V, H) — the lm_head weight matrix
    labels : (B, T) — ground truth token ids (-100 = ignore)
    teacher_topk_indices : (B, T, k) or None
    teacher_topk_probs : (B, T, k) or None
    alpha : float — CE weight (1-alpha = KL weight)
    temperature : float — KL temperature
    chunk_size : int — vocab chunk size
    """
    B, T, H = hidden.shape
    V = lm_head_weight.shape[0]
    N = B * T

    hidden_flat = hidden.reshape(N, H)
    labels_flat = labels.reshape(N)
    weight = lm_head_weight

    has_kl = teacher_topk_indices is not None and teacher_topk_probs is not None
    if has_kl:
        k = teacher_topk_indices.size(-1)
        topk_indices_flat = teacher_topk_indices.reshape(N, k)
        topk_probs_flat = teacher_topk_probs.reshape(N, k)
        student_topk_logits = torch.zeros(N, k, device=hidden.device,
                                          dtype=hidden.dtype)
    else:
        k = 0

    # Accumulators for chunked logsumexp (numerically stable)
    label_logits = torch.zeros(N, device=hidden.device, dtype=torch.float32)
    running_max = torch.full((N,), float('-inf'), device=hidden.device,
                             dtype=torch.float32)
    running_sumexp = torch.zeros(N, device=hidden.device, dtype=torch.float32)

    # --- Single pass over vocab chunks ---
    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)

        # Compute logits for this chunk: (N, chunk)
        logits_chunk = (hidden_flat @ weight[start:end].t()).to(torch.float32)

        # 1) Gather label logits for CE
        in_chunk_labels = (labels_flat >= start) & (labels_flat < end)
        if in_chunk_labels.any():
            local_labels = labels_flat[in_chunk_labels] - start
            label_logits[in_chunk_labels] = logits_chunk[in_chunk_labels, local_labels]

        # 2) Gather top-k logits for KL (at teacher's indices in this chunk)
        if has_kl:
            in_chunk_topk = (topk_indices_flat >= start) & (topk_indices_flat < end)
            if in_chunk_topk.any():
                local_idx = (topk_indices_flat - start).clamp(0, end - start - 1)
                gathered = logits_chunk.gather(1, local_idx)  # (N, k)
                student_topk_logits = torch.where(
                    in_chunk_topk, gathered.to(student_topk_logits.dtype),
                    student_topk_logits
                )

        # 3) Accumulate logsumexp for CE
        chunk_max = logits_chunk.max(dim=-1).values
        chunk_sumexp = torch.exp(logits_chunk - chunk_max.unsqueeze(-1)).sum(dim=-1)
        new_max = torch.maximum(running_max, chunk_max)
        running_sumexp = (running_sumexp * torch.exp(running_max - new_max)
                          + chunk_sumexp * torch.exp(chunk_max - new_max))
        running_max = new_max

        del logits_chunk

    # --- CE loss ---
    logsumexp = running_max + torch.log(running_sumexp.clamp(min=1e-20))
    valid = labels_flat != -100
    ce = -(label_logits - logsumexp)[valid].mean()

    if not has_kl:
        return alpha * ce

    # --- Top-k KL ---
    sl_topk = student_topk_logits / temperature
    log_sl_topk = F.log_softmax(sl_topk, dim=-1)
    log_sl_topk_bt = log_sl_topk.reshape(B, T, k)
    topk_probs_bt = topk_probs_flat.reshape(B, T, k)
    kl = F.kl_div(log_sl_topk_bt, topk_probs_bt, reduction="batchmean")
    kl = kl * (temperature * temperature)

    return alpha * ce + (1.0 - alpha) * kl


def get_hidden_states(model, input_ids, attention_mask=None):
    """Run the base model to get last hidden states (before lm_head)."""
    base = model.model if hasattr(model, 'model') else model
    outputs = base(input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
