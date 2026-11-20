"""Forward-pass helpers, logit-difference metric and bootstrap intervals."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def pad_batch(id_lists, pad_id: int):
    """Right-pad a list of token-id lists. Returns tokens [B, T] and last-index tensor [B]."""
    L = max(len(x) for x in id_lists)
    toks = torch.full((len(id_lists), L), pad_id, dtype=torch.long)
    for i, x in enumerate(id_lists):
        toks[i, : len(x)] = torch.tensor(x)
    last = torch.tensor([len(x) - 1 for x in id_lists])
    return toks, last


def length_groups(seqs, batch_size: int):
    """Yield lists of indices whose sequences share the same length (no padding needed)."""
    groups = defaultdict(list)
    for i, s in enumerate(seqs):
        groups[len(s)].append(i)
    for L, idx in sorted(groups.items()):
        for k in range(0, len(idx), batch_size):
            yield idx[k: k + batch_size]


def last_logits(model, toks, fwd_hooks=(), positions=None):
    """Logits at the last position (or at `positions`, one per row) without unembedding every position.

    Unembedding all positions of a 256k-vocab model dominates the cost of short-prompt forward passes.
    Applies ln_final, W_U and Gemma-2's final logit softcap. Returns float32 [B, d_vocab].
    """
    with model.hooks(fwd_hooks=list(fwd_hooks)):
        resid = model(toks, stop_at_layer=model.cfg.n_layers)
    if positions is None:
        x = resid[:, -1:]
    else:
        x = resid[torch.arange(len(toks)), torch.as_tensor(positions)][:, None]
    lg = model.unembed(model.ln_final(x))[:, 0].float()
    cap = getattr(model.cfg, "output_logits_soft_cap", 0.0) or 0.0
    if cap > 0:
        lg = cap * torch.tanh(lg / cap)
    return lg


@torch.no_grad()
def final_logits(model, id_lists, batch_size: int = 32) -> torch.Tensor:
    """Logits at the last position of each sequence, [N, d_vocab] (float32, on CPU)."""
    out = [None] * len(id_lists)
    for idx in length_groups(id_lists, batch_size):
        toks = torch.tensor([id_lists[i] for i in idx], device=model.cfg.device)
        lg = last_logits(model, toks).cpu()
        for j, i in enumerate(idx):
            out[i] = lg[j]
    return torch.stack(out)


def delta_from_logits(logits: torch.Tensor, pos_ids, neg_ids) -> torch.Tensor:
    """Δ = logsumexp(logits[pos]) - logsumexp(logits[neg]); works on [..., d_vocab]."""
    return torch.logsumexp(logits[..., pos_ids], -1) - torch.logsumexp(logits[..., neg_ids], -1)


def item_deltas(items, logits: torch.Tensor) -> np.ndarray:
    return np.array([delta_from_logits(logits[i], it.pos_ids, it.neg_ids).item() for i, it in enumerate(items)])


@torch.no_grad()
def continuation_logprob(model, prompt_ids, cont_ids) -> float:
    """Sum of log-probs of cont_ids given prompt_ids (teacher forcing)."""
    toks = torch.tensor([prompt_ids + cont_ids], device=model.cfg.device)
    lp = model(toks).float().log_softmax(-1)[0]
    n = len(prompt_ids)
    return float(sum(lp[n - 1 + k, t] for k, t in enumerate(cont_ids)))


def bootstrap_ci(x, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    return float(stat(x)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def bootstrap_ci_axis0(arr, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Bootstrap over axis 0 (items) for an array [N, ...]; returns mean, lo, hi of shape [...]."""
    arr = np.asarray(arr)
    rng = np.random.default_rng(seed)
    boots = np.stack([arr[rng.integers(0, len(arr), len(arr))].mean(0) for _ in range(n_boot)])
    return arr.mean(0), np.quantile(boots, alpha / 2, axis=0), np.quantile(boots, 1 - alpha / 2, axis=0)


def fmt_ci(t) -> str:
    m, lo, hi = t
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
