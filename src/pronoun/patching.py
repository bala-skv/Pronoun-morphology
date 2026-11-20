"""Activation patching + attribution patching between clean/corrupt minimal pairs.

Conventions
-----------
* pairs: list of (clean Item, corrupt Item, sign). Items have identical token length and identical
  token ids before `first_diff` (shared context), so in a causal model only positions >= first_diff
  can differ between the two runs. We therefore only patch those positions; earlier positions have
  R = 0 by construction.
* Δ for an item is logsumexp(pos) − logsumexp(neg); multiplying by `sign` orients it so that the clean
  prompt's correct answer is positive. Recovery is computed from Δ averaged over pairs:
      R = (mean Δ_patch − mean Δ_corrupt) / (mean Δ_clean − mean Δ_corrupt)
  (per-pair ratios explode when Δ_clean ≈ Δ_corrupt).
* Positions are reported by *role* relative to the varying word, so pairs with different noun
  lengths can be pooled: 'w_first', 'w_mid', 'w_last' for the varying word, then 'tail+k:<tok>'.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .metrics import delta_from_logits, last_logits


def first_diff(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def last_diff(a: list[int], b: list[int]) -> int:
    for k in range(1, min(len(a), len(b)) + 1):
        if a[-k] != b[-k]:
            return len(a) - k
    return -1


def roles_for_pair(model, clean, corrupt) -> dict[int, str]:
    """Map patchable positions -> role label, poolable across pairs of different lengths.

    w / w_first / w_mid / w_last : the varying word (noun in C1, pronoun in C2)
    mid / mid_first / mid_inner / mid_last : prompt tokens between the varying word and the prompt end
    pfx : tokens of the shared target prefix (e.g. ' मे'), except the final one
    END : the final position, where Δ is read out
    """
    a, b = clean.ids, corrupt.ids
    assert len(a) == len(b), "pairs must have equal token length"
    s, e = first_diff(a, b), last_diff(a, b)
    roles = {}
    for p in range(s, e + 1):
        roles[p] = "w" if s == e else ("w_first" if p == s else "w_last" if p == e else "w_mid")
    mids = [p for p in range(e + 1, clean.base_len)]
    for k, p in enumerate(mids):
        roles[p] = "mid" if len(mids) == 1 else ("mid_first" if k == 0 else "mid_last" if k == len(mids) - 1 else "mid_inner")
    for p in range(clean.base_len, len(a)):
        roles[p] = "pfx"
    roles[len(a) - 1] = "END"
    return roles


ROLE_ORDER = ["w", "w_first", "w_mid", "w_last", "mid", "mid_first", "mid_inner", "mid_last", "pfx", "END"]


def _group_pairs(pairs):
    """Group pairs by (length, first_diff, last_diff) so each group can be batched with aligned positions."""
    g = defaultdict(list)
    for pr in pairs:
        c, x, _ = pr
        g[(len(c.ids), first_diff(c.ids, x.ids), last_diff(c.ids, x.ids), c.base_len)].append(pr)
    return g


@torch.no_grad()
def _deltas(model, toks, pairs, logits=None):
    if logits is None:
        logits = last_logits(model, toks)
    return torch.stack([sgn * delta_from_logits(logits[i], c.pos_ids, c.neg_ids)
                        for i, (c, _, sgn) in enumerate(pairs)]).cpu()


@torch.no_grad()
def baseline(model, pairs, batch_size=32):
    """Per-pair oriented Δ_clean and Δ_corrupt."""
    out = []
    for which in (0, 1):
        res = np.zeros(len(pairs))
        by_len = defaultdict(list)
        for i, p in enumerate(pairs):
            by_len[len(p[which].ids)].append(i)
        for L, idx in by_len.items():
            for k in range(0, len(idx), batch_size):
                ii = idx[k:k + batch_size]
                toks = torch.tensor([pairs[i][which].ids for i in ii], device=model.cfg.device)
                res[ii] = _deltas(model, toks, [pairs[i] for i in ii]).numpy()
        out.append(res)
    return out[0], out[1]


def _hook_names(model, kind: str):
    L = model.cfg.n_layers
    if kind == "resid_pre":
        return [f"blocks.{l}.hook_resid_pre" for l in range(L)]
    if kind == "attn_out":
        return [f"blocks.{l}.hook_attn_out" for l in range(L)]
    if kind == "mlp_out":
        return [f"blocks.{l}.hook_mlp_out" for l in range(L)]
    if kind == "z":
        return [f"blocks.{l}.attn.hook_z" for l in range(L)]
    raise ValueError(kind)


@torch.no_grad()
def activation_patch(model, pairs, kind="resid_pre", layers=None, batch_size=64, heads=None,
                     role_filter=None, progress=True):
    """Patch clean -> corrupt one (layer, position[, head]) at a time.

    Returns dict with:
      'per_pair': {(layer, role[, head]): np.array of oriented Δ_patch, aligned with `pair_idx`}
      'pair_idx': {(layer, role[, head]): list of pair indices}
    """
    from tqdm.auto import tqdm

    names = _hook_names(model, kind)
    layers = list(range(model.cfg.n_layers)) if layers is None else layers
    groups = _group_pairs(pairs)
    pair_index = {id(p): i for i, p in enumerate(pairs)}
    per, idxs = defaultdict(list), defaultdict(list)
    n_heads = model.cfg.n_heads
    head_list = (list(range(n_heads)) if heads == "all" else heads) if kind == "z" else [None]

    # Pairs of different lengths are batched together with right padding (causal attention: padding after
    # a position cannot affect it) and per-row patch positions. One forward per (chunk, layer, role
    # occurrence, head) instead of one per length group -> far fewer, larger forward passes.
    del groups
    pad = model.tokenizer.pad_token_id or 0
    dev = model.cfg.device
    # per pair: role -> list of positions (a role such as 'mid_inner' can occur more than once)
    role_pos = []
    for c, x, _ in pairs:
        d = defaultdict(list)
        for p, r in sorted(roles_for_pair(model, c, x).items()):
            if not role_filter or role_filter(r):
                d[r].append(p)
        role_pos.append(d)
    role_occ = sorted({(r, j) for d in role_pos for r, ps in d.items() for j in range(len(ps))},
                      key=lambda t: (ROLE_ORDER.index(t[0]) if t[0] in ROLE_ORDER else 99, t[1]))
    chunks = [list(range(k, min(k + batch_size, len(pairs)))) for k in range(0, len(pairs), batch_size)]
    total = len(chunks) * len(layers) * len(role_occ) * len(head_list)
    bar = tqdm(total=total, desc=f"patch {kind}", disable=not progress)

    def padded(seqs):
        T = max(len(s) for s in seqs)
        t = torch.full((len(seqs), T), pad, dtype=torch.long)
        for i, s in enumerate(seqs):
            t[i, : len(s)] = torch.tensor(s)
        return t.to(dev), [len(s) - 1 for s in seqs]

    for ch in chunks:
        clean, _ = padded([pairs[i][0].ids for i in ch])
        corr, last = padded([pairs[i][1].ids for i in ch])
        _, cache = model.run_with_cache(clean, names_filter=lambda n: n in set(names[l] for l in layers),
                                        stop_at_layer=max(layers) + 1)  # no full-vocab logits (OOM)
        for l in layers:
            name = names[l]
            src_all = cache[name]
            for role, j in role_occ:
                rows = [k for k, i in enumerate(ch) if len(role_pos[i].get(role, [])) > j]
                if not rows:
                    bar.update(len(head_list))
                    continue
                pos = [role_pos[ch[k]][role][j] for k in rows]
                r_t, p_t = torch.tensor(rows, device=dev), torch.tensor(pos, device=dev)
                sub_pairs = [pairs[ch[k]] for k in rows]
                local = torch.arange(len(rows), device=dev)  # forward runs on corr[r_t] only
                for h in head_list:
                    def hook(act, hook, h=h):
                        # with the model split over GPUs, act / src_all can live on a different device than r_t
                        lo, pa = local.to(act.device), p_t.to(act.device)
                        rs, ps = r_t.to(src_all.device), p_t.to(src_all.device)
                        if h is None:
                            act[lo, pa] = src_all[rs, ps].to(act.device, act.dtype)
                        else:
                            act[lo, pa, h] = src_all[rs, ps, h].to(act.device, act.dtype)
                        return act

                    logits = last_logits(model, corr[r_t], fwd_hooks=[(name, hook)],
                                         positions=[last[k] for k in rows])
                    d = _deltas(model, None, sub_pairs, logits=logits).numpy()
                    site = (l, role) if h is None else (l, role, h)
                    per[site].extend(d.tolist())
                    idxs[site].extend(ch[k] for k in rows)
                    bar.update(1)
        del cache
    bar.close()
    return {"per_pair": {k: np.array(v) for k, v in per.items()}, "pair_idx": dict(idxs)}


def recovery_table(result, d_clean, d_corr):
    """Aggregate per-pair patched Δ into R per site (mean-based)."""
    rows = []
    for site, dp in result["per_pair"].items():
        idx = result["pair_idx"][site]
        c, x = d_clean[idx].mean(), d_corr[idx].mean()
        R = (dp.mean() - x) / (c - x)
        rows.append(dict(site=site, layer=site[0], role=site[1], head=site[2] if len(site) > 2 else None,
                         R=R, d_patch=dp.mean(), d_clean=c, d_corr=x, n=len(idx)))
    return rows


def recovery_bootstrap(result, d_clean, d_corr, n_boot=500, seed=0):
    """Bootstrap CI for R per site (resampling pairs)."""
    rng = np.random.default_rng(seed)
    out = {}
    for site, dp in result["per_pair"].items():
        idx = np.array(result["pair_idx"][site])
        c, x = d_clean[idx], d_corr[idx]
        bs = []
        for _ in range(n_boot):
            j = rng.integers(0, len(idx), len(idx))
            den = c[j].mean() - x[j].mean()
            bs.append((dp[j].mean() - x[j].mean()) / den if abs(den) > 1e-6 else np.nan)
        out[site] = (np.nanquantile(bs, .025), np.nanquantile(bs, .975))
    return out


def attribution_patch(model, pairs, kind="resid_pre", batch_size=8):
    """Linear (gradient) approximation of activation patching, for all layers & positions at once.

    Returns {(layer, role): np.array of per-pair estimated Δ_patch − Δ_corrupt}.
    """
    names = _hook_names(model, kind)
    groups = _group_pairs(pairs)
    pair_index = {id(p): i for i, p in enumerate(pairs)}
    per, idxs = defaultdict(list), defaultdict(list)
    for key, gp in groups.items():
        roles = roles_for_pair(model, gp[0][0], gp[0][1])
        for k in range(0, len(gp), batch_size):
            sub = gp[k:k + batch_size]
            clean = torch.tensor([p[0].ids for p in sub], device=model.cfg.device)
            corr = torch.tensor([p[1].ids for p in sub], device=model.cfg.device)
            with torch.no_grad():
                _, ccache = model.run_with_cache(clean, names_filter=lambda n: n in names)
            acts, grads = {}, {}

            def fwd_hook(act, hook):
                act.retain_grad()
                acts[hook.name] = act
                return act

            def embed_leaf(act, hook):
                # weights are frozen, so make the embedding output the grad leaf; everything downstream
                # then requires grad and retain_grad() works on the hooked activations
                return act.detach().requires_grad_(True)

            with torch.enable_grad():
                logits = last_logits(model, corr, fwd_hooks=[("hook_embed", embed_leaf)] + [(n, fwd_hook) for n in names])
                metric = torch.stack([sgn * delta_from_logits(logits[i], c.pos_ids, c.neg_ids)
                                      for i, (c, _, sgn) in enumerate(sub)]).sum()
                metric.backward()
            for l, n in enumerate(names):
                diff = (ccache[n].to(acts[n].device) - acts[n].detach())
                g = acts[n].grad
                contrib = (diff * g).sum(-1).float().cpu()  # [B, T] (resid) or [B, T, H] (z: sum over d_head)
                for pos, role in roles.items():
                    vals = contrib[:, pos]
                    if kind == "z":
                        for h in range(vals.shape[-1]):
                            per[(l, role, h)].extend(vals[:, h].tolist())
                            idxs[(l, role, h)].extend(pair_index[id(p)] for p in sub)
                    else:
                        per[(l, role)].extend(vals.tolist())
                        idxs[(l, role)].extend(pair_index[id(p)] for p in sub)
            del acts, ccache
    return {"per_pair": {k: np.array(v) for k, v in per.items()}, "pair_idx": dict(idxs)}
