"""Stage 4 — Gemma Scope SAE features for possessive agreement.

  python scripts/04_sae.py --layers 8,12,16,20

For each layer (resid_post SAE, width 16k, L0 closest to 70):
  1. reconstruction check (fraction of variance explained, L0) on our prompts
  2. differential features: standardized mean difference of each latent, fem-noun vs masc-noun prompts,
     at the noun and at the END position (where Δ is read out), per language
  3. sharing (RQ3): correlation of the latent-difference vectors hi–pa, hi–en, hi–ta; overlap of top-k
     features, against a label-permutation null
  4. causal test: at END, move the top-k Hindi gender features to the opposite gender's mean
     (x' = x + Σ_k (a_k^target − a_k) W_dec[k]) and measure the drop in the Δ gap; same edit using
     Hindi-derived features on Punjabi prompts (cross-script transfer); k random active features as control.
Outputs: results/04_sae/*.csv, *.png
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pronoun.metrics import delta_from_logits, last_logits, length_groups
from pronoun.model import load_model
from pronoun.prompts import C1_FRAMES, c1_items, tok_ids
from pronoun.sae import load_sae, variance_explained

OUT = ROOT / "results" / "04_sae"
OUT.mkdir(parents=True, exist_ok=True)
LANGS = ["hi", "pa", "en", "ta"]


def noun_end(model, it):
    pre = C1_FRAMES[it.lang][it.meta["frame"]].split("{N}")[0] + it.meta["noun"]
    return len(tok_ids(model, pre)) - 1


@torch.no_grad()
def resid_at(model, items, layer, which, bs=16):
    """resid_post of `layer` at the noun or END position for each item -> [N, d]."""
    name = f"blocks.{layer}.hook_resid_post"
    seqs = [it.ids for it in items]
    out = [None] * len(items)
    for grp in length_groups(seqs, bs):
        toks = torch.tensor([seqs[i] for i in grp], device=model.cfg.device)
        _, cache = model.run_with_cache(toks, names_filter=lambda n: n == name, stop_at_layer=layer + 1)
        for j, i in enumerate(grp):
            p = noun_end(model, items[i]) if which == "noun" else len(seqs[i]) - 1
            out[i] = cache[name][j, p].float().cpu()
    return torch.stack(out)


def std_diff(A, y):
    f, m = A[y == 1], A[y == 0]
    pooled = np.sqrt((f.var(0) + m.var(0)) / 2) + 1e-3
    return (f.mean(0) - m.mean(0)) / pooled


@torch.no_grad()
def steer_effect(model, sae, items, y, feats, target_mean, layer, bs=16):
    """Δ for each item with SAE features `feats` at END set to target_mean[gender-flipped]."""
    name = f"blocks.{layer}.hook_resid_post"
    seqs = [it.ids for it in items]
    d0, d1 = np.zeros(len(items)), np.zeros(len(items))
    W = sae.W_dec[feats]  # [k, d]
    for grp in length_groups(seqs, bs):
        toks = torch.tensor([seqs[i] for i in grp], device=model.cfg.device)
        tgt = torch.stack([torch.tensor(target_mean[1 - y[i]][feats]) for i in grp]).float()  # opposite gender

        def hook(act, hook):
            x = act[:, -1].float()
            a = sae.encode(x.to(sae.W_enc.device))[:, feats].to(x.device)
            act[:, -1] = (x + (tgt.to(x.device) - a) @ W.to(x.device)).to(act.dtype)
            return act

        lg0 = last_logits(model, toks)
        lg1 = last_logits(model, toks, fwd_hooks=[(name, hook)])
        for j, i in enumerate(grp):
            it = items[i]
            d0[i] = delta_from_logits(lg0[j], it.pos_ids, it.neg_ids).item()
            d1[i] = delta_from_logits(lg1[j], it.pos_ids, it.neg_ids).item()
    return d0, d1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="6,10,14,18,22")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    model = load_model()
    items = {lang: [i for i in c1_items(model, lang) if i.ok] for lang in LANGS}
    ys = {lang: np.array([1 if i.meta["gender"] == "f" else 0 for i in its]) for lang, its in items.items()}
    rows, srows, feat_rows = [], [], []
    rng = np.random.default_rng(0)
    for layer in layers:
        sae = load_sae(layer, device=model.cfg.device)
        lat, diffs = {}, {}
        for which in ["noun", "END"]:
            for lang in LANGS:
                X = resid_at(model, items[lang], layer, which)
                sdev = sae.W_enc.device
                fve, l0 = variance_explained(sae, X.to(sdev))
                A = sae.encode(X.to(sdev)).cpu().numpy()
                lat[(which, lang)] = A
                diffs[(which, lang)] = std_diff(A, ys[lang])
                rows.append(dict(layer=layer, sae_l0=sae.l0, position=which, lang=lang, fve=fve, l0=l0))
            dh = diffs[(which, "hi")]
            top_hi = set(np.argsort(-np.abs(dh))[: args.k])
            for i, f in enumerate(np.argsort(-np.abs(dh))[: args.k]):
                feat_rows.append(dict(layer=layer, position=which, rank=i, feature=int(f), d_hi=dh[f],
                                      **{f"d_{l}": diffs[(which, l)][f] for l in LANGS if l != "hi"},
                                      neuronpedia=f"https://www.neuronpedia.org/gemma-2-2b/{layer}-gemmascope-res-16k/{f}"))
            for lang in ["pa", "en", "ta"]:
                d = diffs[(which, lang)]
                ok = np.isfinite(dh) & np.isfinite(d)
                corr = float(np.corrcoef(dh[ok], d[ok])[0, 1])
                top_l = set(np.argsort(-np.abs(d))[: args.k])
                # null: permute this language's labels
                null = []
                for _ in range(50):
                    dn = std_diff(lat[(which, lang)], rng.permutation(ys[lang]))
                    null.append(np.corrcoef(dh[ok], dn[ok])[0, 1])
                srows.append(dict(layer=layer, position=which, pair=f"hi-{lang}", corr=corr,
                                  null_mean=float(np.mean(null)), null_95=float(np.quantile(null, .95)),
                                  topk_overlap=len(top_hi & top_l)))
        # ---- causal steering at END with Hindi features (hi and pa prompts)
        dh = diffs[("END", "hi")]
        feats = np.argsort(-np.abs(dh))[: args.k]
        active = np.where((lat[("END", "hi")] > 0).mean(0) > 0.05)[0]
        rand = rng.choice(np.setdiff1d(active, feats), size=min(args.k, len(active)), replace=False)
        for lang in ["hi", "pa"]:
            A = lat[("END", lang)]
            means = {1: A[ys[lang] == 1].mean(0), 0: A[ys[lang] == 0].mean(0)}
            for fname, fs in [("top_hi", feats), ("random", rand)]:
                d0, d1 = steer_effect(model, sae, items[lang], ys[lang], fs, means, layer)
                y = ys[lang]
                gap0 = d0[y == 1].mean() - d0[y == 0].mean()
                gap1 = d1[y == 1].mean() - d1[y == 0].mean()
                acc0 = np.where(y == 1, d0 > 0, d0 < 0).mean()
                acc1 = np.where(y == 1, d1 > 0, d1 < 0).mean()
                srows.append(dict(layer=layer, position="END", pair=f"steer:{fname}->{lang}", gap_before=gap0,
                                  gap_after=gap1, frac_gap_removed=1 - gap1 / gap0, acc_before=acc0, acc_after=acc1))
        print(f"layer {layer} done", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "sae_recon.csv", index=False)
        pd.DataFrame(srows).to_csv(OUT / "sae_sharing_steering.csv", index=False)
        pd.DataFrame(feat_rows).to_csv(OUT / "sae_top_features.csv", index=False)
        del sae

    s = pd.DataFrame(srows)
    sh = s[s.pair.str.startswith("hi-")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for (which, ax) in zip(["noun", "END"], axes):
        for pair, col in [("hi-pa", "#8e44ad"), ("hi-en", "#2471a3"), ("hi-ta", "#16a085")]:
            e = sh[(sh.position == which) & (sh.pair == pair)]
            ax.plot(e.layer, e["corr"], "o-", color=col, label=pair)
            ax.plot(e.layer, e.null_95, ":", color=col, lw=1)
        ax.axhline(0, color="gray", lw=.6)
        ax.set_title(f"[{which}] corr. of SAE gender-difference vectors (dotted: 95% null)")
        ax.set_xlabel("layer")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "sae_sharing.png", dpi=150)
    print(pd.DataFrame(rows).round(3).to_string())
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
