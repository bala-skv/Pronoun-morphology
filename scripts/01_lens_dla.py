"""Stage 1 — logit lens + direct logit attribution.

  python scripts/01_lens_dla.py --contrast c1 --lang hi
  python scripts/01_lens_dla.py --contrast c2 --lang hi

Logit lens: decode resid_post of every layer at the final position through ln_final + W_U (+ softcap),
giving Δ per layer; the "decision layer" is where the contrast between conditions (fem vs masc noun;
मैंने vs मैं) emerges.
DLA: the final residual is exactly embed + Σ_l attn_out_l + mlp_out_l (Gemma-2's post-norms sit
before hook_attn_out / hook_mlp_out in TransformerLens). Each term's contribution to the *pre-softcap*
logit difference is (term / final_rms_scale) · (w_lnf ⊙ (W_U[:,y+] − W_U[:,y−])). Head contributions
use the cached ln1_post scale: head_h = w_ln1post ⊙ (z_h W_O^h) / scale.
We report *differential* DLA: mean contribution for condition A minus condition B (fem − masc noun,
or मैंने − मैं), i.e. which components carry the difference.

A tuned lens is not available pre-trained for gemma-2-2b, so we use the plain logit lens.
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

from pronoun.metrics import bootstrap_ci_axis0, length_groups
from pronoun.model import load_model
from pronoun.prompts import c1_items, c2_items

plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Sans Devanagari", "Noto Sans Gurmukhi", "Noto Sans Tamil"]

# tokens tracked by the lens at the position *before* the possessive (latent-language check, RQ3)
LATENT = {"hi": [" मे", " mine", " my", " मेरे"], "pa": [" ਮ", " mine", " my", " ਮੇਰ"]}


def direction(model, it):
    W_U = model.W_U.float()
    return W_U[:, it.pos_ids].mean(1) - W_U[:, it.neg_ids].mean(1)


@torch.no_grad()
def analyse(model, items, condition_of, bs=16):
    L, H = model.cfg.n_layers, model.cfg.n_heads
    names = {f"blocks.{l}.hook_resid_post" for l in range(L)} | {"blocks.0.hook_resid_pre", "ln_final.hook_scale"}
    names |= {f"blocks.{l}.hook_attn_out" for l in range(L)} | {f"blocks.{l}.hook_mlp_out" for l in range(L)}
    names |= {f"blocks.{l}.attn.hook_z" for l in range(L)} | {f"blocks.{l}.ln1_post.hook_scale" for l in range(L)}
    w_lnf = model.ln_final.w.float()
    lens = np.zeros((len(items), L))
    comps = np.zeros((len(items), 1 + 2 * L))  # embed, attn_0, mlp_0, attn_1, ...
    heads = np.zeros((len(items), L, H))
    recon_err = []
    latent_lp = {}
    seqs = [it.ids for it in items]
    for grp in length_groups(seqs, bs):
        toks = torch.tensor([seqs[i] for i in grp], device=model.cfg.device)
        _, cache = model.run_with_cache(toks, names_filter=lambda n: n in names)
        U = torch.stack([direction(model, items[i]) for i in grp])  # [B, d]
        dev = U.device  # device of W_U (last device when the model is split over GPUs)
        u = U * w_lnf.to(dev)
        s = cache["ln_final.hook_scale"][:, -1].float().to(dev)  # [B, 1]
        # ---- logit lens
        for l in range(L):
            x = cache[f"blocks.{l}.hook_resid_post"][:, -1].to(dev)
            lg = model.unembed(model.ln_final(x[:, None]))[:, 0].float()
            if model.cfg.output_logits_soft_cap > 0:
                c = model.cfg.output_logits_soft_cap
                lg = c * torch.tanh(lg / c)
            for j, i in enumerate(grp):
                it = items[i]
                lens[i, l] = (torch.logsumexp(lg[j, it.pos_ids], -1) - torch.logsumexp(lg[j, it.neg_ids], -1)).item()
        # ---- latent-language lens at the position before the target prefix
        lang = items[grp[0]].lang
        if lang in LATENT and items[grp[0]].contrast == "c1":
            pos = items[grp[0]].base_len - 1
            tids = [model.to_single_token(t) if len(model.to_tokens(t, prepend_bos=False)[0]) == 1 else None
                    for t in LATENT[lang]]
            for l in range(L):
                x = cache[f"blocks.{l}.hook_resid_post"][:, pos].to(dev)
                lp = model.unembed(model.ln_final(x[:, None]))[:, 0].float().log_softmax(-1)
                for t, tid in zip(LATENT[lang], tids):
                    if tid is not None:
                        latent_lp.setdefault(t, np.zeros((len(items), L)))[grp, l] = lp[:, tid].cpu().numpy()
        # ---- DLA
        terms = [cache["blocks.0.hook_resid_pre"][:, -1].to(dev)]
        for l in range(L):
            terms += [cache[f"blocks.{l}.hook_attn_out"][:, -1].to(dev), cache[f"blocks.{l}.hook_mlp_out"][:, -1].to(dev)]
        T = torch.stack(terms, 1).float()  # [B, 1+2L, d]
        final = cache[f"blocks.{L - 1}.hook_resid_post"][:, -1].float().to(dev)
        recon_err.append(((T.sum(1) - final).norm(dim=-1) / final.norm(dim=-1)).max().item())
        comps[grp] = ((T / s[:, None]) * u[:, None]).sum(-1).cpu().numpy()
        for l in range(L):
            z = cache[f"blocks.{l}.attn.hook_z"][:, -1].float()  # [B, H, d_head]
            zd = z.device  # cache and block weights may sit on different GPUs when the model is split
            per_head = torch.einsum("bhe,hed->bhd", z, model.blocks[l].attn.W_O.float().to(zd))
            sc = cache[f"blocks.{l}.ln1_post.hook_scale"][:, -1].float().to(zd)  # [B, 1]
            per_head = per_head * model.blocks[l].ln1_post.w.float().to(zd) / sc[:, None]
            heads[grp, l] = ((per_head.to(dev) / s[:, None]) * u[:, None]).sum(-1).cpu().numpy()
    cond = np.array([condition_of(it) for it in items])
    return dict(lens=lens, comps=comps, heads=heads, cond=cond, recon_err=max(recon_err), latent=latent_lp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", default="c1", choices=["c1", "c2"])
    ap.add_argument("--lang", default="hi")
    args = ap.parse_args()
    out = ROOT / "results" / "01_lens_dla" / f"{args.contrast}_{args.lang}"
    out.mkdir(parents=True, exist_ok=True)
    model = load_model()
    L = model.cfg.n_layers

    if args.contrast == "c1":
        items = [i for i in c1_items(model, args.lang) if i.ok]
        condA, condB = "f", "m"
        cond_of = lambda it: it.meta["gender"]
        labels = {"f": "fem noun", "m": "masc noun"}
    else:
        items = [i for i in c2_items(model, args.lang) if i.ok]
        condA, condB = "clean", "corrupt"
        cond_of = lambda it: it.meta["cond"]
        labels = {"clean": "मैंने O", "corrupt": "मैं O"}
    r = analyse(model, items, cond_of)
    A, B = r["cond"] == condA, r["cond"] == condB
    info = dict(n=len(items), nA=int(A.sum()), nB=int(B.sum()), dla_reconstruction_rel_err=r["recon_err"])

    # ---- logit lens plot
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for m, col in [(A, "#c0392b"), (B, "#2471a3")]:
        if m.sum() == 0:
            continue
        mu, lo, hi = bootstrap_ci_axis0(r["lens"][m])
        ax.plot(range(L), mu, color=col, label=labels[condA if m is A else condB])
        ax.fill_between(range(L), lo, hi, color=col, alpha=.2)
    ax.axhline(0, color="k", lw=.6)
    ax.set_xlabel("layer (resid_post)")
    ax.set_ylabel("lens Δ")
    ax.set_title(f"{args.contrast}/{args.lang} logit lens: Δ at final position")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "logit_lens.png", dpi=150)
    plt.close(fig)
    gap = r["lens"][A].mean(0) - (r["lens"][B].mean(0) if B.sum() else 0)
    pd.DataFrame(dict(layer=range(L), lens_A=r["lens"][A].mean(0),
                      lens_B=r["lens"][B].mean(0) if B.sum() else np.nan, gap=gap)).to_csv(out / "logit_lens.csv", index=False)
    final_gap = gap[-1]
    info["lens_gap_final"] = float(final_gap)
    info["lens_first_layer_gap_over_half_final"] = int(np.argmax(gap > .5 * final_gap)) if final_gap > 0 else None
    info["lens_first_layer_correct_sign_both"] = int(np.argmax((r["lens"][A].mean(0) > 0) & ((r["lens"][B].mean(0) < 0) if B.sum() else True)))

    # ---- latent-language lens
    if r["latent"]:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for t, arr in r["latent"].items():
            ax.plot(range(L), np.exp(arr).mean(0), label=repr(t))
        ax.set_yscale("log")
        ax.set_xlabel("layer")
        ax.set_ylabel("mean prob (lens)")
        ax.set_title(f"{args.lang}: lens probs at last prompt token (before possessive)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "latent_lens.png", dpi=150)
        plt.close(fig)
        pd.DataFrame({t: np.exp(a).mean(0) for t, a in r["latent"].items()}).to_csv(out / "latent_lens.csv")

    # ---- DLA (differential)
    names = ["embed"] + [f"{k}{l}" for l in range(L) for k in ("attn", "mlp")]
    diff = r["comps"][A].mean(0) - (r["comps"][B].mean(0) if B.sum() else 0)
    dla = pd.DataFrame(dict(component=names, mean_A=r["comps"][A].mean(0),
                            mean_B=r["comps"][B].mean(0) if B.sum() else np.nan, diff=diff))
    dla.to_csv(out / "dla_components.csv", index=False)
    hd = r["heads"][A].mean(0) - (r["heads"][B].mean(0) if B.sum() else 0)
    hdf = pd.DataFrame([dict(layer=l, head=h, diff=hd[l, h]) for l in range(L) for h in range(model.cfg.n_heads)])
    hdf.to_csv(out / "dla_heads.csv", index=False)
    info["top_components"] = dla.reindex(dla["diff"].abs().sort_values(ascending=False).index).head(8)[["component", "diff"]].values.tolist()
    info["top_heads"] = hdf.reindex(hdf["diff"].abs().sort_values(ascending=False).index).head(8).values.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[2, 1]))
    x = np.arange(L)
    axes[0].bar(x - .2, diff[1::2], .4, label="attn", color="#e67e22")
    axes[0].bar(x + .2, diff[2::2], .4, label="mlp", color="#16a085")
    axes[0].axhline(0, color="k", lw=.6)
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel(f"DLA({condA}) − DLA({condB})")
    axes[0].set_title(f"{args.contrast}/{args.lang} differential DLA (embed: {diff[0]:+.2f})")
    axes[0].legend()
    lim = np.abs(hd).max()
    im = axes[1].imshow(hd, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
    axes[1].set_xlabel("head")
    axes[1].set_ylabel("layer")
    axes[1].set_title("per-head differential DLA")
    fig.colorbar(im, ax=axes[1])
    fig.tight_layout()
    fig.savefig(out / "dla.png", dpi=150)
    plt.close(fig)
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(info, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
