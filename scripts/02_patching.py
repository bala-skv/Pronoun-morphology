"""Stage 2 — activation patching.

  python scripts/02_patching.py --contrast c1 --lang hi --n-pairs 60
  python scripts/02_patching.py --contrast c2 --lang hi

Passes (each saved as soon as it finishes, so partial results survive):
  0. sanity: identity patch (clean->clean) and full layer-0 patch (should give R = 1)
  1. attribution patching over resid_pre (fast linear sweep)
  2. activation patching of resid_pre at every layer x patchable position
  3. attn_out / mlp_out at every layer x position
  4. individual heads (z) at the layers where attn_out matters (--heads)
Outputs: results/02_patching/<contrast>_<lang>/*.csv, *.png
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pronoun.metrics import bootstrap_ci, fmt_ci
from pronoun.model import load_model
from pronoun.patching import (ROLE_ORDER, activation_patch, attribution_patch, baseline,
                              recovery_bootstrap, recovery_table, roles_for_pair)
from pronoun.prompts import c1_items, c1_pairs, c2_items, c2_pairs

plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Sans Devanagari", "Noto Sans Gurmukhi", "Noto Sans Tamil"]


def heatmap(df, value, title, path, vmin=-1, vmax=1, col="role"):
    roles = [r for r in ROLE_ORDER if r in set(df[col])]
    piv = df.pivot_table(index="layer", columns=col, values=value).reindex(columns=roles)
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(roles), 7))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
    ax.set_xticks(range(len(roles)), roles, rotation=45, ha="right")
    ax.set_yticks(range(0, len(piv), 2), piv.index[::2])
    ax.set_ylabel("layer")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", default="c1", choices=["c1", "c2"])
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--n-pairs", type=int, default=60)
    ap.add_argument("--keep", default="correct", choices=["correct", "distinguishes", "all"],
                    help="correct: model right on both clean and corrupt; distinguishes: Δ_clean > Δ_corrupt")
    ap.add_argument("--passes", default="sanity,attr,attr_am,attr_z,resid,attn_mlp,heads")
    ap.add_argument("--heads-top-layers", type=int, default=6)
    ap.add_argument("--heads-end-only", action="store_true", help="patch heads only at the final position")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--layers", default=None, help="comma-separated subset of layers (default: all)")
    ap.add_argument("--tag", default="", help="suffix for the output directory")
    args = ap.parse_args()
    passes = set(args.passes.split(","))

    out = ROOT / "results" / "02_patching" / f"{args.contrast}_{args.lang}{args.tag}"
    layer_list = [int(x) for x in args.layers.split(",")] if args.layers else None
    out.mkdir(parents=True, exist_ok=True)
    model = load_model()
    t0 = time.time()

    if args.contrast == "c1":
        items = c1_items(model, args.lang)
        pairs = c1_pairs(items, both_directions=True)
    else:
        items = c2_items(model, args.lang)
        pairs = c2_pairs(items)
    pairs = [p for p in pairs if len(p[0].ids) == len(p[1].ids)]
    # Reuse Stage-0 per-prompt Δ (if available) to filter pairs instead of re-running every candidate.
    st0 = ROOT / "results" / "00_behaviour" / ("c1_items_ctx.csv" if args.contrast == "c1" else "c2_items.csv")
    cached = {}
    if st0.exists():
        s0 = pd.read_csv(st0)
        s0 = s0[(s0.lang == args.lang) & s0.ok]
        cached = dict(zip(s0.prompt, s0.delta))
    if cached and all(p[0].prompt in cached and p[1].prompt in cached for p in pairs):
        d_clean = np.array([s * cached[c.prompt] for c, x, s in pairs])
        d_corr = np.array([s * cached[x.prompt] for c, x, s in pairs])
        print("[filter] using Stage-0 deltas", flush=True)
    else:
        d_clean, d_corr = baseline(model, pairs)
    if args.keep == "correct":
        keep = (d_clean > 0) & (d_corr < 0)
    elif args.keep == "distinguishes":
        keep = d_clean > d_corr
    else:
        keep = np.ones(len(pairs), bool)
    n_all = len(pairs)
    pairs = [p for p, k in zip(pairs, keep) if k][: args.n_pairs]
    d_clean, d_corr = baseline(model, pairs)
    info = dict(contrast=args.contrast, lang=args.lang, n_candidate_pairs=n_all,
                n_correct=int(keep.sum()), n_used=len(pairs),
                d_clean=fmt_ci(bootstrap_ci(d_clean)), d_corr=fmt_ci(bootstrap_ci(d_corr)),
                example_clean=pairs[0][0].prompt + pairs[0][0].prefix_str,
                example_corrupt=pairs[0][1].prompt + pairs[0][1].prefix_str,
                example_roles={r: model.tokenizer.convert_ids_to_tokens(pairs[0][0].ids[p])
                               for p, r in roles_for_pair(model, pairs[0][0], pairs[0][1]).items()})
    print(json.dumps(info, ensure_ascii=False, indent=1))
    pd.DataFrame([dict(clean=c.prompt, corrupt=x.prompt, sign=s, d_clean=a, d_corr=b)
                  for (c, x, s), a, b in zip(pairs, d_clean, d_corr)]).to_csv(out / "pairs.csv", index=False)

    # ------------------------------------------------------------------ 0. sanity checks
    if "sanity" in passes:
        from pronoun.metrics import delta_from_logits

        sub = pairs[:16]
        L = len(sub[0][0].ids)
        sub = [p for p in sub if len(p[0].ids) == L]
        clean = torch.tensor([p[0].ids for p in sub], device=model.cfg.device)
        corr = torch.tensor([p[1].ids for p in sub], device=model.cfg.device)
        _, cache = model.run_with_cache(clean, names_filter=lambda n: n == "blocks.0.hook_resid_pre")
        src = cache["blocks.0.hook_resid_pre"]

        def full0(act, hook):
            return src.clone()

        def dl(logits):
            return np.array([s * delta_from_logits(logits[i, -1].float(), c.pos_ids, c.neg_ids).item()
                             for i, (c, _, s) in enumerate(sub)])

        from pronoun.metrics import last_logits

        full = model(clean)[:, -1].float()
        info["sanity_last_logits_vs_full_maxabs"] = float((last_logits(model, clean) - full).abs().max())
        dc = dl(model(clean))
        dfull = dl(model.run_with_hooks(corr, fwd_hooks=[("blocks.0.hook_resid_pre", full0)]))
        did = dl(model.run_with_hooks(clean, fwd_hooks=[("blocks.0.hook_resid_pre", full0)]))
        info["sanity_full_layer0_patch_maxabs_err"] = float(np.abs(dfull - dc).max())
        info["sanity_identity_patch_maxabs_err"] = float(np.abs(did - dc).max())
        print("sanity:", info["sanity_full_layer0_patch_maxabs_err"], info["sanity_identity_patch_maxabs_err"])
    prev = json.loads((out / "info.json").read_text()) if (out / "info.json").exists() else {}
    prev.update(info)  # keep sanity numbers etc. from earlier passes into the same directory
    (out / "info.json").write_text(json.dumps(prev, ensure_ascii=False, indent=1))

    # ------------------------------------------------------------------ 1. attribution patching
    attr_kinds = [k for k, flag in [("resid_pre", "attr"), ("attn_out", "attr_am"), ("mlp_out", "attr_am"), ("z", "attr_z")]
                  if flag in passes]
    for kind in attr_kinds:
        res = attribution_patch(model, pairs, kind, batch_size=8)
        rows = []
        for site, v in res["per_pair"].items():
            idx = res["pair_idx"][site]
            rows.append(dict(layer=site[0], role=site[1], head=site[2] if len(site) > 2 else None,
                             R=v.mean() / (d_clean[idx].mean() - d_corr[idx].mean())))
        df = pd.DataFrame(rows)
        df.to_csv(out / f"attr_{kind}.csv", index=False)
        if kind != "z":
            heatmap(df, "R", f"{args.contrast}/{args.lang} attribution patching ({kind}), est. R", out / f"attr_{kind}.png")
        else:
            piv = df.pivot_table(index="layer", columns="head", values="R", aggfunc="sum")
            fig, ax = plt.subplots(figsize=(5, 7))
            lim = max(abs(piv.values).max(), 1e-3)
            im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
            ax.set_xlabel("head"); ax.set_ylabel("layer")
            ax.set_title(f"{args.contrast}/{args.lang} attribution patching per head\n(summed over positions), est. R", fontsize=9)
            fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out / "attr_z.png", dpi=150); plt.close(fig)
        print(f"[attr {kind}] done {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ 2-3. activation patching
    kinds = (["resid_pre"] if "resid" in passes else []) + (["attn_out", "mlp_out"] if "attn_mlp" in passes else [])
    for kind in kinds:
        res = activation_patch(model, pairs, kind, layers=layer_list, batch_size=args.batch_size)
        df = pd.DataFrame(recovery_table(res, d_clean, d_corr))
        ci = recovery_bootstrap(res, d_clean, d_corr)
        df["R_lo"] = [ci[s][0] for s in df.site]
        df["R_hi"] = [ci[s][1] for s in df.site]
        df.drop(columns="site").to_csv(out / f"act_{kind}.csv", index=False)
        heatmap(df, "R", f"{args.contrast}/{args.lang} activation patching ({kind}), R  n={len(pairs)}",
                out / f"act_{kind}.png")
        print(f"[{kind}] done {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ 4. heads
    if "heads" in passes:
        attn_csv = out / "act_attn_out.csv"
        if attn_csv.exists():
            a = pd.read_csv(attn_csv)
            top = a.groupby("layer").R.apply(lambda r: r.abs().max()).sort_values(ascending=False)
            layers = sorted(top.index[: args.heads_top_layers].tolist())
        else:
            layers = list(range(model.cfg.n_layers))
        print("[heads] layers", layers, flush=True)
        res = activation_patch(model, pairs, "z", layers=layers, heads="all", batch_size=args.batch_size,
                               role_filter=(lambda r: r == "END") if args.heads_end_only else None)
        df = pd.DataFrame(recovery_table(res, d_clean, d_corr)).drop(columns="site")
        df.to_csv(out / "act_heads.csv", index=False)
        for role in sorted(set(df.role), key=ROLE_ORDER.index):
            pass
        piv = df.pivot_table(index=["layer", "head"], columns="role", values="R")
        piv = piv.reindex(columns=[r for r in ROLE_ORDER if r in piv.columns])
        fig, ax = plt.subplots(figsize=(1 + .9 * piv.shape[1], .22 * len(piv) + 1))
        im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=-.5, vmax=.5)
        ax.set_yticks(range(len(piv)), [f"L{l}H{h}" for l, h in piv.index], fontsize=6)
        ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=45, ha="right")
        ax.set_title(f"{args.contrast}/{args.lang} head patching (z), R", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=.05)
        fig.tight_layout()
        fig.savefig(out / "act_heads.png", dpi=150)
        plt.close(fig)
        print(f"[heads] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
