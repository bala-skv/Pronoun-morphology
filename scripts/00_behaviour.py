"""Stage 0 — tokenization + behavioural go/no-go check.

Outputs (results/00_*):
  tokenization.csv     subword counts of every paradigm form (Table 1), tokenized in context
  c1_items.csv         per-prompt Δ for possessive agreement (hi, pa) / log p(mine) controls (en, ta)
  c2_items.csv         per-prompt Δ for the ergative contrast (hi clean/corrupt, pa control)
  summary.md           accuracies (C1 with bare frames and with context frames)
                       + bootstrap CIs, drop reasons
  c1_delta.png, c2_delta.png
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Sans Devanagari", "Noto Sans Gurmukhi", "Noto Sans Tamil"]
import numpy as np
import pandas as pd
import torch

from pronoun.metrics import bootstrap_ci, delta_from_logits, fmt_ci, length_groups
from pronoun.model import load_model
from pronoun.prompts import c1_items, c2_items, continuation_ids, load_lexicon, tok_ids

OUT = ROOT / "results" / "00_behaviour"
OUT.mkdir(parents=True, exist_ok=True)
NEUTRAL = {"en": "and", "hi": "और", "pa": "ਅਤੇ", "ta": "மற்றும்"}


@torch.no_grad()
def run_items(model, items, bs=32):
    """For each ok item: Δ at last position; P(first shared-prefix token); full-form log-prob."""
    seqs, meta = [], []
    for it in items:
        if not it.ok:
            seqs.append(None)
            continue
        base_len = len(tok_ids(model, it.prompt))
        if it.neg_ids:  # contrast item: sequence = base + shared prefix
            seqs.append(it.ids)
        else:  # control item: base + full continuation, to get log p(form)
            seqs.append(it.ids + it.pos_ids)
        meta.append(base_len)
    res = [dict() for _ in items]
    valid = [i for i, s in enumerate(seqs) if s is not None]
    vseqs = [seqs[i] for i in valid]
    for grp in length_groups(vseqs, bs):
        toks = torch.tensor([vseqs[g] for g in grp], device=model.cfg.device)
        # unembed only the positions from the end of the prompt onward (much cheaper than all positions)
        b0 = min(len(tok_ids(model, items[valid[g]].prompt)) for g in grp) - 1
        resid = model(toks, stop_at_layer=model.cfg.n_layers)[:, b0:]
        lg = model.unembed(model.ln_final(resid)).float()
        cap = model.cfg.output_logits_soft_cap
        if cap > 0:
            lg = cap * torch.tanh(lg / cap)
        lp = lg.log_softmax(-1).cpu()  # NB: position index p maps to lp[:, p - b0]
        for j, g in enumerate(grp):
            i = valid[g]
            it = items[i]
            base_len = len(tok_ids(model, it.prompt))
            if it.neg_ids:
                last = lp[j, len(it.ids) - 1 - b0]
                res[i]["delta"] = delta_from_logits(last, it.pos_ids, it.neg_ids).item()
                res[i]["lp_pos"] = torch.logsumexp(last[it.pos_ids], -1).item()
                res[i]["lp_neg"] = torch.logsumexp(last[it.neg_ids], -1).item()
                n_pref = len(it.ids) - base_len
                if n_pref > 0:  # prob. that the model starts the possessive/verb at all
                    res[i]["p_prefix"] = float(lp[j, base_len - 1 - b0, it.ids[base_len]].exp())
                else:
                    res[i]["p_prefix"] = float(torch.logsumexp(last[it.pos_ids + it.neg_ids], -1).exp())
            else:
                cont = it.pos_ids
                res[i]["lp_form"] = float(sum(lp[j, base_len - 1 - b0 + k, t] for k, t in enumerate(cont)))
    return res


def items_df(items, res):
    rows = []
    for it, r in zip(items, res):
        rows.append(dict(lang=it.lang, prompt=it.prompt, ok=it.ok, why=it.why, prefix=it.prefix_str,
                         n_prompt_toks=len(it.ids) if it.ids else None,
                         form_ntoks=str(it.form_ntoks), **it.meta, **r))
    return pd.DataFrame(rows)


def c1_report(c1, variant):
    lines = [f"## Contrast 1 — possessive agreement ({variant} frames)\n",
             "Hindi/Punjabi: Δ = logit(fem possessive) − logit(masc possessive) at the decision token. "
             "Correct = Δ>0 for feminine nouns, Δ<0 for masculine.\n"]
    for lang in ["hi", "pa"]:
        d = c1[(c1.lang == lang) & c1.ok]
        corr = np.where(d.gender == "f", d.delta > 0, d.delta < 0).astype(float)
        df_, dm_ = d[d.gender == "f"].delta.values, d[d.gender == "m"].delta.values
        lines.append(f"**{lang}** (n={len(d)}, dropped={int((~c1[c1.lang == lang].ok).sum())}, "
                     f"prefix='{d.prefix.iloc[0] if len(d) else ''}')  \n"
                     f"- accuracy: {fmt_ci(bootstrap_ci(corr))}  \n"
                     f"- mean Δ | fem noun: {fmt_ci(bootstrap_ci(df_))}  \n"
                     f"- mean Δ | masc noun: {fmt_ci(bootstrap_ci(dm_))}  \n"
                     f"- mean P(possessive prefix): {d.p_prefix.mean():.3f}  \n")
        for fr, g in d.groupby("frame"):
            c = np.where(g.gender == "f", g.delta > 0, g.delta < 0).mean()
            lines.append(f"  - frame {fr}: acc={c:.3f} (n={len(g)}) e.g. `{g.prompt.iloc[0]}`  \n")
    lines.append("\nControls (en, ta): log p(form) for translations of Hindi-fem vs Hindi-masc nouns; "
                 "the difference should be ≈0 (no agreement in these languages).\n")
    for lang in ["en", "ta"]:
        d = c1[(c1.lang == lang) & c1.ok]
        f, m = d[d.gender == "f"].lp_form.values, d[d.gender == "m"].lp_form.values
        rng = np.random.default_rng(0)
        boots = [rng.choice(f, len(f)).mean() - rng.choice(m, len(m)).mean() for _ in range(2000)]
        lines.append(f"**{lang}** (n={len(d)}): mean log p | fem-transl = {f.mean():.3f}, "
                     f"masc-transl = {m.mean():.3f}, diff = {f.mean() - m.mean():.3f} "
                     f"[{np.quantile(boots, .025):.3f}, {np.quantile(boots, .975):.3f}]  \n")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, lang in zip(axes, ["hi", "pa"]):
        d = c1[(c1.lang == lang) & c1.ok]
        for g, col in [("f", "#c0392b"), ("m", "#2471a3")]:
            ax.hist(d[d.gender == g].delta, bins=30, alpha=.6, color=col, label=f"{g} noun")
        ax.axvline(0, color="k", lw=.8)
        ax.set_title(f"{lang} ({variant}): Δ = logit(fem) − logit(masc)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"c1_delta_{variant}.png", dpi=150)
    plt.close(fig)
    return lines


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="tok,c1,c2")
    parts = set(ap.parse_args().parts.split(","))
    model = load_model()
    lines = []
    if "tok" in parts:
        # ---------------------------------------------------------------- tokenization of Table 1
        lex = load_lexicon()
        trows = []
        for _, r in lex.iterrows():
            cont = continuation_ids(model, NEUTRAL[r.lang], r.form)
            trows.append(dict(**r, n_tokens=len(cont) if cont else None,
                              tokens=" | ".join(model.tokenizer.convert_ids_to_tokens(cont)) if cont else "MERGE"))
        tdf = pd.DataFrame(trows)
        tdf.to_csv(OUT / "tokenization.csv", index=False)
        lines += ["## Subword counts of paradigm forms (in context, after a neutral word)\n",
                  tdf[["lang", "function", "form", "n_tokens", "tokens"]].to_markdown(index=False), "\n"]
        (OUT / "summary_tok.md").write_text("\n".join(lines))

    lines = []
    if "c1" in parts:
        # ---------------------------------------------------------------- Contrast 1
        for variant in ["bare", "ctx"]:
            c1 = []
            for lang in ["hi", "pa", "en", "ta"]:
                items = c1_items(model, lang, bare=(variant == "bare"))
                print(f"[c1/{variant}] {lang}: {sum(i.ok for i in items)}/{len(items)} ok", flush=True)
                c1.append(items_df(items, run_items(model, items)))
            c1 = pd.concat(c1)
            c1.to_csv(OUT / f"c1_items_{variant}.csv", index=False)
            lines.extend(c1_report(c1, variant))
        (OUT / "summary_c1.md").write_text("\n".join(lines))

    lines = []
    if "c2" in parts:
        # ---------------------------------------------------------------- Contrast 2
        c2 = []
        for lang in ["hi", "pa"]:
            items = c2_items(model, lang)
            print(f"[c2] {lang}: {sum(i.ok for i in items)}/{len(items)} ok")
            c2.append(items_df(items, run_items(model, items)))
        c2 = pd.concat(c2)
        c2.to_csv(OUT / "c2_items.csv", index=False)
        lines.append("\n## Contrast 2 — ergative cue → verb aspect\n")
        lines.append("Δ = logit(perfective, agreeing with object) − logsumexp(habitual masc, habitual fem). "
                     "Expect Δ>0 after मैंने (clean) and Δ<0 after मैं (corrupt).\n")
        h = c2[(c2.lang == "hi") & c2.ok]
        for cond in ["clean", "corrupt"]:
            d = h[h.cond == cond]
            corr = (d.delta > 0) if cond == "clean" else (d.delta < 0)
            lines.append(f"**hi {cond}** (n={len(d)}): mean Δ = {fmt_ci(bootstrap_ci(d.delta.values))}, "
                         f"acc = {fmt_ci(bootstrap_ci(corr.values.astype(float)))}, "
                         f"mean P(verb prefix)={d.p_prefix.mean():.3f}  \n")
        piv = h.pivot_table(index="row_id", columns="cond", values="delta").dropna()
        if len(piv):
            lines.append(f"**hi paired** Δ_clean − Δ_corrupt (n={len(piv)}): "
                         f"{fmt_ci(bootstrap_ci((piv.clean - piv.corrupt).values))}, "
                         f"frac>0 = {(piv.clean > piv.corrupt).mean():.3f}  \n")
        p = c2[(c2.lang == "pa") & c2.ok]
        if len(p):
            lines.append(f"**pa control** ਮੈਂ O (n={len(p)}): mean Δ = {fmt_ci(bootstrap_ci(p.delta.values))}  \n")
        drops = c2[~c2.ok]
        if len(drops):
            lines.append("\nDropped c2 items: " + "; ".join(f"{r.prompt} ({r.why})" for r in drops.itertuples()) + "\n")

        fig, ax = plt.subplots(figsize=(5, 3.5))
        for cond, col in [("clean", "#c0392b"), ("corrupt", "#2471a3")]:
            ax.hist(h[h.cond == cond].delta, bins=20, alpha=.6, color=col, label={"clean": "मैंने O", "corrupt": "मैं O"}[cond])
        ax.axvline(0, color="k", lw=.8)
        ax.set_title("hi: Δ = logit(perf) − lse(habitual)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "c2_delta.png", dpi=150)
        plt.close(fig)
        (OUT / "summary_c2.md").write_text("\n".join(lines))

    full = ["# Stage 0: tokenization and behaviour\n"] + [
        (OUT / f"summary_{k}.md").read_text() for k in ("tok", "c1", "c2") if (OUT / f"summary_{k}.md").exists()]
    (OUT / "summary.md").write_text("\n".join(full))
    print("\n".join(full))


if __name__ == "__main__":
    main()
