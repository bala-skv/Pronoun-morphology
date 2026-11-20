"""Stage 5 — what do the top agreement heads attend to? (follow-up to DLA / head patching)

  python scripts/05_attention.py --lang hi --heads 21.4,23.1,24.2,18.6

Context frames contain two example sentences ("यह कलम मेरी है। यह घर मेरा है।") before the test noun.
For each head, from the final position (and from the last prompt token), we sum attention to word groups:
  noun            the test noun (whose gender decides the answer)
  ctx_poss_same   the in-context possessive that matches the test noun's gender (e.g. मेरी for किताब)
  ctx_poss_other  the in-context possessive of the other gender
  ctx_noun_same / ctx_noun_other   the in-context example nouns
  filler, bos, other
"Reads gender from the noun" predicts mass on `noun`; "retrieval by analogy" predicts `ctx_poss_same`.
"""
import argparse
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

from pronoun.metrics import length_groups
from pronoun.model import load_model
from pronoun.prompts import C1_FRAMES, c1_items, tok_ids

OUT = ROOT / "results" / "05_attention"
OUT.mkdir(parents=True, exist_ok=True)
CTX_GENDER = {"कलम": "f", "घर": "m", "कुर्सी": "f", "कमरा": "m", "ਕਲਮ": "f", "ਘਰ": "m", "ਕੁਰਸੀ": "f", "ਕਮਰਾ": "m"}
POSS = {"मेरी": "f", "मेरा": "m", "ਮੇਰੀ": "f", "ਮੇਰਾ": "m"}


def word_labels(model, it):
    """Label every token position of it.ids by word group."""
    frame = C1_FRAMES[it.lang][it.meta["frame"]]
    pre, post = frame.split("{N}")
    words = pre.split() + ["{N}"] + post.split()
    g = it.meta["gender"]
    labels = ["bos"] + [None] * (len(it.ids) - 1)
    text = ""
    prev = 1
    for k, w in enumerate(words):
        text = (text + " " if text else "") + (it.meta["noun"] if w == "{N}" else w)
        n = len(tok_ids(model, text))
        wc = w.strip("।.,")
        if w == "{N}":
            lab = "noun"
        elif wc in POSS:
            lab = "ctx_poss_same" if POSS[wc] == g else "ctx_poss_other"
        elif wc in CTX_GENDER:
            lab = "ctx_noun_same" if CTX_GENDER[wc] == g else "ctx_noun_other"
        elif k > words.index("{N}"):
            lab = "filler"
        else:
            lab = "other"
        for p in range(prev, min(n, len(labels))):
            labels[p] = lab
        prev = n
    for p in range(prev, len(labels)):
        labels[p] = "target_prefix"
    return labels


GROUPS = ["noun", "ctx_poss_same", "ctx_poss_other", "ctx_noun_same", "ctx_noun_other", "filler", "other", "bos",
          "target_prefix"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--heads", default="21.4,23.1,24.2,18.6,24.3")
    ap.add_argument("--n", type=int, default=80)
    args = ap.parse_args()
    heads = [tuple(int(x) for x in h.split(".")) for h in args.heads.split(",")]
    model = load_model()
    items = [i for i in c1_items(model, args.lang) if i.ok]
    rng = np.random.default_rng(0)
    items = [items[i] for i in rng.choice(len(items), min(args.n, len(items)), replace=False)]
    labels = [word_labels(model, it) for it in items]
    names = {f"blocks.{l}.attn.hook_pattern" for l, _ in heads}
    rows = []
    seqs = [it.ids for it in items]
    for grp in length_groups(seqs, 16):
        toks = torch.tensor([seqs[i] for i in grp], device=model.cfg.device)
        _, cache = model.run_with_cache(toks, names_filter=lambda n: n in names)
        for l, h in heads:
            pat = cache[f"blocks.{l}.attn.hook_pattern"][:, h].float().cpu()  # [B, q, k]
            for j, i in enumerate(grp):
                it, lab = items[i], labels[i]
                for qname, q in [("END", len(it.ids) - 1), ("last_prompt_tok", it.base_len - 1)]:
                    a = pat[j, q]
                    agg = {gname: 0.0 for gname in GROUPS}
                    for p, gname in enumerate(lab[: q + 1]):
                        agg[gname] += float(a[p])
                    rows.append(dict(layer=l, head=h, query=qname, gender=it.meta["gender"], **agg))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"attention_{args.lang}.csv", index=False)
    summ = df.groupby(["layer", "head", "query"])[GROUPS].mean().round(3)
    summ.to_csv(OUT / f"attention_{args.lang}_summary.csv")
    print(summ.to_string())
    sub = summ.xs("END", level="query")
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(sub) + 1.5))
    left = np.zeros(len(sub))
    for gname, col in zip(GROUPS, plt.cm.tab10.colors):
        ax.barh(range(len(sub)), sub[gname], left=left, color=col, label=gname)
        left += sub[gname].values
    ax.set_yticks(range(len(sub)), [f"L{l}H{h}" for l, h in sub.index])
    ax.set_xlabel("mean attention from END")
    ax.set_title(f"{args.lang}: where do the agreement heads look?")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / f"attention_{args.lang}.png", dpi=150)


if __name__ == "__main__":
    main()
