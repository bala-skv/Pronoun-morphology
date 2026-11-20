"""Stage 3a — where is the possessed noun's gender readable, and is the direction shared across languages?

Template activations from Contrast 1 (context frames). Two positions per prompt:
  noun   = last subword of the possessed noun
  final  = last prompt token (right before the possessive / "mine")
Per layer (0 = embeddings, l = resid_post of block l-1):
  * in-language accuracy, random 5-fold CV                                             [RQ1]
  * control task (random label per noun type), same splits -> selectivity = acc − control   [Hewitt & Liang]
  * acc_unseen: folds split by noun, so test nouns are never seen in training (strict)
  * transfer: probe trained on Hindi, tested on Punjabi / English / Tamil              [RQ3]
      Punjabi labels = Punjabi gender; English/Tamil labels = Hindi gender of the translation
      (no grammatical gender there, so transfer ≈ chance unless semantics leaks)
  * shuffled-label transfer control, and cosine(hi probe, pa probe)
Outputs: results/03_probe_gender/*.csv, *.png
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pronoun.model import load_model
from sklearn.model_selection import StratifiedKFold

from pronoun.probing import (bootstrap_acc, collect_resid, control_labels, cv_probe, majority_baseline,
                             make_probe, probe_direction)
from pronoun.prompts import C1_FRAMES, c1_items, tok_ids

OUT = ROOT / "results" / "03_probe_gender"
OUT.mkdir(parents=True, exist_ok=True)
LANGS = ["hi", "pa", "en", "ta"]


def cv_random(X, y, seed=0):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(np.mean([(make_probe().fit(X[tr], y[tr]).predict(X[te]) == y[te]).mean()
                          for tr, te in skf.split(X, y)]))


def noun_end(model, it):
    frame = C1_FRAMES[it.lang][it.meta["frame"]]
    pre = frame.split("{N}")[0] + it.meta["noun"]
    ids = tok_ids(model, pre)
    assert it.ids[: len(ids)] == ids, (pre, it.prompt)
    return len(ids) - 1


def main():
    model = load_model()
    data = {}
    for lang in LANGS:
        items = [i for i in c1_items(model, lang) if i.ok]
        seqs = [i.ids[: i.base_len] for i in items]
        pos = [[noun_end(model, i), i.base_len - 1] for i in items]
        X, owner = collect_resid(model, seqs, pos)
        X = X.reshape(len(items), 2, *X.shape[1:])  # [N, 2 positions, L+1, d]
        y = np.array([1 if i.meta["gender"] == "f" else 0 for i in items])
        g = np.array([i.meta["noun_id"] for i in items])
        data[lang] = (X, y, g)
        print(lang, X.shape, "f-share", y.mean(), flush=True)
        np.savez_compressed(OUT / f"acts_{lang}.npz", X=X, y=y, g=g)

    nL = data["hi"][0].shape[2]
    rows = []
    for pi, pname in enumerate(["noun", "final"]):
        for l in range(nL):
            Xh, yh, gh = data["hi"]
            Xh_l = Xh[:, pi, l].astype(np.float32)
            w_hi, clf_hi = probe_direction(Xh_l, yh)
            ysh = np.random.default_rng(l).permutation(yh)
            _, clf_sh = probe_direction(Xh_l, ysh)
            for lang in LANGS:
                X, y, g = data[lang]
                Xl = X[:, pi, l].astype(np.float32)
                # random stratified splits: each noun recurs (4 frames) across train/test, which is what the
                # Hewitt–Liang control task needs (it can only be "learned" by memorising noun identity)
                acc = cv_random(Xl, y)
                ctrl = cv_random(Xl, control_labels(y, g, seed=l))
                unseen = cv_probe(Xl, y, g)  # strict: folds split by noun -> test nouns never seen
                r = dict(position=pname, layer=l, lang=lang, acc=acc, control=ctrl, selectivity=acc - ctrl,
                         acc_unseen=unseen, majority=majority_baseline(y))
                if lang != "hi":
                    m, lo, hi = bootstrap_acc(clf_hi.predict(Xl), y)
                    r.update(transfer_from_hi=m, transfer_lo=lo, transfer_hi=hi,
                             shuffled_transfer=float((clf_sh.predict(Xl) == y).mean()))
                    w_l, _ = probe_direction(Xl, y)
                    r["cos_with_hi"] = float(w_l @ w_hi)
                rows.append(r)
        print(f"[{pname}] done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "probe_gender.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    cols = {"hi": "#c0392b", "pa": "#8e44ad", "en": "#2471a3", "ta": "#16a085"}
    for r, pname in enumerate(["noun", "final"]):
        d = df[df.position == pname]
        ax = axes[r, 0]
        for lang in LANGS:
            e = d[d.lang == lang]
            ax.plot(e.layer, e.acc, color=cols[lang], label=lang)
            ax.plot(e.layer, e.control, color=cols[lang], ls=":", lw=1)
            ax.plot(e.layer, e.acc_unseen, color=cols[lang], ls="--", lw=1)
        ax.axhline(d.majority.max(), color="gray", lw=.6)
        ax.set_title(f"[{pname}] in-language acc (dotted: control, dashed: unseen nouns)")
        ax.set_ylim(.3, 1.02)
        ax.legend(fontsize=7)
        ax = axes[r, 1]
        for lang in ["pa", "en", "ta"]:
            e = d[d.lang == lang]
            ax.plot(e.layer, e.transfer_from_hi, color=cols[lang], label=f"hi→{lang}")
            ax.fill_between(e.layer, e.transfer_lo, e.transfer_hi, color=cols[lang], alpha=.15)
            ax.plot(e.layer, e.shuffled_transfer, color=cols[lang], ls=":", lw=1)
        ax.axhline(.5, color="gray", lw=.6)
        ax.set_title(f"[{pname}] transfer of Hindi probe (dotted: shuffled-label)")
        ax.set_ylim(.3, 1.02)
        ax.legend(fontsize=7)
        ax = axes[r, 2]
        for lang in ["pa", "en", "ta"]:
            e = d[d.lang == lang]
            ax.plot(e.layer, e.cos_with_hi, color=cols[lang], label=f"cos(hi, {lang})")
        ax.axhline(0, color="gray", lw=.6)
        ax.set_title(f"[{pname}] cosine of probe directions")
        ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("layer (0 = embeddings)")
    fig.tight_layout()
    fig.savefig(OUT / "probe_gender.png", dpi=150)
    print(df[df.layer.isin([0, 4, 8, 12, 16, 20, 26])].round(3).to_string())


if __name__ == "__main__":
    main()
