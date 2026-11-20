"""Stage 3b — treebank probes for pronoun case/possession.

  python scripts/03_probe_case.py --max-sents 1500        # CPU-friendly
  python scripts/03_probe_case.py --max-sents 6000        # GPU

Labels (shared across languages, from UD): for PRON with PronType=Prs
  ERG  if Case contains Erg                 (Hindi only)
  GEN  if Poss=Yes or Case contains Gen
  NOM  if deprel nsubj*
  OBJ  if deprel obj/iobj or Case contains Dat
  OBL  if deprel obl*
Features: resid at every layer (0 = embeddings), pooled over the word's subwords (first / last / mean).
Reported per layer:
  acc_random     5-fold CV with random splits
  control        same, with Hewitt–Liang control labels (random label per word type)  -> selectivity
  acc_unseen     grouped CV by word form: test forms never seen in training (strict "beyond identity")
  transfer       probe trained on Hindi (or English) tested on the other languages, + shuffled-label control
"""
import argparse
import os

os.environ.setdefault("PRONOUN_PROBE", "torch")  # GPU L-BFGS probes; sklearn took ~25 min/layer on CPU
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import conllu
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pronoun.model import load_model
from pronoun.probing import collect_resid, control_labels, cv_probe, make_probe, majority_baseline

OUT = ROOT / "results" / "03_probe_case"
OUT.mkdir(parents=True, exist_ok=True)
UD = ROOT / "data" / "ud"
TREEBANKS = {"en": "en_gum", "hi": "hi_hdtb", "ta": "ta_ttb", "pa": "pa_cs"}


def label_of(tok):
    if tok["upos"] != "PRON":
        return None
    f = tok["feats"] or {}
    if f.get("PronType") != "Prs":
        return None
    case = f.get("Case", "") or ""
    dep = tok["deprel"] or ""
    if "Erg" in case:
        return "ERG"
    if f.get("Poss") == "Yes" or "Gen" in case or dep.endswith(":poss"):
        return "GEN"
    if dep.startswith("nsubj"):
        return "NOM"
    if dep in ("obj", "iobj") or "Dat" in case:
        return "OBJ"
    if dep.startswith("obl"):
        return "OBL"
    return None


def load_examples(model, lang, max_sents, max_toks=128, seed=0):
    """Returns list of (token ids, [(subword positions, label, form, person)])."""
    files = sorted(UD.glob(f"{TREEBANKS[lang]}-ud-*.conllu"))
    sents = []
    for f in files:
        sents += list(conllu.parse_incr(open(f, encoding="utf-8")))
    rng = np.random.default_rng(seed)
    rng.shuffle(sents)
    out = []
    for s in sents:
        words = [t for t in s if isinstance(t["id"], int)]
        targets = [(i, label_of(t)) for i, t in enumerate(words)]
        targets = [(i, l) for i, l in targets if l]
        if not targets:
            continue
        text, spans = "", []
        for t in words:
            start = len(text)
            text += t["form"]
            spans.append((start, len(text)))
            if not (t["misc"] and t["misc"].get("SpaceAfter") == "No"):
                text += " "
        enc = model.tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
        ids, offs = enc["input_ids"][:max_toks], enc["offset_mapping"][:max_toks]
        tgts = []
        for i, l in targets:
            a, b = spans[i]
            sub = [k for k, (x, y) in enumerate(offs) if y > x and x < b and y > a]
            if sub and sub[-1] < len(ids):
                f = words[i]["feats"] or {}
                tgts.append((sub, l, words[i]["form"].lower(), f.get("Person", "?")))
        if tgts:
            out.append((ids, tgts))
        if len(out) >= max_sents:
            break
    return out


def featurize(model, examples):
    seqs = [ids for ids, _ in examples]
    positions = [[p for sub, *_ in t for p in sub] for _, t in examples]
    X, owner = collect_resid(model, seqs, positions)
    feats = {"first": [], "last": [], "mean": []}
    labels, forms, persons = [], [], []
    k = 0
    for ids, tgts in examples:
        for sub, l, form, per in tgts:
            block = X[k:k + len(sub)].astype(np.float32)
            k += len(sub)
            feats["first"].append(block[0])
            feats["last"].append(block[-1])
            feats["mean"].append(block.mean(0))
            labels.append(l)
            forms.append(form)
            persons.append(per)
    return {p: np.stack(v) for p, v in feats.items()}, np.array(labels), np.array(forms), np.array(persons)


def cv_random(X, y, C=0.1, seed=0):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        accs.append((make_probe(C).fit(X[tr], y[tr]).predict(X[te]) == y[te]).mean())
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sents", type=int, default=1500)
    ap.add_argument("--langs", default="hi,en,ta,pa")
    ap.add_argument("--pooling", default="last,mean,first")
    ap.add_argument("--layer-stride", type=int, default=1)
    args = ap.parse_args()
    model = load_model()
    langs = [l for l in args.langs.split(",") if list(UD.glob(f"{TREEBANKS[l]}-ud-*.conllu"))]
    data = {}
    for lang in langs:
        ex = load_examples(model, lang, args.max_sents)
        feats, y, forms, persons = featurize(model, ex)
        keep = np.isin(y, [l for l, c in zip(*np.unique(y, return_counts=True)) if c >= 10])
        data[lang] = ({p: v[keep] for p, v in feats.items()}, y[keep], forms[keep], persons[keep])
        counts = dict(zip(*np.unique(y[keep], return_counts=True)))
        print(lang, len(ex), "sentences;", counts, flush=True)
        pd.DataFrame(dict(label=y, form=forms, person=persons)).to_csv(OUT / f"examples_{lang}.csv", index=False)

    nL = next(iter(data.values()))[0]["last"].shape[1]
    rows = []
    for pool in args.pooling.split(","):
        for l in range(0, nL, args.layer_stride):
            for lang, (F, y, forms, _) in data.items():
                X = F[pool][:, l]
                acc = cv_random(X, y)
                ctrl = cv_random(X, control_labels(y, forms, seed=l))
                unseen = cv_probe(X, y, forms) if len(np.unique(forms)) >= 5 else np.nan
                rows.append(dict(pool=pool, layer=l, lang=lang, n=len(y), acc_random=acc, control=ctrl,
                                 selectivity=acc - ctrl, acc_unseen=unseen, majority=majority_baseline(y)))
            for src in [s for s in ("hi", "en") if s in data]:
                Fs, ys, _, _ = data[src]
                for tgt, (Ft, yt, _, _) in data.items():
                    if tgt == src:
                        continue
                    shared = np.intersect1d(np.unique(ys), np.unique(yt))
                    ms, mt = np.isin(ys, shared), np.isin(yt, shared)
                    if len(shared) < 2 or mt.sum() < 10:
                        continue
                    clf = make_probe().fit(Fs[pool][ms, l], ys[ms])
                    clf_sh = make_probe().fit(Fs[pool][ms, l], np.random.default_rng(l).permutation(ys[ms]))
                    Xt = Ft[pool][mt, l]
                    rows.append(dict(pool=pool, layer=l, lang=f"{src}->{tgt}", n=int(mt.sum()),
                                     transfer=float((clf.predict(Xt) == yt[mt]).mean()),
                                     shuffled_transfer=float((clf_sh.predict(Xt) == yt[mt]).mean()),
                                     majority=majority_baseline(yt[mt]), labels="/".join(shared)))
            print(pool, l, flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(OUT / "probe_case.csv", index=False)

    df = pd.DataFrame(rows)
    for pool in args.pooling.split(","):
        d = df[df.pool == pool]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for lang in data:
            e = d[d.lang == lang]
            axes[0].plot(e.layer, e.acc_random, label=f"{lang} (maj {e.majority.iloc[0]:.2f})")
            axes[0].plot(e.layer, e.control, ls=":", lw=1, color=axes[0].lines[-1].get_color())
            axes[1].plot(e.layer, e.acc_unseen, label=lang)
        for pair in [p for p in d.lang.unique() if "->" in p]:
            e = d[d.lang == pair]
            axes[2].plot(e.layer, e.transfer, label=f"{pair} ({e.labels.iloc[0]})")
            axes[2].plot(e.layer, e.shuffled_transfer, ls=":", lw=1, color=axes[2].lines[-1].get_color())
        axes[0].set_title(f"case probe, random CV (dotted: control task) [{pool}]")
        axes[1].set_title("case probe, unseen word forms")
        axes[2].set_title("cross-lingual transfer (dotted: shuffled labels)")
        for ax in axes:
            ax.set_xlabel("layer (0 = embeddings)")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / f"probe_case_{pool}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
