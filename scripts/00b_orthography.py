"""Stage 0b — is agreement just suffix matching? (no model needed; reads Stage-0 / Stage-2 / Stage-3 outputs)

Hindi/Punjabi gender is largely predictable from the noun's ending (-ी/-ि feminine, -ा masculine).
We split nouns into
  transparent      ending predicts the gender (fem -ी/-ि, masc -ा)
  opaque           ending says nothing (consonant-final etc.: किताब, घर, दुकान, फ़ोन ...)
  counter          ending predicts the WRONG gender (fem in -ा: कविता, गुड़िया; masc in -ी/-ि)
and recompute behaviour (and, if present, probe accuracy) per group. If agreement survives on opaque and
counter-regular nouns, the model uses lexical gender, not just the orthographic suffix.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pronoun.metrics import bootstrap_ci, fmt_ci  # noqa: E402

FEM_END = {"hi": ("ी", "ि", "ई", "इया", "िया"), "pa": ("ੀ", "ਿ", "ਈ")}
MASC_END = {"hi": ("ा", "आ"), "pa": ("ਾ", "ਆ")}


def ortho_class(noun, gender, lang):
    w = noun.strip()
    f_end = w.endswith(FEM_END[lang])
    m_end = w.endswith(MASC_END[lang])
    if (gender == "f" and f_end) or (gender == "m" and m_end):
        return "transparent"
    if (gender == "f" and m_end) or (gender == "m" and f_end):
        return "counter"
    return "opaque"


def main():
    out = ROOT / "results" / "00_behaviour"
    df = pd.read_csv(out / "c1_items_ctx.csv")
    lines = ["# Stage 0b — orthographic transparency of noun gender\n"]
    rows = []
    for lang in ["hi", "pa"]:
        d = df[(df.lang == lang) & df.ok].copy()
        d["ortho"] = [ortho_class(n, g, lang) for n, g in zip(d.noun, d.gender)]
        d["correct"] = np.where(d.gender == "f", d.delta > 0, d.delta < 0).astype(float)
        lines.append(f"## {lang}\n")
        for cls in ["transparent", "opaque", "counter"]:
            e = d[d.ortho == cls]
            if not len(e):
                continue
            nouns = sorted(set(f"{n}({g})" for n, g in zip(e.noun, e.gender)))
            accs = {g: e[e.gender == g].correct.mean() for g in ["f", "m"] if (e.gender == g).any()}
            lines.append(f"- **{cls}** n={len(e)} ({len(nouns)} nouns): acc {fmt_ci(bootstrap_ci(e.correct.values))}; "
                         + ", ".join(f"acc|{g}={a:.2f}" for g, a in accs.items())
                         + f"  \n  nouns: {' '.join(nouns)}\n")
            rows.append(dict(lang=lang, cls=cls, n=len(e), n_nouns=len(nouns), acc=e.correct.mean(),
                             **{f"acc_{g}": a for g, a in accs.items()}))
        per_noun = d.groupby(["noun", "gender", "ortho"]).correct.mean().reset_index().sort_values("correct")
        per_noun.to_csv(out / f"per_noun_{lang}.csv", index=False)
        lines.append("  worst nouns: " + ", ".join(f"{r.noun}({r.gender},{r.ortho})={r.correct:.2f}"
                                                 for r in per_noun.head(8).itertuples()) + "\n")
    pd.DataFrame(rows).to_csv(out / "orthography.csv", index=False)

    # probe accuracy per class (if Stage 3a activations exist): train on transparent, test on opaque/counter
    pg = ROOT / "results" / "03_probe_gender"
    if (pg / "acts_hi.npz").exists():
        from pronoun.probing import make_probe
        from pronoun.prompts import load_nouns

        nouns = load_nouns().set_index("id")
        lines.append("\n## Probe: train on transparent nouns, test on opaque / counter nouns (final position)\n")
        prow = []
        for lang in ["hi", "pa"]:
            z = np.load(pg / f"acts_{lang}.npz")
            X, y, g = z["X"], z["y"], z["g"]
            gcol = f"{lang}_gender"
            cls = np.array([ortho_class(nouns.loc[i, lang], nouns.loc[i, gcol], lang) for i in g])
            tr = cls == "transparent"
            for pos_i, pos in enumerate(["noun", "final"]):
                for l in range(X.shape[2]):
                    clf = make_probe().fit(X[tr, pos_i, l].astype(np.float32), y[tr])
                    r = dict(lang=lang, position=pos, layer=l)
                    for c in ["opaque", "counter"]:
                        m = cls == c
                        if m.any():
                            r[f"acc_{c}"] = float((clf.predict(X[m, pos_i, l].astype(np.float32)) == y[m]).mean())
                    prow.append(r)
        p = pd.DataFrame(prow)
        p.to_csv(pg / "probe_gender_ortho.csv", index=False)
        for (lang, pos), e in p.groupby(["lang", "position"]):
            b = e.loc[e.acc_opaque.idxmax()]
            lines.append(f"- {lang} [{pos}]: best opaque acc {b.acc_opaque:.2f} @L{int(b.layer)}"
                         + (f", counter acc there {b.acc_counter:.2f}" if "acc_counter" in b and not np.isnan(b.acc_counter) else "")
                         + "\n")
    (out / "summary_orthography.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
