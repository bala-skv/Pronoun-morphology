"""Stage 6 - figures for the report.

Reads the CSV outputs in results/ and writes results/figures/*.pdf. No model needed; runs in a few seconds.
Colours: one per language (Hindi blue, Punjabi orange, English green, Tamil yellow); line style marks the
condition within a language (feminine/masculine noun, noun/final position).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C = {"hi": "#2a78d6", "pa": "#eb6834", "en": "#1baf7a", "ta": "#eda100"}
NAME = {"hi": "Hindi", "pa": "Punjabi", "en": "English", "ta": "Tamil"}
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#9a9993"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "lines.linewidth": 1.6, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
FULL, COL = 6.3, 3.05  # ACL text width and column width in inches


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def panel(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", color=INK)


def end_label(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points", va="center", fontsize=7,
                color=INK2, annotation_clip=False)
    ax.plot([x], [y], "o", ms=3, color=color, clip_on=False)


def fig_overview():
    """Strip chart: where each method places the Hindi agreement decision."""
    L = np.arange(26)
    rows = []
    g = pd.read_csv(R / "03_probe_gender" / "probe_gender.csv")
    pn = g[(g.position == "noun") & (g.lang == "hi")].set_index("layer").acc_unseen
    pf = g[(g.position == "final") & (g.lang == "hi")].set_index("layer").acc_unseen
    rows.append(("Probe: gender at noun", np.clip((pn.reindex(L).values - 0.5) / 0.5, 0, 1)))
    rows.append(("Probe: gender at final", np.clip((pf.reindex(L).values - 0.5) / 0.5, 0, 1)))
    r = pd.read_csv(R / "02_patching" / "c1_hi" / "act_resid_pre.csv")
    rows.append(("Patching: noun holds it", r[r.role == "w_last"].set_index("layer").R.reindex(L).clip(0, 1).values))
    rows.append(("Patching: final token", r[r.role == "END"].set_index("layer").R.reindex(L).clip(0, 1).values))
    d = pd.read_csv(R / "01_lens_dla" / "c1_hi" / "logit_lens.csv").set_index("layer")
    gap = d.gap.reindex(L).values
    rows.append(("Logit lens: fem-masc gap", np.clip(gap / np.nanmax(gap), 0, 1)))
    c = pd.read_csv(R / "01_lens_dla" / "c1_hi" / "dla_components.csv").set_index("component")["diff"]
    dla = np.array([max(c.get(f"mlp{l}", 0), 0) + max(c.get(f"attn{l}", 0), 0) for l in L])
    rows.append(("DLA: attn + MLP write-out", dla / dla.max()))
    s = pd.read_csv(R / "04_sae" / "sae_sharing_steering.csv")
    st = s[s.pair == "steer:top_hi->hi"].set_index("layer").frac_gap_removed
    sae = np.full(26, np.nan)
    for l, v in st.items():
        sae[int(l)] = min(max(v, 0), 1)
    rows.append(("SAE steering flips it", sae))

    fig, ax = plt.subplots(figsize=(COL, 2.15))
    cmap = plt.get_cmap("Blues")
    for i, (_, v) in enumerate(rows):
        for l in L:
            if np.isnan(v[l]):
                ax.add_patch(plt.Rectangle((l - 0.5, i - 0.4), 1, 0.8, facecolor="white", edgecolor=GRID, lw=0.4))
            else:
                ax.add_patch(plt.Rectangle((l - 0.5, i - 0.4), 1, 0.8, facecolor=cmap(0.08 + 0.85 * v[l]),
                                           edgecolor="white", lw=0.4))
    ax.axvspan(20.5, 22.5, color=C["pa"], alpha=0.10, lw=0)
    ax.text(21.5, -0.55, "L21-22", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xlim(-0.5, 25.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _ in rows], fontsize=6.5)
    ax.set_xticks([0, 5, 10, 15, 21, 25])
    ax.set_xlabel("layer (Hindi agreement)")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    save(fig, "fig_overview")


def fig_probes():
    g = pd.read_csv(R / "03_probe_gender" / "probe_gender.csv")
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.1), sharey=True)
    offs = {"noun": {"hi": 4, "pa": 0, "en": -4, "ta": 4}, "final": {"hi": 4, "pa": -3, "en": -4, "ta": 3}}
    for ax, pos, letter, title in [(axes[0], "noun", "a", "Gender at the noun"),
                                   (axes[1], "final", "b", "Gender at the final token")]:
        x = g[g.position == pos]
        for lang in ["hi", "pa", "en", "ta"]:
            y = x[x.lang == lang].sort_values("layer")
            ax.plot(y.layer, y.acc_unseen, color=C[lang])
            end_label(ax, y.layer.iloc[-1], y.acc_unseen.iloc[-1], NAME[lang], C[lang], offs[pos][lang])
        ax.axhline(x.majority.mean(), ls=":", lw=1, color=MUTED)
        panel(ax, letter, title)
        ax.set_xlabel("layer")
    axes[0].set_ylabel("probe accuracy")
    axes[0].plot([], [], ":", lw=1, color=MUTED, label="majority baseline")
    axes[0].legend(loc="lower left", handlelength=1.8)
    ax = axes[2]
    for lang in ["pa", "en", "ta"]:
        for pos, ls in [("noun", "-"), ("final", "--")]:
            y = g[(g.position == pos) & (g.lang == lang)].sort_values("layer")
            ax.plot(y.layer, y.transfer_from_hi, ls=ls, color=C[lang], lw=1.6 if ls == "-" else 1.3)
    for lang, dy in [("pa", 2), ("en", -6), ("ta", 4)]:
        y = g[(g.position == "noun") & (g.lang == lang)].sort_values("layer")
        end_label(ax, y.layer.iloc[-1], y.transfer_from_hi.iloc[-1], NAME[lang], C[lang], dy)
    ax.axhline(0.5, ls=":", lw=1, color=MUTED)
    ax.plot([], [], "-", color=INK2, label="noun")
    ax.plot([], [], "--", color=INK2, lw=1.3, label="final token")
    ax.plot([], [], ":", lw=1, color=MUTED, label="chance (0.5)")
    ax.legend(loc="lower left", ncol=2, handlelength=1.8, columnspacing=0.8)
    panel(ax, "c", "Transfer from Hindi")
    ax.set_xlabel("layer")
    ax.set_ylim(0.3, 1.02)
    fig.subplots_adjust(wspace=0.38)
    save(fig, "fig_probes")


def fig_lens_dla():
    fig = plt.figure(figsize=(FULL, 2.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.35)
    for i, lang in enumerate(["hi", "pa"]):
        ax = fig.add_subplot(gs[0, i])
        d = pd.read_csv(R / "01_lens_dla" / f"c1_{lang}" / "logit_lens.csv")
        ax.plot(d.layer, d.lens_A, color=C[lang])
        ax.plot(d.layer, d.lens_B, color=C[lang], ls="--", lw=1.3)
        ax.axvline(21, ls=":", lw=1, color=MUTED)
        ax.axhline(0, lw=0.8, color=MUTED)
        panel(ax, "ab"[i], f"Logit lens, {NAME[lang]}")
        ax.set_xlabel("layer")
        ax.set_xlim(10, 25.5)
        lo = min(d.lens_A[d.layer >= 10].min(), d.lens_B[d.layer >= 10].min())
        hi = max(d.lens_A[d.layer >= 10].max(), d.lens_B[d.layer >= 10].max())
        ax.set_ylim(lo - 0.5, hi + 0.5)
        if i == 0:
            ax.set_ylabel(r"$\Delta$ = logit(fem) $-$ logit(masc)")
        else:
            ax.plot([], [], "-", color=INK2, label="feminine noun")
            ax.plot([], [], "--", color=INK2, lw=1.3, label="masculine noun")
            ax.plot([], [], ":", lw=1, color=MUTED, label="block 21")
            ax.legend(loc="upper center", bbox_to_anchor=(-0.2, -0.24), ncol=3, handlelength=1.8, fontsize=6.5)
    ax = fig.add_subplot(gs[0, 2])
    w = 0.38
    for lang, off in [("hi", -w / 2), ("pa", w / 2)]:
        d = pd.read_csv(R / "01_lens_dla" / f"c1_{lang}" / "dla_components.csv").set_index("component")
        L = np.arange(26)
        mlp = np.array([d.loc[f"mlp{l}", "diff"] for l in L])
        att = np.array([d.loc[f"attn{l}", "diff"] for l in L])
        ax.bar(L + off, mlp, width=w, color=C[lang], label=f"{NAME[lang]} MLP")
        ax.bar(L + off, att, width=w, bottom=np.where((att > 0) & (mlp > 0), mlp, 0), color="white",
               edgecolor=C[lang], hatch="//////", linewidth=0.6, label=f"{NAME[lang]} attention")
    ax.axhline(0, lw=0.8, color=MUTED)
    ax.set_xlim(14.5, 25.5)
    ax.set_ylim(-0.4, 4.9)
    ax.set_xticks(range(15, 26))
    ax.set_xlabel("layer")
    ax.set_ylabel("fem $-$ masc contribution")
    ax.legend(loc="upper left", ncol=1, fontsize=6.5)
    ax.grid(axis="x", visible=False)
    panel(ax, "c", "DLA (solid MLP, hatched attention)")
    save(fig, "fig_lens_dla")


def fig_latent_english():
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    for lang in ["hi", "pa"]:
        d = pd.read_csv(R / "01_lens_dla" / f"c1_{lang}" / "latent_lens.csv", index_col=0)
        native = d.columns[0]
        ax.plot(d.index, d[" my"] + d[" mine"], color=C[lang])
        ax.plot(d.index, d[native], color=C[lang], ls="--", lw=1.3)
    ax.plot([], [], "-", color=INK2, label="English ' my' + ' mine'")
    ax.plot([], [], "--", color=INK2, lw=1.3, label="native first token")
    ax.plot([], [], "s", color=C["hi"], label="Hindi", ms=5)
    ax.plot([], [], "s", color=C["pa"], label="Punjabi", ms=5)
    ax.legend(loc="upper left", fontsize=6.5, handlelength=1.8)
    ax.set_xlim(14, 25.4)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("layer")
    ax.set_ylabel("logit-lens probability")
    save(fig, "fig_latent_english")


def fig_patching():
    specs = [("c1_hi", "hi", "w_last", "Hindi agreement", 22),
             ("c1_pa", "pa", "w_last", "Punjabi agreement", 22),
             ("c2_hi", "hi", "w", "Hindi ergative", 20)]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.0), sharey=True)
    for ax, (d, lang, src, title, mark), letter in zip(axes, specs, "abc"):
        r = pd.read_csv(R / "02_patching" / d / "act_resid_pre.csv")
        for role, ls in [(src, "--"), ("END", "-")]:
            y = r[r.role == role].sort_values("layer")
            ax.fill_between(y.layer, y.R_lo, y.R_hi, color=C[lang], alpha=0.18, lw=0)
            ax.plot(y.layer, y.R, color=C[lang], ls=ls, lw=1.6 if ls == "-" else 1.3)
        ax.axvline(mark, ls=":", lw=1, color=MUTED)
        ax.text(mark + 0.5, 0.45, f"layer {mark}", rotation=90, ha="left", va="center", fontsize=6.5, color=INK2)
        panel(ax, letter, title)
        ax.set_xlabel("layer (resid_pre)")
        ax.set_ylim(-0.05, 1.08)
    axes[0].set_ylabel("recovery $R$")
    axes[1].plot([], [], "--", color=INK2, lw=1.3, label="noun / pronoun")
    axes[1].plot([], [], "-", color=INK2, label="final token")
    axes[1].plot([], [], ":", lw=1, color=MUTED, label="handoff layer")
    axes[1].legend(loc="lower left", bbox_to_anchor=(0.02, 0.18), handlelength=1.8)
    fig.subplots_adjust(wspace=0.15)
    save(fig, "fig_patching")


def fig_sae():
    s = pd.read_csv(R / "04_sae" / "sae_sharing_steering.csv")
    sh = s[~s.pair.str.startswith("steer")]
    short = {"pa": "Pa", "en": "En", "ta": "Ta"}
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.1))
    for ax, pos, letter, title, dys in [(axes[0], "END", "a", "Sharing, final token", (0, 3, -4)),
                                        (axes[1], "noun", "b", "Sharing, noun", (0, -5, 4))]:
        for tgt, dy in zip(["pa", "en", "ta"], dys):
            y = sh[(sh.position == pos) & (sh.pair == f"hi-{tgt}")].sort_values("layer")
            ax.plot(y.layer, y["corr"], color=C[tgt], marker="o", ms=3)
            ax.plot(y.layer, y.null_95, color=C[tgt], ls=":", lw=1)
            end_label(ax, y.layer.iloc[-1], y["corr"].iloc[-1], f"Hi-{short[tgt]}", C[tgt], dy)
        ax.axhline(0, lw=0.8, color=MUTED)
        ax.set_ylim(-0.1, 0.68)
        ax.set_xlabel("layer")
        panel(ax, letter, title)
    axes[0].set_ylabel("correlation with Hindi")
    axes[0].plot([], [], "-", color=INK2, marker="o", ms=3, label="observed")
    axes[0].plot([], [], ":", lw=1, color=INK2, label="null, 95th pct.")
    axes[0].legend(loc="upper left", handlelength=1.8, fontsize=6.5)
    ax = axes[2]
    st = s[s.pair.str.startswith("steer")]
    for name, ls, col in [("steer:top_hi->hi", "-", C["hi"]), ("steer:top_hi->pa", "--", C["pa"]),
                          ("steer:random->hi", "-", MUTED), ("steer:random->pa", "--", MUTED)]:
        y = st[st.pair == name].sort_values("layer")
        ax.plot(y.layer, y.frac_gap_removed, color=col, ls=ls, marker="o", ms=3, lw=1.6 if col != MUTED else 1.1)
    hi24 = st[(st.pair == "steer:top_hi->hi") & (st.layer == 24)].frac_gap_removed.iloc[0]
    pa24 = st[(st.pair == "steer:top_hi->pa") & (st.layer == 24)].frac_gap_removed.iloc[0]
    end_label(ax, 24, hi24, "Hindi", C["hi"], 5 if hi24 >= pa24 else -5)
    end_label(ax, 24, pa24, "Punjabi", C["pa"], -5 if hi24 >= pa24 else 5)
    ax.annotate("10 random features", (16, 0.0), xytext=(4.5, 0.75), fontsize=6.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.axhline(1.0, ls=":", lw=1, color=MUTED)
    ax.text(4, 1.04, "gap fully removed", fontsize=6.5, color=INK2, va="bottom")
    ax.axhline(0.0, lw=0.8, color=MUTED)
    ax.set_ylim(-0.15, 1.9)
    ax.set_ylabel("fraction of gender gap removed", labelpad=1)
    ax.set_xlabel("layer edited")
    panel(ax, "c", "Steering 10 features")
    fig.subplots_adjust(wspace=0.6)
    save(fig, "fig_sae")


if __name__ == "__main__":
    fig_overview()
    fig_probes()
    fig_lens_dla()
    fig_latent_english()
    fig_patching()
    fig_sae()
