"""Generate manuscript figures from the recorded experimental artefacts.

Sizing discipline (this is what made the first version look poor). IEEE two-column
uses a 3.5 in column and a 7.16 in text block. A figure authored at some other width
and then rescaled by \\includegraphics[width=\\columnwidth] has all of its fonts
rescaled with it, which is why a 7 in two-panel figure dropped into one column came
out unreadable. Every figure here is authored at EXACTLY its target width and is
included at natural size (\\includegraphics without a width key, or width=\\columnwidth
for the single-column ones, which is then a no-op).

Statistics. Arms are PAIRED: for a given seed the two allocations see the same data
partition and the same initialisation, so the correct dispersion for a gain is the
standard error of the paired difference, not the combination of two independent
standard errors.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "manuscript", "figures")
DATA = os.path.join(ROOT, "analysis", "v2")
os.makedirs(OUT, exist_ok=True)

COL, TEXT = 3.5, 7.16          # IEEE column and text-block widths, inches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.3,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    "legend.frameon": True, "legend.framealpha": 0.92,
    "legend.edgecolor": "0.8", "legend.borderpad": 0.4,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})
C_F, C_U, C_R = "#1b4f72", "#a93226", "#7f8c8d"


def _load(fn):
    with open(os.path.join(DATA, fn)) as f:
        return json.load(f)


def _paired(fn):
    """-> (delta, mean gain pp, sem of paired diff, mean misalloc pp, sem, nseeds) per profile."""
    rows = _load(fn)
    by = {}
    for r in rows:
        by.setdefault((round(r["delta"], 4), r["profile"]), {}).setdefault(r["mode"], {})[r["seed"]] = r["acc"]
    out = []
    for (d, p), modes in sorted(by.items()):
        seeds = sorted(set(modes["fulcrum"]) & set(modes["uniform"]) & set(modes.get("random", {})))
        f = np.array([modes["fulcrum"][s] for s in seeds])
        u = np.array([modes["uniform"][s] for s in seeds])
        m = np.array([modes["random"][s] for s in seeds])
        gd, md = 100 * (f - u), 100 * (m - u)
        n = len(seeds)
        sem = lambda v: float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        out.append((d, float(gd.mean()), sem(gd), float(md.mean()), sem(md), n))
    return out


def _pick(base):
    """Prefer the 10-seed artefact when present."""
    s10 = base.replace(".json", "_s10.json")
    return s10 if os.path.exists(os.path.join(DATA, s10)) else base


def fig_gain_vs_delta():
    """Key result. Authored at full text width and placed in a figure* environment."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT, 2.45), sharey=True)
    panels = [(_pick("probe_K0.88.json"), "(a) CIFAR-10, frozen ResNet-18"),
              (_pick("agnews_K0.88.json"), "(b) AG News, frozen MiniLM")]
    for ax, (fn, title) in zip(axes, panels):
        rec = _paired(fn)
        d = np.array([r[0] for r in rec])
        g, ge = np.array([r[1] for r in rec]), np.array([r[2] for r in rec])
        m, me = np.array([r[3] for r in rec]), np.array([r[4] for r in rec])
        ax.axhline(0, color="0.35", lw=0.7, zorder=1)
        ax.errorbar(d, m, yerr=me, marker="s", ms=3.6, ls="--", color=C_R, lw=1.1,
                    capsize=2, elinewidth=0.8, zorder=2, label="Misallocated, equal dispersion")
        ax.errorbar(d, g, yerr=ge, marker="o", ms=4.4, color=C_F, lw=1.5,
                    capsize=2, elinewidth=0.9, zorder=3, label="Exposure-aware allocation")
        nul = d < 1e-9
        ax.scatter(d[nul], g[nul], s=78, facecolors="none", edgecolors=C_F, lw=1.2, zorder=4)
        ax.set_xlabel(r"exposure dispersion $\delta$")
        ax.set_title(title, pad=3)
        ax.set_xlim(-0.06, 0.87)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    axes[0].set_ylabel("accuracy gain over\nuniform allocation (pp)")
    axes[0].annotate("balanced controls,\n" r"$\delta=0$", xy=(0.01, 0.0), xytext=(0.12, -6.0),
                     fontsize=6.6, ha="left", va="center",
                     arrowprops=dict(arrowstyle="->", lw=0.6, color="0.4",
                                     shrinkA=0, shrinkB=3))
    axes[0].legend(loc="upper left", handlelength=1.9)
    fig.subplots_adjust(wspace=0.06)
    fig.savefig(os.path.join(OUT, "fig_gain_vs_delta.pdf"))
    plt.close(fig)
    print("wrote fig_gain_vs_delta.pdf  (full text width, figure*)")


def fig_privacy_utility():
    from fedsim import eps_silo
    cifar = [(0.4, 0.9695), (0.83, 0.970), (1.5, 0.9565), (2, 0.942),
             (4, 0.869), (8, 0.6795), (16, 0.4725), (32, 0.376), (64, 0.3365)]
    ag = [(0.83, 0.9265), (4.0, 0.7640), (16.0, 0.4730)]
    ref = {"CIFAR-10": 0.970, "AG News": 0.9295}
    fig, ax = plt.subplots(figsize=(COL, 2.35))
    for pts, lbl, col, mk in [(cifar, "CIFAR-10", C_F, "o"), (ag, "AG News", C_U, "s")]:
        p = sorted((eps_silo(10, s), a) for s, a in pts)
        ax.plot([x[0] for x in p], [x[1] for x in p], marker=mk, ms=3.6, color=col, label=lbl)
        ax.axhline(ref[lbl], color=col, lw=0.7, ls=":", alpha=0.8)
    ax.axvline(0.99, color="0.35", lw=0.8, ls="--")
    ax.annotate(r"$\varepsilon=0.99$", xy=(0.99, 0.40), xytext=(1.9, 0.365),
                fontsize=6.8, arrowprops=dict(arrowstyle="->", lw=0.6, color="0.4"))
    ax.set_xscale("log")
    ax.set_xlabel(r"privacy parameter $\varepsilon$  ($\delta_{\mathrm{DP}}=10^{-5}$)")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.30, 1.02)
    ax.legend(loc="lower right", handlelength=1.9)
    fig.savefig(os.path.join(OUT, "fig_privacy_utility.pdf"))
    plt.close(fig)
    print("wrote fig_privacy_utility.pdf")


def fig_leverage():
    from lateral_mi import ell, saturation
    fig, ax = plt.subplots(figsize=(COL, 2.35))
    ms = [0, 1, 2, 4, 9, 19, 49]
    for kap, col, mk in [(5, "#7fb3d5", "^"), (20, C_F, "o"), (100, "#0b2545", "s")]:
        ax.plot([m + 1 for m in ms], [ell(m, kap, n_mc=8000) for m in ms],
                marker=mk, ms=3.6, color=col, label=fr"$\kappa={kap}$")
        ax.axhline(saturation(kap, n_mc=60000), color=col, lw=0.7, ls=":", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel(r"organisational group size $|G_i|$")
    ax.set_ylabel(r"lateral floor $\ell_i$ (nats)")
    ax.set_xlim(0.9, 60)
    ax.legend(loc="lower right", title="within-group\ncoupling", title_fontsize=6.6,
              handlelength=1.9)
    fig.savefig(os.path.join(OUT, "fig_leverage.pdf"))
    plt.close(fig)
    print("wrote fig_leverage.pdf")


def fig_delta_real():
    """Structures that admit reallocation. The two structures with delta = 0 are stated
    in the caption rather than drawn, since zero-length bars carry no visual information."""
    labels = ["Clinical sites grouped\nby hospital, $3/1/1/1$",
              "Cellular edge,\n60 base stations",
              "Consortium grouped\nby country"]
    vals, err = [14.4, 85.6, 88.0], [0.0, 2.4, 0.0]
    fig, ax = plt.subplots(figsize=(COL, 1.95))
    y = np.arange(len(vals))
    ax.barh(y, vals, xerr=err, height=0.58, color=[C_U, C_F, C_F],
            error_kw=dict(lw=0.8, capsize=2.2, ecolor="0.25"))
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.8)
    ax.set_xlabel(r"exposure dispersion $\delta$ (\% of budget recoverable)")
    ax.set_xlim(0, 104)
    ax.grid(axis="y", visible=False)
    for i, (v, e) in enumerate(zip(vals, err)):
        ax.text(v + e + 2.0, i, f"{v:.1f}\\%", va="center", fontsize=6.8)
    fig.savefig(os.path.join(OUT, "fig_delta_real.pdf"))
    plt.close(fig)
    print("wrote fig_delta_real.pdf  (zero-delta structures moved to caption)")


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    which = sys.argv[1:] or ["gain", "pu", "lev", "real"]
    if "gain" in which: fig_gain_vs_delta()
    if "pu" in which: fig_privacy_utility()
    if "lev" in which: fig_leverage()
    if "real" in which: fig_delta_real()
