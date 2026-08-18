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
C_H = "#b9770e"   # heuristic series

# Exposure dispersion of real deployment structures (see fulcrum/real_delta.py).
# These are marked on the same axis along which the gain is measured, so a reader can
# locate their own deployment and read off the improvement it implies. This replaces a
# standalone bar chart, which conveyed three numbers and little else.
# The cellular and consortium structures sit at 0.856 and 0.880, too close to label
# separately at this scale, so they share one marker spanning the pair.
DEPLOY = [(0.144, "clinical sites"), (0.868, "cellular edge,\nconsortium")]


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


def _gain_panels(files, outname, note):
    """One full-width figure, two panels (modality), for a single weight regime."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT, 2.5), sharey=True)
    series = [("random",  C_R, "s", "--", 1.3, "Misallocated"),
              ("sqrt_m",  C_H, "^", "-",  3.2, r"Heuristic $\sigma\propto 1/\sqrt{m}$"),
              ("fulcrum", C_F, "o", "-",  1.5, "Optimal allocation")]
    for ax, (fn, title) in zip(axes, files):
        rec = _by_profile(fn)
        d = np.array([x[0] for x in rec])
        ax.axhline(0, color="0.35", lw=0.7, zorder=1)
        for k, (mode, col, mk, ls, lw, lab) in enumerate(series):
            y = np.array([x[1][mode] for x in rec])
            e = np.array([x[2][mode] for x in rec])
            if mode == "sqrt_m":
                # Drawn as a wide translucent band: under equal weights it coincides
                # with the optimum exactly and would otherwise be hidden beneath it.
                ax.plot(d, y, ls=ls, color=col, lw=lw, alpha=0.32,
                        solid_capstyle="round", zorder=2, label=lab)
                ax.plot(d, y, marker=mk, ms=3.4, ls="none", color=col, alpha=0.95, zorder=2.5)
            else:
                ax.errorbar(d, y, yerr=e, marker=mk, ms=3.8, ls=ls, color=col, lw=lw,
                            capsize=2, elinewidth=0.8, zorder=2 + k, label=lab)
        if note == "equal":
            nul = d < 1e-9
            ax.scatter(d[nul], np.array([x[1]["fulcrum"] for x in rec])[nul],
                       s=74, facecolors="none", edgecolors=C_F, lw=1.2, zorder=9)
        ax.set_xlabel(r"exposure dispersion $\delta$")
        ax.set_title(title, pad=3)
    axes[0].set_ylabel("accuracy gain over\nuniform allocation (pp)")
    axes[0].legend(loc="upper left", handlelength=2.1)
    fig.subplots_adjust(wspace=0.07)
    fig.savefig(os.path.join(OUT, outname))
    plt.close(fig)
    print("wrote", outname)


def fig_gain_equal():
    _gain_panels([("probe_base_sp0.0.json",  "(a) CIFAR-10, frozen ResNet-18"),
                  ("agnews_base_sp0.0.json", "(b) AG News, frozen MiniLM")],
                 "fig_gain_equal.pdf", "equal")


def fig_gain_lognormal():
    _gain_panels([("probe_base_sp0.6.json",  "(a) CIFAR-10, frozen ResNet-18"),
                  ("agnews_base_sp0.6.json", "(b) AG News, frozen MiniLM")],
                 "fig_gain_lognormal.pdf", "lognormal")


def _by_profile(fn):
    """-> [(mean delta, {mode: mean gain pp}, {mode: sem})] ordered by delta."""
    rows = _load(fn)
    by = {}
    for r in rows:
        by.setdefault(r["profile"], {}).setdefault(r["mode"], {})[r["seed"]] = r["acc"]
    deltas = {}
    for r in rows:
        deltas.setdefault(r["profile"], []).append(r["delta"])
    out = []
    for p, modes in by.items():
        seeds = sorted(set.intersection(*(set(v) for v in modes.values())))
        u = np.array([modes["uniform"][s] for s in seeds])
        mean, sem = {}, {}
        for m in modes:
            g = 100 * (np.array([modes[m][s] for s in seeds]) - u)
            mean[m] = float(g.mean())
            sem[m] = float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else 0.0
        out.append((float(np.mean(deltas[p])), mean, sem))
    return sorted(out, key=lambda x: x[0])


def fig_privacy_utility():
    """Both modalities on a COMMON sigma grid, so the two series are sampled identically.
    The earlier version sampled CIFAR-10 at nine noise levels and AG News at three, which
    made the two curves look arbitrarily different where they were merely measured differently."""
    grid = _load("pu_grid.json")
    fig, ax = plt.subplots(figsize=(COL, 2.35))
    for key, lbl, col, mk in [("cifar", "CIFAR-10", C_F, "o"), ("agnews", "AG News", C_U, "s")]:
        pts = grid[key]
        ref = [p["acc"] for p in pts if p["sigma"] == 0][0]
        xy = sorted((p["eps"], p["acc"]) for p in pts if p["eps"] is not None)
        ax.plot([x for x, _ in xy], [y for _, y in xy], marker=mk, ms=3.4, color=col, label=lbl)
        ax.axhline(ref, color=col, lw=0.7, ls=":", alpha=0.8)
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


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    which = sys.argv[1:] or ["gain", "pu", "lev"]
    if "gain" in which: fig_gain_equal(); fig_gain_lognormal()
    if "pu" in which: fig_privacy_utility()
    if "lev" in which: fig_leverage()
