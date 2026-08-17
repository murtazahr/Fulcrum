"""Generate the manuscript figures from the recorded experimental artefacts.

Outputs (PDF, vector, into manuscript/figures/):
  fig_gain_vs_delta.pdf   accuracy gain against exposure dispersion, both modalities,
                          with the two delta=0 null controls and the misallocation control
  fig_privacy_utility.pdf accuracy against epsilon for both modalities
  fig_leverage.pdf        lateral floor ell_i saturating in group size
  fig_delta_real.pdf      delta for real deployment structures
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

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
})
C_F, C_U, C_R = "#1f4e79", "#c00000", "#7f7f7f"


def _load(fn):
    with open(os.path.join(DATA, fn)) as f:
        return json.load(f)


def _agg(rows):
    """-> sorted list of (delta, profile, mean acc per mode, sem per mode)."""
    out = {}
    for r in rows:
        out.setdefault((round(r["delta"], 4), r["profile"]), {}).setdefault(r["mode"], []).append(r["acc"])
    res = []
    for (d, p), modes in sorted(out.items()):
        res.append((d, p, {m: float(np.mean(v)) for m, v in modes.items()},
                    {m: float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
                     for m, v in modes.items()}))
    return res


def fig_gain_vs_delta():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, (fn, title) in zip(axes, [("probe_K0.88.json", "CIFAR-10, frozen ResNet-18"),
                                      ("agnews_K0.88.json", "AG News, frozen MiniLM")]):
        rows = _agg(_load(fn))
        d = np.array([r[0] for r in rows])
        gain = np.array([100 * (r[2]["fulcrum"] - r[2]["uniform"]) for r in rows])
        rand = np.array([100 * (r[2]["random"] - r[2]["uniform"]) for r in rows])
        err = np.array([100 * np.hypot(r[3]["fulcrum"], r[3]["uniform"]) for r in rows])
        ax.axhline(0, color="k", lw=0.8, zorder=1)
        ax.errorbar(d, gain, yerr=err, marker="o", ms=5, lw=1.6, color=C_F,
                    capsize=2.5, label="Proposed allocation", zorder=3)
        ax.plot(d, rand, marker="s", ms=4.5, lw=1.4, ls="--", color=C_R,
                label="Misallocation control", zorder=2)
        null = d < 1e-9
        ax.scatter(d[null], gain[null], s=90, facecolors="none",
                   edgecolors=C_F, lw=1.4, zorder=4)
        ax.annotate("null controls\n(theory predicts $0$)", xy=(0.02, 0.6), xytext=(0.10, 6.4),
                    fontsize=7.5, ha="left",
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="0.35"))
        ax.set_xlabel(r"exposure dispersion $\delta$")
        ax.set_title(title)
        ax.set_xlim(-0.05, 0.88)
    axes[0].set_ylabel("accuracy gain over uniform (pp)")
    axes[0].legend(loc="upper left", frameon=True, framealpha=0.95)
    fig.savefig(os.path.join(OUT, "fig_gain_vs_delta.pdf"))
    plt.close(fig)
    print("wrote fig_gain_vs_delta.pdf")


def fig_privacy_utility():
    # measured sweeps (sigma, accuracy) with per-client epsilon at T=10, delta=1e-5
    from fedsim import eps_silo
    cifar = [(0.0, 0.970), (0.4, 0.9695), (0.83, 0.970), (1.5, 0.9565), (2, 0.942),
             (4, 0.869), (8, 0.6795), (16, 0.4725), (32, 0.376), (64, 0.3365)]
    ag = [(0.0, 0.9295), (0.83, 0.9265), (4.0, 0.7640), (16.0, 0.4730)]
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for pts, lbl, col, mk in [(cifar, "CIFAR-10", C_F, "o"), (ag, "AG News", C_U, "s")]:
        p = [(eps_silo(10, s), a) for s, a in pts if s > 0]
        p.sort()
        ax.plot([x[0] for x in p], [x[1] for x in p], marker=mk, ms=4, lw=1.5,
                color=col, label=lbl)
        ax.axhline([a for s, a in pts if s == 0][0], color=col, lw=0.8, ls=":", alpha=0.7)
    ax.axvline(0.99, color="0.3", lw=0.9, ls="--")
    ax.annotate(r"operating point" "\n" r"$\varepsilon=0.99$", xy=(0.99, 0.55),
                xytext=(1.6, 0.50), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=0.7, color="0.35"))
    ax.set_xscale("log")
    ax.set_xlabel(r"privacy parameter $\varepsilon$ (per client, $\delta_{\mathrm{DP}}=10^{-5}$)")
    ax.set_ylabel("test accuracy")
    ax.legend(loc="lower right", frameon=True)
    ax.set_title("Dotted lines: non-private reference", fontsize=8)
    fig.savefig(os.path.join(OUT, "fig_privacy_utility.pdf"))
    plt.close(fig)
    print("wrote fig_privacy_utility.pdf")


def fig_leverage():
    from lateral_mi import ell, saturation
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ms = [0, 1, 2, 4, 9, 19, 49]
    for kap, col in [(5, "#7fb3d5"), (20, C_F), (100, "#0b2545")]:
        y = [ell(m, kap, n_mc=8000) for m in ms]
        ax.plot([m + 1 for m in ms], y, marker="o", ms=4, lw=1.5, color=col,
                label=fr"$\kappa={kap}$")
        ax.axhline(saturation(kap, n_mc=60000), color=col, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("organisational group size $|G_i|$")
    ax.set_ylabel(r"lateral floor $\ell_i$ (nats)")
    ax.legend(loc="lower right", frameon=True, title="coupling")
    ax.set_title(r"Dotted lines: ceiling $I(p_i;\Phi)$", fontsize=8)
    fig.savefig(os.path.join(OUT, "fig_leverage.pdf"))
    plt.close(fig)
    print("wrote fig_leverage.pdf")


def fig_delta_real():
    labels = ["Flat federation\n(no aggregation tier)", "Equally sized\nregions",
              "Clinical sites by\nhospital $3/1/1/1$", "Cellular edge,\n60 base stations",
              "Consortium by\ncountry"]
    vals = [0.0, 0.0, 14.4, 85.6, 88.0]
    err = [0, 0, 0, 2.4, 0]
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    cols = ["0.75", "0.75", C_U, C_F, C_F]
    ax.barh(range(len(vals)), vals, xerr=err, color=cols, height=0.62,
            error_kw=dict(lw=0.9, capsize=2.5))
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel(r"exposure dispersion $\delta$ (\% of budget saveable)")
    ax.set_xlim(0, 100)
    for i, v in enumerate(vals):
        ax.text(v + 2.5, i, f"{v:.1f}\\%", va="center", fontsize=7.5)
    fig.savefig(os.path.join(OUT, "fig_delta_real.pdf"))
    plt.close(fig)
    print("wrote fig_delta_real.pdf")


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    which = sys.argv[1:] or ["gain", "pu", "lev", "real"]
    if "gain" in which: fig_gain_vs_delta()
    if "pu" in which: fig_privacy_utility()
    if "lev" in which: fig_leverage()
    if "real" in which: fig_delta_real()
