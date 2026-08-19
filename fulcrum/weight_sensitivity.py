"""How the saving and the heuristic's shortfall respond to silo weight heterogeneity.

Table 4 reports the operational structures with EQUAL silo weights, because the sources
document how many sites belong to each region without recording how much data each site
holds. Equal weights are the one case in which rho_r = 1/m_r exactly, so sigma ~ 1/sqrt(m)
attains the optimum; a reader comparing Table 4 against Section 6.5 would otherwise
reasonably ask when the optimisation is needed at all.

This quantifies the answer using the per-silo dataset sizes that ARE published, in the two
FLamby federations that report them, and applies their spread to the same structures.
Everything here is analytic: budgets follow from the weights and the region structure, so
no training is involved.
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import rho_stats

ELL, K, A = 0.85, 0.88, 20.0          # a = 2T at T = 10, matching the evaluation
REPS = 200

# Per-centre dataset sizes for all seven federations in FLamby (Terrail et al., Table 1).
# These are measurements, not assumptions: every one of the seven is unbalanced, which is
# the point. Fed-ISIC2019 is quoted at its training split elsewhere in the code; the totals
# are used here because the dispersion is what matters and it is unaffected.
FLAMBY = {
    "Fed-Camelyon16":    np.array([239, 150], float),
    "Fed-LIDC-IDRI":     np.array([670, 205, 69, 74], float),
    "Fed-IXI":           np.array([311, 181, 74], float),
    "Fed-TCGA-BRCA":     np.array([311, 196, 206, 162, 162, 51], float),
    "Fed-KiTS2019":      np.array([12, 14, 12, 12, 16, 30], float),
    "Fed-ISIC2019":      np.array([12413, 3954, 3363, 2259, 819, 439], float),
    "Fed-Heart-Disease": np.array([303, 261, 46, 130], float),
}


def budgets(w, groups):
    """-> (delta, optimal budget, budget the 1/sqrt(m) heuristic needs for the same K)."""
    regs = sorted(set(groups.tolist()))
    _, V, W, delta = rho_stats(w, groups)
    need = np.array([A * W[j] / (K - ELL) for j in range(len(regs))])
    msz = np.array([float((groups == r).sum()) for r in regs])
    S = np.array([np.sum(w[groups == r] ** 2 * (msz[j] ** -1.0)) for j, r in enumerate(regs)])
    S = S * max(need[j] / S[j] for j in range(len(regs)))
    return delta, float(need.sum()), float(S.sum())


def sweep(groups, s_log, reps=REPS):
    d, ratio = [], []
    for rep in range(reps):
        r = np.random.default_rng(rep)
        w = np.ones(len(groups)) if s_log == 0 else r.lognormal(0.0, s_log, len(groups))
        w = w / w.sum()
        dd, uf, uh = budgets(w, groups)
        d.append(dd); ratio.append(uh / uf)
    return float(np.mean(d)), float(np.mean(ratio))


ISIC_W = np.array([9930, 3163, 2691, 1807, 655, 351], float)   # Fed-ISIC2019, as in Table 4
ISIC_G = np.array([0, 0, 0, 1, 2, 3])                          # natural hospital regions 3/1/1/1


if __name__ == "__main__":
    # The one Table 4 entry whose per-silo volumes are published, so nothing is assumed.
    d, uf, uh = budgets(ISIC_W / ISIC_W.sum(), ISIC_G)
    print(f"Fed-ISIC2019 as deployed: delta = {100*d:.1f}%, heuristic needs {uh/uf:.2f}x the "
          f"optimal budget\n  (largest region holds 3 silos, so there is little room for a "
          f"size-based rule to err)\n")
    print(f"{'federation':<20}{'clients':>8}{'max/min':>9}{'sd(log)':>9}")
    for nm, v in FLAMBY.items():
        print(f"{nm:<20}{len(v):>8}{v.max()/v.min():>8.1f}x{np.std(np.log(v)):>9.2f}")
    print()
    sds = {k: float(np.std(np.log(v))) for k, v in FLAMBY.items()}
    med, mx = float(np.median(list(sds.values()))), max(sds.values())
    print(f"across the seven: sd(log size) from {min(sds.values()):.2f} to {mx:.2f}, "
          f"median {med:.2f}; the evaluation's log-normal arm uses 0.60\n")
    spreads = [("equal weights (Table 4)", 0.0),
               (f"weights at the FLamby median (sd={med:.2f})", med),
               (f"weights at the most skewed (sd={mx:.2f})", mx)]
    rng = np.random.default_rng(0)
    cell = np.maximum(1, np.round(rng.lognormal(1.2, 1.3, 60)).astype(int))
    STRUCT = {"Consortium by country 30/10/5/3/1/1":
              np.array(sum([[j] * s for j, s in enumerate([30, 10, 5, 3, 1, 1])], [])),
              "Cellular edge, 60 base stations":
              np.array(sum([[j] * int(x) for j, x in enumerate(cell)], []))}
    for nm, groups in STRUCT.items():
        print(f"\n--- {nm}  ({len(groups)} silos, {len(set(groups.tolist()))} regions)")
        for lbl, s in spreads:
            d, ratio = sweep(groups, s)
            print(f"   {lbl:<36} delta = {100*d:5.1f}%   heuristic needs {ratio:.2f}x the optimal budget")
    print("\nThe evaluation's log-normal arm sits at the median of the seven, not at an extreme.")
