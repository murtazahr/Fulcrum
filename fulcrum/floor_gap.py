"""What the equal-floor assumption of the corollary costs, at the evaluated scale.

The corollary treats the lateral floor as common to all regions; the floor in fact grows
with region size and saturates. This solves the allocation numerically with region-dependent
ell_r and compares the attainable saving against the common-floor expression.

SCALE MATTERS HERE, and an earlier version of the table got it wrong. delta is invariant to
the federation size, since it depends only on the ratios in a profile, but ell_r is not: it
is a function of the ABSOLUTE number of silos sharing a region and saturates once that
number is large. Computing the table from the literal profile ratios (6/6/6/3/3 as 24 silos)
rather than from the evaluated configuration (the same profile as 96 silos) therefore
reports floors that are far from saturation, and overstates the cost of the assumption by
several percentage points. The evaluation runs at n = 96 and this table now follows it.

The consortium is a real 50-silo structure and is unaffected by that choice.

The second half of this script covers the companion question: what happens when the
coupling kappa itself is misspecified, so the allocation is built on ell_r(kappa_assumed)
while the deployment realises ell_r(kappa_true). The target is then overshot by exactly the
error in the floor, which enters additively and is bounded by the saturation ceiling.
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lateral_mi import ell, saturation
from evaluate import rho_stats, profiles

KAP, K, T, NMC = 20.0, 2.2, 10, 60000
A = 2.0 * T


def gap(sizes):
    """-> (common-floor saving %, saving solved for region-dependent ell, ell range)."""
    groups = np.array(sum([[j] * int(s) for j, s in enumerate(sizes)], []))
    n = len(groups)
    w = np.full(n, 1.0 / n)
    regs = sorted(set(groups.tolist()))
    _, V, W, delta_common = rho_stats(w, groups)
    e = np.array([ell(int(s - 1), KAP, n_mc=NMC) for s in sizes])
    U_f = float(np.sum(A * W / (K - e)))
    s2 = max(A * W[j] / ((K - e[j]) * V[j]) for j in range(len(regs)))
    return 100 * delta_common, 100 * (1 - U_f / (s2 * V.sum())), e


def overshoot(sizes, k_assumed, k_true):
    """Worst-case bound actually realised when the floor is built on the wrong coupling."""
    groups = np.array(sum([[j] * int(x) for j, x in enumerate(sizes)], []))
    n = len(groups); w = np.full(n, 1.0 / n)
    _, _, W, _ = rho_stats(w, groups)
    ea = np.array([ell(int(x - 1), k_assumed, n_mc=NMC) for x in sizes])
    et = np.array([ell(int(x - 1), k_true, n_mc=NMC) for x in sizes])
    return float(np.max(A * W / (A * W / (K - ea)) + et))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    rows = [("Mild, 6/6/6/3/3", [int((profiles(n)["mild  6/6/6/3/3"] == r).sum())
                                 for r in sorted(set(profiles(n)["mild  6/6/6/3/3"].tolist()))]),
            ("Severe, 15/5/2/1/1", [int((profiles(n)["severe 15/5/2/1/1"] == r).sum())
                                    for r in sorted(set(profiles(n)["severe 15/5/2/1/1"].tolist()))]),
            ("Consortium by country", [30, 10, 5, 3, 1, 1])]
    print(f"kappa = {KAP:g}, K = {K} nats, evaluated federation size n = {n}\n")
    print(f"{'Region profile':<24}{'regions':>26}{'ell range':>16}{'common':>9}{'solved':>9}{'err':>7}")
    for nm, sz in rows:
        c, s, e = gap(sz)
        print(f"{nm:<24}{str(sz):>26}{f'{e.min():.3f}-{e.max():.3f}':>16}"
              f"{c:>8.1f}%{s:>8.1f}%{c-s:>6.1f}")

    sev = [int((profiles(n)["severe 15/5/2/1/1"] == r).sum())
           for r in sorted(set(profiles(n)["severe 15/5/2/1/1"].tolist()))]
    print(f"\ncoupling misspecification on the severe profile, target K = {K} nats")
    print(f"  saturation ceiling max_kappa I(p;Phi) = "
          f"{max(saturation(k, n_mc=NMC) for k in (5, 20, 100)):.3f} nats")
    for ka in (5, 20):
        r = overshoot(sev, ka, 100)
        print(f"  assume kappa = {ka:<3} deployment realises kappa = 100 -> "
              f"realised {r:.3f} nats, overshoot {r - K:+.2f}")
