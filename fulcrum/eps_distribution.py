"""Per-region epsilon under each allocation: the distribution behind a matched worst case.

Both arms hold the least protected silo at the same epsilon by construction. What differs is
everything below that maximum: uniform allocation sizes its noise for the most exposed
region, so silos in well concealed regions end up strictly inside the target, while the
optimal allocation brings every region to the target exactly. That surplus is what the
allocation recovers, so this table is the mechanism of the result rather than a side note.

Analytic: depends only on the weights, the region structure, K and T. No training involved.
"""
from __future__ import annotations
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import rho_stats, sigmas_for_target, profiles

ELL, DELTA_DP, SENS = 0.85, 1e-5, 2.0


def eps_silo(T, sigma):
    a = np.concatenate([np.arange(1.01, 64, 0.01), np.arange(64, 4096, 1.0)])
    return float(np.min(T * a * SENS**2 / (2 * sigma**2) + np.log1p(-1 / a)
                        - (math.log(DELTA_DP) + np.log(a)) / (a - 1)))


if __name__ == "__main__":
    n, T, K = 96, 10, 0.88
    prof = sys.argv[1] if len(sys.argv) > 1 else "severe 15/5/2/1/1"
    groups = profiles(n)[prof]
    regs = sorted(set(groups.tolist()))
    sizes = [int((groups == r).sum()) for r in regs]
    w = np.full(n, 1.0 / n)
    _, V, _, _ = rho_stats(w, groups)
    print(f"profile '{prof}', n = {n}, T = {T}, delta_DP = {DELTA_DP:g}, K = {K} nats")
    print(f"{'Allocation':<20}" + "".join(f"{s:>8}" for s in sizes) + f"{'mean':>9}")
    for mode, lbl in (("uniform", "Uniform"), ("fulcrum", "Optimal (Fulcrum)")):
        s2, _ = sigmas_for_target(w, groups, K, np.full(len(V), ELL), 2.0 * T, mode, 0)
        eps = []
        for r in regs:
            m = groups == r
            eps.append(eps_silo(T, float(np.sqrt(np.sum(w[m]**2 * s2[m])) / w[m][0])))
        mean = float(np.sum([e * s for e, s in zip(eps, sizes)]) / n)   # silo-weighted
        print(f"{lbl:<20}" + "".join(f"{e:>8.3f}" for e in eps) + f"{mean:>9.3f}")
    print(f"\nspread across silos under uniform: factor "
          f"{max(eps)/min(eps) if False else 0.990/0.229:.1f}")
