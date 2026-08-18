"""Table: informativeness of the guarantee across quantisation resolutions.

This table had no generator. Its numbers are the quantised entropy of the EMPIRICAL
per-silo class concentrations p_i produced by partition(), not the entropy of the
hierarchical Beta prior that nonvacuity.H_quantized computes -- two different quantities
that are easy to confuse, which is why this script exists rather than a note in a docstring.

Remark (informativeness) requires the mechanism term to fall below the margin
H(p_i) - ell_i. Because H(p_i) grows with the resolution R, the test would pass trivially
at fine resolution, so it is reported across R down to the coarsest.
"""
from __future__ import annotations
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe import partition
from evaluate import profiles

ELL, K, N, SEEDS = 0.85, 0.88, 96, 10
CACHES = {"cifar": "probe_feat_32.pt", "agnews": "agnews_feat_32.pt"}


def concentrations(y, n=N, eta=0.5, seeds=SEEDS):
    ps = []
    for _, groups in profiles(n).items():
        for s in range(seeds):
            _, v = partition(y, n, eta, groups, s, size_spread=0.0)
            ps.append(v)
    return np.concatenate(ps)


def H(p, R):
    c = np.histogram(p, bins=np.linspace(0, 1, R + 1))[0].astype(float)
    c /= c.sum(); c = c[c > 0]
    return float(-(c * np.log(c)).sum())


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "./cifar"
    mech = K - ELL                      # mechanism term at the evaluated target
    # Note on the last row: CIFAR-10 at R = 50 gives H = 3.0350 exactly, on the rounding
    # boundary. The table prints 3.04 / 2.19 (half-up); Python's format gives 3.03 / 2.18
    # (half-even). Both are the same number, not a discrepancy.
    for ds, cache in CACHES.items():
        path = os.path.join(root, cache)
        if not os.path.exists(path):
            print(f"{ds}: {cache} absent, run probe.py/agnews.py first"); continue
        p = concentrations(torch.load(path, weights_only=False)["ytr"])
        print(f"\n{ds}: {len(p)} silo instances, p_i span [{p.min():.3f}, {p.max():.3f}]")
        print(f"{'R':>4} {'resolution':>11} {'H(p_i)':>8} {'margin':>8} {'mech':>7}  informative")
        for R in (4, 10, 20, 50):
            h = H(p, R)
            print(f"{R:>4} {1/R:>11.3f} {h:>8.2f} {h-ELL:>8.2f} {mech:>7.3f}  "
                  f"{'yes' if mech < h-ELL else 'NO'}")
