"""Refit the adversary against the recoverable target, reusing the saved observations.

WHY THE TARGET CHANGED. The first two designs asked the adversary to recover a single
silo's p_i from its region's aggregate. Under equal weights the silos in a region are
exchangeable, so the aggregate carries only sum_j w_j Delta_j: p_i is not identifiable
beyond the region mean, for ANY adversary. Both designs therefore returned the prior in
every arm, including a noiseless control, which is a property of the observation model and
not evidence about the mechanism.

A diagnostic settled that the signal exists: from a silo's OWN noiseless clipped update,
ridge recovers p_i with error 0.0375 against a 0.0989 prior baseline, correlation 0.913.
The signal is strong; aggregation is what removes it.

So the adversary is now asked for the region-level concentration p_bar_r, which is what an
observer above the regional tier can actually learn, and which the mechanism term controls.
Samples are (seed, round) pairs with features Y_r^(t) in R^d; the split is BY SEED so no
round of a test federation is ever seen in training. Ridge alpha is cross-validated within
the training seeds, separately per arm and region.

METRICS, unchanged in spirit from the pre-registration.
  PRIMARY   error on the SMALLEST region, which sets the worst-case guarantee in both arms.
  SECONDARY error averaged over regions.
  CONTROL   the noiseless arm. If it does not beat the prior, nothing else here counts.
"""
from __future__ import annotations
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import profiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ridge_cv(A, ya, B, alphas=(1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4)):
    mu, sd = A.mean(0), A.std(0) + 1e-8
    A, B = (A - mu) / sd, (B - mu) / sd
    k = min(5, len(A)); idx = np.arange(len(A)) % k
    best, be = alphas[0], np.inf
    for al in alphas:
        e = []
        for f in range(k):
            tr, va = idx != f, idx == f
            if va.sum() == 0 or tr.sum() < 2: continue
            m = ya[tr].mean()
            c = np.linalg.solve(A[tr].T @ A[tr] + al * np.eye(A.shape[1]), A[tr].T @ (ya[tr] - m))
            e.append(np.mean(np.abs(A[va] @ c + m - ya[va])))
        if e and np.mean(e) < be: be, best = np.mean(e), al
    m = ya.mean()
    c = np.linalg.solve(A.T @ A + best * np.eye(A.shape[1]), A.T @ (ya - m))
    return B @ c + m


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "cifar"
    z = np.load(f"{ROOT}/analysis/v2/attack_{ds}_obs.npz")
    meta = json.load(open(f"{ROOT}/analysis/v2/attack_{ds}_meta.json"))
    groups = profiles(96)["severe 15/5/2/1/1"]
    regs = sorted(set(groups.tolist()))
    sizes = [int((groups == r).sum()) for r in regs]
    NTR = 40
    rows = []
    for mode in ("control", "uniform", "fulcrum"):
        Y, P = z[f"{mode}_Y"], z[f"{mode}_P"]        # (S,T,R,d) and (S,n)
        S, T = Y.shape[0], Y.shape[1]
        pbar = np.stack([P[:, groups == r].mean(1) for r in regs], 1)   # (S,R)
        errs, bases = [], []
        for j, r in enumerate(regs):
            F = Y[:, :, j, :]                                            # (S,T,d)
            tr = np.arange(NTR); te = np.arange(NTR, S)
            Atr = F[tr].reshape(-1, F.shape[-1]); ytr = np.repeat(pbar[tr, j], T)
            Ate = F[te].reshape(-1, F.shape[-1])
            pred = ridge_cv(Atr, ytr, Ate).reshape(len(te), T).mean(1)   # average over rounds
            errs.append(float(np.mean(np.abs(pred - pbar[te, j]))))
            bases.append(float(np.mean(np.abs(pbar[tr, j].mean() - pbar[te, j]))))
        rows.append(dict(mode=mode, dataset=ds, region_sizes=sizes,
                         eps_by_region=meta[mode]["eps_by_region"],
                         err_by_region=errs, baseline_by_region=bases,
                         smallest_region_err=errs[-1], smallest_region_base=bases[-1],
                         mean_region_err=float(np.mean(errs)),
                         mean_region_base=float(np.mean(bases)), n_train=NTR, n_test=int(S - NTR)))
        adv = [b - e for b, e in zip(bases, errs)]
        print(f"{mode:<8} err {[round(e,4) for e in errs]}")
        print(f"{'':<8} base {[round(b,4) for b in bases]}   advantage {[round(a,4) for a in adv]}")
    json.dump(rows, open(f"{ROOT}/analysis/v2/attack_{ds}.json", "w"), indent=1)
    print("DONE ->", f"analysis/v2/attack_{ds}.json")
