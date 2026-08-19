"""Privacy-utility curve, measured through the SAME pipeline that produces the main table.

Why this exists. The first version of this figure swept the RAW injected sigma and labelled
the x axis with eps_silo(T, sigma_raw). That is the wrong accounting basis: a client's release
is concealed by the TOTAL noise in its region, so its effective multiplier is
sqrt(S_r)/w_i, not sigma_i. Labelling the raw sigma overstates eps by sqrt(m) in the noise
argument -- a factor of about 4.5 in eps at m=16 -- and the resulting curve did not pass
through the operating point reported in the main table. It also had no generator in the repo,
so it could not be checked.

This script instead drives sigmas_for_target(..., mode="uniform") on the balanced profile,
exactly as the main sweep does, and reports eps with the same effective-sigma computation.
The eps = 0.99 point therefore reproduces the main table's uniform column by construction,
because it is the same code path with the same seeds.

For the uniform arm on an equal-weight balanced profile the target is analytic:
    S_r = a*W_r/(K - ell),  W_r = max_i w_i^2 = w^2,  sigma_eff = sqrt(S_r)/w = sqrt(a/(K-ell))
so a target eps inverts to K = ell + a/sigma_eff^2. We sweep eps and derive K.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fedsim import eps_silo
from evaluate import rho_stats, sigmas_for_target, profiles
from probe import run, partition

ELL = 0.85


def sigma_eff_for_eps(target: float, T: int) -> float:
    lo, hi = 1e-3, 1e6
    for _ in range(200):
        mid = (lo + hi) / 2
        if eps_silo(T, mid) > target: lo = mid
        else: hi = mid
    return hi


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar", "agnews"], required=True)
    ap.add_argument("--n", type=int, default=96); ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--eta", type=float, default=0.5); ap.add_argument("--fdim", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--local", type=int, default=10)
    ap.add_argument("--root", default="./cifar"); ap.add_argument("--out", required=True)
    ap.add_argument("--eps", default="0.2,0.35,0.5,0.7,0.99,1.5,2.5,5,10,25")
    a = ap.parse_args()

    if a.dataset == "cifar":
        from probe import features
    else:
        from agnews import features
    Xtr, ytr, Xte, yte = features(a.root, a.fdim)
    d = a.fdim * 2 + 2
    aa = 2.0 * a.T
    groups = profiles(a.n)["null: balanced 6 eq"]
    targets = [float(x) for x in a.eps.split(",")]
    print(f"{a.dataset}: features {tuple(Xtr.shape)} d={d} n={a.n} T={a.T} seeds={a.seeds}", flush=True)

    rows = []
    for tgt in targets + [None]:                      # None = non-private reference
        for seed in range(a.seeds):
            parts, _ = partition(ytr, a.n, a.eta, groups, seed, size_spread=0.0)
            w = np.array([len(p) for p in parts], float); w /= w.sum()
            rho, V, W, delta = rho_stats(w, groups)
            se_min = None
            if tgt is None:
                s2 = np.full(a.n, 1e-18); U, eps, K = 0.0, None, None
            else:
                se_t = sigma_eff_for_eps(tgt, a.T)
                K = ELL + aa / se_t ** 2
                s2, U = sigmas_for_target(w, groups, K, np.full(len(V), ELL), aa, "uniform", seed)
                se = np.zeros(a.n)
                for r in sorted(set(groups.tolist())):
                    m = groups == r
                    se[m] = np.sqrt(np.sum(w[m] ** 2 * s2[m])) / w[m]
                eps = eps_silo(a.T, float(se.min())); se_min = float(se.min())
            acc = run(np.sqrt(s2), Xtr, ytr, parts, Xte, yte, groups, a.T, 1.0, 0.5, 64, seed, a.local)
            rows.append(dict(dataset=a.dataset, target_eps=tgt, eps=eps, K=K, seed=seed, acc=acc,
                             U=U, sigma_raw=float(np.sqrt(s2[0])), sigma_eff_min=se_min,
                             spread=0.0, mode="uniform",
                             profile="null: balanced 6 eq", n=a.n, T=a.T, d=d))
            print(f"  eps_tgt={str(tgt):<5} eps={('%.4f'%eps) if eps else 'inf':<8} "
                  f"s={seed} acc={acc:.4f} U={U:.4g}", flush=True)
            json.dump(rows, open(a.out, "w"), indent=1)
    print("DONE ->", a.out)
