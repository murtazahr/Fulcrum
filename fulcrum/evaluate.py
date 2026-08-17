"""Evaluation designed so the result CANNOT be manufactured by choosing a benchmark.

The theory predicts a closed form for the budget saved at equal per-client privacy:

    delta := 1 - <rho>_V / rho_max ,    rho_r = (max_{i in r} w_i^2)/(sum_{i in r} w_i^2)

delta is an identity in the deployment's region-size profile -- no experimental freedom.
So we do NOT pick a favourable benchmark. We SWEEP delta across its whole range,
including delta = 0 where the theory predicts EXACTLY ZERO gain, and test the
falsifiable prediction:

    PRE-DECLARED PREDICTION
    (P1) accuracy gain of Fulcrum over uniform, at MATCHED per-client privacy,
         is monotone increasing in delta;
    (P2) it is ~0 at delta = 0 (null control);
    (P3) a random non-uniform allocation with the SAME dispersion does NOT
         reproduce it (rules out "any asymmetry helps").

Any of P1-P3 failing falsifies the account. Real FLamby-derived region structures
are marked on the delta axis so the reader can locate their own deployment.

Comparison is at MATCHED PRIVACY: both arms hold every client at the same worst-case
per-client bound K (hence the same silo-level (eps,delta)); they differ only in total
noise injected, and therefore in accuracy.
"""
from __future__ import annotations
import itertools, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from fedsim import CNN, eps_silo, flat, setflat, load_cifar_eta, run, DEV

A_COEF = None  # set from T


# --------------------------------------------------------------- structure stats
def rho_stats(w: np.ndarray, groups: np.ndarray):
    """rho_r per region, V_r region weight mass, and delta = 1 - <rho>_V/rho_max."""
    regs = sorted(set(groups.tolist()))
    W = np.array([(w[groups == r] ** 2).max() for r in regs])
    V = np.array([(w[groups == r] ** 2).sum() for r in regs])
    rho = W / V
    delta = 1.0 - float(np.sum(rho * V) / V.sum() / rho.max())
    return rho, V, W, delta


def sigmas_for_target(w, groups, K, ell_r, a, mode, seed=0):
    """Per-client sigma^2 holding EVERY client at worst-case bound <= K.

    Fulcrum : per-region noise S_r = a*W_r/(K-ell_r), split evenly inside the region.
    Uniform : a single sigma for all clients, sized for the worst region.
    Random  : non-uniform per-region noise with the same dispersion as Fulcrum but
              a randomly permuted assignment -- then rescaled so it also meets K.
    """
    rho, V, W, _ = rho_stats(w, groups)
    regs = sorted(set(groups.tolist()))
    need = np.array([a * W[j] / (K - ell_r[j]) for j in range(len(regs))])  # required S_r
    if mode == "fulcrum":
        S = need
    elif mode == "uniform":
        s2 = max(need[j] / V[j] for j in range(len(regs)))
        S = s2 * V
    elif mode == "random":
        # Misallocation control. Permuting `need` is a NO-OP when all regions share the
        # same W_r and ell_r (equal-weight clients): need is then constant across regions.
        # The quantity that actually varies is the per-client sigma^2 = S_r/V_r, so we
        # permute WHICH REGION RECEIVES WHICH per-client sigma, then rescale so every
        # region still meets its constraint. Same dispersion, wrong assignment.
        rng = np.random.default_rng(seed + 1234)
        s2_reg = need / V                                   # fulcrum per-client sigma^2 by region
        # Bounded retry. When all regions are the same effective size s2_reg is CONSTANT,
        # so every permutation is a no-op -- that is correct (nothing to misallocate) and
        # must not be retried forever.
        perm = rng.permutation(len(regs))
        for _ in range(20):
            if not np.allclose(s2_reg[perm], s2_reg):
                break
            perm = rng.permutation(len(regs))
        s2_perm = s2_reg[perm]
        c = max(need[j] / (s2_perm[j] * V[j]) for j in range(len(regs)))
        S = s2_perm * V * c
    else:
        raise ValueError(mode)
    sig2 = np.zeros(len(w))
    for j, r in enumerate(regs):
        m = groups == r
        sig2[m] = S[j] / (w[m] ** 2).sum()      # even split inside region
    return sig2, float(np.sum(w ** 2 * sig2))   # (per-client sigma^2, total noise U)


# --------------------------------------------------------------- region profiles
def profiles(n=48):
    """Region-size profiles spanning delta in [0, ~0.9]. Fractions, so they scale with n.
    'balanced' entries are NULL CONTROLS: theory predicts delta = 0 exactly."""
    P = {
        "null: balanced 6 eq":  [1, 1, 1, 1, 1, 1],
        "null: balanced 4 eq":  [1, 1, 1, 1],
        "mild  6/6/6/3/3":      [6, 6, 6, 3, 3],
        "severe 15/5/2/1/1":    [15, 5, 2, 1, 1],
    }
    out = {}
    for k, frac in P.items():
        tot = sum(frac)
        sizes = [max(1, round(f * n / tot)) for f in frac]
        sizes[0] += n - sum(sizes)
        out[k] = np.array(sum([[g] * s for g, s in enumerate(sizes)], []))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--T", type=int, default=25)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--K", type=float, default=2.2)     # target per-client bound (nats)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--root", default="./cifar")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--local", type=int, default=20)
    args = ap.parse_args()

    a = 2.0 * args.T
    ELL = 0.85                                        # ell saturates; near-constant across regions
    profs = profiles(args.n)
    rows = []
    t_start = time.time()
    for pname, groups in profs.items():
        for seed in range(args.seeds):
            X, y, parts, Xte, yte, mix = load_cifar_eta(args.n, args.eta, groups, seed, args.root)
            w = np.array([len(p) for p in parts], float); w /= w.sum()
            rho, V, W, delta = rho_stats(w, groups)
            ell_r = np.full(len(V), ELL)
            for mode in ["fulcrum", "uniform", "random"]:
                sig2, U = sigmas_for_target(w, groups, args.K, ell_r, a, mode, seed)
                sig = np.sqrt(sig2)
                acc, _ = run(sig, X, y, parts, Xte, yte, groups, args.T, 1.0, 0.05, 64, seed,
                             local_steps=args.local)
                # CORRECT per-client eps under aggregated observation: client i's release
                # is hidden by the TOTAL noise in its region, so its effective noise
                # multiplier is sqrt(S_r)/w_i, not sigma_i. Worst client sets eps.
                regs = sorted(set(groups.tolist()))
                sig_eff = np.zeros(len(w))
                for r in regs:
                    mmask = groups == r
                    S_r = float(np.sum(w[mmask] ** 2 * sig2[mmask]))
                    sig_eff[mmask] = np.sqrt(S_r) / w[mmask]
                eps = eps_silo(args.T, float(sig_eff.min()))
                rows.append(dict(profile=pname, delta=delta, seed=seed, mode=mode,
                                 acc=acc, U=U, eps_worst=eps, sigma_min=float(sig.min()), sigma_eff_min=float(sig_eff.min()),
                                 sigma_max=float(sig.max()), K=args.K, n=args.n, T=args.T, local=args.local))
                print(f"{pname:<22} d={delta:.3f} s={seed} {mode:<8} acc={acc:.4f} "
                      f"U={U:.4g} eps={eps:.2f}  [{time.time()-t_start:.0f}s]", flush=True)
            json.dump(rows, open(args.out, "w"), indent=1)
    print(f"\nDONE {len(rows)} runs in {time.time()-t_start:.0f}s -> {args.out}")
