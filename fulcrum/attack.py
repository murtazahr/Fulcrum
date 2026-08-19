"""Property-inference attack against the regional aggregates.

WHY. Every empirical result in the paper is accuracy at a matched guarantee. The privacy
side rests on Theorem 4.4, an upper bound on I(p_i; Y) that the allocation minimises. An
upper bound being smaller is not evidence that an adversary actually does worse, so this
measures an adversary directly.

PRE-REGISTERED, WRITTEN BEFORE ANY RESULT WAS PRODUCED.

Threat model, matching Section 3. The adversary sits above the regional tier. It observes
the per-round regional aggregates Y_r^(t) = sum_{i in r} w_i * (clipped update_i + noise_i)
and nothing else -- never an individual silo's update. It knows the region structure, the
aggregation weights and the per-silo noise levels (Kerckhoffs), and it may train on shadow
federations drawn from the same generative process with known p_i. It estimates p_i, the
fraction of silo i's data in the sensitive class.

Attack. Ridge regression from the flattened aggregate trajectory of silo i's region
(T rounds x d dimensions) to p_i, fitted on shadow seeds and evaluated on held-out seeds.
The regulariser is chosen by cross-validation WITHIN the training seeds, separately per arm,
so neither arm gets a better-tuned adversary. Both arms see the same seeds, the same
partitions and the same adversary budget.

METRICS. Absolute error |p_hat_i - p_i| on held-out seeds.
  PRIMARY   worst-silo error, min_i |p_hat_i - p_i|, the most successful single attack.
            This is the quantity the min-max allocation claims to control.
  SECONDARY mean-silo error, averaged over all silos.
  BASELINE  the prior-only attack that ignores Y and predicts the training-set mean of p.
            Leakage is only meaningful relative to this.

PREDICTIONS, recorded in advance.
  P1  The worst-silo error is COMPARABLE between uniform and the optimal allocation at a
      matched guarantee. The allocation must not create a new worst case.
  P2  The mean-silo error is LOWER (more leakage) under the optimal allocation than under
      uniform. This is expected and unfavourable to us: uniform over-protects well concealed
      silos, and the allocation deliberately spends that surplus. It is recorded here so it
      cannot be presented as a surprise afterwards.
  P3  Under uniform, attack success varies across regions, being highest in the smallest
      region. Under the optimal allocation it is more nearly equal across regions. This is
      the equalisation the mechanism exists to produce.

A result contradicting P1 would be evidence against the paper's central claim.

REVISION, after a first run that was uninformative. With eta = 0.5 the partitioner pushes
p_i toward a target fixed by region parity, so most of p_i follows from the region index
alone, which is public. With equal weights, silos within a region are exchangeable, so the
best per-silo estimate is the region mean and the residual is unlearnable by construction.
The adversary therefore sat at that identifiability floor (0.046) in BOTH arms, and the
per-region errors agreed to four decimal places even where the two arms differ by 4.3x in
epsilon. The design measured within-region variance, not leakage. Three corrections:

  eta = 0     p_i is now i.i.d. across silos with no region-linked component, so every bit
              of recoverable signal must come from the observations.
  baseline    the prior-only adversary is now the honest comparator, since with eta = 0 the
              region index carries no information about p_i.
  control     a non-private arm (sigma -> 0) is run alongside. Without it, "the adversary
              learns nothing" cannot be distinguished from "the adversary is broken". If the
              control does not succeed, the negative results below carry no weight.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fedsim import eps_silo
from evaluate import rho_stats, sigmas_for_target, profiles
from probe import partition

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ELL = 0.85


def run_observed(sigmas, X, y, parts, groups, T, C, lr, bs, seed, local=10):
    """Train as in probe.run, but return what the adversary sees: the regional aggregates.

    -> Y of shape (T, n_regions, d): per round, the weighted sum within each region.
    """
    torch.manual_seed(seed)
    d = X.shape[1]
    head = nn.Linear(d, 2).to(DEV)
    gv = torch.cat([p.data.flatten() for p in head.parameters()])
    n = len(parts)
    w = np.array([len(p) for p in parts], float); w /= w.sum()
    regs = sorted(set(groups.tolist()))
    yt = torch.tensor(y, dtype=torch.long)
    obs = np.zeros((T, len(regs), gv.numel()), dtype=np.float32)
    for t in range(T):
        D = torch.zeros(n, gv.numel(), device=DEV)
        for i in range(n):
            o = 0
            for pp in head.parameters():
                k = pp.numel(); pp.data.copy_(gv[o:o + k].view_as(pp)); o += k
            opt = torch.optim.SGD(head.parameters(), lr=lr)
            ix = parts[i]
            for _ in range(local):
                b = ix[torch.randperm(len(ix))[:bs].numpy()]
                opt.zero_grad()
                F.cross_entropy(head(X[b].to(DEV)), yt[b].to(DEV)).backward(); opt.step()
            dv = torch.cat([pp.data.flatten() for pp in head.parameters()]) - gv
            dv = dv * min(1.0, C / (dv.norm().item() + 1e-12))
            D[i] = dv + torch.randn_like(dv) * sigmas[i] * C
        wt = torch.tensor(w, device=DEV, dtype=D.dtype)[:, None]
        for j, r in enumerate(regs):
            m = torch.tensor(groups == r, device=DEV)
            obs[t, j] = (wt[m] * D[m]).sum(0).cpu().numpy()
        gv = gv + (wt * D).sum(0)
    return obs


def ridge_cv(Xtr, ytr, Xte, alphas=(1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4)):
    """Ridge in DUAL form, alpha chosen by k-fold CV inside the training set only.

    There are far more features (T*d) than shadow federations, so the dual solves an
    n_train x n_train system instead of a d x d one. Identical predictions, orders of
    magnitude cheaper. Standardisation uses training statistics only.
    """
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    KA, KB = A @ A.T, B @ A.T
    k = min(5, len(A))
    idx = np.arange(len(A)) % k
    best, best_err = alphas[0], np.inf
    for al in alphas:
        err = []
        for f in range(k):
            tr, va = idx != f, idx == f
            if va.sum() == 0 or tr.sum() < 2: continue
            m = ytr[tr].mean()
            dual = np.linalg.solve(KA[np.ix_(tr, tr)] + al * np.eye(tr.sum()), ytr[tr] - m)
            err.append(np.mean(np.abs(KA[np.ix_(va, tr)] @ dual + m - ytr[va])))
        if err and np.mean(err) < best_err: best_err, best = np.mean(err), al
    m = ytr.mean()
    dual = np.linalg.solve(KA + best * np.eye(len(A)), ytr - m)
    return KB @ dual + m, best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar", "agnews"], default="cifar")
    ap.add_argument("--n", type=int, default=96); ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--eta", type=float, default=0.0); ap.add_argument("--fdim", type=int, default=32)
    ap.add_argument("--K", type=float, default=0.88); ap.add_argument("--local", type=int, default=10)
    ap.add_argument("--train", type=int, default=40); ap.add_argument("--test", type=int, default=20)
    ap.add_argument("--profile", default="severe 15/5/2/1/1")
    ap.add_argument("--root", default="./cifar"); ap.add_argument("--out", required=True)
    a = ap.parse_args()

    features = __import__("probe" if a.dataset == "cifar" else "agnews").features
    Xtr, ytr, _, _ = features(a.root, a.fdim)
    groups = profiles(a.n)[a.profile]
    regs = sorted(set(groups.tolist()))
    aa = 2.0 * a.T
    S = a.train + a.test
    print(f"{a.dataset}: profile '{a.profile}', regions {[int((groups==r).sum()) for r in regs]}, "
          f"{S} federations per arm", flush=True)

    store = {}
    for mode in ("control", "uniform", "fulcrum"):
        Ys, Ps = [], []
        for seed in range(S):
            parts, p = partition(ytr, a.n, a.eta, groups, seed, size_spread=0.0)
            w = np.array([len(q) for q in parts], float); w /= w.sum()
            _, V, _, delta = rho_stats(w, groups)
            if mode == "control":                 # positive control: essentially no noise
                s2 = np.full(a.n, 1e-12)
            else:
                s2, U = sigmas_for_target(w, groups, a.K, np.full(len(V), ELL), aa, mode, seed)
            Y = run_observed(np.sqrt(s2), Xtr, ytr, parts, groups, a.T, 1.0, 0.5, 64, seed, a.local)
            Ys.append(Y); Ps.append(p)
            if seed % 10 == 0: print(f"  {mode} seed {seed}/{S}", flush=True)
        se = np.zeros(a.n)
        for r in regs:
            m = groups == r; se[m] = np.sqrt(np.sum(w[m] ** 2 * s2[m])) / w[m]
        ctrl = mode == "control"
        store[mode] = dict(Y=np.array(Ys), P=np.array(Ps), delta=delta,
                           eps_worst=None if ctrl else eps_silo(a.T, float(se.min())),
                           eps_by_region=None if ctrl else
                           [eps_silo(a.T, float(se[groups == r][0])) for r in regs])
    json.dump({m: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in d.items() if k != "Y"} for m, d in store.items()},
              open(a.out.replace(".json", "_meta.json"), "w"), indent=1)
    np.savez_compressed(a.out.replace(".json", "_obs.npz"),
                        **{f"{m}_{k}": store[m][k] for m in store for k in ("Y", "P")})
    print("observations written; fitting adversary", flush=True)

    rows = []
    for mode, d in store.items():
        Y, P = d["Y"], d["P"]
        tr, te = np.arange(a.train), np.arange(a.train, S)
        pred = np.zeros((len(te), a.n))
        for i in range(a.n):
            j = regs.index(groups[i])
            F_ = Y[:, :, j, :].reshape(S, -1)          # this silo's region trajectory
            pred[:, i], _ = ridge_cv(F_[tr], P[tr, i], F_[te])
        err = np.abs(pred - P[te])                      # (test seeds, silos)
        base = np.abs(P[tr].mean() - P[te])             # prior-only adversary
        per_silo = err.mean(0)
        rows.append(dict(mode=mode, delta=d["delta"], eta=a.eta, eps_worst=d["eps_worst"],
                         eps_by_region=d["eps_by_region"],
                         worst_silo_err=float(per_silo.min()),      # PRIMARY
                         mean_silo_err=float(per_silo.mean()),      # SECONDARY
                         baseline_err=float(base.mean()),
                         err_by_region=[float(per_silo[groups == r].mean()) for r in regs],
                         region_sizes=[int((groups == r).sum()) for r in regs],
                         n_train=a.train, n_test=a.test, dataset=a.dataset, K=a.K, T=a.T))
        print(f"  {mode:<8} worst-silo {rows[-1]['worst_silo_err']:.4f}  "
              f"mean-silo {rows[-1]['mean_silo_err']:.4f}  baseline {rows[-1]['baseline_err']:.4f}", flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    print("DONE ->", a.out)
