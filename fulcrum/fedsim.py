"""Silo-level DP federated training + measured distributional-inference attack.

This is the experiment v1 never ran: does the allocation reduce MEASURED attack
success (not just an analytic bound) at a matched noise budget, in a regime where
(eps, delta) is actually meaningful?

Mechanism (matches THEORY_V2 Theorem 1): each round, client i trains locally,
forms Delta_i = theta_i - theta_global, CLIPS THE UPDATE to ||Delta|| <= C, and
adds xi ~ N(0, sigma_i^2 C^2 I). Regional aggregators sum over A(i); the adversary
observes regional aggregates. Silo-level adjacency throughout -- no group-privacy gap.

Accounting: Gaussian mechanism, replace-one-silo sensitivity 2C, noise sigma*C,
T rounds. RDP(alpha) = 2*T*alpha/sigma^2, converted to (eps, delta).
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------- accounting
def eps_silo(T: int, sigma: float, delta: float = 1e-5, sens: float = 2.0) -> float:
    """(eps, delta) for T-fold composition of the silo-level Gaussian mechanism.

    RDP of the Gaussian mechanism with L2 sensitivity `sens*C` and noise `sigma*C`
    is alpha*sens^2/(2 sigma^2) per release; T releases compose additively.
    """
    if sigma <= 0:
        return float("inf")
    best = float("inf")
    for alpha in np.concatenate([np.arange(1.01, 64, 0.01), np.arange(64, 4096, 1.0)]):
        rdp = T * alpha * (sens ** 2) / (2 * sigma ** 2)
        e = rdp + math.log1p(-1 / alpha) - (math.log(delta) + math.log(alpha)) / (alpha - 1)
        best = min(best, e)
    return float(best)


# ---------------------------------------------------------------- allocation
def fulcrum_alloc(ell: np.ndarray, a: float, U: float) -> np.ndarray:
    """sigma_i^2 = a/(K*-ell_i) with sum sigma_i^2 = U (bisection on K)."""
    ell = np.asarray(ell, float)
    if np.allclose(ell, ell[0]):
        return np.full(ell.size, U / ell.size)
    lo, hi = ell.max() + 1e-12, ell.max() + a * ell.size / U + 1.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if np.sum(a / (mid - ell)) > U:
            lo = mid
        else:
            hi = mid
    return a / (0.5 * (lo + hi) - ell)


# ---------------------------------------------------------------- data
def load_cifar_eta(n_clients: int, eta: float, groups: np.ndarray, seed: int, root: str):
    """eta-coupled partition: Delta_i = (1-eta)*Dirichlet + eta*group-determined."""
    cache = os.path.join(root, "cifar_cache.pt")
    if os.path.exists(cache):
        blob = torch.load(cache, weights_only=False)
        X, y, Xte, yte = blob["X"], blob["y"], blob["Xte"], blob["yte"]
    else:
        from torchvision import datasets, transforms
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        tr = datasets.CIFAR10(root, train=True, download=True, transform=tf)
        te = datasets.CIFAR10(root, train=False, download=True, transform=tf)
        X = torch.stack([tr[i][0] for i in range(len(tr))])
        y = np.array(tr.targets)
        Xte = torch.stack([te[i][0] for i in range(2000)])
        yte = torch.tensor(te.targets[:2000])
        torch.save({"X": X, "y": y, "Xte": Xte, "yte": yte}, cache)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    K = 10
    n_groups = int(groups.max()) + 1
    iid = rng.dirichlet(np.full(K, 0.5), size=n_clients)
    tgt = np.zeros((n_clients, K))
    for i in range(n_clients):
        tgt[i, (groups[i] * 3) % K] = 1.0            # group-determined concentration
    mix = (1 - eta) * iid + eta * tgt
    mix = mix / mix.sum(1, keepdims=True)
    by_cls = {c: list(rng.permutation(np.where(y == c)[0])) for c in range(K)}
    per = len(X) // n_clients
    # Draw EXACTLY `per` samples for every client. Sampling classes with replacement
    # from the mixture and refilling exhausted classes keeps |D_i| identical across
    # clients, so w_i is uniform by construction and any delta > 0 in a balanced
    # profile is impossible. (The old code truncated when a class ran out, which
    # silently created weight asymmetry and contaminated the null control.)
    parts = []
    for i in range(n_clients):
        draw = rng.choice(K, size=per, p=mix[i])
        idx = []
        for c in range(K):
            need = int((draw == c).sum())
            while len(by_cls[c]) < need:
                by_cls[c] += list(rng.permutation(np.where(y == c)[0]))
            idx += [by_cls[c].pop() for _ in range(need)]
        parts.append(np.array(idx, dtype=np.int64))
    return X, torch.tensor(y), parts, Xte, yte, mix


class CNN(nn.Module):
    def __init__(self, k=10):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(3, 32, 3, padding=1), nn.Conv2d(32, 64, 3, padding=1)
        self.f1, self.f2 = nn.Linear(64 * 8 * 8, 128), nn.Linear(128, k)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.f2(F.relu(self.f1(x.flatten(1))))


def flat(m):
    return torch.cat([p.data.flatten() for p in m.parameters()])


def setflat(m, v):
    o = 0
    for p in m.parameters():
        n = p.numel(); p.data.copy_(v[o:o + n].view_as(p)); o += n


# ---------------------------------------------------------------- training
def run(sigmas, X, y, parts, Xte, yte, groups, T, C, lr, bs, seed, local_steps=1):
    """Returns (test_acc, observed regional aggregates per round [T, n_regions, d])."""
    torch.manual_seed(seed)
    g = CNN().to(DEV)
    gv = flat(g).clone()
    n = len(parts)
    regions = sorted(set(groups.tolist()))
    w = np.array([len(p) for p in parts], float); w /= w.sum()
    obs = []
    for t in range(T):
        deltas = torch.zeros(n, gv.numel(), device=DEV)
        for i in range(n):
            setflat(g, gv)
            opt = torch.optim.SGD(g.parameters(), lr=lr)
            idx = parts[i]
            for _ in range(local_steps):
                b = idx[torch.randperm(len(idx))[:bs].numpy()]
                xb, yb = X[b].to(DEV), y[b].to(DEV)
                opt.zero_grad(); F.cross_entropy(g(xb), yb).backward(); opt.step()
            d = flat(g) - gv
            nrm = d.norm()
            d = d * min(1.0, C / (nrm.item() + 1e-12))              # SILO-level clip
            d = d + torch.randn_like(d) * sigmas[i] * C             # silo-level noise
            deltas[i] = d
        # regional aggregation: adversary sees only the per-region weighted sum
        ro = []
        for r in regions:
            m = np.where(groups == r)[0]
            ww = torch.tensor(w[m] / w[m].sum(), device=DEV, dtype=deltas.dtype)
            ro.append((ww[:, None] * deltas[m]).sum(0))
        obs.append(torch.stack(ro).cpu().numpy())
        gv = gv + torch.tensor(w, device=DEV, dtype=deltas.dtype)[:, None].mul(deltas).sum(0)
    setflat(g, gv)
    with torch.no_grad():
        acc = (g(Xte.to(DEV)).argmax(1).cpu() == yte).float().mean().item()
    return acc, np.stack(obs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--T", type=int, default=25)
    ap.add_argument("--eta", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--root", default="./cifar")
    a = ap.parse_args()
    groups = np.repeat(np.arange(6), a.n // 6)
    t0 = time.time()
    X, y, parts, Xte, yte, mix = load_cifar_eta(a.n, a.eta, groups, a.seed, a.root)
    print(f"data {time.time()-t0:.1f}s, sizes {[len(p) for p in parts][:6]}...")
    t0 = time.time()
    acc, obs = run(np.full(a.n, a.sigma), X, y, parts, Xte, yte, groups,
                   a.T, 1.0, 0.05, 64, a.seed)
    print(f"T={a.T} sigma={a.sigma} acc={acc:.4f} eps={eps_silo(a.T,a.sigma):.2f} "
          f"obs={obs.shape} time={time.time()-t0:.1f}s")
