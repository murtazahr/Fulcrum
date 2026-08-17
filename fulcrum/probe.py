"""Silo-level DP federated FINE-TUNING of a pretrained backbone (linear probe).

Why this fixes the utility wall. The aggregate noise norm is C*sqrt(sum_i w_i^2 sigma_i^2 * d),
the aggregate signal is ~C*sqrt(a) for average update alignment a, so

    SNR  ~  sqrt(a * n / d) / sigma

Training a CNN from scratch had d=545,098 vs n=96 -- SNR ~ 0.004, hopeless at ANY sigma.
Fine-tuning a head on a frozen pretrained backbone attacks both terms: d drops by ~3 orders
of magnitude, and a rises because the shared backbone makes client updates far more aligned.
This is a standard DP setup (DP linear probing), not a favourable rigging of the benchmark.

Backbone: ImageNet-pretrained ResNet18, frozen. Features precomputed ONCE (so the federated
sweep is fast), then projected to `--fdim` dims. Task: CIFAR-10 animals-vs-vehicles, which
also makes the sensitive-class concentration p_i a natural binary quantity.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from fedsim import eps_silo
from evaluate import rho_stats, sigmas_for_target, profiles

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
VEHICLES = {0, 1, 8, 9}          # airplane, automobile, ship, truck -> label 0; animals -> 1


def features(root: str, fdim: int):
    """Frozen ResNet18 penultimate features for CIFAR-10, projected to `fdim`, cached."""
    cache = os.path.join(root, f"probe_feat_{fdim}.pt")
    if os.path.exists(cache):
        b = torch.load(cache, weights_only=False)
        return b["Xtr"], b["ytr"], b["Xte"], b["yte"]
    from torchvision import datasets, transforms, models
    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity(); net.eval().to(DEV)

    def emb(ds, lim):
        out, lab = [], []
        with torch.no_grad():
            for s in range(0, lim, 256):
                xb = torch.stack([ds[i][0] for i in range(s, min(s + 256, lim))]).to(DEV)
                out.append(net(xb).cpu())
                lab += [ds[i][1] for i in range(s, min(s + 256, lim))]
                print(f"  embed {s + 256}/{lim}", flush=True)
        return torch.cat(out), np.array(lab)

    tr = datasets.CIFAR10(root, train=True, download=True, transform=tf)
    te = datasets.CIFAR10(root, train=False, download=True, transform=tf)
    Ftr, ytr = emb(tr, 20000)
    Fte, yte = emb(te, 2000)
    mu, sd = Ftr.mean(0, keepdim=True), Ftr.std(0, keepdim=True) + 1e-6
    Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
    # PCA to fdim (fit on train)
    U, S, V = torch.pca_lowrank(Ftr, q=fdim)
    Xtr, Xte = Ftr @ V[:, :fdim], Fte @ V[:, :fdim]
    Xtr, Xte = Xtr / Xtr.std(), Xte / Xte.std()
    ytr = np.array([0 if c in VEHICLES else 1 for c in ytr])
    yte = np.array([0 if c in VEHICLES else 1 for c in yte])
    torch.save({"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte}, cache)
    return Xtr, ytr, Xte, yte


def partition(y, n, eta, groups, seed):
    """eta-coupled binary partition: region g pushes p_i toward g%2. Equal |D_i| by construction."""
    rng = np.random.default_rng(seed)
    idx0, idx1 = list(rng.permutation(np.where(y == 0)[0])), list(rng.permutation(np.where(y == 1)[0]))
    per = len(y) // n
    base = rng.uniform(0.3, 0.7, n)
    tgt = np.array([0.05 if g % 2 == 0 else 0.95 for g in groups])
    p = (1 - eta) * base + eta * tgt
    parts, ps = [], []
    for i in range(n):
        k1 = int(round(p[i] * per)); k0 = per - k1
        while len(idx1) < k1: idx1 += list(rng.permutation(np.where(y == 1)[0]))
        while len(idx0) < k0: idx0 += list(rng.permutation(np.where(y == 0)[0]))
        take = [idx1.pop() for _ in range(k1)] + [idx0.pop() for _ in range(k0)]
        parts.append(np.array(take)); ps.append(k1 / per)
    return parts, np.array(ps)


def run(sigmas, X, y, parts, Xte, yte, groups, T, C, lr, bs, seed, local=10):
    torch.manual_seed(seed)
    d = X.shape[1]
    head = nn.Linear(d, 2).to(DEV)
    gv = torch.cat([p.data.flatten() for p in head.parameters()])
    n = len(parts)
    w = np.array([len(p) for p in parts], float); w /= w.sum()
    yt = torch.tensor(y, dtype=torch.long)
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
        gv = gv + torch.tensor(w, device=DEV, dtype=D.dtype)[:, None].mul(D).sum(0)
    o = 0
    for pp in head.parameters():
        k = pp.numel(); pp.data.copy_(gv[o:o + k].view_as(pp)); o += k
    with torch.no_grad():
        acc = (head(Xte.to(DEV)).argmax(1).cpu().numpy() == yte).mean()
    return float(acc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96); ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--K", type=float, default=2.65); ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--fdim", type=int, default=32); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--local", type=int, default=10); ap.add_argument("--root", default="./cifar")
    ap.add_argument("--out", default="probe_results.json"); ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    Xtr, ytr, Xte, yte = features(a.root, a.fdim)
    d = a.fdim * 2 + 2
    print(f"features {tuple(Xtr.shape)}  head dim d={d}  n={a.n}  sqrt(n/d)={np.sqrt(a.n/d):.2f}")
    if a.probe:                      # quick SNR probe before committing to the sweep
        g = np.repeat(np.arange(6), a.n // 6)
        parts, ps = partition(ytr, a.n, a.eta, g, 0)
        for sg in [0.0, 0.4, 0.83, 1.5]:
            acc = run(np.full(a.n, max(sg, 1e-9)), Xtr, ytr, parts, Xte, yte, g, a.T, 1.0, 0.5, 64, 0, a.local)
            print(f"  sigma={sg:<5} acc={acc:.4f}", flush=True)
        sys.exit()
    rows = []
    for pname, groups in profiles(a.n).items():
        for seed in range(a.seeds):
            parts, ps = partition(ytr, a.n, a.eta, groups, seed)
            w = np.array([len(p) for p in parts], float); w /= w.sum()
            rho, V, W, delta = rho_stats(w, groups)
            for mode in ["fulcrum", "uniform", "random"]:
                s2, U = sigmas_for_target(w, groups, a.K, np.full(len(V), 0.85), 2.0 * a.T, mode, seed)
                acc = run(np.sqrt(s2), Xtr, ytr, parts, Xte, yte, groups, a.T, 1.0, 0.5, 64, seed, a.local)
                se = np.zeros(len(w))
                for r in sorted(set(groups.tolist())):
                    m = groups == r; se[m] = np.sqrt(np.sum(w[m] ** 2 * s2[m])) / w[m]
                eps = eps_silo(a.T, float(se.min()))
                rows.append(dict(profile=pname, delta=delta, seed=seed, mode=mode, acc=acc,
                                 U=U, eps=eps, K=a.K, n=a.n, T=a.T, d=d))
                print(f"{pname:<20} d={delta:.3f} s={seed} {mode:<8} acc={acc:.4f} U={U:.4g} eps={eps:.2f}", flush=True)
            json.dump(rows, open(a.out, "w"), indent=1)
    print("DONE ->", a.out)
