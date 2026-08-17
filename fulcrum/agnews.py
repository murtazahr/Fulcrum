"""Second modality: AG News + frozen sentence encoder (federated fine-tuning).

Same experiment as probe.py, different modality. The point is narrow and specific: delta is a
property of the REGION STRUCTURE, not of the data, so this does not re-test the theory -- it
tests only that the budget saving converts to accuracy outside vision.

PRE-DECLARED (identical to probe.py):
  P1 accuracy gain at matched privacy is monotone increasing in delta
  P2 gain is exactly 0 at the delta=0 null controls
  P3 a random allocation of the same dispersion does NOT reproduce it

Expected: the STRUCTURE replicates (0 at delta=0, monotone, random negative). The MAGNITUDE
will differ from CIFAR because it depends on the local slope of accuracy-vs-noise.

Task: AG News 4-class -> binary (World+Sports vs Business+Sci/Tech), matching probe.py's head
dimension d = fdim*2+2 so the SNR ~ sqrt(a*n/d)/sigma regime is comparable.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from fedsim import eps_silo
from evaluate import rho_stats, sigmas_for_target, profiles
from probe import run, partition          # identical training loop + partitioner

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def features(root: str, fdim: int, n_train: int = 20000, n_test: int = 2000):
    """Frozen MiniLM sentence embeddings for AG News, PCA'd to `fdim`, cached."""
    cache = os.path.join(root, f"agnews_feat_{fdim}.pt")
    if os.path.exists(cache):
        b = torch.load(cache, weights_only=False)
        return b["Xtr"], b["ytr"], b["Xte"], b["yte"]
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    for _rid in ["fancyzhx/ag_news", "ag_news", "SetFit/ag_news"]:
        try:
            ds = load_dataset(_rid); print(f"loaded {_rid}", flush=True); break
        except Exception as _e:
            print(f"  {_rid} failed: {type(_e).__name__}", flush=True)
    else:
        raise RuntimeError("could not load AG News from any known repo id")
    tr, te = ds["train"].select(range(n_train)), ds["test"].select(range(n_test))
    enc = SentenceTransformer("all-MiniLM-L6-v2", device=str(DEV))
    Ftr = torch.tensor(enc.encode(tr["text"], batch_size=256, show_progress_bar=True))
    Fte = torch.tensor(enc.encode(te["text"], batch_size=256, show_progress_bar=True))
    mu, sd = Ftr.mean(0, keepdim=True), Ftr.std(0, keepdim=True) + 1e-6
    Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
    U, S, V = torch.pca_lowrank(Ftr, q=fdim)
    Xtr, Xte = Ftr @ V[:, :fdim], Fte @ V[:, :fdim]
    Xtr, Xte = Xtr / Xtr.std(), Xte / Xte.std()
    # 4-class -> binary: {World(0), Sports(1)} = 0 ; {Business(2), Sci/Tech(3)} = 1
    ytr = np.array([0 if c < 2 else 1 for c in tr["label"]])
    yte = np.array([0 if c < 2 else 1 for c in te["label"]])
    torch.save({"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte}, cache)
    return Xtr, ytr, Xte, yte


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96); ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--K", type=float, default=0.88); ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--fdim", type=int, default=32); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--local", type=int, default=10); ap.add_argument("--root", default="./cifar")
    ap.add_argument("--out", default="agnews_results.json"); ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    Xtr, ytr, Xte, yte = features(a.root, a.fdim)
    print(f"AG News features {tuple(Xtr.shape)}  d={a.fdim*2+2}  n={a.n}  "
          f"sqrt(n/d)={np.sqrt(a.n/(a.fdim*2+2)):.2f}", flush=True)
    if a.probe:
        g = np.repeat(np.arange(6), a.n // 6)
        parts, _ = partition(ytr, a.n, a.eta, g, 0)
        for sg in [0.0, 0.83, 4.0, 16.0]:
            acc = run(np.full(a.n, max(sg, 1e-9)), Xtr, ytr, parts, Xte, yte, g, a.T, 1.0, 0.5, 64, 0, a.local)
            print(f"  sigma={sg:<6} acc={acc:.4f}", flush=True)
        sys.exit()
    rows = []
    for pname, groups in profiles(a.n).items():
        for seed in range(a.seeds):
            parts, _ = partition(ytr, a.n, a.eta, groups, seed)
            w = np.array([len(p) for p in parts], float); w /= w.sum()
            rho, V, W, delta = rho_stats(w, groups)
            for mode in ["fulcrum", "uniform", "random"]:
                s2, U = sigmas_for_target(w, groups, a.K, np.full(len(V), 0.85), 2.0 * a.T, mode, seed)
                acc = run(np.sqrt(s2), Xtr, ytr, parts, Xte, yte, groups, a.T, 1.0, 0.5, 64, seed, a.local)
                se = np.zeros(len(w))
                for r in sorted(set(groups.tolist())):
                    m = groups == r; se[m] = np.sqrt(np.sum(w[m] ** 2 * s2[m])) / w[m]
                rows.append(dict(profile=pname, delta=delta, seed=seed, mode=mode, acc=acc, U=U,
                                 eps=eps_silo(a.T, float(se.min())), K=a.K, n=a.n, T=a.T,
                                 d=a.fdim * 2 + 2, dataset="agnews"))
                print(f"{pname:<20} d={delta:.3f} s={seed} {mode:<8} acc={acc:.4f} "
                      f"U={U:.4g} eps={rows[-1]['eps']:.2f}", flush=True)
            json.dump(rows, open(a.out, "w"), indent=1)
    print("DONE ->", a.out)
