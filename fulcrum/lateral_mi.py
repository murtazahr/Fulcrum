"""ell_i = I(p_i ; D_{-i}) -- the lateral (prior-coupling) leakage floor, in NATS.

D_{-i} reveals other silos' concentrations {p_j} exactly; since
p_i -- p_{-i} -- (any statistic of D_{-i}) is Markov, I(p_i;D_{-i}) <= I(p_i;p_{-i}),
so this upper-bounds lateral leakage for ANY observation of the other silos.

Hierarchical prior:  Phi_g ~ Beta(a0,b0);  p_i | Phi_{g(i)} ~ Beta(k*Phi, k*(1-Phi)).
Only i's m_i = |G_{g(i)}|-1 siblings matter.  Beta is an exponential family in p, so
(sum_j log p_j, sum_j log(1-p_j)) are SUFFICIENT for the Phi-posterior:
  log L(Phi) = (kPhi-1) S1 + (k(1-Phi)-1) S2 - m log B(kPhi, k(1-Phi))
which makes the m-loop O(1) and the whole computation exact up to quadrature.
"""
import numpy as np
from scipy import stats
from scipy.special import betaln

NPHI, NP = 1500, 4000
GRID = np.linspace(1e-4, 1 - 1e-4, NPHI)
PGRID = np.linspace(1e-6, 1 - 1e-6, NP)
DPHI = GRID[1] - GRID[0]


def _tables(kappa, a0, b0):
    prior = stats.beta.pdf(GRID, a0, b0); prior /= np.trapezoid(prior, GRID)
    A, B = kappa * GRID[:, None], kappa * (1 - GRID)[:, None]
    M = np.exp(stats.beta.logpdf(PGRID[None, :], A, B))
    log_qm = np.log(np.maximum(np.trapezoid(M * prior[:, None], GRID, axis=0), 1e-300))
    return prior, M, log_qm


def saturation(kappa, a0=2.0, b0=2.0, n_mc=200000, seed=1):
    """I(p_i ; Phi): exact ceiling on ell_i.

    Estimated by Monte Carlo with the EXACT conditional density, consistent with
    ell(). A quadrature of -int q log q over a uniform p-grid is not usable here:
    for kappa*Phi < 1 the Beta density has integrable singularities at p -> 0,1
    which a uniform grid mishandles (it returned a negative MI at kappa=2).
    """
    rng = np.random.default_rng(seed)
    prior, M, log_qm = _tables(kappa, a0, b0)
    phi = rng.beta(a0, b0, n_mc)
    p = np.clip(rng.beta(kappa * phi, kappa * (1 - phi)), 1e-12, 1 - 1e-12)
    log_cond = stats.beta.logpdf(p, kappa * phi, kappa * (1 - phi))
    lo = np.clip(np.searchsorted(PGRID, p) - 1, 0, NP - 2)
    w = (p - PGRID[lo]) / (PGRID[lo + 1] - PGRID[lo])
    log_marg = (1 - w) * log_qm[lo] + w * log_qm[lo + 1]
    return float(np.mean(log_cond - log_marg))


def ell(m, kappa, a0=2.0, b0=2.0, n_mc=20000, seed=0):
    if m == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    prior, M, log_qm = _tables(kappa, a0, b0)
    phi = rng.beta(a0, b0, n_mc)
    p_i = rng.beta(kappa * phi, kappa * (1 - phi))
    ps = np.clip(rng.beta((kappa * phi)[:, None], (kappa * (1 - phi))[:, None], size=(n_mc, m)), 1e-12, 1 - 1e-12)
    S1, S2 = np.log(ps).sum(1), np.log1p(-ps).sum(1)          # sufficient statistics
    kA, kB = kappa * GRID, kappa * (1 - GRID)
    logL = np.outer(S1, kA - 1) + np.outer(S2, kB - 1) - m * betaln(kA, kB)[None, :]
    logL -= logL.max(axis=1, keepdims=True)
    post = prior[None, :] * np.exp(logL)
    post /= (post.sum(1, keepdims=True) * DPHI)
    q_cond = np.maximum((post @ M) * DPHI, 1e-300)
    lo = np.clip(np.searchsorted(PGRID, p_i) - 1, 0, NP - 2)
    w = (p_i - PGRID[lo]) / (PGRID[lo + 1] - PGRID[lo]); r = np.arange(n_mc)
    lq_c = (1 - w) * np.log(q_cond[r, lo]) + w * np.log(q_cond[r, lo + 1])
    lq_m = (1 - w) * log_qm[lo] + w * log_qm[lo + 1]
    return float(np.mean(lq_c - lq_m))


if __name__ == "__main__":
    kaps = [2, 5, 20, 100]
    sat = {k: saturation(k) for k in kaps}
    ms = [0, 1, 2, 4, 9, 19, 49, 199]
    print("ell_i = I(p_i ; p_{-i})  [NATS] -- lateral leakage floor, hierarchical Beta prior (a0=b0=2)")
    print("rows = siblings m_i in i's organisational group; cols = within-group coupling kappa\n")
    print(f"{'siblings':>9} |" + "".join(f"  kappa={k:<6g}" for k in kaps))
    print("-" * 58)
    tab = {m: [ell(m, k) for k in kaps] for m in ms}
    for m in ms:
        print(f"{m:>9} |" + "".join(f"{v:12.4f}" for v in tab[m]))
    print("-" * 58)
    print(f"{'CEILING':>9} |" + "".join(f"{sat[k]:12.4f}" for k in kaps) + "   <- I(p_i;Phi), exact")
    print()
    ok = True
    for ki, k in enumerate(kaps):
        col = [tab[m][ki] for m in ms]
        mono = all(col[i] <= col[i + 1] + 5e-3 for i in range(len(col) - 1))
        under = all(v <= sat[k] + 5e-3 for v in col)
        conv = abs(col[-1] - sat[k]) < 0.05 * max(sat[k], 1e-3)
        print(f"kappa={k:<5g}  monotone={mono}  under_ceiling={under}  converges={conv}")
        ok &= mono and under and conv
    print("\nALL SELF-CHECKS PASS" if ok else "\n*** SELF-CHECK FAILURE ***")
