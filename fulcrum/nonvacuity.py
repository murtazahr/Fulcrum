"""Corrected bound: non-vacuity regime + DUAL-form gain of the Fulcrum allocation.

CORRECTED per-round mechanism term (silo-level adjacency, silo-level DP):
  client i clips its UPDATE to ||Delta||<=C and adds N(0, sigma_i^2 C^2).
  swapping D_i entirely moves the clipped update by <= 2C, so
      per-round KL <= (2C)^2 / (2 sigma_i^2 C^2) = 2 / sigma_i^2                (per-client obs)
      per-round KL <= 2 w_i^2 / sum_{j in A(i)} w_j^2 sigma_j^2                 (aggregated obs)
  NOTE the paper's 1/(2 sigma^2 |B|^2) uses RECORD-level sensitivity C/|B| with a
  DATASET-level MI on the left -- that is the group-privacy gap. It also mis-places
  the Opacus noise (Opacus adds sigma*C to the SUM, so scale on the mean is sigma*C/|B|).

Total:  I(p_i; phat_i) <= min{ H(p_i),  T * kappa_i(sigma) + ell_i }
Margin: M_i := H(p_i) - ell_i  = information the mechanism actually has to protect.
Non-vacuous iff  T * kappa_i(sigma) < M_i.
"""
import numpy as np
from scipy import stats
from lateral_mi import ell

def H_quantized(kappa, a0=2.0, b0=2.0, r=20, n=400000, seed=3):
    """Prior entropy of p_i quantized to r levels (nats) -- the vacuity ceiling."""
    rng = np.random.default_rng(seed)
    phi = rng.beta(a0, b0, n)
    p = rng.beta(kappa*phi, kappa*(1-phi))
    c = np.histogram(p, bins=np.linspace(0,1,r+1))[0].astype(float); c /= c.sum()
    c = c[c>0]
    return float(-(c*np.log(c)).sum())

def sigma_needed(T, margin, frac=0.5, agg=1):
    """sigma^2 needed so the mechanism term is frac*margin. agg = aggregation-set size."""
    return 2.0*T/(agg*frac*margin)

def dual_gain(ells, K_target):
    """Budget U = sum sigma_i^2 needed to hold EVERY client at <= K_target.
    Fulcrum: sigma_i^2 = 2T/(K-ell_i).  Uniform: must satisfy the worst client."""
    ells = np.asarray(ells, float)
    if K_target <= ells.max():
        return np.nan, np.nan, np.nan
    u_f = np.sum(1.0/(K_target-ells))          # in units of 2T
    u_u = len(ells)/(K_target-ells.max())
    return u_f, u_u, u_u/u_f

if __name__ == "__main__":
    KAPPA = 20
    Hq = H_quantized(KAPPA)
    print(f"Prior entropy ceiling H(p_i) at 20-level quantisation, kappa={KAPPA}: {Hq:.4f} nats")
    print(f"Lateral floor saturates at ell_max = {ell(199,KAPPA):.4f} nats\n")

    print("=== 1. NON-VACUITY: sigma required for the mechanism term to use half the margin ===")
    print(f"{'group |G|':>9} {'ell_i':>8} {'margin M_i':>11} | " +
          "".join(f"{'T='+str(T):>12}" for T in [10,25,50,100]))
    print("-"*74)
    for g in [2,5,10,50]:
        e = ell(g-1, KAPPA); M = Hq - e
        row = "".join(f"{np.sqrt(sigma_needed(T,M)):>12.2f}" for T in [10,25,50,100])
        print(f"{g:>9} {e:>8.4f} {M:>11.4f} | {row}")
    print("   (values are the per-client noise multiplier sigma; per-client observation model)")
    print()
    print("   Same, but with neighbourhood aggregation over |A| silos (secure agg. within region):")
    print(f"{'|A|':>9} " + "".join(f"{'T='+str(T):>12}" for T in [10,25,50,100]))
    M = Hq - ell(4, KAPPA)
    for A in [1,4,10,50]:
        print(f"{A:>9} " + "".join(f"{np.sqrt(sigma_needed(T,M,agg=A)):>12.2f}" for T in [10,25,50,100]))

    print("\n=== 2. DUAL GAIN: budget to hold every client at K_target ===")
    print("Setting A-like hierarchy: group sizes [1,2,3] -> siblings [0,1,1,2,2,2]")
    groups = [1,2,2,3,3,3]
    ells_true = np.array([ell(g-1, KAPPA) for g in groups])
    ells_lin  = np.array([float(g) for g in groups])          # the paper's linear proxy
    print(f"  true ell   = {np.round(ells_true,4)}")
    print(f"  linear proxy (paper) = {ells_lin}   <- unnormalised, and wrong SHAPE")
    print()
    print(f"{'K_target':>9} {'U_fulcrum':>11} {'U_uniform':>11} {'budget saving':>14}")
    for Kt in [Hq*0.9, Hq*0.75, Hq*0.6, ells_true.max()*1.5, ells_true.max()*1.2]:
        uf,uu,r = dual_gain(ells_true, Kt)
        print(f"{Kt:>9.3f} {uf:>11.3f} {uu:>11.3f} {(1-uf/uu)*100:>13.1f}%")
    print("  (U in units of 2T; saving = fraction of total noise budget saved at equal privacy)")
