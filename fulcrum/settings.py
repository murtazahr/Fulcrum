"""Corrected allocation on the paper's three real settings.

Two corrections change the allocation materially:

(1) UTILITY BUDGET.  The noise actually injected into the global model under
    weighted FedAvg is sum_i w_i^2 sigma_i^2, NOT sum_i sigma_i^2. Using the
    latter (v1) both mis-states utility cost and, under aggregated observation,
    makes the program degenerate (all noise concentrates on the top-weight silo).
    Correct budget:      U := sum_i w_i^2 sigma_i^2
    Budget equation:     sum_i w_i^2 * a/(K - ell_i) = U      (a := 2T)
    Reduces to v1's when w_i = 1/n.

(2) WHERE DATASET SIZE ENTERS.  v1 used |D_i| as a LEVERAGE proxy (ell_i), i.e.
    claimed big silos are more predictable from others. That is not what dataset
    size does. It sets the aggregation weight w_i, which enters the MECHANISM term
    m_i = 2 w_i^2 / sum_j w_j^2 sigma_j^2. So big silos need more noise because they
    are more VISIBLE in the aggregate, not because they are more PREDICTABLE.
    Same qualitative conclusion as v1, rigorous derivation, different exponent.
"""
import numpy as np
from lateral_mi import ell

SETTINGS = {
    "A (Fed-ISIC2019)":  dict(sizes=[9930,3163,2691,1807,655,351], groups=[0,0,1,1,2,2]),
    "B (Fed-Heart)":     dict(sizes=[199,172,30,85],                groups=[0,1,2,3]),
    "C (synthetic)":     dict(sizes=[1000]*50, groups=sum([[g]*s for g,s in
                              enumerate([20,15,10,3,2])],[])),
}

def alloc(ell_v, w, a, U):
    """sigma_i^2 = a/(K-ell_i) with sum_i w_i^2 sigma_i^2 = U."""
    ell_v, w = np.asarray(ell_v,float), np.asarray(w,float)
    lo, hi = ell_v.max()+1e-12, ell_v.max()+a*np.sum(w**2)/U*len(w)+1e3
    for _ in range(400):
        mid=.5*(lo+hi)
        if np.sum(w**2*a/(mid-ell_v))>U: lo=mid
        else: hi=mid
    K=.5*(lo+hi); return a/(K-ell_v), K

def uniform_K(ell_v, w, a, U):
    """Uniform sigma^2 = U/sum(w^2); worst-case bound set by the worst client."""
    s2 = U/np.sum(np.asarray(w,float)**2)
    return np.max(a/s2 + np.asarray(ell_v,float)), s2

if __name__ == "__main__":
    KAP, T = 20.0, 25
    a = 2.0*T
    print(f"kappa={KAP}, T={T} (a=2T={a}), utility budget U := sum_i w_i^2 sigma_i^2\n")
    for name,cfg in SETTINGS.items():
        sz=np.array(cfg["sizes"],float); gr=np.array(cfg["groups"])
        w=sz/sz.sum()
        # siblings = other members of the same organisational group
        sib=np.array([np.sum(gr==g)-1 for g in gr])
        ell_true=np.array([ell(int(m),KAP) for m in sib])
        ell_v1_ds=sz/sz.mean()                       # v1 dataset-size proxy
        ell_v1_gs=np.array([np.sum(gr==g) for g in gr],float)   # v1 group-size proxy
        print(f"--- {name}   n={len(sz)}  weights w_i in [{w.min():.4f},{w.max():.4f}]")
        print(f"    siblings m_i      : {sib}")
        print(f"    ell_true (nats)   : {np.round(ell_true,3)}   spread {ell_true.max()-ell_true.min():.3f}")
        print(f"    v1 dataset proxy  : {np.round(ell_v1_ds,2)}   spread {ell_v1_ds.max()-ell_v1_ds.min():.2f}")
        print(f"    v1 group proxy    : {np.round(ell_v1_gs,2)}")
        for U in [0.05, 0.2]:
            s_f,Kf = alloc(ell_true,w,a,U)
            Ku,s_u = uniform_K(ell_true,w,a,U)
            # dual: budget needed to hold every client at K_target
            Kt = Kf*1.0
            Uu_needed = np.sum(w**2)*a/(Kt-ell_true.max())
            print(f"    U={U:<5g}: K_fulcrum={Kf:8.2f}  K_uniform={Ku:8.2f}   "
                  f"budget saving at equal privacy = {(1-U/Uu_needed)*100:5.1f}%")
        print()
