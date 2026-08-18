import numpy as np, functools
from lateral_mi import ell as _ell, saturation
from nonvacuity import H_quantized
ell=functools.lru_cache(maxsize=None)(lambda m,k: _ell(m,k,n_mc=6000))
def saving(ells,K):
    ells=np.asarray(ells,float)
    if K<=ells.max()+1e-9: return np.nan
    n=len(ells); w=np.ones(n)/n
    return (1-np.sum(w**2/(K-ells))/(np.sum(w**2)/(K-ells.max())))*100
STRUCT={"balanced 3x2 (Setting A)":[2]*6,"ring/no groups (Setting B)":[1]*4,
 "hier 20/15/10/3/2 (Setting C)":[20]*20+[15]*15+[10]*10+[3]*3+[2]*2,
 "extreme 1 + 49":[1]+[50]*49,"geometric 1,2,4,8,16,32":sum([[s]*s for s in [1,2,4,8,16,32]],[])}
for kap in [5,20,100]:
    H=H_quantized(kap,n=120000); sat=saturation(kap,n_mc=60000)
    print(f"\n=== kappa={kap}  H(p_i)={H:.3f}  ell ceiling={sat:.3f} ===")
    print(f"{'structure':<31}{'spread':>8}" + "".join(f"{'K='+f'{f:g}'+'H':>9}" for f in [0.95,0.75,0.5]))
    for nm,gs in STRUCT.items():
        e=np.array([ell(g-1,kap) for g in gs])
        print(f"{nm:<31}{e.max()-e.min():>8.3f}" + "".join(
            (f"{saving(e,f*H):>9.1f}" if not np.isnan(saving(e,f*H)) else f"{'--':>9}") for f in [0.95,0.75,0.5]))
print("\n% of noise budget saved at equal per-client privacy; '--' = target below ell_max (infeasible)")
