import numpy as np
def delta_of(w,groups):
    w=np.asarray(w,float); w=w/w.sum(); groups=np.asarray(groups)
    regs=sorted(set(groups.tolist()))
    W=np.array([(w[groups==r]**2).max() for r in regs]); V=np.array([(w[groups==r]**2).sum() for r in regs])
    rho=W/V; return 100*(1-np.sum(rho*V)/V.sum()/rho.max())
print(f"{'hierarchical structure':<54}{'regions':>8}{'silos':>7}{'delta':>8}")
print("-"*77)
for s in range(3):
    r=np.random.default_rng(s)
    sizes=np.maximum(1,np.round(r.lognormal(1.2,1.3,60)).astype(int))
    gr=np.array(sum([[j]*int(x) for j,x in enumerate(sizes)],[]))
    lbl="Cellular MEC, 60 base stations (long-tailed devices/BS)" if s==0 else f"   (seed {s})"
    print(f"{lbl:<54}{60:>8}{len(gr):>7}{delta_of(np.ones(len(gr)),gr):>7.1f}%")
print(f"{'Fed-ISIC2019, natural hospital regions [3,1,1,1]':<54}{4:>8}{6:>7}"
      f"{delta_of([9930,3163,2691,1807,655,351],[0,0,0,1,2,3]):>7.1f}%")
sites=[30,10,5,3,1,1]; gr=np.array(sum([[j]*x for j,x in enumerate(sites)],[]))
print(f"{'Multinational consortium, sites/country 30/10/5/3/1/1':<54}{6:>8}{sum(sites):>7}"
      f"{delta_of(np.ones(sum(sites)),gr):>7.1f}%")
print(f"{'[scope boundary] flat cross-silo, no aggregation tier':<54}{6:>8}{6:>7}"
      f"{delta_of([9930,3163,2691,1807,655,351],list(range(6))):>7.1f}%")
