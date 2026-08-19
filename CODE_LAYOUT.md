# Fulcrum — Implementation (v2)

Optimal per-region DP noise allocation for **hierarchical** federated learning.

A silo's exposure is set by the aggregation region it hides in. Regions differ in effective
size, so uniform noise over-provisions large regions and under-provisions small ones. Fulcrum
allocates per-region noise to the exposure profile and attains the same per-client privacy at
strictly lower total noise. The saving is a closed form:

    delta = 1 - <rho>_V / rho_max ,    rho_r = (max_{i in r} w_i^2) / (sum_{i in r} w_i^2)

`rho_r` is region `r`'s inverse participation ratio (its *effective* size). `delta` is an
identity in the deployment's region-size profile — computable **before training** — and is zero
exactly when regions are equally sized.

## Layout

```
fulcrum/
├── lateral_mi.py    ell_i = I(p_i; D_-i): lateral leakage floor under a hierarchical Beta
│                    prior. Exact up to quadrature; self-checks for monotonicity and the
│                    saturation ceiling I(p_i; Phi).
├── nonvacuity.py    Non-vacuity regime (bound vs H(p_i)) and the dual budget form.
├── settings.py      Corrected allocation on concrete deployment profiles.
├── real_delta.py    delta for real deployment structures (MEC, consortia, FLamby).
├── sweep.py         Sensitivity of the gain to coupling and group structure.
├── fedsim.py        Silo-level DP federated trainer + (eps, delta) accounting.
├── evaluate.py      rho/delta stats, the three allocation modes, region profiles.
├── probe.py         CIFAR-10 experiment: frozen ResNet18 -> PCA -> linear head.
└── agnews.py        AG News experiment: frozen MiniLM -> PCA -> linear head.

analysis/v2/         Result JSONs for the confirmed runs.
analysis/*.parquet   Archived run records retained as the evidence base for documented
                     epsilon and batch-size corrections.
```

## The three allocation modes (`evaluate.sigmas_for_target`)

All three hold **every** client at the same worst-case per-client bound `K`, so the comparison
is at matched privacy; they differ only in total noise, hence in accuracy.

- `fulcrum` — per-region `S_r = a*W_r/(K - ell_r)`.
- `uniform` — a single sigma for all clients, sized for the worst region (standard practice).
- `random`  — misallocation control: same dispersion, permuted across regions, rescaled to
  still meet `K`. Rules out "any asymmetry helps".

## Reproducing the headline result

```bash
pip install -e .
python fulcrum/probe.py  --K 0.88 --n 96 --T 10 --fdim 32 --seeds 3 --out probe_K0.88.json
python fulcrum/agnews.py --K 0.88 --n 96 --T 10 --fdim 32 --seeds 3 --out agnews_K0.88.json
```

Both at eps = 0.99. Expected: exactly 0.000 pp gain at
the two `delta = 0` null controls, +4 pp at `delta = 0.375`, +13.5 pp at `delta = 0.792`, and a
*negative* gain for the `random` control in every non-null cell.
