# Fulcrum

**Optimal per-region differential-privacy noise allocation for hierarchical federated learning.**

In hierarchical FL, clients report to regional/edge aggregators before the cloud. A silo's
privacy exposure is therefore set not by its own noise alone, but by the **aggregation region it
hides in**. Regions differ in effective size, so a single uniform noise multiplier
over-provisions large regions and under-provisions small ones.

Fulcrum allocates per-region noise to the exposure profile. It attains the **same per-client
privacy at strictly lower total noise**, and the saving has a closed form:

```
delta = 1 - <rho>_V / rho_max        rho_r = (max_{i in r} w_i^2) / (sum_{i in r} w_i^2)
```

`rho_r` is region `r`'s inverse participation ratio — its *effective* size. Two properties make
this practical:

- **`delta` is computable before training.** It depends only on the region-size profile, not on
  the data or the model. A deployer can decide in advance whether this is worth doing.
- **`delta = 0` exactly when regions are equally sized.** The method provably does nothing in
  that case, which is an explicit applicability test rather than a hidden failure mode.

## Result

At **eps = 0.99**, n = 96 silos, T = 10 rounds, 3 seeds, federated fine-tuning of a frozen
backbone. Gain is Fulcrum minus uniform at **matched per-client privacy**.

| region profile | delta | budget saved | CIFAR-10 | AG News |
|---|---|---|---|---|
| balanced 4 (null control) | 0.000 | 0.0% | **+0.00 pp** | **+0.00 pp** |
| balanced 6 (null control) | 0.000 | 0.0% | **+0.00 pp** | **+0.00 pp** |
| mild 6/6/6/3/3 | 0.375 | 37.5% | +4.33 pp | +4.02 pp |
| severe 15/5/2/1/1 | 0.792 | 79.2% | **+13.75 pp** | **+13.55 pp** |

A **misallocation control** — the same noise dispersion, permuted across regions — is *worse
than uniform* (−2 to −3 pp) in every non-null cell, ruling out "any asymmetry helps".

Real hierarchical deployments sit high on the `delta` axis: cellular MEC with long-tailed
devices-per-base-station gives 82.8–87.6%, a multinational consortium by sites-per-country 88.0%.
Flat cross-silo FL with no aggregation tier gives `delta = 0` — that is the scope boundary.

## Scope

- **Hierarchical FL** (clients → regional aggregators → cloud). Flat topologies gain nothing.
- **Silo-level DP**: each silo clips its own update to `||Delta|| <= C` and adds
  `N(0, sigma_i^2 C^2)` before transmitting. No trust in the aggregator is required.
  Under a *trusted* aggregator that injects noise once, per-client sigma scales as `1/m` rather
  than `1/sqrt(m)` for a region of `m` silos; the `sqrt(m)` gap is the price of not trusting it.
- **Federated fine-tuning**, not training from scratch. Aggregate SNR goes as
  `sqrt(a*n/d)/sigma`, so the parameter count `d` is the binding constraint: a frozen backbone
  with a small head trains at privacy levels where an end-to-end CNN sits at chance.

## Quick start

```bash
pip install -e .
python fulcrum/probe.py  --K 0.88 --n 96 --T 10 --fdim 32 --seeds 3 --out probe_K0.88.json
python fulcrum/agnews.py --K 0.88 --n 96 --T 10 --fdim 32 --seeds 3 --out agnews_K0.88.json
```

See [CODE_LAYOUT.md](CODE_LAYOUT.md) for module structure and
[manuscript/](manuscript/) for the theory, revision plan, and positioning.

## Relation to v1

This supersedes the v1 preprint, *Topology-Aware Differential Privacy in Federated Learning*
(arXiv:2506.19260), which is separately archived under its own DOI. The v1 pipeline has been
removed. Two v1 claims do **not** carry over: that communication topology acts as an information
channel, and that per-client noise heterogeneity is itself novel. v1 run records are retained in
`analysis/v1/` as the evidence base for the corrections documented in
[manuscript/REVISION_PLAN.md](manuscript/REVISION_PLAN.md).
