# Stage 1 — Research Framing

## Status: Closed

## Headline claim

> Even under DP-SGD with deployment-grade noise, an adversary with topology and
> organizational knowledge can recover per-client sensitive-class concentration
> $\hat{p}_i$ with mean squared error meaningfully below the non-topology
> baseline. We characterize the regime where this attack succeeds, propose
> topology-aware DP noise allocation paired with observation-window bounding as
> a defense, prove a mutual-information bound on what any adversary in this
> threat model can recover under our defense, and measure the privacy–utility
> frontier across three federated benchmarks.

## What this paper contributes

1. **Empirical characterization of topology-conditional leakage** under
   DP-SGD: how much marginal information the adversary gains from topology and
   organizational labels beyond observing parameter sequences.
2. **A defense with theoretical guarantee** — topology-aware noise allocation
   and observation-window bounding, with a Medium-tier MI bound on adversary
   recovery, derived from standard RDP composition and KL–MI conversion.
3. **A privacy–utility characterization** sweeping noise allocation strength
   and observation window length, on real and synthetic FL benchmarks.

## What this paper does NOT claim

- That topology is a privacy attack surface in isolation; we make no claim
  about deployments where data placement is independent of topology.
- That the attack works against arbitrary defenses; the threat model and
  defense are co-designed (see Stage 2 and Stage 4).
- That the attack defeats secure aggregation in its standard form; we treat
  secure aggregation as an extension experiment with a separate threat model.

## Why this is a TOPS-level contribution

- A novel privacy attack alone is not enough — the original paper attempted
  this with the proxy metrics critique (see `decisions_log.md`).
- A novel defense without a guarantee is not enough — TOPS reviewers expect
  theoretical bounds for defenses.
- The combination — characterized empirical attack, defense with proven MI
  bound, measured privacy–utility frontier — is the publishable structure.

## Out-of-scope (acknowledged)

- Active adversaries (message modification, collusion, malicious aggregator).
- Architecture-blind attacks; we assume the model architecture is public to
  the adversary (consistent with FLamby and most open FL benchmarks).
- Star-only deployments; if topology has no non-trivial structure, our attack
  reduces to known content-only leakage and adds nothing.

## What replaces what in the original manuscript

| Original | Replacement | Rationale |
|---|---|---|
| Three "complementary attack vectors" with proxy metrics (cluster-coherence, silhouette, max&#124;ρ&#124;) | One attack TADI with channel ablations and ground-truth-grounded MSE / AUROC | Original metrics did not measure inference of $\Delta_i$ |
| SG/TC/IS synthetic partitionings | Native Fed-ISIC2019 + Fed-Heart-Disease partitions; CIFAR-10 + Dirichlet for ablation | Synthetic partitionings injected the correlations the attacks then "discovered" |
| $\mathcal{K}_{\text{complete}}$ adversary in §3, structural noise defense in §6 | One adversary throughout; defense compatible with adversary knowing $\mathcal{G}$ | Original defense contradicted the threat model |
| "Defense effectiveness" reported as % attack reduction only | Privacy–utility curve: bound + measured model accuracy / convergence cost | Original paper had no utility metric |
