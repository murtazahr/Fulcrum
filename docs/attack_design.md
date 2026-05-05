# Stage 3 — Attack Design

## Status: Closed

## 3.1 Single attack architecture: TADI

**Topology-Aware Distributional Inference (TADI):**
$$f_\phi : (\Theta_i, x_i) \mapsto \hat{p}_i \in [0, 1]$$
where:
- $\Theta_i = (\theta_i^{(1)}, \ldots, \theta_i^{(T)})$ is client $i$'s observable parameter sequence
- $x_i \in \mathbb{R}^k$ is a structural feature vector built from $(\mathcal{G}, \omega)$:
  - degree $d_i = |\{j : (i, j) \in E\}|$
  - hierarchical position (depth from root) for hierarchical topologies
  - shortest-path distance to nearest aggregator
  - one-hot encoding of $\omega_i$ (organizational label)
  - betweenness centrality (optional, computed from $\mathcal{G}$)
- $f_\phi$ is a learned regressor (gradient-boosted trees baseline; MLP and
  linear regression as robustness checks) with parameters $\phi$ trained offline
  via the shadow-model framework.

This replaces the original paper's three attacks (cluster-coherence, silhouette,
correlation) with one attack that produces a ground-truth-comparable estimate.

## 3.2 Parameter feature extraction

From $\Theta_i$, extract a fixed-dimensional feature vector combining
per-round and temporal-aggregate signals:

- $\{\|\theta_i^{(t)}\|_2\}_{t=1}^T$ — raw norms (per-round)
- $\bar{n}_i = \frac{1}{T}\sum_t \|\theta_i^{(t)}\|_2$ — mean norm
- $\sigma_{n,i} = \text{std}(\{\|\theta_i^{(t)}\|_2\})$ — temporal std
- $\beta_i = \text{LinearReg slope}(\{\|\theta_i^{(t)}\|_2\})$ — drift trend
- Last-round norm $\|\theta_i^{(T)}\|_2$
- Layer-wise mean norms (one feature per layer)
- Pairwise cosine similarity to neighbors $\{\langle \theta_i^{(t)}, \theta_j^{(t)} \rangle / (\|\theta_i^{(t)}\| \|\theta_j^{(t)}\|) : (i,j) \in E\}$ aggregated to mean/max

Regressor selects which features are predictive; we do not pre-commit.

## 3.3 Channel ablations (replace "three attacks")

| Ablation | Inputs to $f_\phi$ | Purpose |
|---|---|---|
| $\mathcal{A}_1$ — Parameter-only | $\Theta_i$ only | Topology-blind baseline (replicates Melis-style leakage) |
| $\mathcal{A}_2^{\text{topo}}$ — Topology-only | $x_i^{\text{topo}}$ (structural features only, no $\omega$) | Pure topology signal |
| $\mathcal{A}_2^{\text{org}}$ — Org-only | $\omega_i$ only | Pure organizational signal |
| $\mathcal{A}_2^{\text{full}}$ — Combined | $\Theta_i + x_i$ | Full threat |

**Topology contribution** is decomposed as:
- $L_{\text{cal}}(\mathcal{A}_1) - L_{\text{cal}}(\mathcal{A}_2^{\text{full}})$ — total contribution
- $L_{\text{cal}}(\mathcal{A}_2^{\text{full}}) - L_{\text{cal}}(\mathcal{A}_2^{\text{topo}})$ — marginal value of label
- $L_{\text{cal}}(\mathcal{A}_2^{\text{full}}) - L_{\text{cal}}(\mathcal{A}_2^{\text{org}})$ — marginal value of structure

## 3.4 Shadow-model training protocol

Following Shokri et al. (S&P 2017), the adversary trains $f_\phi$ offline on
simulated FL runs where ground-truth $p_i$ is known.

**Procedure:**
1. Adversary uses a public proxy dataset (CIFAR-10 for vision target settings;
   Fed-ISIC2019 itself for Setting A — public dataset with synthetic re-partitioning;
   Fed-Heart-Disease for Setting B).
2. Run many simulated FL trainings, sweeping topology, $\alpha$, $\eta$, DP level.
3. For each simulated client, record $(\Theta_i, x_i, p_i)$.
4. Train $f_\phi$ on the resulting (input, $p_i$) pairs with a held-out
   validation split.
5. At test time apply $f_\phi$ to the real federation; no ground truth needed
   at test time.

**Domain mismatch acknowledgement.** For Setting C, shadow and target are the
same dataset, no mismatch. For Settings A and B, shadow training is performed on
public re-partitionings of the target dataset — same dataset family, different
partitioning. We document this as a methodological commitment.

**Architecture assumption.** Adversary knows the model architecture. This is
realistic for FLamby benchmarks and most cross-silo healthcare consortia, where
the architecture is shared with participants. Architecture-blind attack is
out of scope.

## 3.5 Comparator baselines

| Baseline | Description | Purpose |
|---|---|---|
| Constant-mean | $\hat{p}_i = \bar{p} = \frac{1}{n}\sum_j p_j$ | Defines attack lift; should be optimal under IID (sanity) |
| Random | $\hat{p}_i \sim \text{Unif}(0, 1)$ | Lower bound on adversary capability |
| Gradient inversion | Geiping et al. (NeurIPS 2020) attack run at our DP levels | Demonstrates complementarity (per-sample reconstruction vs. aggregate inference) |

**Attack lift** $= L_{\text{cal}}(\text{constant-mean}) - L_{\text{cal}}(f_\phi)$.
Positive lift indicates client-level information beyond the federation mean.
Reported as primary headline number.

## 3.6 Secure-aggregation extension (separate threat model)

Define $\mathcal{A}_2^{\text{SA}}$: adversary observes only neighborhood-aggregated
updates $\Theta_i^{\text{agg}} = \sum_{j \in N(i)} \theta_j^{(t)}$, plus
$(\mathcal{G}, \omega)$. Run TADI with $\Theta_i^{\text{agg}}$ replacing $\Theta_i$.

This is **one extension experiment**, not a third primary attack. It addresses
the natural reviewer question "what about secure aggregation?" by quantifying
remaining leakage.

## 3.7 Robustness experiments

- Regressor architecture: gradient-boosted trees (primary), MLP, linear regression
- Feature set: raw-only, aggregate-only, full
- Shadow data quantity: 100, 500, 1000, 5000 simulated FL runs
- Random seeds: $\geq 5$ per configuration

## 3.8 What this attack contributes

The regressor itself is standard supervised learning. The **contribution** is the
empirical characterization:
- How much marginal information topology and organizational labels add beyond
  parameter observation under DP-SGD.
- How this scales with topology type, network size, and DP noise level.
- Identifying when structural information dominates organizational labels and
  vice versa (the channel ablations).

This framing matches how Melis et al. (S&P 2019) was accepted: standard ML
machinery, novel characterization. Combined with the Stage 4 defense, it's a
TOPS-level paper.

## 3.9 Open commitments carried forward

- **Title** — "Network Structures as an Attack Surface" overclaims given what
  TADI actually proves. Defer retitle to Stage 8. Working title:
  "Topology-Conditional Distribution Inference in Differentially-Private
  Federated Learning."
- **Coupling parameter $\eta$** — formal definition deferred to Stage 5;
  must match what TADI's structural features exploit.
