# Stage 2 — Threat Model, Inference Target, and Partitioning

## Status: Closed

## 2.1 System model

Federated learning system with $n$ clients $\mathcal{P} = \{P_1, \ldots, P_n\}$
connected via topology $\mathcal{G} = (\mathcal{P}, E)$. Each client $P_i$ holds
a local dataset $\mathcal{D}_i$ inducing class distribution
$\Delta_i \in \Delta(\mathcal{C})$ over class set $\mathcal{C}$. Clients are
partitioned into organizations by labelling $\omega : \mathcal{P} \to [k_{\text{org}}]$.
Training proceeds for $T$ rounds. At round $t$, client $P_i$ transmits parameter
update $\theta_i^{(t)}$. Local DP-SGD is applied at each client with per-round
budget $\varepsilon$ tracked via Rényi Differential Privacy.

## 2.2 Adversary $\mathcal{A} = (\mathcal{K}, \mathcal{O}, \mathcal{I}, \mathcal{R})$

**Knowledge $\mathcal{K}$.** The adversary knows: (i) the full topology
$\mathcal{G}$; (ii) the organizational labelling $\omega$; (iii) public protocol
parameters (DP budget, model architecture, training schedule). The adversary
does **not** know any $\Delta_i$ or $\mathcal{D}_i$.

**Observation $\mathcal{O}$.** The adversary observes the sequence of
transmitted parameter updates $\Theta = \{\theta_i^{(t)} : i \in [n], t \in [T]\}$
and communication metadata. This models a passive infrastructure observer or
curious aggregator.

**Inference goal $\mathcal{I}$.** Recover, for each client $P_i$, the sensitive-class
concentration $p_i = \Delta_i(\mathcal{C}_s)$ where $\mathcal{C}_s \subset \mathcal{C}$
is the designated sensitive class set.

**Restriction $\mathcal{R}$.** Honest-but-curious. Cannot modify messages, inject
traffic, collude with clients, or adaptively choose corruptions. The adversary's
view is fixed by the protocol.

**Why this adversary.** Strong enough to be interesting (knows $\mathcal{G}$,
$\omega$); weak enough to be realistic (no protocol modification, no compromise).
This adversary is consistent throughout the paper — defenses that hide
$\mathcal{G}$ are off the table.

## 2.3 Inference target and metric

**Per-dataset specification of $\mathcal{C}_s$:**

| Setting | $\mathcal{C}_s$ |
|---|---|
| A — Fed-ISIC2019 | $\{\text{melanoma}\}$ |
| B — Fed-Heart-Disease | $\{\text{heart disease positive}\}$ (binary) |
| C — CIFAR-10 | $\{\text{designated rare class}\}$ |

**Primary metric — calibration loss / MSE.**
$$L_{\text{cal}} = \frac{1}{n} \sum_{i=1}^n (\hat{p}_i - p_i)^2$$
Has statistical power even at small $N$. Used across all three settings.

**Secondary metric — top-$k$ recovery.** Of the $k$ clients with highest true
$p_i$, what fraction does the adversary correctly identify in the top $k$ of
$\hat{p}_i$? Bounded, interpretable.

**Tertiary metric — AUROC.** Used for Setting C only ($N \in [20, 500]$, where
AUROC has meaningful statistical power). For binary $y_i = \mathbb{1}[p_i > \tau]$.

## 2.4 Reference adversaries (for isolating topology contribution)

| Label | Knowledge | Purpose |
|---|---|---|
| $\mathcal{A}_0$ | None | Random-guess / constant-mean baseline |
| $\mathcal{A}_1$ | $\Theta$ only — no $\mathcal{G}$, no $\omega$ | Non-topology attacker |
| $\mathcal{A}_2$ | Full $\mathcal{K} = (\mathcal{G}, \omega)$ + $\Theta$ | Topology-aware attacker (the threat) |

**Topology contribution = $L_{\text{cal}}(\mathcal{A}_1) - L_{\text{cal}}(\mathcal{A}_2)$.**

The claim "topology is an attack surface" is only validated if this gap is
positive and statistically significant. Under IID partitioning, the gap should
collapse — this is the sanity check, baked into Setting C at $\eta = 0$.

## 2.5 Success criterion

Per topology × partitioning × DP setting, compute the null distribution of
$L_{\text{cal}}(\mathcal{A}_2)$ under IID partitioning (Setting C, $\eta = 0$)
across seeds. Attack success is $L_{\text{cal}}$ below the 5th percentile of the
null. This is standard hypothesis-testing framing; replaces the arbitrary
$\delta = 0.30$ threshold.

## 2.6 Three experimental settings

### Setting A — Hierarchical Healthcare FL (realism anchor)

| Attribute | Value |
|---|---|
| Dataset | Fed-ISIC2019 (FLamby) — 23,247 dermoscopy images, 6 hospital sites, 8 lesion classes |
| Topology | 2-level hierarchy: hospitals → regional aggregators → global aggregator. Aggregator assignment by geography. |
| Partitioning | **Native** site partitioning. No synthetic SG/TC/IS. |
| Sample size | 6 sites — too small for AUROC; use calibration loss + top-$k$ |
| Citations | Ogier du Terrail et al. (FLamby, NeurIPS 2022); Liu et al. (ICC 2020); Sheller et al. (Sci. Rep. 2020); Dayan et al. (Nat. Med. 2021) |
| Role | Realism check: "this attack succeeds on a real federated healthcare benchmark" |

### Setting B — Decentralized FL (forward-looking)

| Attribute | Value |
|---|---|
| Dataset | Fed-Heart-Disease (FLamby) — 740 records, 4 centers, binary heart disease classification |
| Topology | $k$-NN proximity graph (explicit hypothesis about future deployments) |
| Partitioning | **Native** site partitioning |
| Sample size | 4 sites |
| Citations | FLamby; Roy et al. (BrainTorrent, MICCAI 2019); Hegedűs et al. (JPDC 2021); Wang et al. (Field Guide, 2021) |
| Role | Forward-looking: "as decentralized FL matures, this attack will apply" |
| Limitation | Proximity-decentralized FL is research-stage in production; explicitly noted |

### Setting C — Controlled Synthetic (statistical vehicle)

| Attribute | Value |
|---|---|
| Dataset | CIFAR-10 |
| Heterogeneity | Dirichlet$(\alpha)$ partitioning (Hsu et al. 2019), $\alpha \in \{0.1, 0.5, 1.0, \infty\}$ |
| Topology coupling | Parametric $\eta \in [0, 1]$ where $\eta = 0$ is IID-equivalent and $\eta = 1$ is fully position-determined. **Formal definition deferred to Stage 5** so that the coupling matches what the attack actually exploits. |
| Sample size | $N \in \{20, 50, 100, 250, 500\}$ |
| Topologies | star, complete, ring, line, hierarchical |
| Citations | Hsu et al. (arXiv:1909.06335) |
| Role | Statistical vehicle for headline claims; declares synthetic upfront |

### Cross-setting validation

Setting C with $\eta, \alpha$ tuned to qualitatively match Setting A's hierarchical
behaviour at $N = 6$. Report consistency, not equivalence. Documents the transfer
claim from synthetic-scale to realistic deployment.

## 2.7 Paper structure implication

Settings A and B are **realism anchors** with limited statistical power.
Setting C carries the statistical claims. The manuscript states this plainly
in the experimental section — the alternative ("hide that C is synthetic") is
the original paper's failure mode and is rejected.
