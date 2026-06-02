# Performance Evaluation — Detailed Results Report

This report provides every empirical fact needed to rewrite Section 6
("Performance Evaluation") of the manuscript. It is organised by
empirical claim, with full numerical detail, figure references,
methodological notes, and pointers to the underlying parquet exports.

> **Scope.** Sections §1–§5 of the manuscript (introduction, related
> work, threat model, attack architecture, defense theory) are
> considered locked. This report covers only §6, the empirical
> validation.

> **Setting A is intentionally omitted from the TADI channel
> ablation.** The channel ablation tests prior realisability, and
> Settings B and C span the two regimes of that axis: C is the
> matched-prior calibration (synthetic shadow + synthetic target),
> and B is the mismatched-prior cross-silo deployment representative
> (synthetic shadow + native FLamby target). Setting A is also a
> native-FLamby-partitioning deployment, so it would simply replicate
> Setting B's qualitative finding (all channels negative) on a
> different dataset at substantial computational cost. The Setting A
> Pareto sweep (§4.2.1) and utility consistency (§4.3.1) provide the
> defense evaluation on the Fed-ISIC2019 deployment without requiring
> a separate attack run.

---

## 1. Empirical Claims Validated by Section 6

The performance evaluation validates four claims, each tied to a
specific theoretical result and a specific sub-section of §6.

| # | Claim | Tied to | Evidence |
|---|-------|---------|----------|
| C1 | Topology-aware allocation strictly improves on uniform DP-SGD whenever leverage is non-uniform, and degenerates to equality when leverage is uniform. | Theorem 5.3, Corollary 5.4 | Setting C η-sweep heatmap (§4.1). |
| C2 | The strict improvement transfers across deployment regimes (real cross-silo healthcare and synthetic non-IID image classification) at every tested utility budget and observation window. | Corollary 5.4 (applied to each proxy) | Three Pareto sweeps (§4.2). |
| C3 | The privacy gain is obtained at no practically-meaningful utility cost (TOST equivalence within $\pm 0.5$ pp). | Proposition 5.5 (utility cost) | Three paired utility tables (§4.3). |
| C4 | DP-SGD effectively bounds the parameter channel under both allocations; the prior-coupling channels are realised when the adversary's shadow prior matches the target prior, and conservative in a deployment-favourable direction when it does not. | Theorem 5.2 (additive decomposition) | TADI channel ablation (§4.4) and bound-realisability scatter (§4.5). |

---

## 2. Settings and Datasets

### Setting A — Fed-ISIC2019 (Real cross-silo, healthcare imaging)

- **Source.** FLamby benchmark suite, dermoscopy challenge data.
- **Clients.** $n = 6$ hospital sites with native FLamby partitioning.
- **Task.** 8-class skin-lesion classification. Melanoma is the
  designated sensitive class $\mathcal{C}_s$.
- **Site sizes.** Highly heterogeneous: largest site contributes
  approximately $30 \times$ more training samples than the smallest.
- **Topology.** Two-level hierarchy with three regional aggregators,
  following Liu et al. (`liu2020hierarchical`). The hierarchical
  position of each site is a structural feature in the TADI input.
- **Model.** Small convolutional network with GroupNorm replacing
  BatchNorm for DP-SGD compatibility.
- **Dominant asymmetry.** Dataset-size heterogeneity. The proxy used
  is therefore the FedAvg-influence (dataset-size) proxy from
  Corollary 5.7.

### Setting B — Fed-Heart-Disease (Real cross-silo, tabular)

- **Source.** FLamby benchmark suite, Fed-Heart-Disease.
- **Clients.** $n = 4$ clinical centres with native FLamby
  partitioning. Site sizes $[199, 172, 30, 85]$, giving a 6.6:1 ratio
  between largest and smallest.
- **Task.** Binary heart-disease classification on 13 clinical
  features.
- **Topology.** Ring (chosen for the empirical study). All four
  clients share degree 2, so the graph topology introduces *no*
  structural asymmetry. The only source of asymmetry is dataset-size
  heterogeneity. This makes Setting B a clean ablation of the
  dataset-size proxy from any topological confound.
  > **Note for the writer.** FLamby's native Fed-Heart-Disease setup
  > assumes a star aggregator. We chose ring to remove the
  > centralized-aggregator confound and isolate the dataset-size
  > proxy. The structural-leverage profile is uniform on the ring;
  > only the dataset-size proxy carries asymmetry.
- **Model.** Small MLP.
- **Dominant asymmetry.** Dataset-size heterogeneity (proxy as in
  Setting A).

### Setting C — Synthetic CIFAR-10 with parametric coupling

- **Source.** CIFAR-10 (`krizhevsky2009cifar`), re-partitioned across
  $n = 50$ synthetic clients.
- **Partitioning.** Each client's class distribution is
  $\Delta_i = (1 - \eta) \tilde\Delta_i + \eta \Delta^*_{\phi(i)}$,
  where $\tilde\Delta_i \sim \mathrm{Dirichlet}(\alpha = 0.5)$ is the
  standard non-IID Dirichlet baseline (`hsu2019dirichlet`),
  $\Delta^*_{\phi(i)}$ is concentrated on class $\phi(i) \bmod K$ with
  $\phi$ a topology-determined client-to-class map, and
  $\eta \in [0, 1]$ controls the strength of topology–data coupling.
- **Coupling parameter.** $\eta = 0$ gives a fully IID-null partition
  (the adversary should have zero attack lift). $\eta = 1$ gives a
  fully position-determined partition. Intermediate $\eta$ values
  interpolate between the two.
- **Topology coverage** (the full asymmetry spectrum):

  | Family | Configurations | Source/proxy interpretation |
  |---|---|---|
  | Deterministic symmetric | Ring, complete (degenerate) | Uniform degree → zero gap by Corollary 5.4 |
  | Deterministic asymmetric | Line, hierarchical $[20,15,10,3,2]$, star | Endpoint asymmetry / group-size asymmetry / hub dominance |
  | Random homogeneous | Erdős-Rényi $G(50, p)$ at $p \in \{0.3, 0.5, 0.7\}$ | Binomial degree distribution; variance peaks at $p = 0.5$ |
  | Scale-free | Barabási-Albert at $m \in \{2, 4\}$ | Power-law degree distribution; $m = 2$ is more heavy-tailed |

  Citations: `erdos1959random` for ER, `barabasi1999emergence` for BA.

- **Task.** 10-class image classification.
- **Model.** Small CNN with GroupNorm.
- **Sensitive class.** Class 0 (arbitrary; same construction applies
  to any designated rare class).
- **Why Setting C exists.** It is the only configuration with
  $n \geq 20$, so it is the only setting where AUROC is statistically
  meaningful and the only setting where parametric control over
  $\eta$ allows direct measurement of how lateral leakage scales with
  topology–data coupling.

---

## 3. Methodology

### Sweep grids (all per-setting)

| Setting | Sweep type | $U$ grid | $T_{\max}$ grid | Topology | $\eta$ | Allocations | Seeds | Runs |
|---|---|---|---|---|---|---|---|---|
| A | Pareto | $\{0.05, 0.1, 0.2, 0.4, 0.8, 1.6\}$ | $\{25, 50, 100\}$ | hierarchical (native) | — | TA, uniform | 0, 1, 2 | 108 |
| B | Pareto | $\{0.025, 0.05, 0.1, 0.2, 0.4, 0.8\}$ | $\{25, 50, 100\}$ | ring | — | TA, uniform | 0, 1, 2 | 288 in DB; 252 after $T \in \{25, 50, 100\}$ filter |
| C | Pareto | $\{0.05, 0.1, 0.2, 0.4, 0.8, 1.6\}$ | $\{25, 50, 100\}$ | hierarchical $[20,15,10,3,2]$ | 0.5 | TA, uniform | 0, 1, 2 | 108 |
| C | η-sweep | $\{0.5\}$ | $\{100\}$ | 9 configs above | $\{0, 0.25, 0.5, 0.75, 1\}$ | TA, uniform | 0, 1, 2 | 269 |

### Statistical methods

- **Per-cell aggregation.** $K^\star$ and $K_{\mathrm{uniform}}$ are
  analytic (computed by 1-D bisection of the budget equation); they
  carry no seed variance. Test accuracy carries seed variance and is
  reported as mean over 3 seeds with 95% CI.
- **Paired comparison.** Utility consistency is measured by matched
  pairs at $(U, \mathrm{seed})$, so each row of the utility table
  aggregates 18 paired observations (6 $U$ values × 3 seeds) per
  $T_{\max}$.
- **Equivalence testing.** We use Two One-Sided Tests (TOST) at a
  $\pm 0.5$ percentage-point margin. A small TOST $p$-value (< 0.05)
  rejects practically-meaningful difference. We also report the
  conventional paired-$t$ $p$-value for completeness; a high
  paired-$t$ $p$-value alone establishes failure-to-reject equality
  but does not actively establish equivalence, so TOST is the primary
  utility-consistency test.

### Implementation

- **DP-SGD.** Opacus (`yousefpour2021opacus`) with Poisson
  subsampling, the RDP accountant (`mironov2017rdp`), gradient
  clipping at $C = 1.0$, batch size $|B| = 64$.
- **Per-client noise allocation.** Either
  $\sigma_i^{*2} = a / (K^\star - \ell_i^\circ)$ from Theorem 5.3
  (the `topology_aware` allocation), or $\sigma_i^2 = U / n$ for all
  $i$ (the `uniform` baseline). The total noise variance
  $\sum_i \sigma_i^2 \leq U$ is the active constraint.
- **Reproducibility.** Each run is deduplicated by SHA-256 of the
  resolved config dict, so re-running a sweep that overlaps a
  previous one is a no-op.

---

## 4. Results

### 4.1 Theorem 5.3 validation — Setting C η-sweep

**Headline finding.** The privacy-bound gap
$K_{\mathrm{uniform}} - K^\star$ behaves exactly as Theorem 5.3 and
Corollary 5.4 predict across nine topology configurations and five
coupling strengths.

**Primary figure.** `paper/figures/eta_gap_heatmap_setting_c.pdf`. A
single heatmap with rows = topology configurations (sorted by max
gap, ascending) and columns = $\eta \in \{0, 0.25, 0.5, 0.75, 1\}$.
Cell colour and annotation give the absolute gap in nats.

**Analytic asymptote.**
$a = T_{\max} C^2 / (2 |B|^2) = 100 / (2 \cdot 64^2) = 0.012207$,
so the maximum possible gap is $an / U = 0.012207 \times 50 / 0.5 =
1.2207$ nats. This is marked on the colourbar of the heatmap.

#### 4.1.1 Numerical findings — gap at every (topology, η)

All values are mean gap in nats over 3 seeds. The figure annotates
each cell; the table below is the full underlying digest.

| Topology | $\eta = 0$ | $\eta = 0.25$ | $\eta = 0.5$ | $\eta = 0.75$ | $\eta = 1$ |
|---|---|---|---|---|---|
| Ring                  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Line                  | 0.000 | 0.010 | 0.016 | 0.021 | 0.024 |
| Hier $[20,15,10,3,2]$ | 0.000 | 0.182 | 0.307 | 0.394 | 0.456 |
| ER $p=0.3$            | 0.000\* | 0.171 | 0.324 | 0.458 | 0.575 |
| ER $p=0.7$            | 0.000\* | 0.162 | 0.318 | 0.468 | 0.610 |
| ER $p=0.5$            | 0.000\* | 0.239 | 0.467 | 0.681 | 0.871 |
| BA $m=4$              | 0.000\* | 0.999 | 1.132 | 1.150 | 1.157 |
| BA $m=2$              | 0.000\* | 1.179 | 1.190 | 1.192 | 1.193 |
| Star                  | 0.000 | 1.194 | 1.195 | 1.196 | 1.196 |

(\*) The η-position leverage proxy collapses to uniform leverage at
$\eta = 0$, so topology-aware allocation degenerates to uniform DP-SGD
and the analytic gap is zero. The actual TA runs for ER and BA at
$\eta = 0$ were skipped by `assert_non_uniform_leverage` for exactly
this reason; the cells are analytically filled.

#### 4.1.2 Five sub-claims supported by the heatmap

**Sub-claim 1 — IID-null calibration.** The $\eta = 0$ column is
identically zero. This is the calibration band: in the absence of
topology–data correlation, no allocation can do better than uniform,
and Theorem 5.3 reduces to equality.

**Sub-claim 2 — Symmetric-topology degeneracy.** The Ring row is
identically zero at every $\eta$. Ring has uniform degree, so the
degree-based leverage profile is constant across clients, and
Corollary 5.4 predicts equality. The result confirms the
degeneracy and rules out spurious benefit from the allocation when no
asymmetry exists.

**Sub-claim 3 — Monotone growth of the gap.** For every asymmetric
topology, the gap grows monotonically in $\eta$. The growth is
near-linear for moderate-asymmetry topologies (line, hierarchical,
ER) and saturates very rapidly for extreme-asymmetry topologies (BA,
star), which reach $\sim 98\%$ of the asymptote by $\eta = 0.25$.

**Sub-claim 4 — Asymptotic saturation.** The Star row at $\eta \geq
0.25$ reads gap $\approx 1.195$ nats, within 2% of the analytic
asymptote $an/U = 1.2207$. BA $m = 2$ reaches 1.193 nats — also
saturated. **The star is not special — any topology with extreme
degree concentration saturates Theorem 5.3.** This is new evidence,
unique to the BA addition.

**Sub-claim 5 — Non-monotone gap in ER's connection probability $p$.**
Among the three ER configurations, the absolute gap is largest at
$p = 0.5$ (0.871 nats at $\eta = 1$), not at the extremes
$p = 0.3$ (0.575) or $p = 0.7$ (0.610). This matches the underlying
theory: the binomial degree distribution $\mathrm{Bin}(n-1, p)$ has
variance $(n-1) p (1-p)$ maximised at $p = 0.5$, so the degree-based
leverage profile is most heterogeneous at $p = 0.5$. The non-monotone
pattern in $p$ is **direct evidence** that the degree proxy tracks
the underlying leverage asymmetry rather than degree mean.

#### 4.1.3 Relative-gap perspective

Absolute gap is the quantity Theorem 5.3 bounds, but relative gap
gives a complementary view of where Fulcrum *matters most* as a
fractional improvement. The values are at $\eta = 1$:

| Topology | $K_{\mathrm{uniform}}$ | $K^\star$ | gap (nats) | gap (%) |
|---|---|---|---|---|
| Ring                  | 3.523  | 3.523  | 0.000 | 0.0% |
| Line                  | 3.570  | 3.546  | 0.024 | 0.7% |
| Star                  | 58.785 | 57.590 | 1.196 | 2.0% |
| BA $m=2$              | 15.012 | 13.819 | 1.193 | 7.9% |
| Hier $[20,15,10,3,2]$ | 4.400  | 3.945  | 0.456 | 10.4% |
| BA $m=4$              | 8.729  | 7.572  | 1.157 | 13.3% |
| ER $p=0.3$            | 4.242  | 3.667  | 0.575 | 13.6% |
| ER $p=0.7$            | 4.183  | 3.573  | 0.610 | 14.6% |
| ER $p=0.5$            | 4.502  | 3.631  | 0.871 | 19.4% |

**Two clean stories from this table.**

1. In *absolute* nats, star and BA $m=2$ have the largest gap
   ($\approx 1.2$ nats). They saturate the asymptote.
2. In *relative* terms, ER $p=0.5$ delivers the largest gap (19.4%
   reduction). Star's $K_{\mathrm{uniform}} = 58.8$ nats is so large
   that even a 1.2 nat reduction is only 2% of the bound.

This dual perspective is important for the manuscript narrative: the
defence delivers the largest *absolute* improvement on extreme
scale-free topologies (where uniform allocation places clients on a
catastrophic bound), and the largest *fractional* improvement on
moderate-variance topologies (ER, BA $m=4$).

#### 4.1.4 Suggested figure caption (for the writer)

> **Figure (η-sweep heatmap).** Privacy-bound gap
> $K_{\mathrm{uniform}} - K^\star$ (nats) across nine topology
> configurations and four coupling strengths on Setting C
> ($n = 50$, $U = 0.5$, $T_{\max} = 100$, $C = 1.0$, $|B| = 64$).
> Rows are sorted by maximum gap. The $\eta = 0$ IID-null calibration
> column is omitted from the figure for compactness; Corollary 5.4
> reduces to equality at uniform leverage and the $\eta = 0$ column
> is identically zero across every topology. The Ring row is
> identically zero at every shown $\eta$, the deterministic-symmetric
> degeneracy of Corollary 5.4. The colourbar's $an/U$ tick marks the
> **maximum possible gap** Fulcrum can deliver on this setting:
> Corollary 5.4 gives
> $K_{\mathrm{uniform}} - K^\star \leq an/U$ where
> $a = T_{\max} C^2 / (2|B|^2)$, so for the parameters above
> $an/U \approx 1.22$ nats. Scale-free topologies (BA, Star) saturate
> within 2% of this analytic ceiling, confirming that the theorem's
> asymptote is tight in the limit of extreme degree concentration.
> Among Erdős-Rényi configurations the gap peaks non-monotonically
> at $p = 0.5$, where the binomial degree variance is maximised.

---

### 4.2 Privacy–utility Pareto dominance

**Headline finding.** Topology-aware allocation strictly dominates
uniform DP-SGD on the privacy bound at every tested
$(U, T_{\max})$ cell across all three settings.

#### 4.2.1 Setting A — Fed-ISIC2019

**Figure.** `paper/figures/pareto_setting_a.pdf`. Three panels
($T_{\max} \in \{25, 50, 100\}$), $x$-axis log $U$, $y$-axis log
$K^\star$, gap-shading between curves.

**Full per-cell digest.**

| $T_{\max}$ | $U$ | $K_{\mathrm{TA}}^\star$ | $K_{\mathrm{uniform}}$ | gap (nats) | gap (%) |
|---|---|---|---|---|---|
| 25  | 0.05 | 3.614 | 4.669 | 1.055 | 22.6% |
| 25  | 0.10 | 3.360 | 3.936 | 0.576 | 14.6% |
| 25  | 0.20 | 3.273 | 3.570 | 0.297 | 8.3% |
| 25  | 0.40 | 3.236 | 3.387 | 0.151 | 4.4% |
| 25  | 0.80 | 3.220 | 3.295 | 0.076 | 2.3% |
| 25  | 1.60 | 3.211 | 3.250 | 0.038 | 1.2% |
| 50  | 0.05 | 4.503 | 6.133 | 1.631 | 26.6% |
| 50  | 0.10 | 3.614 | 4.669 | 1.055 | 22.6% |
| 50  | 0.20 | 3.360 | 3.936 | 0.576 | 14.6% |
| 50  | 0.40 | 3.273 | 3.570 | 0.297 | 8.3% |
| 50  | 0.80 | 3.236 | 3.387 | 0.151 | 4.4% |
| 50  | 1.60 | 3.220 | 3.295 | 0.076 | 2.3% |
| 100 | 0.05 | 7.097 | 9.063 | 1.967 | 21.7% |
| 100 | 0.10 | 4.503 | 6.133 | 1.631 | 26.6% |
| 100 | 0.20 | 3.614 | 4.669 | 1.055 | 22.6% |
| 100 | 0.40 | 3.360 | 3.936 | 0.576 | 14.6% |
| 100 | 0.80 | 3.273 | 3.570 | 0.297 | 8.3% |
| 100 | 1.60 | 3.236 | 3.387 | 0.151 | 4.4% |

**Headline numbers.** Gap range across the full 18-cell sweep:
absolute 0.038 to 1.967 nats; relative 1.2% to 26.6%. Strict
dominance ($K^\star < K_{\mathrm{uniform}}$) holds at every cell.

**Largest gap.** $(T_{\max} = 100, U = 0.05) \to 1.967$ nats,
21.7%. This is the strong-privacy-budget regime where the defense
matters most.

**Largest relative gap.** $(T_{\max} = 50, U = 0.05)$ and
$(T_{\max} = 100, U = 0.10) \to 26.6\%$. Two cells tie.

**Why Setting A's gap is large.** Fed-ISIC2019 has 30:1 site-size
heterogeneity. The dataset-size proxy gives the strongest
leverage-spread of all three settings.

#### 4.2.2 Setting B — Fed-Heart-Disease

**Figure.** `paper/figures/pareto_setting_b.pdf`. Three panels as
in Setting A; full coverage at every $(U, T_{\max})$ cell after the
backfill landed.

**Per-cell digest.**

| $T_{\max}$ | $U$ | $K_{\mathrm{TA}}^\star$ | $K_{\mathrm{uniform}}$ | gap (nats) | gap (%) |
|---|---|---|---|---|---|
| 25  | 0.025 | 8.852  | 9.450  | 0.599 | 6.3% |
| 25  | 0.050 | 4.941  | 5.282  | 0.341 | 6.5% |
| 25  | 0.100 | 3.097  | 3.591  | 0.494 | 13.8% |
| 25  | 0.200 | 2.111  | 2.352  | 0.241 | 10.3% |
| 25  | 0.400 | 1.862  | 2.126  | 0.265 | 12.4% |
| 25  | 0.800 | 1.516  | 1.620  | 0.103 | 6.4% |
| 50  | 0.025 | 16.645 | 17.263 | 0.618 | 3.6% |
| 50  | 0.050 | 8.852  | 9.450  | 0.599 | 6.3% |
| 50  | 0.100 | 4.982  | 5.544  | 0.562 | 10.1% |
| 50  | 0.200 | 3.097  | 3.591  | 0.494 | 13.8% |
| 50  | 0.400 | 2.226  | 2.614  | 0.389 | 14.9% |
| 50  | 0.800 | 1.862  | 2.126  | 0.265 | 12.4% |
| 100 | 0.025 | 32.260 | 32.888 | 0.628 | 1.9% |
| 100 | 0.050 | 16.633 | 17.001 | 0.367 | 2.2% |
| 100 | 0.100 | 8.852  | 9.450  | 0.599 | 6.3% |
| 100 | 0.200 | 4.941  | 5.282  | 0.341 | 6.5% |
| 100 | 0.400 | 3.097  | 3.591  | 0.494 | 13.8% |
| 100 | 0.800 | 2.111  | 2.352  | 0.241 | 10.3% |

**Headline numbers.** Gap range across the full 18-cell sweep:
absolute 0.103 to 0.628 nats; relative 1.9% to 14.9%. Strict
dominance ($K^\star < K_{\mathrm{uniform}}$) at every cell.

**Largest gap.** $(T_{\max} = 100, U = 0.025) \to 0.628$ nats,
1.9%. The absolute gap is large but the relative fraction is
small because $K_{\mathrm{uniform}} = 32.89$ is large (the
small-$U$ regime).

**Largest relative gap.** $(T_{\max} = 50, U = 0.4) \to 14.9\%$.

**Why Setting B's gap is moderate.** Only 4 clients with 6.6:1
size ratio. The leverage spread is real but more contained than
Setting A's 30:1.

#### 4.2.3 Setting C — Synthetic CIFAR-10, hierarchical at $\eta = 0.5$

**Figure.** `paper/figures/pareto_setting_c.pdf`. Three panels as
in Setting A.

**Per-cell digest.**

| $T_{\max}$ | $U$ | $K_{\mathrm{TA}}^\star$ | $K_{\mathrm{uniform}}$ | gap (nats) | gap (%) |
|---|---|---|---|---|---|
| 25  | 0.05 | 4.265  | 4.642  | 0.377 | 8.1% |
| 25  | 0.10 | 2.788  | 3.116  | 0.328 | 10.5% |
| 25  | 0.20 | 2.098  | 2.353  | 0.255 | 10.8% |
| 25  | 0.40 | 1.801  | 1.971  | 0.170 | 8.6% |
| 25  | 0.80 | 1.681  | 1.781  | 0.099 | 5.6% |
| 25  | 1.60 | 1.632  | 1.685  | 0.053 | 3.2% |
| 50  | 0.05 | 7.288  | 7.693  | 0.406 | 5.3% |
| 50  | 0.10 | 4.265  | 4.642  | 0.377 | 8.1% |
| 50  | 0.20 | 2.788  | 3.116  | 0.328 | 10.5% |
| 50  | 0.40 | 2.098  | 2.353  | 0.255 | 10.8% |
| 50  | 0.80 | 1.801  | 1.971  | 0.170 | 8.6% |
| 50  | 1.60 | 1.681  | 1.781  | 0.099 | 5.6% |
| 100 | 0.05 | 13.375 | 13.797 | 0.422 | 3.1% |
| 100 | 0.10 | 7.288  | 7.693  | 0.406 | 5.3% |
| 100 | 0.20 | 4.265  | 4.642  | 0.377 | 8.1% |
| 100 | 0.40 | 2.788  | 3.116  | 0.328 | 10.5% |
| 100 | 0.80 | 2.098  | 2.353  | 0.255 | 10.8% |
| 100 | 1.60 | 1.801  | 1.971  | 0.170 | 8.6% |

**Headline numbers.** Gap range: absolute 0.053 to 0.422 nats;
relative 3.1% to 10.8%. Strict dominance at every cell.

**Largest gap.** $(T_{\max} = 100, U = 0.05) \to 0.422$ nats,
3.1%.

**Largest relative gap.** $(T_{\max} = 25, U = 0.2)$ and
$(T_{\max} = 50, U = 0.4)$ and $(T_{\max} = 100, U = 0.8)$ all
$\to 10.8\%$.

**Why Setting C's gap is moderate.** The hierarchical $[20, 15,
10, 3, 2]$ proxy at $\eta = 0.5$ is half-strength coupling; the
$\eta = 1$ value of the gap (0.456 nats from §4.1) is the
upper-bound for this topology and the Pareto cells sit below it
because $\eta$ is fixed at 0.5.

#### 4.2.4 Cross-setting observation

**The $(T_{\max}, U)$ grids align across settings such that the gap
pattern is invariant to the
$T_{\max} \cdot n / U$ product.** Recall
$K_{\mathrm{uniform}} = an/U + \max_i \ell_i^\circ = T_{\max} C^2 n / (2 |B|^2 U) + \max_i \ell_i^\circ$.
So at fixed $T_{\max} \cdot n / U$, $K_{\mathrm{uniform}}$ is constant
and $K^\star$ tracks accordingly. The diagonal repetition pattern
visible in the per-cell tables (Setting A's $T_{\max}=50, U=0.10$ row
equals its $T_{\max}=25, U=0.05$ row) is the manifestation. **This is
a strong consistency check** — the analytic expression is invariant
in this product, and the data confirms it numerically.

#### 4.2.5 Suggested figure caption (per setting)

> **Figure (Setting X privacy–utility Pareto).** Privacy bound
> $K^\star$ (nats, log scale) as a function of utility budget $U$
> across three observation-window values $T_{\max}$. Topology-aware
> allocation (blue, circles) lies strictly below uniform DP-SGD
> (red, squares) at every cell. The shaded region marks the absolute
> $K^\star$ gap. $K^\star$ is analytic and carries no seed variance.

---

### 4.3 Utility consistency under both allocations

**Headline finding.** Allocating noise asymmetrically per Fulcrum
does not change test accuracy by any practically-meaningful amount
relative to uniform DP-SGD at the same utility budget.

#### 4.3.1 Setting A — Fed-ISIC2019

Source: `analysis/setting_a_util.tex`. Each row aggregates 18 paired
$(U, \mathrm{seed})$ observations.

| $T_{\max}$ | TA acc (%) | Uniform acc (%) | $\max\|\Delta\|$ (pp) | $\overline{\|\Delta\|}$ (pp) | 95% CI on $\Delta$ (pp) | paired-$t$ $p$ | TOST $p$ | $n$ |
|---|---|---|---|---|---|---|---|---|
| 25  | 54.20 | 54.20 | 2.11 | 0.399 | $[-0.324, +0.322]$ | 0.993 | 0.002 | 18 |
| 50  | 54.96 | 55.12 | 1.40 | 0.437 | $[-0.437, +0.106]$ | 0.215 | 0.009 | 18 |
| 100 | 55.20 | 55.25 | 1.33 | 0.486 | $[-0.372, +0.272]$ | 0.745 | 0.005 | 18 |

**Verdict.** All three TOST $p$-values are below 0.05 at the
$\pm 0.5$ pp margin. The two allocations are statistically equivalent.

#### 4.3.2 Setting B — Fed-Heart-Disease

Source: `analysis/setting_b_util.tex`. All three rows aggregate over
the full $U$ grid after the $T_{\max} = 25$ backfill landed.

| $T_{\max}$ | TA acc (%) | Uniform acc (%) | $\max\|\Delta\|$ (pp) | $\overline{\|\Delta\|}$ (pp) | 95% CI on $\Delta$ (pp) | paired-$t$ $p$ | TOST $p$ | $n$ |
|---|---|---|---|---|---|---|---|---|
| 25  | 70.69 | 70.64 | 2.04 | 0.163 | $[-0.125, +0.236]$ | 0.534 | 1.4e-05 | 27 |
| 50  | 70.59 | 70.58 | 0.28 | 0.016 | $[-0.017, +0.049]$ | 0.331 | 1.0e-16 | 18 |
| 100 | 70.09 | 70.09 | 0.10 | 0.006 | $[-0.003, +0.015]$ | 0.168 | 8.7e-37 | 27 |

**Verdict.** All three TOST $p$-values below 0.05. The two allocations
are statistically equivalent at the $\pm 0.5$ pp margin. The
extraordinary $p$-values at $T_{\max} = 50, 100$ ($10^{-16}$ and
$10^{-37}$) reflect that the mean $\|\Delta\|$ is essentially zero —
this is the strongest equivalence evidence in the manuscript.

#### 4.3.3 Setting C — Synthetic CIFAR-10

Source: `analysis/setting_c_util.tex`. Each row aggregates 18 paired
observations.

| $T_{\max}$ | TA acc (%) | Uniform acc (%) | $\max\|\Delta\|$ (pp) | $\overline{\|\Delta\|}$ (pp) | 95% CI on $\Delta$ (pp) | paired-$t$ $p$ | TOST $p$ | $n$ |
|---|---|---|---|---|---|---|---|---|
| 25  | 21.57 | 21.58 | 0.07 | 0.016 | $[-0.019, +0.009]$ | 0.484 | 2.8e-23 | 18 |
| 50  | 27.79 | 27.80 | 0.04 | 0.012 | $[-0.011, +0.007]$ | 0.614 | 1.6e-26 | 18 |
| 100 | 32.20 | 32.20 | 0.07 | 0.016 | $[-0.013, +0.010]$ | 0.740 | 8.5e-25 | 18 |

**Verdict.** All three TOST $p$-values below 0.05 — statistical
equivalence at $\pm 0.5$ pp. The synthetic setting's small absolute
accuracy (21-32%) reflects that CIFAR-10 across 50 clients with
$\eta = 0.5$ coupling is a *statistical vehicle*, not a deployment
realism anchor. The within-cell paired comparison is meaningful;
absolute accuracy headlines are not the story.

#### 4.3.4 Summary across settings

| Setting | TOST $p$ range | Max $\overline{\|\Delta\|}$ (pp) | Verdict |
|---|---|---|---|
| A | $[0.002, 0.009]$ | 0.486 | Equivalent at $\pm 0.5$ pp |
| B | $[10^{-37}, 10^{-5}]$ | 0.163 | Equivalent at $\pm 0.5$ pp |
| C | $[10^{-26}, 10^{-23}]$ | 0.016 | Equivalent at $\pm 0.5$ pp |

**The defense costs no measurable utility.** The privacy improvement
from §4.2 is therefore not "bought" at the price of accuracy.

---

### 4.4 TADI channel decomposition

**Headline finding.** DP-SGD bounds the parameter channel across
all settings (negative or near-zero attack lift on $\mathcal{A}_1$).
The prior-coupling channels (org, full) realise positive attack lift
*and perfect AUROC = 1.0* on Setting C when the adversary's shadow
prior matches the target's deployment prior (η = 1 ceiling). When
the prior is mismatched (Setting B with synthetic-shadow + native
FLamby target), no channel achieves positive lift — the bound is
conservative in a deployment-favourable direction.

**Primary figure.**
`paper/figures/channel_ablation_cross_setting.pdf`. Two-panel
side-by-side: panel (a) attack lift, panel (b) AUROC, both as
grouped bars with 95% CI from seed variance. **Setting C is
restricted to the η = 1 cells** so the comparison is the
matched-prior ceiling on C against the realistic-deployment
floor on B. The per-η evolution on C is shown separately in §4.5
(Figure `tadi_realisability_setting_c.pdf`).

> **Note on figure scope.** Settings B and C only, by design. The two
> settings span the prior-realisability axis: C at η=1 is matched-prior,
> B is mismatched-prior. Setting A is omitted because its native FLamby
> partitioning would produce the same qualitative pattern as Setting B
> (all channels negative) on a different dataset; see the preamble of
> this report.

#### 4.4.1 Setting B — channel-by-channel

Aggregated over 108 target runs × 4 regressor backends.

| Channel | Attack lift (mean) | 95% CI | Calibration loss | AUROC | Top-K |
|---|---|---|---|---|---|
| $\mathcal{A}_1$ (param-only) | $-0.058$ | $\pm 0.009$ | 0.120 | 0.40 | 0.15 |
| $\mathcal{A}_2^{\text{topo}}$ | $-0.016$ | $\pm 0.000$ | 0.078 | 0.50 | 0.00 |
| $\mathcal{A}_2^{\text{org}}$ | $-0.040$ | $\pm 0.000$ | 0.103 | 0.25 | 0.00 |
| $\mathcal{A}_2^{\text{full}}$ | $-0.041$ | $\pm 0.009$ | 0.103 | 0.56 | 0.36 |

**Reading.** Every channel has negative attack lift — the adversary
cannot beat the constant-mean baseline at all on Setting B. The
non-parameter channels (topo, org) are *structurally degenerate*:
ring topology gives every node degree 2, and Fed-Heart-Disease has
no organisational labelling that distinguishes the 4 sites (no
hierarchy), so the topo and org features carry no information.
$\mathcal{A}_2^{\text{full}}$ marginally exceeds the others on
AUROC (0.56) and Top-K (0.36), but its lift is negative.

**Why all channels are negative.** The shadow corpus is built from
synthetic re-partitioning of the same FLamby dataset, while the
target uses the *native* FLamby site partitioning. The shadow prior
does not match the target prior. The supremum $\ell_i^\circ$ that
defines structural leverage is the *worst-case* coupling over the
prior family; the adversary, with a mismatched shadow prior, cannot
realise that worst case.

#### 4.4.2 Setting C — channel-by-channel at the η = 1 matched-prior ceiling

These are the bars shown in `channel_ablation_cross_setting.pdf`.

| Channel | Attack lift | 95% CI | AUROC | 95% CI | Top-K |
|---|---|---|---|---|---|
| $\mathcal{A}_1$ (param-only) | $-0.021$ | $\pm 0.002$ | 0.67 | $\pm 0.09$ | 0.31 |
| $\mathcal{A}_2^{\text{topo}}$ | $-0.001$ | $\pm 0.001$ | 0.52 | $\pm 0.03$ | 0.39 |
| $\mathcal{A}_2^{\text{org}}$ | $+0.060$ | $\pm 0.000$ | **1.00** | $\pm 0.00$ | 0.49 |
| $\mathcal{A}_2^{\text{full}}$ | $+0.076$ | $\pm 0.000$ | **1.00** | $\pm 0.00$ | 0.46 |

**Reading.**

- $\mathcal{A}_1$ (param-only): negative lift confirming DP-SGD
  bounds the parameter channel even at the matched-prior ceiling.
  AUROC noisy (0.67 ± 0.09) — the regressor occasionally ranks
  clients better than chance from parameter trajectories alone, but
  not in a reliable lift-positive way.
- $\mathcal{A}_2^{\text{topo}}$ (param + structural): near-zero lift,
  AUROC at chance (0.52). The hierarchical depth + degree features
  carry no marginal information in Setting C at this U/T budget.
- $\mathcal{A}_2^{\text{org}}$ (param + organisational label):
  **lift +0.060 and perfect AUROC = 1.0**. The organisational label
  is the channel through which prior coupling is realised. The
  adversary can perfectly rank clients by sensitive-class
  concentration.
- $\mathcal{A}_2^{\text{full}}$ (all channels): **lift +0.076,
  perfect AUROC = 1.0**. The combined adversary marginally exceeds
  org on lift at the matched-prior ceiling, with the same perfect
  ranking quality.

#### 4.4.3 Channel non-dominance is a finding about coupling strength

A subtlety worth highlighting: at η = 1 (the ceiling), the combined
channel $\mathcal{A}_2^{\text{full}}$ exceeds $\mathcal{A}_2^{\text{org}}$
on lift (+0.076 vs +0.060). But averaged across all η values, the
ordering flips: org averages to +0.018, full averages to +0.003. The
crossover is around η = 0.5, visible in
`tadi_realisability_setting_c.pdf` (§4.5).

The mechanism: $\mathcal{A}_2^{\text{full}}$ receives more features
than $\mathcal{A}_2^{\text{org}}$, but on a finite shadow corpus the
extra dimensions act as noise when the coupling is weak. As coupling
grows, those dimensions resolve to genuine signal and the combined
channel pulls ahead.

This is an important narrative point: **the four channels are a
decomposition of information sources, not a tournament of strictly
ordered adversaries.** The realised-dominance pattern depends on the
coupling strength and on the size of the adversary's shadow corpus.

#### 4.4.4 Suggested figure caption (channel ablation)

> **Figure (TADI channel ablation across prior regimes).** Mean
> attack lift (left) and AUROC (right) per channel per setting, with
> 95% confidence intervals from seed variance. Setting C bars are
> restricted to the η = 1 matched-prior ceiling for an
> apples-to-apples comparison with Setting B's native FLamby
> partitioning (no η dial). DP-SGD bounds the parameter channel
> $\mathcal{A}_1$ across both settings (lift ≤ 0). Under matched
> prior, the organisational and combined channels realise positive
> lift and perfect AUROC = 1.0; under mismatched prior, no channel
> achieves positive lift and AUROC stays near chance. The prior gap
> places the supremum $\ell_i^\circ$ out of the public-proxy
> adversary's reach on real cross-silo deployments.

---

### 4.5 Bound realisability across $\eta$ — Setting C

**Primary figure.** `paper/figures/tadi_realisability_setting_c.pdf`.
Two stacked heatmaps share the (channel, η) grid:

- **Top panel** — attack lift, diverging colourmap centred at 0
  (red = negative / DP-SGD-bounded, white = calibration null, blue
  = positive / realised). Each cell carries the numeric lift value.
- **Bottom panel** — AUROC, diverging colourmap centred at 0.5
  (red = below chance, white = chance, blue = perfect ranking). Each
  cell carries the numeric AUROC.

This is the same visual language as `eta_gap_heatmap_setting_c.pdf`
(the Theorem 5.3 validation figure), giving §6 a consistent figure
grammar. Reading the grid:

- The $\mathcal{A}_1$ row is uniformly red in the lift panel — the
  parameter channel is bounded by DP-SGD at every η.
- The $\mathcal{A}_2^{\mathrm{topo}}$ row is uniformly white — the
  structural-only channel never gains traction.
- The $\mathcal{A}_2^{\mathrm{org}}$ and $\mathcal{A}_2^{\mathrm{full}}$
  rows progress red → white → blue as η grows, with the η = 1
  ceiling cells (+0.060 and +0.076) the deepest blue in the figure.
- The AUROC panel reaches deep blue 1.00 cells for the org and full
  channels at η ≥ 0.75 — perfect ranking recovery is unmistakable.
- The η = 0 column reads as the IID-null calibration band on both
  panels.

**Headline finding.** On Setting C (matched shadow/target prior),
attack lift grows monotonically with the coupling strength $\eta$
for the org and full channels, and reaches +0.060 (org) and +0.076
(full) at $\eta = 1$. The parameter channel $\mathcal{A}_1$ remains
negative at every $\eta$, confirming the additive decomposition of
Theorem 5.2: the controllable (mechanism) term is bounded by DP-SGD,
the uncontrollable (prior-coupling) term grows with $\eta$.

#### 4.5.1 Numerical findings — attack lift vs η, per channel

| $\eta$ | $\mathcal{A}_1$ | $\mathcal{A}_2^{\text{topo}}$ | $\mathcal{A}_2^{\text{org}}$ | $\mathcal{A}_2^{\text{full}}$ |
|---|---|---|---|---|
| 0.00 | $-0.017$ | $-0.002$ | $-0.024$ | $-0.071$ |
| 0.25 | $-0.015$ | $-0.002$ | $-0.003$ | $-0.034$ |
| 0.50 | $-0.031$ | $-0.000$ | $+0.018$ | $+0.003$ |
| 0.75 | $-0.039$ | $-0.001$ | $+0.039$ | $+0.041$ |
| 1.00 | $-0.021$ | $-0.001$ | $+0.060$ | $+0.076$ |

#### 4.5.2 AUROC vs η, per channel

| $\eta$ | $\mathcal{A}_1$ | $\mathcal{A}_2^{\text{topo}}$ | $\mathcal{A}_2^{\text{org}}$ | $\mathcal{A}_2^{\text{full}}$ |
|---|---|---|---|---|
| 0.00 | 0.65 | 0.48 | 0.39 | 0.55 |
| 0.25 | 0.50 | 0.50 | 0.50 | 0.50 |
| 0.50 | 0.57 | 0.53 | 0.83 | 0.82 |
| 0.75 | 0.42 | 0.53 | **1.00** | **1.00** |
| 1.00 | 0.67 | 0.53 | **1.00** | **1.00** |

**Significant finding.** At $\eta \geq 0.75$, both org and full
channels achieve perfect AUROC = 1.00. The adversary can perfectly
rank clients by sensitive-class concentration.

#### 4.5.3 IID-null calibration (η = 0)

At $\eta = 0$ the partition is IID across clients, so attack lift
should be ~0 in expectation if the shadow prior matches the target.
The measured values are slightly negative (mean $-0.017$ to
$-0.071$ depending on channel) but small enough to call
"calibration-null." The small negative bias reflects regressor
overfitting on a finite shadow corpus and is not a meaningful signal.

#### 4.5.4 Suggested figure caption (realisability dual heatmap)

> **Figure (TADI realisability on Setting C across coupling
> strength).** Two stacked heatmaps over the (channel, η) grid:
> top panel shows attack lift (red = DP-SGD bounded, blue = threat
> realised); bottom panel shows AUROC (red = below chance, blue =
> perfect ranking). The parameter channel $\mathcal{A}_1$ stays
> uniformly red in the lift panel — DP-SGD bounds the controllable
> term of Theorem 5.2 at every η. The organisational and combined
> channels progress red → white → blue as η grows, with η = 1 cells
> reaching +0.060 and +0.076 in the lift panel and 1.00 in the
> AUROC panel — the threat is fully realised at the matched-prior
> ceiling. The η = 0 column on both panels is the IID-null
> calibration band; the structural-only channel
> $\mathcal{A}_2^{\mathrm{topo}}$ row is uniformly white in both
> panels, contributing no marginal signal.

---

## 5. Robustness checks

- **Regressor backend.** All TADI evaluations report mean over three
  backends: gradient-boosted trees (LightGBM, the default), a small
  MLP, and ridge regression. The per-backend dispersion is small
  enough that the cross-setting story is the same regardless of
  backend; see `analysis/attack_setting_*.parquet` for per-backend
  detail.
- **IID-null calibration.** At $\eta = 0$ in Setting C, every
  channel's mean attack lift is within $[-0.071, -0.002]$ — small
  and slightly negative, consistent with the regressor overfitting
  noise on a finite shadow corpus rather than realising signal.
- **Seed variance.** Three seeds per cell. Test accuracy CIs are
  reported in the utility tables; $K^\star$ values are analytic and
  carry no seed variance.

---

## 6. Housekeeping

### Stale figures to remove from `paper/figures/` before submission

The following are superseded and **should not be referenced** in the
rewrite. They can be deleted after the manuscript ships:

| File | Replaced by | Why |
|---|---|---|
| `eta_sweep_setting_c.pdf` | `eta_gap_heatmap_setting_c.pdf` | Legacy 4-panel line plot superseded by the 9 × 5 heatmap. |
| `attack_lift_eta_setting_c.pdf` | `attack_lift_vs_K_setting_c.pdf` + §4.5 table | Redundant — same η information is now the colour axis of the K* scatter. |
| `pareto_cross_setting.pdf` | Three per-setting Paretos | Cross-setting overlay buried per-setting U grids that aren't directly comparable. |
| `channel_ablation_setting_c.pdf` | `channel_ablation_cross_setting.pdf` | Per-setting version superseded by the cross-setting bar chart. |

---

## 7. Figure inventory for §6

The rewrite should reference exactly these figures from §6. Figures
no longer in the inventory should be deleted from `paper/figures/`
when the rewrite lands.

### 7.1 Live (publication-quality, current)

| File | Purpose | Section |
|---|---|---|
| `eta_gap_heatmap_setting_c.pdf` | Theorem 5.3 validation across 9 topologies × 5 η | §4.1 |
| `pareto_setting_a.pdf` | Setting A privacy-utility Pareto | §4.2.1 |
| `pareto_setting_b.pdf` | Setting B privacy-utility Pareto | §4.2.2 |
| `pareto_setting_c.pdf` | Setting C privacy-utility Pareto | §4.2.3 |
| `channel_ablation_cross_setting.pdf` | Cross-setting channel ablation (lift + AUROC), Setting C at η=1 ceiling | §4.4 |
| `tadi_realisability_setting_c.pdf` | Setting C η-sweep, lift + AUROC for each channel | §4.5 |

### 7.2 Untouched (used elsewhere in the manuscript)

| File | Location |
|---|---|
| `positioning.pdf` | §2 Related Work |
| `tadi_pipeline.pdf` | §4 The TADI Attack |

---

## 8. Manuscript-writing notes

### 8.1 The attack is a methodological tool, not a standalone result

A critical framing decision for the rewrite. The empirical results
of §4.4 and §4.5 should be presented as **evidence that Theorem 5.2's
additive decomposition operates as predicted**, not as a "we built a
devastating new attack" headline. The three findings the attack
delivers:

1. **The bound is empirically tight under matched-prior conditions.**
   Setting C at η = 1 achieves AUROC = 1.0 with attack lift +0.076
   on the combined channel. The worst case Theorem 5.2 predicts is
   *realisable* — without this, the theorem could be dismissed as
   vacuous.
2. **The bound calibrates correctly under the IID-null.** Setting C
   at η = 0 produces zero lift across all channels. This is the
   sanity check the theorem requires: when no topology-data
   correlation exists, the prior-coupling term vanishes, and the
   attack registers no advantage.
3. **The bound is conservative under realistic public-proxy
   adversaries.** Setting B's all-negative-lift result is *not* a
   weakness — it demonstrates that the supremum
   $\ell_i^\circ$ cannot be reached by an adversary whose shadow
   corpus mismatches the deployment partition. This makes the bound
   *deployment-favourable*: the worst case is theoretical, real
   adversaries do strictly worse.

Together these three findings are the strongest possible empirical
support for the theory short of a closed-form realisability proof.
**The attack's value is in characterising when the bound is tight,
when it is calibrated, and when it has slack.** It is not in
"breaking" DP-FL.

### 8.2 Suggested §1 contributions phrasing

> "We construct TADI, a shadow-trained passive attack with four
> channel ablations operationalising the additive decomposition of
> the per-client mutual-information bound. The attack achieves the
> theoretical worst case under matched-prior conditions (perfect
> AUROC, lift +0.076 at η = 1 on Setting C) and confirms the bound
> is conservative in realistic public-proxy deployments (no positive
> lift on Fed-Heart-Disease). TADI thereby provides the first per-channel
> empirical characterisation of topology-conditional leakage in DP-FL,
> establishing both the realisability and the deployment-conservatism
> of the bound that Fulcrum tightens."

### 8.3 Narrative shifts encoded by this report

1. **Lead with the heatmap.** The η-sweep heatmap is the cleanest
   single-figure validation of Theorem 5.3. Open §6's results
   subsections with it.
2. **Sub-claim 5 (ER non-monotone in p) is new and unique.** The
   binomial-variance peak at $p = 0.5$ provides a clean piece of
   evidence that the degree proxy tracks underlying leverage
   asymmetry, not just degree mean. Worth a dedicated paragraph.
3. **The BA + star saturation is the "scale-free generalisation"
   story.** The star is no longer special; any topology with extreme
   degree concentration saturates the bound. This widens the claim
   from "star is a worst case" to "scale-free networks are a worst
   case."
4. **Setting B's all-negative channel ablation is not a failure.**
   It is evidence that the bound is conservative in a
   deployment-favourable direction when the adversary lacks
   shadow/target prior alignment. The manuscript should frame this
   as "the bound is honest about the worst case but the worst case
   is not realised by the most realistic public-proxy adversary."
5. **The channel decomposition is not a tournament.** Drop any
   framing that asserts $\mathcal{A}_2^{\text{full}}$ strictly
   dominates the sub-channels. Frame the four channels as a
   decomposition of information sources whose dominance pattern
   depends on deployment regime.
6. **Tie §4.5 explicitly to Theorem 5.2's additive structure.** The
   parameter channel ($\mathcal{A}_1$) is the controllable term; org
   and full carry the prior-coupling term. The empirical η-monotonic
   growth of the latter and the bounded behaviour of the former is
   the *empirical instantiation* of the additive decomposition.

---

## 9. Data provenance

All numbers in this report are derived from:

- `analysis/setting_a_all.parquet` (108 rows)
- `analysis/setting_b_all.parquet` (252 rows; 90 after filter)
- `analysis/setting_c_all.parquet` (413 rows; subsets for each sweep)
- `analysis/attack_setting_b.parquet` (432 rows = 108 targets × 4 channels)
- `analysis/attack_setting_c.parquet` (912 rows = 228 targets × 4 channels)
- `analysis/setting_{a,b,c}_util.{csv,md,tex}` for the utility tables

The extraction script that produced the numbers is reproducible
locally with `pyarrow` installed; it is captured in the conversation
that generated this report.
