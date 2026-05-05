# Stage 4 — Defense Design and Theoretical Guarantee

## Status: In progress

## 4.1 Theoretical tier commitment: Medium

Statement form:
$$I(\hat{p}_i ; p_i) \leq B(\sigma, T_{\max}, \mathcal{G}, \omega)$$
for any adversary $\mathcal{A}$ in the threat model of §2.2, where $\sigma$ is
the topology-aware noise scale, $T_{\max}$ is the observation-window length,
$\mathcal{G}$ is the topology, and $\omega$ is the organizational labelling.

The bound is derived from RDP composition and the KL–MI conversion (Pinsker
or related). Standard machinery throughout.

**Why Medium and not Light or Strong:**

- **Light** ($\varepsilon$-DP composition only) is well-known DP machinery with
  no genuine differentiation; reviewers would dismiss it as application of
  existing tools.
- **Strong** (minimax lower bound on adversary MSE via Le Cam / Fano) carries
  real risk of an irreparable proof gap; if the bound is loose or has a hidden
  flaw, the paper collapses.
- **Medium** (MI bound conditional on topology) is a real theoretical
  contribution achievable with standard machinery, with bounded risk.

## 4.2 Defense architecture: D1 + D3

### D1 — Topology-aware DP noise allocation

Instead of uniform per-node DP noise $\sigma$, allocate $\sigma_i$ per node $i$
as a function of structural leverage:
$$\sigma_i = \sigma_0 \cdot g(\ell_i)$$
where $\ell_i$ is a structural-leverage score (precise definition pending) and
$g$ is a monotone-increasing scaling function.

**Intuition.** Nodes whose position contributes more to the topology-conditional
information channel get more noise, allocating the privacy budget where it
buys the most reduction in $I(\hat{p}_i ; p_i)$.

**Constraint.** Total privacy budget $\sum_i \sigma_i^{-2}$ (or its RDP
equivalent) is bounded, so additional noise at high-leverage nodes comes at the
cost of lower noise (better utility) at low-leverage nodes.

### D3 — Observation-window bounding

Cap the adversary's observation window to $T_{\max}$ rounds, after which
participants are re-keyed or model is rotated. This defeats the temporal
aggregation that makes parameter-magnitude attacks robust to per-round noise.

**Mechanism options (pending evaluation):**
- Periodic model checkpointing and reinitialization
- Participant rotation (subset of clients in each $T_{\max}$-round window)
- Re-keyed encryption / fresh client identifiers per window

## 4.3 Stage 4 work plan (revised after self-review of 4.1)

| Step | Task | Output | Status |
|---|---|---|---|
| 4.1 | Formal threat model recap in proof-friendly notation | Section 4.1 of paper | **Done** |
| 4.2 | Choice of theoretical machinery: RDP composition + KL–MI conversion (Mironov 2017, Cuff & Yu 2016, Asoodeh et al. 2021 backup) | Methods section | **Done** |
| 4.3 | Definition of structural-leverage score $\ell_i$ as supremum-over-priors of $I(p_i; (p_j)_{j \neq i})$, with practical instantiations as corollaries | Theorem prerequisite | **Done** |
| 4.4 | Theorem 1 statement (additive form: $T_{\max}\alpha/(2\sigma_i^2) + \ell_i^\circ$) **and** Theorem 2 statement (balanced min-max allocation $\sigma_i^{*2} = a/(K^\star - \ell_i^\circ)$) | Theorems 1 & 2 | **Done** |
| 4.5 | Theorem 1 proof (full): chain rule + RDP composition + Cuff-Yu KL-MI conversion + leverage upper-bound for SBM (rigorous) + bounded-degree (heuristic, empirically validated) | Appendix proof + corollaries | **Done**; Corollary 2 reframed as heuristic after self-review caught MI subadditivity error |
| 4.6 | Theorem 2 proof (full): KKT reformulation + uniqueness of $K^\star$ + strict improvement over uniform (Corollary 3) | Appendix proof | **Done** |
| 4.7 | Cross-theorem adversarial self-review: assumption consistency + reviewer-perspective check + three framing clarifications | Loophole-finding round | **Done** — no math bugs, three clarifications applied |
| 4.8 | Utility analysis: convergence cost as function of $(\sigma_i, T_{\max})$ — standard FL convergence machinery (per-client adaptation of Wei+ 2020 + McMahan+ 2018 + Li+ 2020) | Proposition 1 | **Done** — strongly convex + non-convex bounds, per-setting application |
| 4.9 | Pareto frontier specification: $K^\star$ vs. test error, swept across $(U, T_{\max})$, topology-aware vs uniform allocation | Plot spec for Stage 7 + 540-run experimental commitment | **Done** |

**Key insight from Step 4.1 self-review:** the theoretical contribution is split across two theorems. Theorem 1 is a standard DP-SGD MI bound (clean, low risk). Theorem 2 — the allocation-optimality result — is the topology-aware novelty. This split keeps the theorem statements honest about what's standard vs. novel.

## 4.4 Structural leverage definition (Step 4.3 closure — corrected after Step 4.4 derivation)

**Abstract definition.** For any $i$,
$$\ell_i^\circ = \ell(\mathcal{G}, \omega, i) := \sup_{\mathbb{P} \in \mathcal{F}_{\mathcal{G}, \omega}} I_\mathbb{P}(p_i; D_{-i})$$
where $\mathcal{F}_{\mathcal{G}, \omega}$ is the family of priors over $(\Delta_1, \ldots, \Delta_n)$ that:
(a) factorize over connected components of $\mathcal{G}$;
(b) within each component, satisfy a Markov property w.r.t. $\mathcal{G}$;
(c) have bounded second moments (ensures finite supremum).

**Note:** the original Step 4.3 definition used $I(p_i; (p_j)_{j \neq i})$. Working through Step 4.4's chain-rule derivation revealed that the cleaner, derivation-aligned quantity is $I(p_i; D_{-i})$. Under (IA2) and standard prior structure, $I(p_i; D_{-i}) = I(p_i; \Delta_{-i})$, so the supremum is well-defined. Practical instantiations (group size, degree) bound this quantity under SBM and bounded-degree priors respectively (proofs in Step 4.5).

This quantity is deterministic in $(\mathcal{G}, \omega, i)$, finite under (a)–(c), and computable in closed form for the practical instantiations below.

**Practical instantiations** (proved as corollaries in Step 4.5):

| Setting | Instantiation | Domain |
|---|---|---|
| A — Hierarchical FL | $\ell_i^{\text{org}} \propto \|\omega^{-1}(\omega_i)\|$ — organizational group size | Stochastic block model prior on $\omega$-clusters |
| B — Decentralized k-NN | $\ell_i^{\text{deg}} \propto \deg_\mathcal{G}(i)$ — graph degree | Bounded-degree prior with neighbor-similarity assumption |
| C — Synthetic | $\ell_i^{\eta}$ — function of coupling parameter $\eta$ | Defined fully in Stage 5 |

The instantiations are **hypotheses about which proxy best approximates $\ell$ in each setting**; we evaluate them empirically (Stage 6). The formal theorems are stated abstractly.

## 4.5 Theorem 1 statement (Step 4.4 closure)

**Theorem 1 (per-client conditional MI bound).** For any adversary $\mathcal{A}$ in the threat model of §4.1, any prior $\mathbb{P} \in \mathcal{F}_{\mathcal{G},\omega}$, and any $i$:
$$I_{\mathbb{P}}(p_i; \hat{p}_i \mid \mathcal{G}, \omega, \{\sigma_j\}) \;\leq\; \frac{T_{\max}\,\alpha_i^\star}{2\sigma_i^2} \;+\; \ell_i^\circ$$
where $\alpha_i^\star$ is the RDP order optimizing the Cuff-Yu / Asoodeh KL-MI conversion and $\ell_i^\circ$ is the structural leverage from §4.4.

**Decomposition.** First term is **controllable** (DP-SGD on client $i$). Second term is **uncontrollable** (prior coupling between $p_i$ and other clients' data, a property of the world). The defense reduces only the controllable term.

**Proof sketch.** (1) Data processing: $\hat{p}_i = f(\Theta) \Rightarrow I(p_i; \hat{p}_i) \leq I(p_i; \Theta)$. (2) Conditioning identity with $Z = D_{-i}$ + dropping non-negative subtractive term: $I(p_i; \Theta) \leq I(p_i; \Theta \mid D_{-i}) + I(p_i; D_{-i})$. (3) Conditional on $D_{-i}$, $\Theta_{-i}$ adds nothing about $D_i$, so $I(p_i; \Theta \mid D_{-i}) = I(p_i; \Theta_i \mid D_{-i}) \leq I(D_i; \Theta_i \mid D_{-i})$. (4) RDP composition + KL-MI conversion: $I(D_i; \Theta_i \mid D_{-i}) \leq T_{\max} \alpha_i^\star / (2\sigma_i^2)$. (5) The lateral term is bounded by leverage: $I(p_i; D_{-i}) \leq \ell_i^\circ$ by definition of $\sup$. Full proof in Step 4.6.

## 4.6 Theorem 2 statement (Step 4.4 closure — corrected)

**Theorem 2 (balanced min-max allocation).** Let $a := T_{\max}\,\alpha^\star / 2$. Under utility budget $\sum_i \sigma_i^2 \leq U$, with $U > $ feasibility threshold, the worst-case-optimal allocation is
$$\sigma_i^{*2} = \frac{a}{K^\star - \ell_i^\circ},$$
where $K^\star$ is the unique value $> \max_i \ell_i^\circ$ satisfying the budget equation
$$\sum_i \frac{a}{K^\star - \ell_i^\circ} = U.$$
The corresponding worst-case MI bound is $\max_i I = K^\star$, balanced uniformly across clients.

**Improvement over uniform allocation.** Under uniform $\sigma_i^2 = U/n$, the worst-case MI is $an/U + \max_i \ell_i^\circ$, dominated by the highest-leverage client. The topology-aware allocation gives $K^\star < an/U + \max_i \ell_i^\circ$ strictly when leverage scores are not all equal — quantifiable per topology.

**Proof sketch.** Lagrangian $\mathcal{L} = \max_i [a/\sigma_i^2 + \ell_i^\circ] + \lambda(\sum_i \sigma_i^2 - U)$; first-order condition gives $a/\sigma_i^{*2} + \ell_i^\circ = K^\star$; budget equation determines $K^\star$. Strict improvement via Jensen-type argument. Full proof in Step 4.6.

**Edge cases:** when $U$ is below feasibility threshold (or $\ell_i^\circ$ are highly skewed), optimal allocation is degenerate ($\sigma_i = 0$ for low-leverage clients). Documented in §4.7 as deployment caveat.

**Reduction to uniform under symmetry.** When all $\ell_i^\circ$ are equal, $\sigma_i^* = \sqrt{U/n}$ — uniform allocation. The defense provides marginal benefit *exactly when* topology and organization create asymmetric leverage. Frame as feature.

## 4.6.5 Theorem 1 full proof (Step 4.5 closure)

**Lemma 1 (chain-rule decomposition).** For random variables $X, Y, Z$: $I(X;Y) \leq I(X;Z) + I(X;Y \mid Z)$.
*Proof:* Apply identity $I(X;Y) = I(X;Z) + I(X;Y \mid Z) - I(X;Z \mid Y)$ and drop $I(X;Z \mid Y) \geq 0$. $\square$

**Lemma 2 (per-round Gaussian-mechanism KL).** Under (IA1)–(IA3):
$$D_{\mathrm{KL}}\!\left(\mathbb{P}_{\theta_i^{(t)} \mid D_i, \theta^{(t-1)}, D_{-i}} \big\| \mathbb{P}_{\theta_i^{(t)} \mid D_i', \theta^{(t-1)}, D_{-i}}\right) \leq \frac{C^2}{2\sigma_i^2 |B|^2}.$$
By Cuff & Yu 2016 Theorem 1 (max-KL ⇒ MI-DP):
$$I(D_i; \theta_i^{(t)} \mid \theta^{(t-1)}, D_{-i}) \leq \frac{C^2}{2\sigma_i^2 |B|^2}.$$

**Lemma 3 (sequential composition).** Under (IA1)–(IA3):
$$I(D_i; \Theta_i \mid D_{-i}) \leq \frac{T_{\max} C^2}{2\sigma_i^2 |B|^2}.$$
*Proof:* MI chain rule + Lemma 2 per round. The global state $\theta^{(t-1)}$ is post-processing of past observed $\Theta_i^{[<t]}$, so conditioning on $\theta^{(t-1)}$ adds no information about $D_i$ beyond what's already observed. $\square$

**Lemma 4 (lateral-leakage bound).** Under (IA2): $I(p_i; D_{-i}) \leq \ell_i^\circ$. Direct from the supremum definition. $\square$

**Theorem 1 (assembled).** Combining Lemmas 1–4:
$$I_\mathbb{P}(p_i; \hat{p}_i \mid \mathcal{G}, \omega, \{\sigma_j\}) \leq \ell_i^\circ + \frac{T_{\max} C^2}{2\sigma_i^2 |B|^2}. \quad \square$$

**Citations used:** Mironov 2017 (RDP of Gaussian mechanism, Prop. 7); Cuff & Yu 2016 Theorem 1 (DP ⇒ MI-DP).

## 4.6.6 Leverage corollaries (Step 4.5 closure)

**Corollary 1 (SBM upper bound).** Under stochastic-block-model prior with finite-entropy block parameter $\Phi$:
$$\ell_i^\circ \leq H(\Phi_{\omega_i})$$
*independent* of block size $|G_i|$ (asymptotic); for finite blocks, leverage grows with $|G_i|$ until saturating at $H(\Phi)$.
*Proof:* By SBM conditional-independence structure, $D_{-i}$ provides info about $\Delta_i$ only through $\Phi_{\omega_i}$. Markov chain + data processing. $\square$

**Practical proxy $\ell_i^{\text{org}} \propto |G_i|$**: tight in small-block regime ($|G_i| \cdot \kappa_1 < H(\Phi)$); loose for large blocks. For Setting A (Fed-ISIC2019, $|G_i| \in \{2, 3\}$ per regional aggregator), firmly in small-block regime.

**Corollary 2 (bounded-degree, heuristic).** Under bounded-degree Markov prior with sufficient regularity assumptions, leverage is bounded by a function of $\deg_\mathcal{G}(i)$; rigorous linear-in-degree bound requires strong conditional-independence assumptions on the prior. The proxy $\ell_i^{\text{deg}} \propto \deg_\mathcal{G}(i)$ is **treated as a heuristic** validated empirically in Stage 6, **not as a rigorous theoretical bound**.

**Honest framing in the paper:** Corollary 1 has a clean rigorous bound (saturation at $H(\Phi)$); Corollary 2 is heuristic with empirical validation. Reviewers will accept this asymmetry if framed transparently.

## 4.6.7 Theorem 2 full proof (Step 4.6 closure)

**Reformulation.** Original objective $\max_i [a/\sigma_i^2 + \ell_i^\circ]$ is non-smooth. Reformulate via slack $K$:
$$\min_{K, \{\sigma_i^2 > 0\}} K \quad \text{s.t.} \quad a/\sigma_i^2 + \ell_i^\circ \leq K \;\forall i, \quad \sum_i \sigma_i^2 \leq U.$$
Linear objective + convex constraints + Slater's condition (interior point $\sigma_i^2 = U/(2n)$) ⇒ strong duality + KKT necessary and sufficient.

**Lemma 5 (KKT optimality form).** From the Lagrangian $\mathcal{L} = K + \sum_i \mu_i(a/\sigma_i^2 + \ell_i^\circ - K) + \lambda(\sum_i \sigma_i^2 - U)$:
- Stationarity: $\sum_i \mu_i = 1$, $\sigma_i^{*2} = \sqrt{\mu_i a / \lambda}$.
- All clients active (since $\sigma_i = 0$ violates the $\leq K$ constraint), so:
$$\sigma_i^{*2} = \frac{a}{K^\star - \ell_i^\circ}.$$

**Lemma 6 (uniqueness of $K^\star$).** Define $g(K) := \sum_i a/(K - \ell_i^\circ)$ on $K > \max_i \ell_i^\circ$. $g$ is strictly decreasing, $g \to +\infty$ at $\max_i \ell_i^\circ$, $g \to 0$ at $\infty$. Unique $K^\star$ with $g(K^\star) = U$.

**Theorem 2.** For $a = T_{\max} C^2/(2|B|^2)$ and $U > 0$:
$$\sigma_i^{*2} = \frac{a}{K^\star - \ell_i^\circ}, \quad K^\star \text{ uniquely solves } \sum_i \frac{a}{K^\star - \ell_i^\circ} = U.$$
Worst-case MI is $K^\star$, balanced across clients.

**Corollary 3 (strict improvement).** $K^\star \leq K_{\text{uniform}} := an/U + \max_i \ell_i^\circ$, with equality iff all $\ell_i^\circ$ are equal.

**Citations used:** Convex optimization machinery (Boyd & Vandenberghe textbook); no new external citations.

**Computational note:** $K^\star$ computed by 1D bisection on $g$; $O(n \log(1/\epsilon))$ per allocation. Trivial for $n \leq 500$.

## 4.6.8 Cross-theorem clarifications (Step 4.7 closure)

Three clarifications surfaced during the cross-theorem self-review. None are bugs; all are framing fixes for the manuscript.

**Clarification 1 — Joint $(\sigma_i, T_{\max})$ optimization scope.** Theorem 2 fixes $T_{\max}$ at a chosen value and gives the optimal $\sigma_i$ allocation. $T_{\max}$ is not jointly optimized in the closed-form theorem — it is swept as the second Pareto axis in §4.9. The paper states: "For each $T_{\max}$, Theorem 2 gives the optimal noise allocation; sweeping $T_{\max}$ traces the Pareto frontier."

**Clarification 2 — Shadow-training data is public by assumption.** The TADI adversary trains its regressor $f_\phi$ via shadow simulations on public data (CIFAR-10 for Setting C; public splits of Fed-ISIC2019 / Fed-Heart-Disease for Settings A and B). Theorem 1 formalizes the test-time strategy: $f_\phi$ is treated as a deterministic function $f$ at deployment. Shadow training does not violate Theorem 1 because (i) shadow data is public, (ii) the trained $f_\phi$ at test time depends only on public inputs $(\mathcal{G}, \omega, \{\sigma_j\}, \Theta)$.

**Clarification 3 — Bounds are conservative; framed as guarantee.** Theorem 1 uses the chain-rule upper bound (drops $I(p_i; D_{-i} \mid \Theta) \geq 0$) and the Cuff-Yu KL-to-MI conversion (also an upper bound). Theorem 2 minimizes this upper bound. The resulting allocation is *conservative*: it may over-protect relative to the actual minimum. We frame this as a feature — the appropriate posture for a privacy guarantee is to provide upper bounds on adversary leakage. Tightness analysis is acknowledged as future work.

## 4.6.9 Proposition 1 — Utility analysis (Step 4.8 closure)

**Setup.** FedAvg with per-client DP-SGD (noise $\{\sigma_i\}$, clipping $C$, batch $|B|$) over $T_{\max}$ rounds. Standard FL convergence assumptions: $L$-smooth loss, bounded local gradient variance $G^2$, bounded heterogeneity $H^2$, full participation. Aggregated noise variance per coordinate: $V = (C^2 / |B|^2 n^2) \sum_i \sigma_i^2$.

**Proposition 1 (utility cost of D1+D3).**

Strongly convex (parameter $\mu$):
$$\mathbb{E}[\mathcal{L}(\theta^{(T_{\max})})] - \mathcal{L}^* \leq c_1 \log T_{\max} / (\mu T_{\max}) + c_2 d V / \mu$$

Non-convex smooth:
$$\min_{t \leq T_{\max}} \mathbb{E}\|\nabla\mathcal{L}(\theta^{(t)})\|^2 \leq c_3/\sqrt{T_{\max}} + c_4 \sqrt{d V}$$

In both cases utility cost decomposes as $f_T(T_{\max}) + f_\sigma(\sum_i \sigma_i^2)$, $f_T$ decreasing and $f_\sigma$ increasing.

**Implications.**
- $\sum_i \sigma_i^2 \leq U$ (Theorem 2's budget) directly bounds the noise contribution to convergence error.
- $T_{\max}$ trades off Theorem 1's privacy (linear in $T_{\max}$) against utility ($f_T$ decreasing in $T_{\max}$).
- Pareto frontier (§4.9) is a 2D sweep over $(U, T_{\max})$.

**Per-setting application:**
- Setting A (Fed-ISIC2019, ResNet, cross-entropy): non-convex smooth bound.
- Setting B (Fed-Heart-Disease, regularized logistic regression): strongly convex bound.
- Setting C (CIFAR-10, ResNet, cross-entropy): non-convex smooth bound.

**Citations:** Abadi+ 2016 (DP-SGD); McMahan+ 2018 (DP-FedAvg per-client); Wei+ 2020 (DP-FL convergence form); Li+ 2020 (FedAvg non-IID convergence).

**Caveat:** Wei+ 2020's NbAFL noises the aggregate (not per-client). We use the per-client formulation matching McMahan+ 2018, deriving aggregated variance $\sum_i \sigma_i^2 / n^2$ from independent client noises. Convergence analysis follows the standard SGD-with-noise-variance template.

## 4.6.10 Pareto frontier specification (Step 4.9 closure)

**Per-setting figure design.** One figure per setting (A, B, C):
- X-axis: privacy MI bound $K^\star$ in nats (Theorem 2, computed from $(U, T_{\max})$).
- Y-axis: utility cost = empirical test error (1 − accuracy).

**Sweep grid:**
- $U$: 6 log-spaced values (weak DP $\varepsilon \approx 8$ → strong DP $\varepsilon \approx 0.5$)
- $T_{\max}$: 3 values $\{T/4, T/2, T\}$ where $T$ is convergence-saturating count
- Allocation: topology-aware (Theorem 2) AND uniform baseline
- Seeds: 5 per configuration

Total: 6 × 3 × 2 × 5 = 180 runs/setting × 3 settings = **540 runs**, within Stage 5 compute budget.

**Curves per figure:** Four curves (color = $T_{\max}$, line style = allocation), with shaded 95% CI bands from seed variance.

**Reference markers:**
- No-defense baseline (standard FedAvg, no DP) as single point.
- TADI empirical attack calibration loss overlaid for selected $(U, T_{\max})$ points — validates that theoretical $K^\star$ tracks empirical adversary performance.

**Headline claim:** topology-aware Pareto curve dominates uniform Pareto when leverage scores are non-uniform (per Corollary 3). Empirical confirmation or refutation of this claim is the central evaluation question of §6 of the manuscript.

**Quantitative summary:** report area between topology-aware and uniform Pareto curves per setting as scalar measure of the defense's marginal value.

**What this gives the manuscript:** the headline visual for §6 (defense evaluation), supporting the central claim that topology-aware allocation outperforms uniform under fixed privacy or fixed utility constraints.

## 4.7 Regimes where the defense matters

The defense's value is concentrated where the controllable term $a/\sigma_i^2$ is comparable to or larger than the uncontrollable $\ell_i^\circ$. We characterize this regime explicitly:
- **High noise + low coupling**: defense dominant; large per-client improvement.
- **Low noise + high coupling**: defense provides small relative improvement; deployment alternative is to reduce $\ell_i^\circ$ via partitioning changes.
- **Mixed regime**: typical for our experimental settings; quantitative characterization in Stage 6 results.

## 4.4 Citations to verify before Stage 4 begins

- Mironov (CSF 2017) — Rényi Differential Privacy
- Wang, Balle, Kasiviswanathan (2019) — Subsampled Rényi DP and Analytical Moments Accountant
- Bonawitz et al. (CCS 2017) — Practical Secure Aggregation
- Cuff & Yu (CCS 2016) — Differential Privacy as a Mutual Information Constraint (for KL–MI conversion route)
- Possible: Asoodeh et al. (ITW 2020) — Three Variants of Differential Privacy (MI formulation)

These will be verified at the start of Stage 4 work, before any claim relying
on them is committed.

## 4.5 Open questions for the user before proceeding

None at the start of Stage 4 — all key decisions taken (Medium tier, D1+D3
architecture, MI bound formulation). Proceeding will be incremental: each step
above will be drafted, self-reviewed, then presented for sign-off.
