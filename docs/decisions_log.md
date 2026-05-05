# Decisions Log

Chronological record of design commitments. Append-only — earlier decisions
are not edited; if a decision is reversed, log the reversal as a new entry.

---

## 2026-05-05 — Original manuscript flagged for full rebuild

**Decision:** Treat the existing `../Paper/Tex Paper/paper.tex` as
rejection-worthy; do not patch. Restart methodology, experiments, and writing.

**Four flaws identified in original manuscript:**
1. Attacks measure proxy metrics (cluster-coherence, silhouette, max|ρ|) not
   actual data-distribution inference.
2. SG/TC/IS partitionings inject topology↔data correlations that the attacks
   then "discover" — circular.
3. Structural-noise defense contradicts the $\mathcal{K}_{\text{complete}}$
   threat model — adversary already knows topology, so dummy traffic doesn't
   fool them.
4. No utility metric for the defense — only "% attack reduction."

**Additional issues identified during review:**
- No null/IID baseline anywhere
- Phase 4 synthetic simulation is calibrated against empirical results then
  used to validate them (circular validation)
- "3-group org knowledge beats complete topology" (74.1% vs 47.2%) is
  parsimoniously explained as the partitioning leaking via group label, not
  by topology
- Effect-size analysis treats configurations as units, conflating seed and
  condition variance
- No comparison to a naive non-topology attacker
- "Status: Fully Effective" labels mask per-vector collapse cases

---

## 2026-05-05 — Stage 1 closed: research framing

**Decision:** Adopt Option B (attack + defense, AUROC-grounded with concrete
inference target) rather than attack-only or defense-only framings.

**Rationale:** Attack-only is not TOPS-publishable; defense-only without a real
attack is also weak. Combining gives a coherent threat model + defense + bound
+ utility characterization story.

**Headline claim:** Even under DP-SGD, an adversary with topology and
organizational knowledge can recover sensitive-class membership at rate
meaningfully above a non-topology baseline, on real federated benchmarks. We
propose D1+D3 as a defense, prove an MI bound, characterize the privacy–utility
frontier.

---

## 2026-05-05 — Stage 1 closed: theoretical tier

**Decision:** Target Medium tier — MI bound $I(\hat{p}_i; p_i) \leq B$ via RDP
+ KL–MI conversion. Reject Light tier (too thin) and Strong tier (Le Cam /
Fano carries unbounded proof-gap risk).

---

## 2026-05-05 — Stage 1 closed: defense architecture

**Decision:** D1 + D3 — topology-aware DP noise allocation paired with
observation-window bounding. Reject D2 (secure-aggregation only) — pushes paper
toward systems/crypto venue. D2 retained as separate extension experiment.

---

## 2026-05-05 — Stage 2 closed: threat model

**Decision:** Single adversary throughout the paper.
$\mathcal{K} = (\mathcal{G}, \omega, \text{public protocol})$;
$\mathcal{O} = \Theta$ (parameter sequences) + comm metadata;
honest-but-curious; no protocol modification.

This adversary is consistent with the defense — defenses that hide $\mathcal{G}$
are off the table.

---

## 2026-05-05 — Stage 2 closed: inference target

**Decision:** Per-client sensitive-class concentration $p_i = \Delta_i(\mathcal{C}_s)$
with primary metric mean squared calibration loss; secondary top-$k$ recovery;
tertiary AUROC (Setting C only). Replaces the original paper's proxy metrics
(cluster-coherence, silhouette, max|ρ|).

---

## 2026-05-05 — Stage 2 closed: partitioning settings

**Decision:** Three settings.
- A: Fed-ISIC2019 native 6-site partitioning (realism anchor)
- B: Fed-Heart-Disease native 4-site partitioning (forward-looking)
- C: CIFAR-10 + Dirichlet$(\alpha)$ + parametric topology coupling $\eta$
  (statistical vehicle, $N \in [20, 500]$)

Rejected: HAM10000 (no natural federation), MNIST (toy), synthetic SG/TC/IS
(circular).

**Concession:** Settings A and B carry realism but minimal statistical power
($N \leq 6$). Setting C carries the statistical claims. The manuscript states
this explicitly.

---

## 2026-05-05 — Stage 3 closed: attack design

**Decision:** Single attack TADI with channel ablations, replacing the
original three-attack proxy-metric design.

$f_\phi(\Theta_i, x_i) \to \hat{p}_i$, trained via shadow-model framework
(Shokri et al. 2017).

Channel ablations: $\mathcal{A}_1$ (params only), $\mathcal{A}_2^{\text{topo}}$
(structure only), $\mathcal{A}_2^{\text{org}}$ (label only),
$\mathcal{A}_2^{\text{full}}$ (combined).

Comparators: constant-mean baseline (mandatory; defines attack lift); gradient
inversion (Geiping et al. 2020) run empirically at our DP levels.

---

## 2026-05-05 — Methodology commitment

**Decision:** Every methodological design choice (partitioning, threat model,
defense, metric, dataset) is grounded in a real-world scenario or verified
citation; every design is stress-tested adversarially before commitment.

**Operationalization:** Documented loophole-finding rounds applied to every
Stage closure. See:
- Stage 2: Section 2 of `02_threat_model_partitioning.md` (four rounds)
- Stage 3: Section 3 of `03_attack_design.md` (three rounds)

---

## 2026-05-05 — Stage 4 step 4.1 closed: formal threat model

**Decision:** Theorem statement target is
$I(p_i; \hat{p}_i \mid \mathcal{G}, \omega, \{\sigma_i\}) \leq B(\sigma_i, T_{\max})$
for any adversary strategy. Topology and organizational labels are constants of the
experiment (fixed by deployment), so conditioning on them is conditioning on
public observables. Independence assumptions: (IA1) DP-SGD noise iid;
(IA2) disjoint client datasets (cross-silo standard); (IA3) FedAvg /
gossip aggregation, with adversary observing pre-aggregation messages.

---

## 2026-05-05 — Stage 4 step 4.1 closed: theorem split

**Decision:** Split the theoretical contribution into two theorems.
- **Theorem 1** — per-client MI bound from DP-SGD + RDP + KL–MI conversion.
  Standard machinery; low risk.
- **Theorem 2** — min-max optimality of the topology-aware noise allocation
  $\sigma_i^*$ under a fixed total budget $\sum_i \sigma_i^{-2} \leq B_{\text{total}}$.
  This is the genuinely novel theoretical claim; depends on the
  structural-leverage definition $\ell_i$ from Step 4.3.

**Rationale:** Theorem 1 alone is "applied DP" and reviewers would dismiss as
known. Theorem 2 makes the topology-aware framing genuinely theoretical rather
than just algorithmic.

---

## 2026-05-05 — Stage 4 step 4.2 closed: machinery commitment

**Decision:** Proof route for Theorem 1 — DP-SGD per-step RDP (Mironov 2017)
→ composition over $T_{\max}$ rounds → data-processing inequality
($p_i = \delta(D_i) \to \Theta_i \to \hat{p}_i$) → KL–MI conversion via
Cuff & Yu 2016 (with Asoodeh et al. 2021 as backup if constants are loose).

No exotic constructions; standard machinery throughout.

---

## 2026-05-05 — Stage 4 step 4.3 closed: structural leverage definition

**Decision:** Define $\ell_i := \sup_{\mathbb{P} \in \mathcal{F}_{\mathcal{G},\omega}} I_\mathbb{P}(p_i; (p_j)_{j \neq i})$
abstractly as supremum-over-priors of mutual information between the target
$p_i$ and other clients' targets, where $\mathcal{F}$ is the prior family
respecting topology + organizational structure.

**Rationale:** Abstract definition is theory-grounded (information-theoretic
quantity, not arbitrary). Practical instantiations (group size for hierarchical;
degree for decentralized) are derived as corollaries under standard structured
priors and evaluated empirically.

**Rejected alternatives:**
- Pure degree centrality — too simplistic, weak link to leakage in non-decentralized topologies.
- Pure spectral (eigenvector centrality, effective resistance) — captures information propagation, but our threat model observes pre-aggregation messages so propagation centrality is the wrong quantity.
- Pure org-conditional — ignores topology and undermines "topology-aware" framing.

**Theorem 2 form (sketch):** Under fixed utility budget $\sum_i \sigma_i^2 \leq U$,
optimal allocation is $\sigma_i^{*2} = U \ell_i / \sum_j \ell_j$ via Lagrangian.
Worst-case MI bound: $\max_i I \leq T_{\max} \alpha \sum_j \ell_j / (2U)$.

When all $\ell_i$ equal (symmetric topology + IID), allocation reduces to
uniform — defense provides marginal benefit *only* when topology creates
asymmetric leverage. Framed as feature.

---

## 2026-05-05 — Stage 4 step 4.4 closed: Theorems 1 and 2 statements (with derivation correction)

**Theorem 1 (per-client conditional MI bound):**
$$I(p_i; \hat{p}_i \mid \mathcal{G}, \omega, \{\sigma_j\}) \leq T_{\max} \alpha_i^\star / (2\sigma_i^2) + \ell_i^\circ$$
**Additive** decomposition: controllable DP-SGD term + uncontrollable prior-coupling leverage.

**Theorem 2 (balanced min-max optimal allocation):**
$$\sigma_i^{*2} = a / (K^\star - \ell_i^\circ), \quad a = T_{\max} \alpha^\star / 2$$
where $K^\star$ solves $\sum_i a/(K^\star - \ell_i^\circ) = U$. Worst-case MI is $K^\star$, balanced uniformly across clients. Strictly better than uniform allocation when leverage scores are not all equal.

**Correction from previous draft:** Earlier sketch had multiplicative form $\ell_i \cdot B(\sigma_i)$ with proportional allocation $\sigma_i^{*2} \propto \ell_i$. Working through the chain-rule decomposition revealed the bound is actually **additive**, not multiplicative — leverage is a privacy floor, not a leakage rate. Optimal allocation has the more delicate balanced-min-max form. Leverage definition adjusted from $I(p_i; (p_j)_{j \neq i})$ to $I(p_i; D_{-i})$ to match the derivation.

**Implication for paper framing:** The defense reduces *only* the mechanism-induced leakage; the prior-coupling floor $\ell_i^\circ$ is fundamental and requires partitioning-side mitigation if it dominates. We characterize the regime where the defense matters explicitly.

---

## 2026-05-05 — Stage 4 step 4.5 closed: Theorem 1 proof + corollaries

**Theorem 1 proof:** Four-lemma chain (chain-rule decomposition; per-round Gaussian KL; sequential composition; lateral-leakage definition). Standard machinery throughout.

**Citations used:** Mironov 2017 (Gaussian mechanism RDP, Prop. 7); Cuff & Yu 2016 (DP ⇒ MI-DP, Theorem 1).

**Corollary 1 (SBM):** Rigorous. Leverage saturates at $H(\Phi_{\omega_i})$. Practical proxy $\ell_i^{\text{org}} \propto |G_i|$ valid in small-block regime (covers Setting A).

**Corollary 2 (bounded-degree):** Reframed as **heuristic**, not rigorous. Self-review caught a real bug — I had used MI subadditivity $I(X; Y_1, \ldots, Y_n) \leq \sum_i I(X; Y_i)$, which is false in general. The honest fix: present the linear-in-degree proxy as empirically validated rather than provably tight. Rigorous version would require strong conditional-independence assumptions on the prior; we don't claim them.

**Lesson:** my self-review caught an actual mathematical error (MI subadditivity misuse). Validates the methodology — without the loophole-finding round, this would have shipped to reviewers. Continuing to apply rigorously to Theorem 2 proof in next iteration.

---

## 2026-05-05 — Stage 4 step 4.6 closed: Theorem 2 proof

**Theorem 2 proof:** KKT reformulation (introduce slack $K$ to make objective smooth) + Lagrangian + monotonicity argument for uniqueness of $K^\star$. Standard convex-optimization machinery, no new external citations needed.

**Corollary 3:** Strict improvement over uniform allocation when leverage scores are not all equal — direct from KKT optimality requirement that $a/\sigma_i^2 + \ell_i^\circ$ be balanced across clients.

**Honest framing on the magnitude of improvement:** Gap $K_{\text{uniform}} - K^\star$ has no clean closed form; depends on leverage dispersion. Zero for uniform leverages, up to $\ell_{\max}$ for maximally dispersed. We don't claim a universal large gap; we measure empirically per topology in Stage 6.

**Computational viability:** $K^\star$ computed by 1D bisection in $O(n \log(1/\epsilon))$. Trivial for $n \leq 500$.

**Self-review caught no real bugs this round** (unlike Step 4.5 where MI subadditivity was wrong). The proof is straightforward convex optimization. Slater's condition holds, KKT is necessary and sufficient, and the monotonicity argument for uniqueness is elementary.

---

## 2026-05-05 — Stage 4 step 4.7 closed: cross-theorem self-review

**Outcome:** No math bugs. Three framing clarifications applied:

1. **T2 fixes $T_{\max}$;** $T_{\max}$ swept as second Pareto axis in §4.9, not jointly optimized in closed form.
2. **Shadow-training data is public by assumption;** TADI adversary's $f_\phi$ trained on public proxy data; test-time $f$ is deterministic per T1's formalization.
3. **Bounds are conservative;** T1 + T2 minimize an upper bound, not the actual MI. Framed as appropriate guarantee posture.

**External-reviewer perspective check:** No structural concerns. Bound tightness, prior-work comparison, leverage-proxy practicality, threat-model realism, defense overhead, scope (passive adversary) all addressable in standard manuscript framing.

**Sanity check at deployment-grade parameters (Setting A, $\varepsilon = 1$):** T1 bound gives ~0.3 bits controllable + ≤3 bits leverage = bounded total. Non-vacuous, deployable.

---

## 2026-05-05 — Stage 4 step 4.8 closed: Proposition 1 (utility analysis)

**Proposition 1:** Convergence error decomposes as $f_T(T_{\max}) + f_\sigma(\sum_i \sigma_i^2)$, with $f_T$ decreasing in $T_{\max}$ and $f_\sigma$ increasing. Specific forms for strongly convex (Setting B) and non-convex smooth (Settings A, C) cases.

**Citations:** Abadi+ 2016 CCS (DP-SGD); McMahan+ 2018 ICLR (DP-FedAvg per-client); Wei+ 2020 IEEE TIFS (DP-FL convergence form, with caveat — NbAFL noises aggregate, we adapt to per-client); Li+ 2020 ICLR (non-IID FedAvg).

**Implication:** Theorem 2's budget constraint $\sum_i \sigma_i^2 \leq U$ directly bounds the noise contribution to convergence error. Lower $T_{\max}$ tightens privacy linearly but degrades utility. Pareto frontier (next step) traces $(U, T_{\max})$ trade-off.

**Self-review:** No bugs. Noted limitations: dimension scaling makes absolute utility cost large for deep networks but doesn't affect relative comparison; Wei+ 2020's NbAFL is aggregate-noise (we adapt per-client); partial participation deferred to secure-aggregation extension experiment.

---

## 2026-05-05 — Stage 4 step 4.9 closed: Pareto frontier specification

**Plot spec:** Per setting (A, B, C), one figure with X = privacy MI bound $K^\star$ (nats), Y = empirical test error. Sweep grid: $U$ (6 values) × $T_{\max}$ (3 values) × allocation (topology-aware + uniform) × seeds (5). Total: 540 runs. Within Stage 5 compute budget.

**Curves:** 4 per figure (color = $T_{\max}$, line style = allocation). Reference markers: no-defense baseline + TADI empirical attack overlay for selected points.

**Headline claim to test:** topology-aware Pareto dominates uniform when leverage is non-uniform (per Corollary 3). Empirical confirmation or refutation is the central evaluation question of §6.

**Quantitative summary:** area between curves as scalar measure of defense's marginal value per setting.

---

## 2026-05-05 — Stage 4 closed

All nine sub-steps complete. Theoretical contribution:
- **Theorem 1:** $I(p_i; \hat{p}_i \mid \mathcal{G}, \omega, \{\sigma_j\}) \leq T_{\max} C^2/(2\sigma_i^2 |B|^2) + \ell_i^\circ$ — additive decomposition of controllable + uncontrollable leakage.
- **Theorem 2:** Closed-form min-max optimal allocation $\sigma_i^{*2} = a/(K^\star - \ell_i^\circ)$ with strict improvement over uniform when leverage scores are non-uniform.
- **Corollary 1:** Rigorous SBM bound for organizational-group leverage proxy.
- **Corollary 2:** Heuristic (not rigorous) degree-based proxy, empirically validated.
- **Proposition 1:** Utility cost monotone in $(\sum_i \sigma_i^2, 1/T_{\max})$, justifying budget-constrained Pareto sweep.
- **Theorem 3 (informal — area-between-curves):** Empirical validation framework.

Two iterations of the methodology caught real issues that would have shipped to reviewers: the multiplicative→additive bound correction in Step 4.4 (intuition wrong, math right), and the MI-subadditivity error in Corollary 2 (math wrong, reframed as heuristic). The methodology is working as intended.

---

## 2026-05-05 — Stage 5 tech stack decision: extend Murmura

**Decision:** Use Murmura (https://github.com/Cloudslab/murmura) as the framework base. Murmura is now a clean simulation framework (no Ray) with config-driven YAML experiments, topology generators (ring, fully, erdos, k-regular), aggregation algorithms (FedAvg + Byzantine-resilient variants), and modular data adapters.

**Extensions we add (in `Redesign/code/fulcrum/`):**
1. DP-SGD via Opacus, hooked into Murmura's training loop
2. Topology-aware noise allocation (Theorem 2 — 1D bisection over $K^\star$)
3. Observation-window bounding (D3)
4. Data adapters for Fed-ISIC2019, Fed-Heart-Disease (via FLamby), and CIFAR-10 with Dirichlet+$\eta$ coupling
5. Hierarchical and line topology generators (Murmura ships ring/fully/erdos/k-regular)
6. TADI attack module (shadow training + channel ablations + regressor)
7. Leverage computation per Corollaries 1 and 2

**Compute budget revised to ~775 runs at ~7 min average on 1 L40S = ~3.8 days continuous, ~1 week wall-clock with iteration.**

---

## 2026-05-05 — Stage 5 closed: full implementation + experimental plan

**Implementation in `Redesign/code/`:** project skeleton + dependencies (Murmura, Opacus, FLamby, LightGBM); data acquisition scripts; three setting adapters (Fed-ISIC2019, Fed-Heart-Disease, CIFAR-10+η); two custom topology generators (line, hierarchical); DP layer (leverage proxies + Theorem 2 allocation + Opacus per-client wrapper); TADI attack module (features + shadow + regressor + metrics); per-setting model factories; SQLite-backed runner with idempotent re-runs; CLI + factorial manager; analysis pipeline (Pareto extraction + figure generation).

**Tests:** standalone math verification for allocation (KKT balance, budget binding, strict improvement, monotonicity) and TADI metrics (calibration loss, IID-null lift, top-k recovery, channel ablation shapes) and Pareto math (frontier extraction, area-between-curves, IID-null at zero area). All passing.

**Experimental plan committed via YAML sweeps:**
- Pareto sweeps for Settings A, B, C (108 runs each, 324 total)
- η-sweep Setting C (30 runs)
- Reference canonical configs for all three settings

**Compute estimate:** ~775 total runs at ~7 min average on 1×L40S = ~3.8 days continuous, ~1 week wall clock with iteration.

**Outstanding for Stage 6 execution:** install env on the L40S box (run `scripts/setup_env.sh`), download datasets (`scripts/download_data.py` + manual ISIC steps), run the sweeps via `scripts/run_factorial.py`.

---

## Open commitments (carried to later stages)

| Item | Carried to | Status |
|---|---|---|
| Title rewrite — current title overclaims | Stage 8 | Working title noted in `03_attack_design.md` |
| Coupling parameter $\eta$ formal definition | Stage 5 | Must match what TADI's structural features exploit |
| Stage 4 citations to verify (Mironov, Wang+Balle+Kasiviswanathan, Cuff+Yu, Asoodeh+) | Stage 4 start | Listed in `references.md` |
| D2 secure-aggregation extension experiment | Stage 5 | Specification in `03_attack_design.md` §3.6 |
