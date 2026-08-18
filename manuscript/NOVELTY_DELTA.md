# Novelty delta — precise positioning for v2

## The one-sentence claim

> Prior work either **fixes** the mechanism and computes how much the aggregation structure
> amplifies privacy, or varies per-client budgets **heuristically**. We do neither: we derive a
> per-client bound whose adjacency matches the quantity being protected, solve for the
> **min–max optimal** allocation against the aggregation-exposure profile in closed form, and
> show the achievable gain is exactly `δ = 1 − ⟨ρ⟩_V/ρ_max` — computable from the deployment
> **before training**.

Novelty is **optimality + the closed-form gain**, not "non-uniform budgets in hierarchical FL"
(which exists). State it this narrowly and it holds; state it broadly and it does not.

---

## The four adjacent lines, and the delta against each

**1. Personalised / individualised DP budgets** — Jorgensen et al. (2015); Boenisch et al.;
Liu et al. (2024) personalised DP-FL.
*They:* per-user budgets driven by **declared user preference**; allocation is an input.
*We:* budgets derived from **structural position** (aggregation-region exposure); allocation is
an output of an optimisation. Complementary, not competing — a combined preference-plus-structure
allocation is genuine future work.

**2. Clustered / hierarchical FL with heterogeneous DP** — HDP-FedCD; clustered FL with
heterogeneous DP on non-IID data; intra-cluster privacy-budget weighting.
*They:* non-uniform per-cluster budgets weighted by **data quality, noise level, or training
progress**. Heuristic; no optimality theorem; no statement of what gain is achievable.
*We:* the weighting is *derived*, provably min–max optimal, and the attainable gain is a closed
form. **This is the closest line and must be cited and distinguished explicitly.**

### ✅ VERDICT — both papers read in full; novelty stands unchanged

**FedCDP** (Guo et al., *Computer Communications* 244:108339, 2025) — "Clustered FL with
heterogeneous DP on Non-IID data".
- Per-client budgets `(ε_m, δ_m)` are **exogenous client preferences** (their Def. 2), not
  derived. Same input-side stance as personalised DP.
- Their "intra-cluster privacy budget weighting" adjusts **aggregation weights** as a function of
  a client's noise level, via a chosen **logarithmic function**. That is the *opposite direction
  to ours*: given noise, choose weights. We: given structure, choose noise.
- Clusters exist for **model personalisation** (clients with similar distributions share a model),
  not as aggregation regions that confer privacy amplification.
- Record-level DP-SGD (Abadi moments accountant). No budget-constrained optimisation, no
  optimality theorem, no dependence on group size.

**HDP-FedCD** (Yin et al., *FGCS* 176:108140, 2026) — "Data-quality-driven hierarchical FL".
- ⚠️ **False positive from keyword search.** "Hierarchical" here means layering each client's
  **local dataset** into core / non-core layers by a data-quality metric ("core-degree"). The FL
  topology is **flat** (their Fig. 1: clients ↔ central server). There are no edge aggregators.
- Noise is allocated by **data quality per training example**, not by structural position.
- Adaptive-threshold heuristic; no optimality theorem, no budget constraint, no group size.

**Conclusion.** Neither paper (a) states an optimality result, (b) allocates noise by
aggregation-group size, (c) imposes a total-noise budget, nor (d) uses silo-level adjacency.
All four contributions in the list below survive intact. Cite both as the closest heuristic
allocation work and distinguish on **optimality + the `δ` identity**.

**3. Privacy amplification by shuffling / secure aggregation** — shuffle model, `O(√n)`
amplification; Bonawitz et al. secure aggregation.
*They:* compute how much a **fixed, uniform** mechanism is amplified by aggregation over `n`.
*We:* treat the amplification profile as the thing to **allocate against**. When regions have
unequal effective size the amplification is unequal, and uniform noise is provably wasteful by
exactly `δ`. Amplification is our premise, not our result.

### ⚠️ CLOSEST PRIOR WORK — Chandrasekaran et al., "Hierarchical Federated Learning with
### Privacy" (arXiv:2206.05209, Telefonica Research / UW-Madison, 2022)

**This is a partial hit and the contribution framing must change accordingly.**

*What they already have — do NOT claim any of it as novel:*
- The **same architecture**: clients (level 0) → *zones* with elected *super-nodes* (level 1) →
  central aggregator (level 2). Their Fig. 1 is our system model.
- **Hierarchical DP (HDP)**: calibrated Gaussian noise added at the super-node / zone level.
- **Per-zone noise that depends on the number of clients in the zone.** Algorithm 1, line 14:
  `σ_i ← z·S_i/(q·W)` (FlatClip) or `2z·S_i/(q·W_min)` (PerLayerClip), where
  `W = Σ_{k∈C_i} w_k` is the zone's *own* weight mass. They state explicitly that σ is computed
  from the online clients **per zone**, not the global client fraction.
- Privacy **amplification from intermediate aggregation** ("natural composability of the Gaussian
  mechanism provides more privacy at the central aggregator when noise is added at the
  super-node").

So "topology enters the controllable term through the aggregation region", and "clients in larger
zones need less noise", are **theirs, not ours**. v2 must cite this as the primary related work.

*What remains genuinely ours:*
1. **Optimality.** Their `σ_i` is a *calibration formula* — each zone independently computes the
   noise needed to hit a given DP level. There is no cross-zone allocation problem, no budget
   constraint, and no optimality claim. We pose `min_σ max_i [T·m_i(σ) + ℓ_i]` s.t.
   `Σ w_i²σ_i² ≤ U` and prove the solution is **min–max optimal**.
2. **The `δ` identity.** `δ = 1 − ⟨ρ⟩_V/ρ_max` characterises the *achievable gain* in closed form,
   computable before training. They have no analogue — no statement of how much is gained, or
   when the approach is worth using at all.
3. **The applicability test.** `δ = 0` exactly when regions are equally sized. Nothing comparable.
4. **The per-client MI bound with silo-level adjacency**, targeting a *distributional* property
   `p_i`. They use standard `(ε,δ)`-DP against record-level inference/reconstruction.
5. **The σ-exponent question is RESOLVED — both are correct, for different trust models.**
   Verified analytically and by Monte Carlo (closed form vs MC agree to 0.4%):

   | injection model | who adds noise | zone-sum noise variance | σ(m) |
   |---|---|---|---|
   | **(A) per-client (ours)** | each of the `m` silos, locally | `m·w²σ²C²` | **σ ∝ 1/√m** (fitted p = −0.500) |
   | **(B) super-node (theirs)** | one draw at the zone aggregator | `σ²C²` | **σ ∝ 1/m** (fitted p = −1.000) |

   Sensitivity to replacing one silo is `2wC` in both cases; only the noise accumulation differs.
   Under (B) the aggregator must be **trusted** to add the noise honestly; under (A) no trust in
   the super-node is required, because each silo protects itself before transmitting.

   **The `√m` gap is exactly the price of not trusting the aggregator.** This is a clean,
   quotable result and belongs in the paper — it is not a correction to their work, and their
   work is not a refutation of ours. State both models, name the trust assumption each requires,
   and position v2 in the untrusted-aggregator regime (which is also the regime in which
   silo-level DP against a *distributional* adversary is meaningful at all).

   ⚠️ Note for §V: the `δ` identity is derived under model (A). Under (B) the exposure profile
   differs and `δ` would need re-deriving. Scope the corollary to (A) explicitly.

*Revised one-sentence claim:*
> Hierarchical DP already calibrates per-zone noise to zone size (Chandrasekaran et al.). We show
> that the resulting allocation problem has a **min–max optimal closed-form solution**, and that
> the gain over uniform allocation is exactly `δ = 1 − ⟨ρ⟩_V/ρ_max` — a quantity computable from
> the deployment before training, which is zero precisely when zones are equally sized.

**4. Network / topology DP** — Cyffers & Bellet (2022); Muffliato (Cyffers, Bellet, Even et al.,
NeurIPS 2022) — *confirmed*: "Muffliato: Peer-to-Peer Privacy Amplification for Decentralized
Optimization and Averaging", Cyffers, Even, Bellet, Massoulié. Pairwise network DP in **gossip**;
amplification is a function of **graph distance** between nodes. Noise is not allocated or
optimised. Distinct from ours (hierarchical aggregation, allocation optimised), but must be cited.
*They:* pairwise topology-dependent accounting in **gossip**; topology as an amplification asset
for a fixed mechanism.
*We:* hierarchical (not gossip); topology enters through the **aggregation set**, and we optimise
the mechanism against it. **Currently absent from v1's related work — must be added; it is the
closest work on the amplification axis.**

---

## Supporting evidence for the claim

A 2024 systematic review of DP-FL (arXiv:2405.08299, 70+ papers) reports adaptive and heuristic
allocation (Andrew et al. adaptive clipping; Liu et al. personalised DP) but **no closed-form or
provably optimal per-client allocation under a budget**, and treats group-size amplification only
as the known shuffle-model result. This is the citation that supports "optimality is open".

---

## What v1 claimed that v2 must NOT re-claim

- "First to treat topology as an information channel" — the structural channel showed null lift in
  every setting, and under a global per-silo observer topology carries no channel at all.
- "First per-client noise heterogeneity in FL" — false; see lines 1 and 2 above.
- Any framing in which the novelty is *that* noise should be non-uniform.

## Replacement contribution list

1. A per-client MI bound for hierarchical FL with **matched silo-level adjacency** (v1's bound
   mixed record-level sensitivity with dataset-level MI).
2. The **min–max optimal** per-region allocation in closed form, with the utility budget correctly
   identified as `U = Σ_i w_i²σ_i²`.
3. `δ = 1 − ⟨ρ⟩_V/ρ_max`: an exact, pre-computable characterisation of the achievable gain, with
   `δ = 0` precisely when regions are equally sized — an explicit **applicability test**.
4. Empirical confirmation with a null control (exactly 0.000 pp at `δ=0`) and an adversarial
   control (same dispersion, permuted: **−2 to −3 pp**, i.e. worse than uniform).
5. A deployment guideline: silo-level DP is practical for federated **fine-tuning**, not training
   from scratch, because `SNR ≈ √(a·n/d)/σ`.

## Positioning risk register

| risk | mitigation |
|---|---|
| Clustered-DP work already claims optimal allocation | Read both papers (paywalled); if so, fall back to the `δ` identity + applicability test as the contribution |
| "Non-uniform budgets are known" | Never claim heterogeneity as novel; claim optimality and the closed form |
| v1 preprint (arXiv:2506.19260) is public with different claims | Frame v2 as a substantive revision with a changed thesis; do not silently drop the old claims |
| Reviewer asks why gains need uneven regions | This is a feature — `δ` is the applicability test, stated up front, not a limitation discovered late |
