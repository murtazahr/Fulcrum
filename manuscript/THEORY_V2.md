# Corrected theory for Fulcrum (v2)

## What was wrong in v1

| # | Defect | Consequence |
|---|--------|-------------|
| D1 | Lemma A.2 computes a KL over **record-adjacent** datasets (sensitivity `C/|B|`) but states the conclusion as `I(D_i; ·)`, a **dataset-level** MI. | Group-privacy gap. Bound not established. `p_i` is a whole-dataset property; record-level DP provably does not control it. |
| D2 | Assembly Step 4 asserts `Θ_{-i} ⊥ D_i \| D_{-i}` under (IA1). | False for `T>1`: `θ_j^(t)` depends on `θ^(t-1)` which aggregates `i`'s past updates. Assumes away the very coupling the paper studies. |
| D3 | Eq. (1) adds `N(0,σ²C²)` to the per-sample **average**; Opacus adds it to the **sum**. | Factor `\|B\|²` between analysis and code. |
| D4 | `Def. V.1` takes `sup` over an unconstrained prior family. | For any non-isolated `i` one can pick a prior making `p_i` a deterministic function of a neighbour's data ⇒ `ℓ_i° = H(p_i)` for all `i` ⇒ constant ⇒ allocation collapses to uniform. |
| D5 | Proxies specified only up to `∝`. | The gap `K_uniform − K*` scales with the unfixed constant; "1.967 nats" is not a well-defined quantity. |
| D6 | No comparison of `K*` against `H(p_i)`. | Reported `K*` of 3–30 nats vs a ceiling of ~2–3 nats: the bound is vacuous at every operating point evaluated. |
| D7 | Eq. (1) is one SGD step/round; FedAvg runs many local steps. | Composition undercounts mechanism invocations. |

## v2 setup

`n` silos, topology `G`, organisational labelling `ω`. Client `i` holds `D_i`; the
inference target is the sensitive-class concentration `p_i = Δ_i(C_s)`, **quantised to
`r` levels** (so `H(p_i)` is a finite, operationally meaningful ceiling: `r` levels =
resolution `1/r` on a concentration).

**Mechanism — silo-level DP.** At round `t`, client `i` forms its local update
`Δ_i^(t) = θ_i^(t) − θ^(t−1)`, clips it to `‖Δ‖ ≤ C`, and adds `ξ_i^(t) ~ N(0, σ_i² C² I)`.
Clipping the *update* (not the per-sample gradient) is what makes the adjacency match
the quantity being protected — this is the D1 fix, and it is exactly DP-FedAvg
(McMahan et al.), already cited in the paper as [10].

**Aggregation set `A(i)`.** The smallest set of silos whose contributions the adversary
sees only in aggregate:
- per-client observer (v1's model): `A(i) = {i}`
- hierarchical FL with regional aggregators: `A(i)` = `i`'s region
- full secure aggregation: `A(i) = [n]`

Adversary observes `Y^(t) = Σ_{j∈A} w_j (clip(Δ_j^(t)) + ξ_j^(t))` for each region `A`,
plus `G, ω, {σ_j}`.

**This is where topology genuinely enters the controllable term.** Under v1's global
per-client observer, topology could not carry any channel at all — it only indexed the
prior. That is why v1's structural channel `A₂^topo` showed null lift in every setting.

## Definition 1 (lateral leakage)

Relative to an **explicitly stated** deployment prior `P` (not a sup over an
unconstrained family — that is D4):

    ℓ_i := I_P(p_i ; D_{−i})

Hierarchical instantiation used throughout:

    Φ_g ~ Beta(a₀,b₀)                      group-g latent propensity
    p_i | Φ_{g(i)} ~ Beta(κΦ, κ(1−Φ))      within-group coupling κ

Conditional on `Φ_{g(i)}`, `p_i` ⫫ everything else, so only `i`'s `m_i = |G_{g(i)}|−1`
siblings matter: `ℓ_i = I(p_i ; p_sib(1..m_i))`.

Properties (all verified numerically, `theory/lateral_mi.py`):
- `ℓ_i = 0` for an isolated client
- monotone increasing in `m_i`
- **saturating**, with exact ceiling `I(p_i; Φ) < H(p_i)`
- computable in nats — fixes D5

If a worst-case flavour is wanted, take the sup over a *parametric* family with coupling
`κ ≤ κ_max`; that is finite and client-dependent, unlike D4's unconstrained sup.

## Theorem 1 (per-client bound, corrected)

For any deterministic adversary `f` with `p̂_i = f(Y)`:

    I(p_i ; p̂_i | G, ω, {σ_j})  ≤  min{ H(p_i),  T · m_i(σ) + ℓ_i }

with the mechanism term

    m_i(σ) = 2 w_i² / Σ_{j∈A(i)} w_j² σ_j²

**Proof.**
1. DPI: `I(p_i;p̂_i) ≤ I(p_i;Y)`.
2. `I(p_i;Y,D_{−i})` expanded two ways gives
   `I(p_i;Y) = I(p_i;D_{−i}) + I(p_i;Y|D_{−i}) − I(p_i;D_{−i}|Y) ≤ I(p_i;D_{−i}) + I(p_i;Y|D_{−i})`.
3. First term `≤ ℓ_i` by Definition 1.
4. `p_i` is a function of `D_i`, so DPI gives `I(p_i;Y|D_{−i}) ≤ I(D_i;Y|D_{−i})`.
5. Chain rule over rounds: `I(D_i;Y|D_{−i}) = Σ_t I(D_i; Y^(t) | Y^(<t), D_{−i})`.
6. **(D2 fix.)** Condition on the past. `θ^(t−1)` is a deterministic function of `Y^(<t)`.
   For every `j ≠ i`, `Δ_j^(t) = F(θ^(t−1), D_j)` is deterministic and `ξ_j^(t)` is
   independent by (IA1). So
   `Y^(t) = w_i(clip Δ_i^(t) + ξ_i^(t)) + R`,
   where `R = R(Y^(<t), D_{−i}, {ξ_j}_{j≠i})` is conditionally independent of `D_i`
   given `(Y^(<t), D_{−i})`.
   We never claim the false marginal independence `Θ_{−i} ⫫ D_i | D_{−i}`; conditioning
   on the past is what makes the step valid.
7. Max-KL (elementary; **replaces Cuff–Yu**): if `sup_{x,x'} KL(P_{Y|x}‖P_{Y|x'}) ≤ κ`
   then `I(X;Y) ≤ κ`, since `I(X;Y) = E_X KL(P_{Y|X}‖P_Y)` and `P_Y = E_{X'} P_{Y|X'}`,
   so convexity of KL gives `KL(P_{Y|x}‖P_Y) ≤ E_{X'} KL(P_{Y|x}‖P_{Y|X'}) ≤ κ`.
   Two **silo-adjacent** inputs move the clipped update by `≤ 2C`, hence the observed
   mean by `≤ 2w_i C`, against noise covariance `Σ_j w_j²σ_j²C² I`:
   `KL ≤ (2w_i C)² / (2 Σ_j w_j²σ_j²C²) = m_i(σ)`.
   Adjacency on the left (`I(D_i;·)`) and on the right (silo replacement) now **match** —
   this is the D1 fix. No group-privacy factor is needed.
8. Sum over `T` rounds. `T` counts **mechanism invocations** = rounds × local steps (D7 fix).
9. `I(p_i;p̂_i) ≤ H(p_i)` always, giving the `min` (D6 fix).                     ∎

## Theorem 2 (allocation) — v1's Theorem V.3 survives, with corrected constants

`min_σ max_i [T·m_i(σ) + ℓ_i]` s.t. `Σ σ_i² ≤ U`.

- **Per-client observation** (`A(i)={i}`, `w_i=1`): `m_i = 2/σ_i²`, so the problem is
  exactly v1's with `a := 2T` (not `T/(2|B|²)`). Closed form `σ_i*² = a/(K*−ℓ_i)` and
  Corollary V.4 carry over verbatim. The KKT derivation was correct and is retained.
- **Disjoint regions**: `1/(linear)` is convex, so the program stays convex; it decouples
  into an across-region and a within-region allocation.

## Corollary 1 (non-vacuity)

The bound is informative for client `i` iff

    T · m_i(σ)  <  M_i := H(p_i) − ℓ_i

`M_i` = the information about `p_i` **not** already fixed by prior structure — i.e. what
the mechanism actually has to protect. Clean interpretation, and it makes D6 checkable.

## Corollary 2 (dual / budget form) — the headline claim, restated

Instead of "shave nats off a possibly-vacuous bound at fixed `U`", fix a **non-vacuous
target** `K_target < H(p_i)` and minimise total noise:

    U_fulcrum = Σ_i 2T/(K_target − ℓ_i)        U_uniform = 2nT/(K_target − max_i ℓ_i)

Reported quantity: **% of total noise budget saved at equal per-client privacy.**
Directly tied to utility, non-vacuous by construction, and independent of any
proportionality constant.

## Numerical findings so far

`κ=20`, `H(p_i)=2.94` nats at 20-level quantisation, `ℓ` saturates at `0.914`.

**(a) Under per-client observation the bound is vacuous at any deployable noise.**
Required `σ` for the mechanism term to consume half the margin:

| `T` | 10 | 25 | 50 | 100 |
|---|---|---|---|---|
| `σ` needed | 4.2 | 6.6 | 9.3 | 13.1 |

No model trains at `σ≈13`. **This is an impossibility result, not a tuning problem** — and
it explains v1's measured `ε` of 50 … 4×10⁶.

**(b) Aggregation restores non-vacuity.** Required `σ` with aggregation set `|A|`:

| `|A|` \ `T` | 10 | 25 | 50 | 100 |
|---|---|---|---|---|
| 1 | 4.34 | 6.86 | 9.70 | 13.72 |
| 4 | 2.17 | 3.43 | 4.85 | 6.86 |
| 10 | 1.37 | 2.17 | 3.07 | 4.34 |
| 50 | 0.61 | 0.97 | 1.37 | 1.94 |

Deployable at `|A| ≥ 10`, `T ≤ 25`. **The non-vacuous regime is exactly the one where
topology matters** — which is the paper's thesis, now actually earned.

**(c) The paper's linear proxy is the wrong shape, not just unnormalised.**
True `ℓ` saturates; `ℓ ∝ |G_i|` does not:

| `|G|` | 2 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| true `ℓ` | 0.616 | 0.726 | 0.814 | 0.869 | 0.893 | 0.907 |
| linear proxy | 0.616 | 0.924 | 1.540 | 3.080 | 6.161 | 15.40 |
| overstatement | 1.0× | 1.3× | 1.9× | 3.5× | 6.9× | **17×** |

At `|G|=50` the proxy returns 15.4 nats — above the entropy ceiling, hence impossible.
The v1 headline gains are largely an artifact of this over-dispersion.

**(d) Honest gain with the corrected `ℓ`.** Hierarchy with group sizes `[1,2,3]`:
budget saving at equal privacy is **6–28%**, ~10% at `K_target = 0.6·H`.
Real and defensible; far from v1's claimed 21.7% *nats* reduction, which measured
something else.

## Consequences for the paper's claims

- "up to 1.967 nats" — **withdrawn**; not well defined (D5) and vacuous (D6).
- "no measurable utility cost" — **withdrawn as stated**; at `σ ≈ 0.03–0.5` both arms add
  negligible noise, so equivalence was measuring the absence of noise. Must be re-tested
  at non-vacuous `σ`, where a real cost is expected and must be reported.
- "DP-SGD bounds the parameter channel" (from negative TADI lift) — **withdrawn**;
  a failed attack is not a bound, and at these `σ` the noise was not doing the bounding.
- Theorem V.3 / Corollary V.4 — **retained**, constants corrected.
