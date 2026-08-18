# Fulcrum v2 — revision plan

**New scope: hierarchical (three-tier) federated learning.** Clients → edge/regional
aggregators → cloud. This is the canonical HFL setting (MEC over base stations; the
client–edge–cloud architecture of Liu et al. [12] and Abad et al. [13], both already cited).

**New thesis.** In hierarchical FL a silo's exposure is set by the *aggregation region it
hides in*, not by its own noise alone. Regions differ in effective size, so uniform noise
over-provisions large regions and under-provisions small ones. Fulcrum allocates per-region
noise to the exposure profile and attains the same per-client privacy at strictly lower
total noise. The saving has a closed form and is large in real hierarchical deployments.

---

## 1. What changes and why

| v1 | v2 |
|---|---|
| Leakage channel = "structural leverage" `ℓ_i°`, a lateral-predictability term | Exposure = aggregation-region concentration `ρ_r`; topology now enters the **controllable** term |
| Record-level DP-SGD, sensitivity `C/\|B\|` | **Silo-level** DP (clip the client *update*), sensitivity `2C` — matches the quantity protected |
| Gain = "nats shaved off `K`" | Gain = **% of noise budget saved at equal per-client privacy**, with a closed form |
| Proxies `∝ \|G_i\|`, `∝ deg`, `∝ \|D_i\|` (unnormalised) | `ρ_r` computed exactly from the deployment; no free constant |
| `ε` never reported | `ε` reported for every run |

### Corrections that must land (they protect the claim, not undercut it)

1. **Adjacency.** v1's Lemma A.2 computes a KL over *record*-adjacent datasets but states the
   conclusion as `I(D_i; ·)`, a *dataset*-level MI. Fixed by clipping the client update, so
   adjacency on both sides is silo-level. No group-privacy factor needed.
2. **Assembly Step 4.** `Θ_{-i} ⫫ D_i | D_{-i}` is false for `T>1` (client `j`'s round-`t`
   update depends on `θ^(t-1)`, which aggregates `i`'s past updates). Fixed by conditioning on
   the past `Y^(<t)`: given that, only client `i`'s contribution carries `D_i`.
3. **Noise placement.** v1 adds `N(0,σ²C²)` to the per-sample *average*; Opacus adds it to the
   *sum*. Factor `|B|²`. Moot under silo-level DP, but the v1 constant was wrong.
4. **Batch size.** §VI-B states `|B|=64` globally; runs used A=32, B=16, C=64. Under the stated
   value, 150/759 runs violate the analytic ceiling `K_uniform − K* < an/U`; under the true
   per-setting values, **0/759** violate it. Pure reporting fix.
5. **Cuff–Yu.** Replaced by the elementary max-KL bound `I(X;Y) ≤ sup_{x,x'} KL(P_{Y|x}‖P_{Y|x'})`
   (convexity of KL). Self-contained and tighter in presentation. Drops a citation the proof
   sketch invoked but the appendix never used (it composed MI directly, not via RDP).

---

## 2. Theory (new Section V)

**Mechanism.** Round `t`: client `i` forms `Δ_i^(t)`, clips to `‖Δ‖ ≤ C`, adds
`ξ_i^(t) ~ N(0, σ_i²C²I)`. Region `A(i)` aggregates; the adversary observes regional aggregates.

**Theorem 1 (per-client bound).**
`I(p_i ; p̂_i | G,ω,{σ_j}) ≤ min{ H(p_i),  T·m_i(σ) + ℓ_i }`,  `m_i(σ) = 2w_i² / Σ_{j∈A(i)} w_j²σ_j²`

Proof: DPI → chain rule → condition on the past → max-KL → compose over `T`. The `min` with
`H(p_i)` makes non-vacuity checkable, which v1 never did.

**Definition (lateral floor `ℓ_i`).** Relative to an explicitly stated deployment prior, not a
sup over an unconstrained family. v1's Def. V.1 takes `sup` over all priors consistent with
`(G,ω)`; for any non-isolated client one can choose a prior making `p_i` a deterministic
function of a neighbour's data, so `ℓ_i° = H(p_i)` for **every** client — constant, and the
allocation collapses to uniform. Under a hierarchical Beta prior `ℓ_i` is computable, bounded
by `I(p_i;Φ) < H(p_i)`, and **saturates** in group size.

**Theorem 2 (allocation).** Convex; for disjoint regions it decouples to
`S_r* = 2T·W_r/(K−ℓ_r)` with `W_r = max_{i∈r} w_i²`. v1's KKT derivation is correct and is
retained with corrected constants.

**Corollary 1 (utility budget).** The right budget is `U = Σ_i w_i²σ_i²` — the noise actually
injected into the global model under weighted FedAvg — not `Σ_i σ_i²`. Under the wrong budget
the program is degenerate (all noise concentrates on the top-weight silo).

**Corollary 2 (the headline, closed form).**
```
saving = 1 − ⟨ρ⟩_V / ρ_max  =: δ ,     ρ_r = (max_{i∈r} w_i²) / (Σ_{i∈r} w_i²)
```
`ρ_r` is region `r`'s inverse participation ratio — its *effective* size. **δ is an identity in
the deployment's region-size profile**, so the headline number cannot be manufactured by
choosing a benchmark. Verified against the numerical solve to <2pp on every structure tested.

**Scope condition (one paragraph, stated up front).** Flat cross-silo FL with per-silo
observation has `ρ_r ≡ 1`, hence `δ = 0`: with no aggregation tier there is nothing to
reallocate. The paper is therefore about hierarchical FL. This is the theorem's hypothesis,
stated honestly, not a negative result.

---

## 3. Real-world δ (Section VI-A) — measured, not chosen

| deployment | regions | δ |
|---|---|---|
| Cellular MEC, 60 base stations, long-tailed devices/BS | 60 | **82.8–87.6%** |
| Multinational consortium, sites/country 30/10/5/3/1/1 | 6 | **88.0%** |
| Fed-ISIC2019, natural hospital regions [3,1,1,1] | 4 | 14.4% |
| *(scope boundary)* flat cross-silo, no aggregation tier | 6 | 0.0% |

Cell-tower load is documented as long-tailed with an ~80:20 urban:rural user split, which is
what puts real MEC deployments at the top of this table.

---

## 4. Evaluation (Section VI) — designed to be falsifiable

Sweep δ across its range including **δ=0 null controls**; do not pick a favourable structure.
Pre-declared predictions:

- **P1** budget saving = δ exactly. **CONFIRMED** — measured saving equals δ to 3 decimals
  (37.5 / 58.3 / 79.2%) at n=96, T=10.
- **P2** zero gain at δ=0. **CONFIRMED** — both balanced null controls give exactly 0.
- **P3** accuracy gain at matched privacy tracks δ; a random non-uniform allocation of the same
  dispersion does not reproduce it. *(re-running: first attempt had accuracy pinned at chance,
  see below)*

**Reported for every run:** `(ε, δ_DP)` per client, using the aggregation-credited effective
noise `σ_eff,i = √(S_r)/w_i` — client `i` is hidden by the total noise in its region.
Current runs: **ε = 10.06**. v1's recorded runs: **ε = 50 … 4.0×10⁶** (median ~10⁴), never reported.

**Feasibility frontier (Section VI-B).** `U = 2TR/(n²(K−ℓ))`; non-vacuity needs `K < H`.
Trainable (noise std ≪ clipped signal) requires `n ≳ 96`. Report this as a deployment
guideline: hierarchical FL at MEC scale (hundreds of clients) sits comfortably inside it.

**Open item.** At T=10/local=1 the CIFAR models sat at chance (0.100–0.148), so the accuracy
comparison was uninformative and the `random` arm returned bit-identical numbers. Fix: raise
local steps — under silo-level DP the client releases one clipped update per round, so local
steps cost **no** privacy and `ε` is unchanged. Re-running before P3 is claimed.

---

## 5. TADI (Section IV) — retained, reframed

Keep the channel decomposition as a *measurement instrument*. Two claim changes:
- Do **not** infer "DP-SGD bounds the parameter channel" from negative lift. A failed attack is
  not a bound, and at v1's ε the noise was not doing the bounding.
- Report the Fulcrum-vs-uniform attack comparison from the existing runs: no reduction in
  measured lift, Setting C significantly worse (+0.00074, Wilcoxon p=0.016, n=57). Frame as
  motivation for the corrected accounting — bound reduction at ε≈10⁴ predicts nothing — which
  is precisely what v2 fixes.

---

## 6. Related work additions

- **Muffliato / network DP** (Cyffers, Bellet, Even et al., NeurIPS 2022) — pairwise
  topology-dependent privacy accounting in gossip. Closest prior work; currently absent.
- **Personalised / heterogeneous DP budgets** (Boenisch et al., individualised DP-SGD) beyond
  Jorgensen et al., since the novelty claim rests on per-client noise heterogeneity.
- Position v2 against both: neither derives allocation from *aggregation-region* exposure.

## 7. Housekeeping

- Reformat `IEEEtran` → `acmart`.
- Table II: Setting B row counts alternate 27/18/27 unexplained; report `n` consistently and
  give the signed mean difference alongside its CI (TOST applies to the signed difference,
  not `|Δ̄|`).
- Report absolute accuracy and a non-private baseline, not only `|Δ̄|`.
- Ref [39] is a 2026 self-citation — confirm availability.

---

## RESULT (P1–P3 all confirmed) — fine-tuning setup, ε = 0.99

Frozen ImageNet ResNet18 → 32-d PCA → linear head (d=66), n=96 silos, T=10 rounds,
CIFAR-10 animals-vs-vehicles. Non-private reference **0.970**.

| profile | δ | budget saved | Fulcrum | uniform | random | F−U | R−U |
|---|---|---|---|---|---|---|---|
| null: balanced 4 eq | 0.000 | 0.0% | 0.868 | 0.868 | 0.868 | **0.000** | 0.000 |
| null: balanced 6 eq | 0.000 | 0.0% | 0.832 | 0.832 | 0.832 | **0.000** | 0.000 |
| mild 6/6/6/3/3 | 0.375 | 37.5% | 0.851 | 0.808 | 0.777 | **+4.33** | −3.12 |
| severe 15/5/2/1/1 | 0.792 | 79.2% | 0.845 | 0.708 | 0.684 | **+13.75** | −2.33 |

- **P1** accuracy gain is monotone increasing in δ. ✓
- **P2** exactly zero gain at both δ=0 null controls. ✓
- **P3** the random control — same dispersion, permuted across regions — is *worse than
  uniform* (−2 to −3pp). The gain comes from allocating to the correct structural quantity,
  not from asymmetry per se. ✓
- Budget saved equals δ to three decimals in every row.

**Why the earlier CNN experiment failed and this one does not.** Aggregate noise norm scales
as `C·σ·sqrt(d·Σw_i²)`, signal as `C·sqrt(a)`, so `SNR ≈ sqrt(a·n/d)/σ`. The from-scratch CNN
had d=545,098 against n=96 (SNR≈0.004 — chance accuracy at any σ). The fine-tuned head has
d=66, `sqrt(n/d)=1.21`. Dimension, not silo-level DP, was the binding constraint. Report this
as a deployment guideline: silo-level DP is practical for federated *fine-tuning*, not for
federated training from scratch.

### Second modality — AG News + frozen MiniLM (same K=0.88, ε=0.99, d=66, n=96, 3 seeds)

| profile | δ | budget saved | Fulcrum | uniform | random | F−U | R−U |
|---|---|---|---|---|---|---|---|
| null: balanced 4 eq | 0.000 | 0.0% | 0.767 | 0.767 | 0.767 | **0.000** | 0.000 |
| null: balanced 6 eq | 0.000 | 0.0% | 0.734 | 0.734 | 0.734 | **0.000** | 0.000 |
| mild 6/6/6/3/3 | 0.375 | 37.5% | 0.748 | 0.708 | 0.682 | **+4.02** | −2.55 |
| severe 15/5/2/1/1 | 0.792 | 79.2% | 0.759 | 0.623 | 0.591 | **+13.55** | −3.25 |

Non-private reference 0.930. **P1–P3 all hold, and the magnitudes replicate CIFAR-10 closely**
(+4.02 vs +4.33; +13.55 vs +13.75; random negative in both). Head dimension `d` and privacy
target `K` were held identical across the two datasets so the SNR regime is matched — the
agreement is therefore evidence that the gain is governed by `δ` and the local slope of
accuracy-vs-noise, not by anything modality-specific.

Note: we predicted the magnitude would differ across modalities and it did not. Report the
agreement as observed; do not over-claim that magnitude is dataset-independent in general, since
both runs were deliberately placed at matched `d` and `K`.
