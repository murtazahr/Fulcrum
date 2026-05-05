# NSAV Rebuild — Overview

This directory contains the redesigned methodology, theory, experiments, and
draft material for the NSAV manuscript. The original manuscript lives at
`../Paper/Tex Paper/paper.tex` and is considered rejection-worthy due to four
methodological flaws documented in `decisions_log.md`. This rebuild starts
from scratch.

## Working principles

- Every methodological choice is grounded in a real-world scenario or verified
  citation (see `references.md`). Synthetic constructions are explicitly labelled
  as ablations.
- Each design choice is stress-tested adversarially (proposed → attacked as
  reviewer → refined → repeat) before commitment.
- Research framing is one paragraph, narrow and defensible; the original paper's
  three-claim framing is retired.

## Stage status

| Stage | Title | Status | Document |
|---|---|---|---|
| 1 | Research framing | Closed | [`01_research_framing.md`](01_research_framing.md) |
| 2 | Threat model + partitioning | Closed | [`02_threat_model_partitioning.md`](02_threat_model_partitioning.md) |
| 3 | Attack design | Closed | [`03_attack_design.md`](03_attack_design.md) |
| 4 | Defense design + theoretical guarantee | Closed | [`04_defense_design.md`](04_defense_design.md) |
| 5 | Experimental design | Closed | `code/` (configs + sweeps + analysis) |
| 6 | Implementation + execution | In progress | — |
| 7 | Statistical analysis | Pending | — |
| 8 | Manuscript rewrite | Pending | — |

## Headline claim (Stage 1)

Even under DP-SGD, an adversary with topology + organizational knowledge can
recover sensitive-class membership at rate $X$ above a non-topology baseline,
on real federated benchmarks. We propose **topology-aware DP noise allocation
+ observation-window bounding** as a defense, prove a mutual-information bound
on what any adversary can recover under this defense, and characterize the
privacy-utility frontier empirically.

## Datasets

- **Setting A** — Fed-ISIC2019 (FLamby), 6 sites, hierarchical aggregation. Realism anchor.
- **Setting B** — Fed-Heart-Disease (FLamby), 4 sites, decentralized k-NN proximity. Forward-looking.
- **Setting C** — CIFAR-10 + Dirichlet($\alpha$) + parametric coupling $\eta$. Statistical vehicle.

## Open commitments

- Title rewrite (current title overclaims; defer to Stage 8).
- Coupling parameter $\eta$ formal definition (defer to Stage 5 once attack inputs are coded).
- Theoretical contribution targeted at **Medium tier**: MI bound
  $I(\hat{p}_i; p_i) \leq B(\sigma, T_{\max}, \mathcal{G})$ via RDP composition
  and Pinsker / KL–MI conversion.
