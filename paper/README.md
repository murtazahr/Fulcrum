# Fulcrum manuscript

Source for the paper *Fulcrum: Topology-Aware Differential Privacy
Allocation for Federated Learning*, targeting ACM Transactions on
Privacy and Security (TOPS).

## Layout

```
paper/
├── main.tex              top-level document, includes everything below
├── references.bib        bibliography (ALL entries verified — see ../docs/references.md)
├── Makefile              latexmk-based build
├── sections/
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_threat_model.tex
│   ├── 04_attack.tex            TADI, the attack
│   ├── 05_defense.tex           Theorems 1+2, Corollaries 1-3, Proposition 1
│   ├── 06_experiments.tex       PLACEHOLDER figures + tables
│   ├── 07_discussion.tex
│   ├── 08_conclusion.tex
│   └── appendix_proofs.tex      Full proofs of Theorems 1 and 2
└── figures/                     PLACEHOLDER — populate from ../analysis/
```

## Build

```bash
make             # produces main.pdf
make watch       # continuous rebuild as you edit
make clean       # remove .aux/.log/etc., keep main.pdf
make distclean   # remove everything, including main.pdf
```

Requires a TeX distribution with `latexmk` (TeX Live full, MacTeX, MikTeX
with the appropriate packages).

## Placeholders to fill once experiments complete

Search for `\PLACE{...}` in any `.tex` file. Each marker includes a short
description of what should replace it. In rough priority order:

1. **§6.3 Figure 1** — η-sweep figure across four topologies (run
   `python scripts/analyze.py eta-sweep` after the η-sweep completes).
2. **§6.4 Figure 2** — Privacy-utility Pareto frontier per setting.
3. **§6.5 Figure 3** — TADI channel-ablation bar chart.
4. **§6.6 Table 2** — Robustness ablations (regressor, feature set, shadow size).
5. **§5.7 Table 1** — Replace qualitative gap entries with measured values.
6. **§6.7** — Secure-aggregation extension results.
7. **§6.8** — Cross-setting transfer validation.
8. **Abstract + Conclusion** — Headline empirical numbers.

## Citation discipline

Every entry in `references.bib` has been verified against primary sources
(ACM/IEEE/ArXiv/journal pages) as part of the manuscript-writing
methodology — see `../docs/references.md` for verification provenance.
**Do not add references without verifying them first**: search the
primary source, confirm the venue/year/authors, and append a note to
`../docs/references.md` describing the verification step.

When adding new claims that need backing:
- If a verified reference covers the claim, use it.
- If no verified reference covers the claim, do NOT invent one — flag
  the claim with `\PLACE{cite needed: ...}` and add the verification to
  the to-do list.
