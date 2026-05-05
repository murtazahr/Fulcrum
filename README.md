# Fulcrum

**Topology-aware differential privacy allocation for federated learning.**

Fulcrum is a research codebase implementing the methodology and experiments for
a paper on topology-conditional distribution inference in differentially-private
federated learning. The defense is a **balanced min-max optimal noise
allocation** — under fixed utility budget, per-client DP noise is allocated to
high-leverage clients (those whose position in the topology + organizational
structure exposes them to more lateral leakage), driving every client's
worst-case mutual-information bound to the same value $K^\star$.

The name "Fulcrum" reflects this balance point: the lever where uneven privacy
loads find equilibrium under structural leverage.

## Headline contributions

- **Theorem 1.** A per-client conditional mutual-information bound on
  topology-aware adversaries under DP-SGD: the bound decomposes additively into
  a controllable mechanism-induced term $T_{\max} C^2 / (2\sigma_i^2|B|^2)$ and
  an uncontrollable prior-coupling term $\ell_i^\circ$ (structural leverage).
- **Theorem 2.** A closed-form min-max optimal noise allocation
  $\sigma_i^{*2} = a/(K^\star - \ell_i^\circ)$ with strict improvement over
  uniform DP-SGD allocation when leverage scores are non-uniform.
- **TADI.** A topology-aware distributional inference attack with four channel
  ablations to isolate the marginal privacy contribution of topology
  ($\mathcal{G}$) versus organizational labels ($\omega$) versus parameter
  observation ($\Theta$).
- **Empirical evaluation.** Three settings — Fed-ISIC2019 (realistic
  hierarchical), Fed-Heart-Disease (forward-looking decentralized), and
  CIFAR-10 with parametric $\eta$ coupling (statistical vehicle).

## Repository layout

```
Fulcrum/
├── docs/                     research design + theorem proofs + decisions log
│   ├── overview.md
│   ├── research_framing.md
│   ├── threat_model_partitioning.md
│   ├── attack_design.md
│   ├── defense_design.md
│   ├── references.md         every cited paper, verified against primary sources
│   └── decisions_log.md      chronological record of design commitments
├── fulcrum/                  the Python package
│   ├── data/                 Setting A/B/C data adapters (Fed-ISIC2019, Fed-Heart-Disease, CIFAR-10+η)
│   ├── topology/             line + hierarchical generators (Murmura ships ring/fully/erdos/k-regular)
│   ├── dp/                   leverage proxies + Theorem 2 allocation + Opacus per-client wrapper
│   ├── attacks/              TADI: feature extraction, shadow training, regressors, metrics
│   ├── analysis/             Pareto frontier extraction + figure generation
│   ├── models.py             per-setting model factories (Opacus-compatible)
│   ├── runner.py             end-to-end FL training with DP + feature collection
│   ├── storage.py            SQLite experiments DB + per-run NPZ/JSON artifact I/O
│   ├── config.py             dataclass-based YAML config schema
│   └── cli.py                `fulcrum run|status|fit-tadi` commands
├── scripts/                  shell-callable wrappers
│   ├── setup_env.sh          venv + dependencies + FLamby from source
│   ├── download_data.py      CIFAR-10 + Fed-Heart-Disease auto; Fed-ISIC2019 manual instructions
│   ├── run_experiment.py     single-config runner
│   ├── run_factorial.py      sweep manager with hash-based skip
│   └── analyze.py            Pareto + η-sweep figures + Parquet exports
├── configs/                  reference configs per setting
├── sweeps/                   factorial spec YAMLs
├── tests/                    standalone math verification (allocation, TADI metrics, topology, Pareto)
├── pyproject.toml            package definition
└── CODE_LAYOUT.md            developer-oriented layout reference
```

## Quick start

```bash
# Set up environment (uses uv)
bash scripts/setup_env.sh
source .venv/bin/activate

# Download datasets (CIFAR-10 + Fed-Heart-Disease auto; Fed-ISIC2019 manual)
python scripts/download_data.py

# Run one canonical experiment
python scripts/run_experiment.py configs/setting_b_canonical.yaml
fulcrum status

# Run a sweep
python scripts/run_factorial.py sweeps/eta_sweep_setting_c.yaml

# Generate the headline figures
python scripts/analyze.py pareto --setting C
python scripts/analyze.py eta-sweep
```

## Design references

Every design decision is documented in `docs/`. Notable entries:

- `docs/research_framing.md` — what the paper claims and what it deliberately does not.
- `docs/threat_model_partitioning.md` — the formal adversary $\mathcal{A} = (\mathcal{K}, \mathcal{O}, \mathcal{I}, \mathcal{R})$, the three experimental settings, and the IID-null calibration.
- `docs/defense_design.md` — Theorem 1 + Theorem 2 statements and full proofs, plus the SBM-derived leverage corollary and the heuristic degree-based proxy.
- `docs/decisions_log.md` — a chronological record of every commitment, including two real math errors caught by adversarial self-review (multiplicative→additive bound correction; MI subadditivity misuse in the bounded-degree corollary).

## Status

Implementation complete; experiments pending execution on a single L40S GPU
(48GB VRAM). Estimated wall-clock for the headline sweeps: ~1 week with iteration.

## Citation

If you use this work, please cite:

```bibtex
@article{rangwala2026fulcrum,
  title   = {Fulcrum: Topology-Aware Differential Privacy Allocation for Federated Learning},
  author  = {Rangwala, Murtaza and Sinnott, Richard O. and Buyya, Rajkumar},
  journal = {Submitted},
  year    = {2026},
}
```

## Authors

- Murtaza Rangwala (University of Melbourne)
- Richard O. Sinnott (University of Melbourne)
- Rajkumar Buyya (University of Melbourne)

## License

To be determined — see `LICENSE` once added.
