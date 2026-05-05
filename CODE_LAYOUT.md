# NSAV — Implementation

Code for the rebuild of *Network Structures as an Attack Surface*. Extends Murmura
(https://github.com/Cloudslab/murmura) with DP-SGD, topology-aware noise
allocation, FLamby + CIFAR-η data adapters, and the TADI attack.

Design specification lives in [../00_overview.md](../00_overview.md).

## Layout

```
code/
├── pyproject.toml          # fulcrum package definition + dependencies
├── fulcrum/                   # our extensions on top of Murmura
│   ├── data/               # Fed-ISIC2019, Fed-Heart-Disease, CIFAR-10+η adapters
│   ├── topology/           # hierarchical + line generators (Murmura ships ring/fully/erdos/k-regular)
│   ├── dp/                 # DP-SGD wrapper, leverage computation, Theorem 2 allocation
│   ├── attacks/            # TADI implementation: features, shadow training, regressors
│   ├── analysis/           # privacy/utility metrics, Pareto frontier
│   └── cli.py              # fulcrum command-line entrypoint
├── scripts/
│   ├── setup_env.sh        # creates venv, installs deps, downloads FLamby
│   ├── download_data.py    # fetches Fed-ISIC2019, Fed-Heart-Disease, CIFAR-10
│   ├── run_experiment.py   # runs one experiment from a YAML config
│   ├── run_factorial.py    # submits/manages many runs, skips completed
│   └── analyze.py          # loads experiments.db → analysis tables + figures
├── configs/                # YAML experiment configs (one per (setting × condition))
└── tests/                  # unit tests
```

Output layout (created at runtime):

```
data/                       # downloaded datasets
runs/<run_id>/              # per-run artifacts (config.json, features.npz, metrics.json, log.txt)
experiments.db              # SQLite master log of all runs
analysis/                   # derived tables (Parquet), figures (PNG/PDF)
```

## Quick start

```bash
cd code
bash scripts/setup_env.sh           # creates .venv, installs fulcrum + deps + FLamby
source .venv/bin/activate
python scripts/download_data.py     # fetches Fed-ISIC2019, Fed-Heart-Disease, CIFAR-10
fulcrum run configs/setting_a_canonical.yaml
```

## Stage status

This is Stage 5 (experimental design) → Stage 6 (implementation) work. See
[../decisions_log.md](../decisions_log.md) for the chronological commitments.
