# Fulcrum — Deployment & Execution Guide

End-to-end instructions for setting up Fulcrum on a single-GPU VM and running
the experiments. Written for the L40S target (48 GB VRAM, 32 vCPU, 236 GB RAM,
3 TB storage) but the steps work on any reasonably-equipped Linux machine with
an NVIDIA GPU.

Read this **once through before you start** — there's a manual step in the
dataset section (Fed-ISIC2019 download is gated by the ISIC Archive) that
takes time you'll want to kick off early.

---

## 0. Prerequisites

The VM should have:

| Requirement | Why |
|---|---|
| Linux (Ubuntu 22.04+ assumed) | Tested target |
| NVIDIA driver ≥ 535 + CUDA 12.x | Matches PyTorch 2.1+ wheels |
| Python 3.10 or 3.11 | `fulcrum` minimum, broadly compatible |
| Git | Clone the repo |
| ~50 GB free disk | Datasets (~12 GB) + dependencies (~10 GB) + experiment artifacts (~22 GB) + headroom |
| Outbound HTTPS to github.com, pypi.org, isic-archive.com | Code and data fetches |

Quick sanity check on the GPU:

```bash
nvidia-smi
# You should see "NVIDIA L40S" and CUDA Version: 12.x
```

If `nvidia-smi` prints driver/version info and shows ~48 GB of memory, you're good.

---

## 1. Clone and set up the environment

```bash
# Clone
cd ~
git clone https://github.com/murtazahr/Fulcrum.git
cd Fulcrum

# One-shot environment setup — installs uv, creates .venv, installs fulcrum + Murmura + FLamby
bash scripts/setup_env.sh
source .venv/bin/activate
```

The `setup_env.sh` script does:

1. Installs `uv` (a fast Python package manager) if not present.
2. Creates `.venv/` with Python 3.10.
3. Installs `fulcrum` editable, which pulls in `torch`, `opacus`, `lightgbm`,
   `numpy`, `pandas`, etc. *and* Murmura from GitHub.
4. Clones FLamby into `.flamby_src/` and installs it editable (FLamby is not on PyPI).

Expected first-run time: **5–10 minutes** depending on network speed.

### Verify PyTorch sees the GPU

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected output:
```
CUDA available: True
Device: NVIDIA L40S
```

If `CUDA available: False`, the PyTorch wheel doesn't match your CUDA driver.
Fix:

```bash
uv pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
```

(Adjust `cu121` to match your CUDA major version: `cu118`, `cu124`, etc.)

### Verify Opacus, FLamby, Murmura

```bash
python -c "import opacus; import flamby; import murmura; import fulcrum; print('All imports OK')"
```

If any import fails, re-run `bash scripts/setup_env.sh` — it's idempotent.

### Run the standalone math tests (no datasets needed)

```bash
pip install pytest
pytest tests/ -v
```

You should see ~30 tests pass across `test_allocation.py`, `test_topology.py`,
`test_tadi.py`. If any of these fail, **stop and report the error** — they are
math sanity checks that should always pass. Running 775 GPU experiments on top
of broken math wastes a week of compute.

---

## 2. Download the datasets

The repo expects datasets under `data/`. Each setting has its own subdirectory.

### CIFAR-10 (Setting C) — automatic, ~200 MB

Triggered by `download_data.py`:

```bash
python scripts/download_data.py --skip-isic
```

You should see torchvision's progress bar followed by:
```
==> CIFAR-10 → /home/.../Fulcrum/data/cifar10
    OK
==> Fed-Heart-Disease → /home/.../Fulcrum/data/fed_heart_disease
    OK (199 samples in center 0, train split)
```

### Fed-Heart-Disease (Setting B) — automatic, ~1 MB

Already triggered by the same `download_data.py` call above. FLamby handles it.

### Fed-ISIC2019 (Setting A) — **manual, ~9 GB**

This dataset is gated behind the ISIC Archive. Steps:

1. **Create a free account** at https://challenge.isic-archive.com/
2. Navigate to the 2019 challenge and download three files:
   - `ISIC_2019_Training_Input.zip` (~9 GB)
   - `ISIC_2019_Training_GroundTruth.csv`
   - `ISIC_2019_Training_Metadata.csv`
3. Move all three into `data/fed_isic2019/`:
   ```bash
   mkdir -p data/fed_isic2019
   mv ~/Downloads/ISIC_2019_Training_Input.zip data/fed_isic2019/
   mv ~/Downloads/ISIC_2019_Training_GroundTruth.csv data/fed_isic2019/
   mv ~/Downloads/ISIC_2019_Training_Metadata.csv data/fed_isic2019/
   ```
4. Run the FLamby preprocessing scripts:
   ```bash
   cd .flamby_src/flamby/datasets/fed_isic2019/dataset_creation_scripts
   python download_isic.py --output-folder ../../../../../data/fed_isic2019
   python resize_images.py --input-folder ../../../../../data/fed_isic2019 \
                           --output-folder ../../../../../data/fed_isic2019/resized
   cd ~/Fulcrum
   ```
5. Verify:
   ```bash
   ls data/fed_isic2019/resized | head
   # Should show per-center subdirectories (center_0, center_1, ...)
   ```
6. Re-run the script to confirm FLamby sees it:
   ```bash
   python scripts/download_data.py
   # Should now print: "==> Fed-ISIC2019 → ...  (already present)"
   ```

If you want to skip Setting A initially (the manual step takes time), you can
run Settings B and C first — they don't need ISIC.

---

## 3. End-to-end smoke test

Before kicking off the long sweeps, run **one canonical config** to verify the
whole pipeline works on this VM. Setting B is the fastest (~1 minute on L40S):

```bash
python scripts/run_experiment.py configs/setting_b_canonical.yaml
```

Expected output (last few lines):

```
{
  "run_id": "...",
  "status": "done",
  "test_accuracy": 0.7...,
  "test_loss": 0.5...,
  "K_star": 1.2..,
  "K_uniform": 1.5..
}
```

Verify the artifacts landed:

```bash
fulcrum status                  # should show one done row
ls runs/                        # one subdirectory named with the run_id
ls runs/<run_id>/               # config.json, features.npz, metrics.json
```

If this works, you're cleared to run the full sweeps.

### Common smoke-test failures

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleValidator` error from Opacus about BatchNorm | Model has incompatible layers | Should not happen in the shipped models — file an issue if you see it |
| `CUDA out of memory` | Batch size too large for some operation | Reduce `training.batch_size` in the config (default 32 → try 16) |
| Hangs at "Running config=..." | Dataset not found or FLamby reading slowly | Check `data/` exists and has expected contents |
| `RuntimeError: CUDA error: invalid device function` | PyTorch / CUDA version mismatch | Reinstall torch as in §1 |

---

## 4. Run the experimental sweeps

Order matters — run the cheap-and-revealing sweeps first so you discover
problems early. Approximate wall-clock estimates assume the L40S; halve them if
you have a faster GPU, double if slower.

### Sweep 1 — η-sweep on Setting C (cheapest, ~3 hours, 30 runs)

This is the **central topology-coupling experiment** and includes the IID-null
condition at η=0. Run it first because:

- It's small enough that any pipeline issues surface within an hour.
- The η=0 condition is the sanity check from Stage 2 design — if Fulcrum
  doesn't collapse to uniform allocation here, something is wrong.

```bash
python scripts/run_factorial.py sweeps/eta_sweep_setting_c.yaml
```

Watch progress in another terminal:

```bash
watch -n 5 'fulcrum status --setting C | tail -10'
```

You should see the `done` count climb from 0 to 30. If you see `failed`
entries, check `runs/<run_id>/log.txt` for the stack trace.

Generate the figure once it's done:

```bash
python scripts/analyze.py eta-sweep
# → analysis/eta_sweep_setting_c.{png,pdf}
```

The PNG should show **two curves that coincide at η=0** and **diverge as η
grows** — topology-aware allocation lying below uniform. If the curves don't
diverge meaningfully, the Setting C model may need more training rounds or a
different leverage proxy; come back with the figure and we'll diagnose.

### Sweep 2 — Pareto on Setting B (~2 hours, 108 runs)

Smallest model (logistic regression on tabular), so fastest of the three Pareto
sweeps:

```bash
python scripts/run_factorial.py sweeps/pareto_setting_b.yaml
python scripts/analyze.py pareto --setting B
```

### Sweep 3 — Pareto on Setting C (~9 hours, 108 runs)

```bash
python scripts/run_factorial.py sweeps/pareto_setting_c.yaml
python scripts/analyze.py pareto --setting C
```

### Sweep 4 — Pareto on Setting A (~54 hours, 108 runs) — needs ISIC

This is the realism-anchor experiment. **Long-running** because of the ResNet-class
model on dermoscopy images. Run it inside `tmux` or `screen` so a dropped SSH
session doesn't kill it:

```bash
tmux new -s fulcrum
source .venv/bin/activate
python scripts/run_factorial.py sweeps/pareto_setting_a.yaml
# Detach with Ctrl-b d; reattach with: tmux attach -t fulcrum
```

Monitor:

```bash
fulcrum status --setting A
nvidia-smi                       # confirm GPU utilization > 50%
df -h .                          # ensure you're not running out of disk
```

### Resuming after interruption

The runner is **idempotent** by config hash. If a sweep is interrupted (kernel
panic, you killed the tmux session, etc.), just re-run the same factorial
command. Already-completed runs are skipped automatically.

```bash
python scripts/run_factorial.py sweeps/pareto_setting_a.yaml
# "Sweep expanded to 108 configs ... skipped: 47, submitted: 61, failed: 0"
```

To re-attempt failed runs only:

```bash
python scripts/run_factorial.py sweeps/pareto_setting_a.yaml --retry-failed
```

To preview without running:

```bash
python scripts/run_factorial.py sweeps/pareto_setting_a.yaml --dry-run
```

---

## 5. Analysis: generate the headline figures

Once the sweeps are done:

```bash
# Per-setting Pareto figures (PNG + PDF in analysis/)
python scripts/analyze.py pareto --setting A
python scripts/analyze.py pareto --setting B
python scripts/analyze.py pareto --setting C

# η-sweep figure
python scripts/analyze.py eta-sweep

# Quick text summary across all settings
python scripts/analyze.py summary

# Export Parquet tables for downstream manuscript work
python scripts/analyze.py export-table --setting C --out analysis/setting_c.parquet
```

Each `analyze pareto` call prints a headline number:

```
Topology-aware advantage (area between curves): 0.0427
  topology-aware Pareto points: 8
  uniform Pareto points:        6
```

Positive area = topology-aware allocation dominates uniform on the Pareto front
(the desired outcome from Theorem 2 + Corollary 3). Numbers near zero mean
either:

- Leverage scores were near-uniform in this setting (expected for some configs).
- The defense's marginal value is small in this regime (deployment caveat — see
  `docs/defense_design.md` §4.7).

---

## 6. Sending results back for Stage 7 + 8

When you want me to do the statistical analysis and start the manuscript
rewrite, send back:

| Artifact | Purpose | Approx size |
|---|---|---|
| `experiments.db` | All run metadata + summary metrics — the queryable source of truth | ~5–10 MB |
| `analysis/` directory (figures + Parquet exports) | The figures themselves + Parquet tables | ~5–20 MB |
| `runs/*/metrics.json` (just these, not the npz files) | Detailed per-run metrics — needed for some analysis steps | ~10 MB |
| Stdout/stderr from any failed runs | For debugging | varies |

Compress and ship with:

```bash
tar -czf fulcrum_results.tar.gz experiments.db analysis/ \
  $(find runs -name 'metrics.json')
```

Upload the tar somewhere I can fetch (SCP back to your laptop, S3, GitHub
release, etc.) and tell me the URL or path. **Do not include `runs/*/features.npz`**
— those are large (50–100 MB total) and only needed for re-running the TADI
attack, which we'll handle on the VM.

For the **shadow → TADI evaluation pipeline** (Stage 7 work), I'll add a
follow-up commit with the relevant analysis script; you'll re-pull and run
it on the VM where the features are, then send back the post-TADI metric
tables.

---

## 7. Compute and storage budget at a glance

| Sweep | Runs | Per-run | Wall clock | Disk |
|---|---|---|---|---|
| η-sweep Setting C | 30 | ~6 min | ~3 hr | ~150 MB |
| Pareto Setting B | 108 | ~1 min | ~2 hr | ~150 MB |
| Pareto Setting C | 108 | ~5 min | ~9 hr | ~600 MB |
| Pareto Setting A | 108 | ~30 min | ~54 hr | ~5 GB |
| **Total** | **354** | | **~70 hr (~3 days)** | **~6 GB** |

Total ~3 days continuous; budget **1 week wall clock** with iteration.
Storage well under the 3 TB available.

---

## 8. Troubleshooting

### "Out of memory" during ResNet-class training (Setting A)

Reduce batch size in `configs/setting_a_canonical.yaml`:

```yaml
training:
  batch_size: 16   # was 32
```

Re-run the sweep — already-completed runs are skipped, only failed ones retry.

### Disk filling up

Check what's growing:

```bash
du -sh data/ runs/ analysis/ .venv/ .flamby_src/
```

`runs/` grows ~50–100 KB per run; not a concern at 354 runs total.
`data/` is dominated by Fed-ISIC2019 (~9 GB raw + ~3 GB resized).

### Process killed by OOM on the system side

Setting A's ResNet has high RAM peak during evaluation. If `dmesg` shows
OOM kills, reduce evaluation batch size by editing `runner.py` line ~231
(`per_client_test_loaders`) — change `batch_size=cfg.training.batch_size` to a
fixed smaller value.

### A specific run keeps failing

```bash
fulcrum status --status failed
cat runs/<run_id>/log.txt | tail -50
```

If it's a deterministic failure on a specific config, share the log and the
run_id with me and I'll patch.

### Want to restart from scratch

**Destructive — only do this if you really want to wipe results:**

```bash
rm experiments.db
rm -rf runs/ analysis/
```

The next factorial call will re-run everything.

---

## 9. What to do next

After Sweep 1 (η-sweep) completes:

1. Scp the figure (`analysis/eta_sweep_setting_c.png`) to your laptop and look
   at it — the IID-null at η=0 should show curves coinciding.
2. If it looks right, kick off the rest of the sweeps.
3. If it looks off (curves diverge at η=0, or topology-aware lies *above* uniform
   anywhere), share the figure + `experiments.db` and I'll diagnose before you
   commit a week of compute to a broken pipeline.

After all sweeps complete:

1. Generate all four figures (`pareto --setting {A,B,C}` + `eta-sweep`).
2. Send the bundle described in §6.
3. We move to Stage 7 (statistical significance, additional figure polish,
   shadow→TADI evaluation) and Stage 8 (manuscript rewrite).
