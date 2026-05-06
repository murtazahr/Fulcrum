"""Command-line interface for fulcrum.

Three subcommands:

- ``fulcrum run CONFIG.yaml``  — execute one experiment.
- ``fulcrum status``           — list runs in the experiments database.
- ``fulcrum fit-tadi SETTING`` — train a TADI regressor from completed shadow runs.

Factorial sweeps are managed by ``scripts/run_factorial.py``; that script reads
a sweep spec and calls :func:`fulcrum.runner.run_experiment` for each generated
config (skipping completed ones via the SQLite ``status`` field).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click


@click.group()
def main():
    """Fulcrum experiment CLI."""


@main.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--db-path", default="experiments.db", help="SQLite experiments DB path")
@click.option("--device", default=None, help="Override device (cuda/cpu)")
def run(config_path: Path, db_path: str, device: str | None):
    """Execute one experiment from a YAML config."""
    from fulcrum.config import load_config
    from fulcrum.runner import run_experiment

    cfg = load_config(config_path)
    click.echo(f"Running config={config_path.name} setting={cfg.setting} mode={cfg.experiment.mode}")
    result = run_experiment(cfg, db_path=db_path, device=device)
    click.echo(json.dumps(
        {k: v for k, v in result.items() if not isinstance(v, list)},
        indent=2, default=str,
    ))


@main.command()
@click.option("--db-path", default="experiments.db")
@click.option("--setting", default=None, help="Filter to a specific setting (A/B/C)")
@click.option("--mode", default=None, help="Filter to target/shadow")
@click.option("--status", default=None, help="Filter to queued/running/done/failed")
def status(db_path: str, setting: str | None, mode: str | None, status: str | None):
    """List runs in the experiments DB."""
    from fulcrum import storage as st

    rows = st.list_runs(db_path, setting=setting, mode=mode, status=status)
    if not rows:
        click.echo("No matching runs.")
        return
    click.echo(f"{'run_id':<18} {'setting':<8} {'mode':<8} {'status':<10} {'acc':<8} {'K*':<10}")
    for r in rows:
        acc = f"{r['test_accuracy']:.3f}" if r["test_accuracy"] is not None else "-"
        k = f"{r['K_star']:.3f}" if r["K_star"] is not None else "-"
        click.echo(f"{r['run_id']:<18} {r['setting']:<8} {r['mode']:<8} {r['status']:<10} {acc:<8} {k:<10}")


@main.command("fit-tadi")
@click.argument("setting", type=click.Choice(["A", "B", "C"]))
@click.option("--db-path", default="experiments.db")
@click.option("--runs-root", default="runs")
@click.option("--channel", default="A2_full", type=click.Choice(["A1", "A2_topo", "A2_org", "A2_full"]))
@click.option("--regressor", default="lightgbm", type=click.Choice(["lightgbm", "mlp", "linear"]))
@click.option("--out", default=None, help="Path to save the fitted regressor (joblib)")
def fit_tadi(setting: str, db_path: str, runs_root: str, channel: str, regressor: str, out: str | None):
    """Train a TADI regressor from completed shadow runs."""
    from fulcrum.attacks.shadow import load_shadow_dataset
    from fulcrum.attacks.tadi import TADI

    X, p, run_ids = load_shadow_dataset(db_path, setting=setting, runs_root=runs_root, channel=channel)
    if X.shape[0] == 0:
        click.echo(f"No completed shadow runs for setting={setting}, channel={channel}", err=True)
        sys.exit(1)
    click.echo(f"Loaded {X.shape[0]} shadow examples ({len(set(run_ids))} runs) X.shape={X.shape}")
    attack = TADI(regressor=regressor)
    attack.fit(X, p)
    click.echo(f"TADI fitted. (Save to {out!r} on demand — not auto-persisted.)")
    if out:
        try:
            import joblib
        except ImportError:
            click.echo("joblib not installed; cannot save", err=True)
            sys.exit(1)
        joblib.dump(attack, out)
        click.echo(f"Saved → {out}")


if __name__ == "__main__":
    main()
