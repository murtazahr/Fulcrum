"""Pareto frontier extraction and area-between-curves computation.

The Pareto figure (one per setting) plots privacy bound $K^\\star$ on the X axis
vs. utility cost (test loss or 1 − accuracy) on the Y axis. Across the
$(U, T_{\\max})$ × allocation sweep, the topology-aware allocation should trace
a curve dominating (lying below-left of) the uniform allocation curve when
leverage scores are non-uniform — Corollary 3.

Functions here:

- :func:`pareto_frontier_indices`  — non-dominated points (lower X *and* lower Y).
- :func:`area_between_curves`      — scalar measure of the gap between two
  curves; the headline number for "how much does topology-aware allocation help."
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pareto_frontier_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return indices of non-dominated points (smaller is better on both axes).

    A point $(x_i, y_i)$ is dominated if some other $(x_j, y_j)$ has
    $x_j \\leq x_i$ and $y_j \\leq y_i$ with at least one strict inequality.

    Args:
        x: shape ``(N,)``. Lower = better (e.g., privacy bound $K^\\star$).
        y: shape ``(N,)``. Lower = better (e.g., test loss).

    Returns:
        Sorted indices of Pareto-optimal points (sorted by ``x``).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    keep = np.ones(n, dtype=bool)
    # Sort by x ascending, break ties by y ascending; iterate to drop dominated points
    order = np.lexsort((y, x))
    best_y = np.inf
    for idx in order:
        if y[idx] < best_y:
            best_y = y[idx]
        else:
            keep[idx] = False
    pareto_idx = np.where(keep)[0]
    return pareto_idx[np.argsort(x[pareto_idx])]


def area_between_curves(
    x_a: np.ndarray, y_a: np.ndarray,
    x_b: np.ndarray, y_b: np.ndarray,
    x_grid: np.ndarray | None = None,
) -> float:
    """Compute $\\int |y_a(x) - y_b(x)|\\, dx$ over the overlap of the two curves.

    Uses linear interpolation on a shared grid spanning the intersection of the
    two curves' x-ranges. If the curves don't overlap, returns 0.

    Args:
        x_a, y_a: first curve (sorted by ``x_a``).
        x_b, y_b: second curve (sorted by ``x_b``).
        x_grid: optional explicit grid; defaults to the union of inputs clipped
            to the overlap region.

    Returns:
        Non-negative scalar — area between the two curves.
    """
    x_a = np.asarray(x_a, dtype=np.float64)
    y_a = np.asarray(y_a, dtype=np.float64)
    x_b = np.asarray(x_b, dtype=np.float64)
    y_b = np.asarray(y_b, dtype=np.float64)

    lo = max(x_a.min(), x_b.min())
    hi = min(x_a.max(), x_b.max())
    if hi <= lo:
        return 0.0

    if x_grid is None:
        x_grid = np.unique(np.concatenate([x_a, x_b]))
        x_grid = x_grid[(x_grid >= lo) & (x_grid <= hi)]
        if x_grid.size < 2:
            x_grid = np.linspace(lo, hi, 50)

    ya_interp = np.interp(x_grid, x_a, y_a)
    yb_interp = np.interp(x_grid, x_b, y_b)
    diff = np.abs(ya_interp - yb_interp)
    # numpy.trapezoid is the post-2.0 spelling; fall back to trapz on older versions
    integrate = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    return float(integrate(diff, x_grid))


def extract_pareto_per_group(
    df: pd.DataFrame,
    group_col: str = "dp.allocation",
    x_col: str = "K_star",
    y_col: str = "final_loss",
) -> dict[str, pd.DataFrame]:
    """For each value of ``group_col``, extract the Pareto frontier subset.

    Returns a dict mapping group value → Pareto-frontier DataFrame.
    """
    out: dict[str, pd.DataFrame] = {}
    for value, sub in df.groupby(group_col):
        sub = sub.dropna(subset=[x_col, y_col])
        if sub.empty:
            continue
        idx = pareto_frontier_indices(sub[x_col].to_numpy(), sub[y_col].to_numpy())
        out[value] = sub.iloc[idx].copy()
    return out


def topology_aware_advantage(
    df: pd.DataFrame,
    x_col: str = "K_star",
    y_col: str = "final_loss",
    allocation_col: str = "dp.allocation",
) -> dict[str, float]:
    """Quantitative summary: area between topology-aware and uniform Pareto curves.

    Returns:
        Dict with keys ``area``, ``ta_pareto_size``, ``unif_pareto_size``.
        ``area`` is positive when topology-aware lies below uniform (the
        expected direction). NaN if either curve has < 2 points.
    """
    fronts = extract_pareto_per_group(df, group_col=allocation_col, x_col=x_col, y_col=y_col)
    if "topology_aware" not in fronts or "uniform" not in fronts:
        return {"area": float("nan"), "ta_pareto_size": 0, "unif_pareto_size": 0}
    ta = fronts["topology_aware"]
    un = fronts["uniform"]
    if len(ta) < 2 or len(un) < 2:
        return {"area": float("nan"), "ta_pareto_size": len(ta), "unif_pareto_size": len(un)}
    area = area_between_curves(
        ta[x_col].to_numpy(), ta[y_col].to_numpy(),
        un[x_col].to_numpy(), un[y_col].to_numpy(),
    )
    return {"area": area, "ta_pareto_size": len(ta), "unif_pareto_size": len(un)}
