"""TADI — Topology-Aware Distributional Inference attack.

The single-attack architecture from §3.1 of the design doc. Trains a regressor
$f_\\phi: (\\Theta_i, x_i) \\to \\hat p_i$ on shadow simulations (where $p_i$ is
known), then applies $f_\\phi$ to a target federation. Channel ablations
($\\mathcal{A}_1, \\mathcal{A}_2^{\\text{topo}}, \\mathcal{A}_2^{\\text{org}},
\\mathcal{A}_2^{\\text{full}}$) are selected by the ``channel`` argument to
:func:`~fulcrum.attacks.features.build_attack_features` upstream.

Three regressor backends:
- ``"lightgbm"`` (default; gradient-boosted trees, primary)
- ``"mlp"`` (PyTorch, robustness check)
- ``"linear"`` (sklearn ridge regression, robustness check)

Metrics computed by :func:`TADI.metrics`:
- **Calibration loss** $\\frac{1}{n}\\sum_i (\\hat p_i - p_i)^2$ — primary (Stage 2)
- **Top-k recovery** — fraction of top-$k$ true-$p_i$ clients correctly identified
- **AUROC** — only meaningful at $n \\geq 20$, used in Setting C
- **Attack lift** — $L_{\\text{cal}}(\\bar p) - L_{\\text{cal}}(\\hat p)$ where
  $\\bar p$ is the constant-mean baseline. Positive lift = client-level signal.

Design ref: Redesign/03_attack_design.md §3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


RegressorName = Literal["lightgbm", "mlp", "linear"]


# ---------------------------------------------------------------------------
# Metrics (pure-numpy — testable without training)
# ---------------------------------------------------------------------------

def calibration_loss(p_hat: np.ndarray, p_true: np.ndarray) -> float:
    """Mean squared error between predicted and true sensitive-class concentrations."""
    p_hat = np.asarray(p_hat, dtype=np.float64)
    p_true = np.asarray(p_true, dtype=np.float64)
    if p_hat.shape != p_true.shape:
        raise ValueError(f"shape mismatch: {p_hat.shape} vs {p_true.shape}")
    return float(np.mean((p_hat - p_true) ** 2))


def top_k_recovery(p_hat: np.ndarray, p_true: np.ndarray, k: int) -> float:
    """Fraction of the true top-$k$ clients that the adversary places in their predicted top-$k$."""
    p_hat = np.asarray(p_hat, dtype=np.float64)
    p_true = np.asarray(p_true, dtype=np.float64)
    n = p_true.size
    if k <= 0 or k > n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    top_true = set(np.argsort(p_true)[-k:].tolist())
    top_pred = set(np.argsort(p_hat)[-k:].tolist())
    return len(top_true & top_pred) / k


def auroc_membership(p_hat: np.ndarray, p_true: np.ndarray, threshold: float = 0.5) -> float:
    """AUROC for predicting $\\mathbb{1}[p_i > \\text{threshold}]$ from $\\hat p_i$.

    Only meaningful at $n \\geq \\sim 20$ (Stage 2 design).

    Returns 0.5 if all true labels collapse to one class (undefined AUROC).
    """
    p_hat = np.asarray(p_hat, dtype=np.float64)
    p_true = np.asarray(p_true, dtype=np.float64)
    y = (p_true > threshold).astype(np.int64)
    if y.sum() == 0 or y.sum() == y.size:
        return 0.5
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p_hat))


def constant_mean_baseline_loss(p_true: np.ndarray) -> float:
    """Loss of the trivial baseline $\\hat p_i = \\bar p$ for all $i$.

    Equal to the variance of $p_true$ (since it is the MSE-minimizing constant predictor).
    """
    p_true = np.asarray(p_true, dtype=np.float64)
    return float(np.var(p_true))


def attack_lift(p_hat: np.ndarray, p_true: np.ndarray) -> float:
    """Reduction in calibration loss vs the constant-mean baseline.

    Positive lift = the attack extracts client-level information beyond the
    federation mean. Under the IID null condition the lift should be near zero.
    """
    return constant_mean_baseline_loss(p_true) - calibration_loss(p_hat, p_true)


# ---------------------------------------------------------------------------
# Regressor backends
# ---------------------------------------------------------------------------

@dataclass
class _LightGBMBackend:
    """LightGBM gradient-boosted trees — primary regressor."""

    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 5
    random_state: int = 0
    _model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "lightgbm not installed. Install with `pip install lightgbm` or use "
                "regressor='linear' / 'mlp'."
            ) from exc
        self._model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_data_in_leaf=self.min_data_in_leaf,
            random_state=self.random_state,
            verbose=-1,
        )
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Regressor not fitted")
        return np.clip(self._model.predict(X), 0.0, 1.0)


@dataclass
class _LinearBackend:
    """Ridge regression — fallback / robustness check."""

    alpha: float = 1.0
    random_state: int = 0
    _model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import Ridge
        self._model = Ridge(alpha=self.alpha, random_state=self.random_state)
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Regressor not fitted")
        return np.clip(self._model.predict(X), 0.0, 1.0)


@dataclass
class _MLPBackend:
    """Small MLP via sklearn — robustness check.

    Kept on sklearn (not PyTorch) to avoid GPU bookkeeping for an attack that
    runs on tiny tabular data (a few hundred to a few thousand examples).
    """

    hidden_layer_sizes: tuple[int, ...] = (64, 32)
    max_iter: int = 500
    random_state: int = 0
    _model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        self._model = Pipeline([
            ("scale", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )),
        ])
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Regressor not fitted")
        return np.clip(self._model.predict(X), 0.0, 1.0)


def _make_backend(name: RegressorName, random_state: int = 0):
    if name == "lightgbm":
        return _LightGBMBackend(random_state=random_state)
    if name == "linear":
        return _LinearBackend(random_state=random_state)
    if name == "mlp":
        return _MLPBackend(random_state=random_state)
    raise ValueError(f"Unknown regressor: {name!r}")


# ---------------------------------------------------------------------------
# TADI main class
# ---------------------------------------------------------------------------

@dataclass
class TADI:
    """The TADI attack with channel-ablation support.

    Usage::

        attack = TADI(regressor="lightgbm")
        attack.fit(X_shadow, p_shadow)
        p_hat = attack.predict(X_target)
        metrics = attack.metrics(p_hat, p_target_ground_truth)
    """

    regressor: RegressorName = "lightgbm"
    random_state: int = 0
    _backend: Any = None

    def fit(self, X: np.ndarray, p: np.ndarray) -> "TADI":
        """Train on shadow data ``(X, p)``."""
        if X.shape[0] != p.shape[0]:
            raise ValueError(f"X and p row mismatch: {X.shape[0]} vs {p.shape[0]}")
        self._backend = _make_backend(self.regressor, random_state=self.random_state)
        self._backend.fit(X, p)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Apply the trained regressor to target features."""
        if self._backend is None:
            raise RuntimeError("TADI not fitted; call .fit() first")
        return self._backend.predict(X)

    @staticmethod
    def metrics(
        p_hat: np.ndarray,
        p_true: np.ndarray,
        top_k: int | None = None,
        auroc_threshold: float = 0.5,
    ) -> dict[str, float]:
        """Compute the standard TADI metric panel.

        Args:
            p_hat: predicted sensitive-class concentrations.
            p_true: ground-truth concentrations.
            top_k: ``k`` for top-$k$ recovery; defaults to $\\lceil n/3 \\rceil$.
            auroc_threshold: AUROC binary-label threshold (median of $p_true$ is
                a sensible default; this argument lets callers override).

        Returns:
            Dict with keys ``calibration_loss``, ``constant_mean_baseline``,
            ``attack_lift``, ``top_k_recovery``, ``auroc``.
        """
        p_hat = np.asarray(p_hat, dtype=np.float64)
        p_true = np.asarray(p_true, dtype=np.float64)
        n = p_true.size
        if top_k is None:
            top_k = max(1, n // 3)
        return {
            "calibration_loss": calibration_loss(p_hat, p_true),
            "constant_mean_baseline": constant_mean_baseline_loss(p_true),
            "attack_lift": attack_lift(p_hat, p_true),
            "top_k_recovery": top_k_recovery(p_hat, p_true, top_k),
            "auroc": auroc_membership(p_hat, p_true, auroc_threshold),
            "n_clients": int(n),
            "top_k": int(top_k),
        }
