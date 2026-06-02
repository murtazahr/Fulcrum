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
    """LightGBM gradient-boosted trees with cross-validated hyperparameter tuning.

    With small shadow corpora (a few hundred training examples) and high-
    dimensional features (per-channel feature widths in the tens), the
    default LightGBM hyperparameters overfit. Cross-validated grid search
    over the four most impactful hyperparameters (n_estimators,
    learning_rate, num_leaves, min_data_in_leaf) consistently selects
    simpler configurations on shadow training sets of this size, and
    improves transferability of the regressor to target features.

    Tuning is on by default; pass ``cv_tune=False`` to fall back to fixed
    hyperparameters (e.g., for ablation or smoke tests).

    Both ``fit`` and ``predict`` wrap the input array in a DataFrame with
    stable column names so LightGBM's ``feature_names_in_`` agrees at
    train and predict time and sklearn's validator stays quiet.
    """

    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 5
    random_state: int = 0
    cv_tune: bool = True
    cv_folds: int = 5
    _model: Any = None
    _best_params: Any = None
    _best_cv_score: Any = None

    @staticmethod
    def _wrap(X: np.ndarray):
        import pandas as pd
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "lightgbm not installed. Install with `pip install lightgbm` or use "
                "regressor='linear' / 'mlp'."
            ) from exc

        X_df = self._wrap(X)

        # Fall back to fixed hyperparams for very small corpora where CV
        # would split into folds smaller than ``min_data_in_leaf`` and
        # fits degenerate.
        n_per_fold = len(y) // max(self.cv_folds, 1)
        if not self.cv_tune or n_per_fold < 8:
            self._model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                min_data_in_leaf=self.min_data_in_leaf,
                random_state=self.random_state,
                verbose=-1,
            )
            self._model.fit(X_df, y)
            return

        from sklearn.model_selection import GridSearchCV, KFold

        # Compact grid focused on regularisation knobs that matter most
        # for small training sets: shallower trees and larger leaf-data
        # minimums prevent the classifier from memorising shadow quirks.
        param_grid = {
            "n_estimators":     [100, 200, 400],
            "learning_rate":    [0.05, 0.1],
            "num_leaves":       [7, 15, 31],
            "min_data_in_leaf": [5, 10, 20],
        }
        base = lgb.LGBMRegressor(random_state=self.random_state, verbose=-1)
        # KFold with shuffling so contiguous (run_id-ordered) shadow rows
        # don't cluster in folds; random_state fixed for reproducibility.
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        gs = GridSearchCV(
            base, param_grid,
            cv=kf,
            scoring="neg_mean_squared_error",
            n_jobs=1,     # LightGBM is itself parallelised; avoid oversubscription
            refit=True,
        )
        gs.fit(X_df, y)
        self._model = gs.best_estimator_
        self._best_params = dict(gs.best_params_)
        self._best_cv_score = float(-gs.best_score_)
        # One line of provenance per channel is useful when diagnosing
        # whether tuning is doing anything; printed by attack-eval driver.

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Regressor not fitted")
        return np.clip(self._model.predict(self._wrap(X)), 0.0, 1.0)


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
