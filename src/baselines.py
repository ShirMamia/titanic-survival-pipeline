"""Classical ML baselines for comparison with the PyTorch MLP.

Trains Logistic Regression, Random Forest, and Gradient Boosting classifiers
on the same preprocessed feature matrix used by the MLP.  Each model is
selected via cross-validated grid search on the training split only; the
held-out test split is used exactly once for the final comparison.

These models are for **comparison only**.  The required submission model
remains the PyTorch MLP defined in ``src/model.py``.
Results are saved to ``artifacts/baseline_metrics.json`` by ``train.py``.
"""
from __future__ import annotations

import logging

import numpy as np
import scipy.sparse
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .evaluate import compute_metrics

logger = logging.getLogger(__name__)

_CV_SCORING = "roc_auc"
_CV_SPLITS = 5


def _to_dense(X) -> np.ndarray:
    """Convert sparse matrix to dense; pass dense arrays through unchanged."""
    if scipy.sparse.issparse(X):
        return X.toarray()
    return np.asarray(X, dtype=float)


def _validate_inputs(X_train, y_train, X_test, y_test) -> None:
    """Raise ValueError with a clear message for any shape / label mismatch."""
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"X_train and X_test must have the same number of columns; "
            f"got {X_train.shape[1]} vs {X_test.shape[1]}."
        )
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train has {len(X_train)} rows but y_train has {len(y_train)} labels."
        )
    if len(X_test) != len(y_test):
        raise ValueError(
            f"X_test has {len(X_test)} rows but y_test has {len(y_test)} labels."
        )
    for name, y in (("y_train", y_train), ("y_test", y_test)):
        y_arr = np.asarray(y, dtype=float).ravel()
        if not np.all(np.isfinite(y_arr)):
            raise ValueError(f"{name} contains NaN or infinite values.")
        bad = set(np.unique(y_arr.astype(int))) - {0, 1}
        if bad:
            raise ValueError(
                f"{name} must contain only binary labels {{0, 1}}; "
                f"found unexpected values: {sorted(bad)}."
            )


def train_baselines(
    X_train,
    y_train: np.ndarray,
    X_test,
    y_test: np.ndarray,
    *,
    seed: int = 42,
) -> dict:
    """Train three classical classifiers with CV-tuned hyperparameters.

    Cross-validated grid search is performed **on the training split only**.
    The held-out test split is used exactly once for the final comparison.

    Parameters
    ----------
    X_train, y_train : preprocessed training features and labels.
    X_test, y_test   : preprocessed held-out test features and labels.
    seed             : random state for CV splits and estimators.

    Returns
    -------
    dict keyed by model name.  Each value contains the test metrics from
    ``compute_metrics`` plus ``best_params``, ``best_cv_score``,
    ``cv_scoring``, and ``n_cv_splits``.
    """
    y_train = np.asarray(y_train).astype(int).ravel()
    y_test = np.asarray(y_test).astype(int).ravel()

    _validate_inputs(X_train, y_train, X_test, y_test)

    X_train_dense = _to_dense(X_train)
    X_test_dense = _to_dense(X_test)

    cv = StratifiedKFold(n_splits=_CV_SPLITS, shuffle=True, random_state=seed)

    # Each spec: estimator created fresh (avoids shared-state mutations),
    # param_grid, and whether sparse input is acceptable.
    model_specs = [
        (
            "logistic_regression",
            LogisticRegression(solver="lbfgs", max_iter=2000, random_state=seed),
            {"C": [0.1, 1.0, 10.0]},
            True,   # sparse_ok
        ),
        (
            "random_forest",
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            {
                "n_estimators": [200, 300],
                "max_depth": [3, 5, None],
                "min_samples_leaf": [2, 3, 5],
            },
            False,  # needs dense
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(random_state=seed),
            {
                "n_estimators": [100, 150],
                "learning_rate": [0.03, 0.05, 0.10],
                "max_depth": [2, 3],
            },
            False,  # needs dense
        ),
    ]

    results: dict = {}
    for model_name, estimator, param_grid, sparse_ok in model_specs:
        logger.info(
            "Tuning %s via %d-fold CV (scoring=%s)...",
            model_name, _CV_SPLITS, _CV_SCORING,
        )
        X_tr = X_train if sparse_ok else X_train_dense
        X_te = X_test if sparse_ok else X_test_dense

        search = GridSearchCV(
            estimator,
            param_grid=param_grid,
            scoring=_CV_SCORING,
            cv=cv,
            refit=True,
            n_jobs=-1,
        )
        search.fit(X_tr, y_train)

        y_prob = search.best_estimator_.predict_proba(X_te)[:, 1]
        test_metrics = compute_metrics(y_test, y_prob, threshold=0.5)

        results[model_name] = {
            **test_metrics,
            "best_params": search.best_params_,
            "best_cv_score": round(float(search.best_score_), 4),
            "cv_scoring": _CV_SCORING,
            "n_cv_splits": _CV_SPLITS,
        }
        logger.info(
            "%s — best CV %s=%.4f | test F1=%.4f, AUC=%s",
            model_name, _CV_SCORING, search.best_score_,
            test_metrics["f1"],
            f"{test_metrics['roc_auc']:.4f}" if test_metrics["roc_auc"] else "n/a",
        )

    return results
