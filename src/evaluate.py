"""Evaluation: metric computation and plot generation.

The metric functions return plain dicts (JSON-serialisable). The plotting
functions return matplotlib ``Figure`` objects so they can either be saved to
disk by ``train.py`` or rendered directly in Streamlit.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend; safe for scripts and Streamlit
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _validate_binary_inputs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce inputs shared by all public functions.

    Checks:
      * Both arrays are non-empty and have the same length.
      * ``y_true`` contains only binary labels {0, 1}.
      * ``y_prob`` contains finite values in [0, 1] (no NaN / inf).
      * ``threshold`` (if given) is in [0, 1].

    Returns
    -------
    (y_true_1d, y_prob_1d) as flat numpy arrays.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()

    if y_true.size == 0:
        raise ValueError("y_true is empty — nothing to evaluate.")
    if y_prob.size == 0:
        raise ValueError("y_prob is empty — nothing to evaluate.")
    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"y_true and y_prob must have the same length; "
            f"got {y_true.shape[0]} vs {y_prob.shape[0]}."
        )

    bad_labels = set(np.unique(y_true)) - {0, 1}
    if bad_labels:
        raise ValueError(
            f"y_true must contain only binary labels {{0, 1}}; "
            f"found unexpected values: {sorted(bad_labels)}."
        )

    if not np.all(np.isfinite(y_prob)):
        raise ValueError("y_prob contains NaN or infinite values.")
    if y_prob.min() < 0.0 or y_prob.max() > 1.0:
        raise ValueError(
            f"y_prob values must be in [0, 1]; "
            f"got range [{y_prob.min():.4f}, {y_prob.max():.4f}]."
        )

    if threshold is not None and not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1]; got {threshold}.")

    return y_true, y_prob


# --------------------------------------------------------------------------- #
# Metric functions
# --------------------------------------------------------------------------- #

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Return a dict of headline classification metrics.

    ``roc_auc`` is set to ``None`` if only one class is present in ``y_true``
    (AUC is undefined in that degenerate case) rather than raising.
    """
    y_true, y_prob = _validate_binary_inputs(y_true, y_prob, threshold)
    y_pred = (y_prob >= threshold).astype(int)

    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
    except ValueError:
        auc = None

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "n_samples": int(len(y_true)),
        "threshold": float(threshold),
    }


def classification_text_report(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> str:
    """Return sklearn's text classification report.

    ``labels=[0, 1]`` is passed explicitly so the report is always two-row
    even when only one class appears in ``y_true`` or ``y_pred``.
    """
    y_true, y_prob = _validate_binary_inputs(y_true, y_prob, threshold)
    y_pred = (y_prob >= threshold).astype(int)
    return classification_report(
        y_true, y_pred,
        labels=[0, 1],
        target_names=config.CLASS_NAMES,
        zero_division=0,
    )


def tune_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, n_steps: int = 81
) -> tuple[float, float]:
    """Sweep thresholds in [0.10, 0.90] and return the one that maximises F1.

    Parameters
    ----------
    y_true  : ground-truth binary labels (0/1).
    y_prob  : predicted probabilities for the positive class.
    n_steps : number of candidate thresholds to evaluate (must be >= 2).

    Returns
    -------
    (best_threshold, best_f1) — both rounded to 4 decimal places.
    """
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2; got {n_steps}.")
    y_true, y_prob = _validate_binary_inputs(y_true, y_prob)
    thresholds = np.linspace(0.10, 0.90, n_steps)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return round(best_t, 4), round(best_f1, 4)


# --------------------------------------------------------------------------- #
# Plotting functions
# --------------------------------------------------------------------------- #

def plot_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """Return a confusion-matrix heatmap figure."""
    y_true, y_prob = _validate_binary_inputs(y_true, y_prob, threshold)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=config.CLASS_NAMES, rotation=20, ha="right")
    ax.set_yticks([0, 1], labels=config.CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray):
    """Return a ROC-curve figure, or ``None`` if AUC is undefined."""
    y_true, y_prob = _validate_binary_inputs(y_true, y_prob)
    if len(np.unique(y_true)) < 2:
        return None

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_training_curves(history) -> plt.Figure:
    """Return train/val loss + val accuracy curves from a TrainHistory.

    Raises
    ------
    ValueError
        If ``history`` is missing required attributes, or if the loss/accuracy
        lists have different lengths or are empty.
    """
    for attr in ("train_loss", "val_loss", "val_acc"):
        if not hasattr(history, attr):
            raise ValueError(f"history is missing required attribute '{attr}'.")
    n = len(history.train_loss)
    if n == 0:
        raise ValueError("history.train_loss is empty; no training epochs recorded.")
    if len(history.val_loss) != n or len(history.val_acc) != n:
        raise ValueError(
            "history.train_loss, val_loss, and val_acc must all have the same length."
        )

    epochs = range(1, n + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    ax1.plot(epochs, history.train_loss, label="train")
    ax1.plot(epochs, history.val_loss, label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(epochs, history.val_acc)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation accuracy")

    fig.tight_layout()
    return fig
