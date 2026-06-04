"""Training utilities: reproducible seeding, DataLoaders, and the training loop.

Updated for ``TitanicEmbeddingMLP`` which takes ``(x_cat, x_cont)`` inputs
instead of a single flat feature tensor.  Batches yielded by DataLoaders are
3-tuples ``(x_cat: LongTensor, x_cont: FloatTensor, y: FloatTensor)``.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible CPU runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_loader(
    X_cat: np.ndarray,
    X_cont: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
) -> DataLoader:
    """Wrap category IDs, continuous features, and labels in a DataLoader.

    Yields batches of ``(x_cat: LongTensor, x_cont: FloatTensor, y: FloatTensor)``.

    Parameters
    ----------
    X_cat     : integer array, shape (n, n_cat_features) — category IDs >= 0.
    X_cont    : float array, shape (n, n_cont_features) — scaled continuous.
    y         : binary label array, shape (n,) or (n, 1) — values exactly 0 or 1.
    batch_size: mini-batch size (> 0).
    shuffle   : whether to shuffle each epoch (use ``True`` for training).
    drop_last : drop the final incomplete batch.

    Raises
    ------
    ValueError
        For shape mismatches, non-positive batch_size, non-finite values,
        non-integer category IDs, or non-binary labels.
    """
    X_cat  = np.asarray(X_cat)
    X_cont = np.asarray(X_cont, dtype=float)
    y_float = np.asarray(y, dtype=float).ravel()

    # ---- Shape checks --------------------------------------------------- #
    if X_cat.ndim != 2:
        raise ValueError(f"X_cat must be 2-D; got shape {X_cat.shape}.")
    if X_cont.ndim != 2:
        raise ValueError(f"X_cont must be 2-D; got shape {X_cont.shape}.")
    if not (X_cat.shape[0] == X_cont.shape[0] == y_float.shape[0]):
        raise ValueError(
            f"X_cat, X_cont, and y must have the same number of rows; "
            f"got {X_cat.shape[0]}, {X_cont.shape[0]}, {y_float.shape[0]}."
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0; got {batch_size}.")

    # ---- X_cat: must be finite, non-negative, and integer-valued --------- #
    cat_float = X_cat.astype(float)
    if not np.all(np.isfinite(cat_float)):
        raise ValueError("X_cat contains NaN or infinite values.")
    if np.any(cat_float < 0):
        raise ValueError("X_cat contains negative category IDs.")
    if not np.all(cat_float == np.floor(cat_float)):
        raise ValueError(
            "X_cat must contain integer-valued category IDs (e.g. 0, 1, 2); "
            "found non-integer values.  Check that TabularPreprocessor.transform "
            "was applied before calling make_loader."
        )

    # ---- X_cont: must be finite ----------------------------------------- #
    if not np.all(np.isfinite(X_cont)):
        raise ValueError("X_cont contains NaN or infinite values.")

    # ---- y: finite and exactly in {0.0, 1.0} ----------------------------- #
    if not np.all(np.isfinite(y_float)):
        raise ValueError("y contains NaN or infinite values.")
    bad = set(np.unique(y_float)) - {0.0, 1.0}
    if bad:
        raise ValueError(
            f"y must contain only binary labels {{0, 1}}; "
            f"found unexpected values: {sorted(bad)}.  "
            "Non-integer values such as 0.2 or 1.9 are not accepted."
        )
    y_int = y_float.astype(int)

    # ---- Build TensorDataset -------------------------------------------- #
    dataset = TensorDataset(
        torch.as_tensor(X_cat,  dtype=torch.long),
        torch.as_tensor(X_cont, dtype=torch.float32),
        torch.as_tensor(y_int,  dtype=torch.float32).view(-1, 1),
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )


@dataclass
class TrainHistory:
    """Per-epoch metric traces collected during training."""
    train_loss: list[float] = field(default_factory=list)
    val_loss:   list[float] = field(default_factory=list)
    val_acc:    list[float] = field(default_factory=list)


def _evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Return ``(mean_loss, accuracy)`` over a loader without gradient tracking.

    Raises
    ------
    ValueError
        If the loader contains no samples.
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x_cat, x_cont, yb in loader:
            x_cat, x_cont, yb = x_cat.to(device), x_cont.to(device), yb.to(device)
            logits      = model(x_cat, x_cont)
            total_loss += criterion(logits, yb).item() * x_cat.size(0)
            preds       = (torch.sigmoid(logits) >= 0.5).float()
            correct    += (preds == yb).sum().item()
            n          += x_cat.size(0)
    if n == 0:
        raise ValueError(
            "_evaluate_loss: loader contains no samples.  "
            "Cannot compute loss or accuracy."
        )
    return total_loss / n, correct / n


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    device: str | torch.device = "cpu",
) -> TrainHistory:
    """Train ``model`` in place with Adam + early stopping on validation loss.

    DataLoaders must yield ``(x_cat, x_cont, y)`` triples as produced by
    ``make_loader``.  The best (lowest-val-loss) weights are restored before
    return.

    Parameters
    ----------
    model        : ``TitanicEmbeddingMLP`` instance.
    train_loader : DataLoader for the training split.
    val_loader   : DataLoader for the validation split.
    lr           : Adam learning rate (> 0).
    weight_decay : L2 regularisation coefficient (>= 0).
    epochs       : maximum number of training epochs (> 0).
    patience     : early-stopping patience in epochs (> 0).
    device       : device to train on (default ``"cpu"``).

    Raises
    ------
    ValueError
        For invalid hyperparameters or empty DataLoaders.
    """
    if lr <= 0.0:
        raise ValueError(f"lr must be > 0; got {lr}.")
    if weight_decay < 0.0:
        raise ValueError(f"weight_decay must be >= 0; got {weight_decay}.")
    if epochs <= 0:
        raise ValueError(f"epochs must be > 0; got {epochs}.")
    if patience <= 0:
        raise ValueError(f"patience must be > 0; got {patience}.")

    device = torch.device(device)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history           = TrainHistory()
    best_val_loss     = float("inf")
    best_state        = {k: v.clone() for k, v in model.state_dict().items()}
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # ---- Training -------------------------------------------------- #
        model.train()
        running, seen = 0.0, 0
        for x_cat, x_cont, yb in train_loader:
            x_cat, x_cont, yb = x_cat.to(device), x_cont.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_cat, x_cont), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * x_cat.size(0)
            seen    += x_cat.size(0)

        if seen == 0:
            raise ValueError(
                "train_loader contained no samples on epoch 1.  "
                "Check that the DataLoader is correctly constructed."
            )
        train_loss = running / seen

        # ---- Validation ------------------------------------------------ #
        val_loss, val_acc = _evaluate_loss(model, val_loader, criterion, device)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)

        # ---- Early stopping -------------------------------------------- #
        if val_loss < best_val_loss - 1e-5:
            best_val_loss     = val_loss
            best_state        = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch == 1 or epoch % 10 == 0 or epochs_no_improve >= patience:
            logger.info(
                "Epoch %3d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
                epoch, train_loss, val_loss, val_acc,
            )

        if epochs_no_improve >= patience:
            logger.info(
                "Early stopping at epoch %d (best val_loss=%.4f).",
                epoch, best_val_loss,
            )
            break

    model.load_state_dict(best_state)
    return history