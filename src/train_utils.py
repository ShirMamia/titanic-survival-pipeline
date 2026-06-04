"""Training utilities: reproducible seeding, tensor datasets, and the loop.

Kept framework-light and CPU-friendly. The training loop tracks train/val loss
and validation accuracy each epoch and applies early stopping on validation
loss, restoring the best weights before returning.
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
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
) -> DataLoader:
    """Wrap feature/label arrays in a float32 TensorDataset DataLoader."""
    dataset = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32).view(-1, 1),  # (N, 1) for BCE
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


@dataclass
class TrainHistory:
    """Per-epoch metric traces collected during training."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


def _evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    """Return (mean loss, accuracy) over a loader without gradient tracking."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * xb.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == yb).sum().item()
            n += xb.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
) -> TrainHistory:
    """Train ``model`` in place with Adam + early stopping on validation loss.

    The best (lowest-val-loss) weights are restored into ``model`` before return.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # drop_last avoids a trailing batch of size 1 breaking BatchNorm in train mode.
    drop_last = len(X_train) % batch_size == 1
    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True, drop_last=drop_last)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)

    history = TrainHistory()
    best_val_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            seen += xb.size(0)
        train_loss = running / max(seen, 1)

        val_loss, val_acc = _evaluate_loss(model, val_loader, criterion)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0 or epochs_without_improvement >= patience:
            logger.info(
                "Epoch %3d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
                epoch, train_loss, val_loss, val_acc,
            )

        if epochs_without_improvement >= patience:
            logger.info("Early stopping at epoch %d (best val_loss=%.4f).", epoch, best_val_loss)
            break

    model.load_state_dict(best_state)
    return history
