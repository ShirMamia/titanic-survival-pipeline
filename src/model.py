"""PyTorch model definition: a small MLP for binary classification.

The network outputs a single raw logit per sample (no sigmoid), which pairs
with ``nn.BCEWithLogitsLoss`` for numerically stable training. Probabilities
are obtained with ``torch.sigmoid`` at evaluation/inference time.
"""
from __future__ import annotations

import torch
from torch import nn


class TitanicMLP(nn.Module):
    """Configurable multi-layer perceptron.

    Parameters
    ----------
    input_dim:
        Number of input features (equals the preprocessor's output width).
    hidden_dims:
        Widths of the hidden layers, e.g. ``[64, 32]``.
    dropout:
        Dropout probability applied after each hidden activation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))  # single logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns shape (batch, 1) of raw logits.
        return self.net(x)
