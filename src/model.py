"""Model definitions.

Active model (trained by ``train.py``):
    ``TitanicEmbeddingMLP`` — categorical embeddings + continuous MLP head.

Reference-only (kept for backward compatibility, not trained by default):
    ``TitanicMLP`` — flat one-hot / scaled MLP from the original pipeline.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# TitanicEmbeddingMLP — active model
# --------------------------------------------------------------------------- #

class TitanicEmbeddingMLP(nn.Module):
    """Tabular embedding MLP for binary survival classification.

    Architecture (inspired by the fastai tabular model pattern)
    -----------------------------------------------------------
    * One ``nn.Embedding`` layer per categorical feature, in *fixed* order.
      ``x_cat[:, i]`` must correspond to ``categorical_features[i]``.
    * Concatenation of all embedding outputs + standardised continuous features.
    * MLP head: ``Linear → LayerNorm → ReLU → Dropout`` × ``len(hidden_dims)``,
      followed by a single-logit output layer.
    * ``LayerNorm`` is used instead of ``BatchNorm1d`` because it does not depend
      on batch statistics and therefore works correctly with any batch size,
      including batch size 1 during inference.
    * Output is a raw logit (no sigmoid); use ``BCEWithLogitsLoss`` during
      training and ``torch.sigmoid`` at inference.

    Parameters
    ----------
    categorical_features : list[str]
        Feature names in the *exact* column order used by the preprocessor.
        Stored as ``self.cat_features``; the forward pass depends on this order.
    category_sizes : dict[str, int]
        Vocabulary size per feature (OOV slot 0 + known ids 1…N).
    embedding_dims : dict[str, int]
        Embedding dimension per feature.
    n_cont : int
        Number of continuous input features (>= 0).
    hidden_dims : list[int] or None
        Width of each hidden MLP layer.  Defaults to ``[64, 32]`` when
        ``None``.  Must be non-empty when provided explicitly.
    dropout : float
        Dropout probability in [0, 1).
    """

    def __init__(
        self,
        categorical_features: list[str],
        category_sizes: dict[str, int],
        embedding_dims: dict[str, int],
        n_cont: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Apply default before validation so the validator sees a concrete list.
        if hidden_dims is None:
            hidden_dims = [64, 32]

        # ------------------------------------------------------------------ #
        # Configuration validation
        # ------------------------------------------------------------------ #
        if not categorical_features:
            raise ValueError("categorical_features must be a non-empty list.")
        for f in categorical_features:
            if f not in category_sizes:
                raise ValueError(f"category_sizes is missing key '{f}'.")
            if not isinstance(category_sizes[f], int) or category_sizes[f] < 1:
                raise ValueError(
                    f"category_sizes['{f}'] must be an integer >= 1; "
                    f"got {category_sizes[f]}."
                )
            if f not in embedding_dims:
                raise ValueError(f"embedding_dims is missing key '{f}'.")
            if not isinstance(embedding_dims[f], int) or embedding_dims[f] < 1:
                raise ValueError(
                    f"embedding_dims['{f}'] must be an integer >= 1; "
                    f"got {embedding_dims[f]}."
                )
        if not isinstance(n_cont, int) or n_cont < 0:
            raise ValueError(f"n_cont must be a non-negative integer; got {n_cont}.")
        if not hidden_dims:
            raise ValueError("hidden_dims must be a non-empty list.")
        for i, h in enumerate(hidden_dims):
            if not isinstance(h, int) or h <= 0:
                raise ValueError(
                    f"hidden_dims[{i}] must be a positive integer; got {h}."
                )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1); got {dropout}.")

        # ------------------------------------------------------------------ #
        # Stored attributes (useful for reconstruction from metadata)
        # ------------------------------------------------------------------ #
        self.cat_features:   list[str]       = list(categorical_features)
        self.category_sizes: dict[str, int]  = dict(category_sizes)
        self.embedding_dims: dict[str, int]  = dict(embedding_dims)
        self.n_cont:         int             = n_cont
        self.hidden_dims:    list[int]       = list(hidden_dims)
        self.dropout:        float           = dropout

        # ------------------------------------------------------------------ #
        # Embedding layers — one per categorical feature, in fixed order
        # ------------------------------------------------------------------ #
        self.embeddings = nn.ModuleList([
            nn.Embedding(category_sizes[f], embedding_dims[f])
            for f in self.cat_features
        ])

        emb_total = sum(embedding_dims[f] for f in self.cat_features)
        input_dim = emb_total + n_cont

        # ------------------------------------------------------------------ #
        # MLP head with LayerNorm (robust to any batch size, incl. batch = 1)
        # ------------------------------------------------------------------ #
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # single raw logit
        self.network = nn.Sequential(*layers)

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        x_cat:  torch.Tensor,   # (batch, n_cat)  — dtype: torch.long
        x_cont: torch.Tensor,   # (batch, n_cont) — dtype: torch.float32
    ) -> torch.Tensor:           # (batch, 1)      — raw logit
        """Forward pass.

        ``x_cat[:, i]`` must correspond to ``self.cat_features[i]``.

        Raises
        ------
        ValueError / TypeError
            For empty batches, mismatched shapes, wrong dtypes, out-of-range
            category IDs, or tensors on different devices.
        """
        # ---- Structural checks ------------------------------------------ #
        if x_cat.dim() != 2:
            raise ValueError(
                f"x_cat must be 2-D (batch, n_cat); got shape {tuple(x_cat.shape)}."
            )
        if x_cont.dim() != 2:
            raise ValueError(
                f"x_cont must be 2-D (batch, n_cont); got shape {tuple(x_cont.shape)}."
            )
        if x_cat.shape[0] == 0:
            raise ValueError("Batch size must be > 0.")
        if x_cat.shape[0] != x_cont.shape[0]:
            raise ValueError(
                f"x_cat and x_cont must have the same batch size; "
                f"got {x_cat.shape[0]} vs {x_cont.shape[0]}."
            )
        if x_cat.shape[1] != len(self.cat_features):
            raise ValueError(
                f"x_cat has {x_cat.shape[1]} columns but model expects "
                f"{len(self.cat_features)} categorical features."
            )
        if x_cont.shape[1] != self.n_cont:
            raise ValueError(
                f"x_cont has {x_cont.shape[1]} columns but model expects "
                f"{self.n_cont} continuous features."
            )
        # ---- Dtype checks ----------------------------------------------- #
        if x_cat.dtype != torch.long:
            raise TypeError(
                f"x_cat must have dtype torch.long; got {x_cat.dtype}."
            )
        if not x_cont.is_floating_point():
            raise TypeError(
                f"x_cont must be a floating-point tensor; got {x_cont.dtype}."
            )
        # ---- Device check ----------------------------------------------- #
        if x_cat.device != x_cont.device:
            raise ValueError(
                f"x_cat and x_cont must be on the same device; "
                f"got {x_cat.device} vs {x_cont.device}."
            )
        # ---- Categorical ID range checks --------------------------------- #
        for i, feat in enumerate(self.cat_features):
            col = x_cat[:, i]
            min_id = int(col.min().item())
            max_id = int(col.max().item())
            vocab_size = self.category_sizes[feat]
            if min_id < 0 or max_id >= vocab_size:
                raise ValueError(
                    f"Categorical feature '{feat}' (column {i}) has IDs in "
                    f"[{min_id}, {max_id}], but the valid range is "
                    f"[0, {vocab_size - 1}]."
                )

        # ---- Forward computation ----------------------------------------- #
        embs = [self.embeddings[i](x_cat[:, i]) for i in range(len(self.cat_features))]
        x = torch.cat(embs + [x_cont], dim=1)
        return self.network(x)


# --------------------------------------------------------------------------- #
# TitanicMLP — reference only; not trained by default
# --------------------------------------------------------------------------- #
# Kept so that any external model.pt checkpoint saved by the original flat-MLP
# pipeline can still be loaded manually for inspection.
# This class is NOT imported or used by train.py or inference.py.

class TitanicMLP(nn.Module):
    """Flat MLP on one-hot / scaled features (original pipeline, reference only).

    Not trained by default.  Use ``TitanicEmbeddingMLP`` for all new runs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
