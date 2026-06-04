"""Inference: load saved artifacts and predict on new data.

Used by the Streamlit app (and usable from any script).  Loading is split from
prediction so the app can load once and reuse the bundle across runs.

The final inference model is ``TitanicEmbeddingMLP``.
Baseline models (LR, RF, GB) are comparison-only and are not used here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch

from . import config
from .model import TitanicEmbeddingMLP
from .preprocessing import TabularPreprocessor, engineer_features

# --------------------------------------------------------------------------- #
# Required keys that must be present in metadata.json for reconstruction.
# --------------------------------------------------------------------------- #
_REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "model_type",
    "categorical_features",
    "continuous_features",
    "category_sizes",
    "embedding_dims",
    "hidden_dims",
    "dropout",
    "threshold",
    "pytorch_preprocessor_path",
    "feature_engineering_version",
)


class InferenceError(RuntimeError):
    """Raised when artifacts are missing/corrupt or input data is unusable."""


@dataclass
class ArtifactBundle:
    """Everything needed to reproduce a prediction."""
    model:        TitanicEmbeddingMLP
    preprocessor: TabularPreprocessor
    metadata:     dict


def load_artifacts(
    model_path:        Path = config.MODEL_PATH,
    preprocessor_path: Optional[Path] = None,   # resolved from metadata if None
    metadata_path:     Path = config.METADATA_PATH,
) -> ArtifactBundle:
    """Load model weights, the fitted PyTorch preprocessor, and metadata.

    The preprocessor path is read from ``metadata.json`` by default and
    resolved relative to ``config.ROOT_DIR``.  Pass ``preprocessor_path``
    explicitly only to override.

    Raises
    ------
    InferenceError
        If any required artifact is missing, metadata keys are absent,
        the model_type is unexpected, the feature_engineering_version does not
        match, the preprocessor type/contents are inconsistent, or the
        stored threshold is invalid.
    """
    # ---- 1. Metadata must exist and be readable first ------------------- #
    if not Path(metadata_path).exists():
        raise InferenceError(
            f"Required artifact 'metadata.json' not found at {metadata_path}.  "
            "Train the model first with `python train.py`."
        )

    try:
        with open(metadata_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)
    except Exception as exc:
        raise InferenceError(f"Failed to read metadata.json: {exc}") from exc

    # ---- 2. Check all required metadata keys are present ---------------- #
    missing_keys = [k for k in _REQUIRED_METADATA_KEYS if k not in metadata]
    if missing_keys:
        raise InferenceError(
            f"metadata.json is missing required keys: {missing_keys}.  "
            "Re-run `python train.py` to regenerate artifacts."
        )

    # ---- 3. Validate model type ----------------------------------------- #
    model_type = metadata["model_type"]
    if model_type != config.MODEL_TYPE:
        raise InferenceError(
            f"Unexpected model_type '{model_type}' in metadata.json.  "
            f"Expected '{config.MODEL_TYPE}'.  "
            "Re-run `python train.py` to regenerate artifacts."
        )

    # ---- 4. Validate feature-engineering version ------------------------ #
    fev = metadata["feature_engineering_version"]
    if fev != config.FEATURE_ENGINEERING_VERSION:
        raise InferenceError(
            f"Artifact feature_engineering_version={fev} does not match "
            f"code version={config.FEATURE_ENGINEERING_VERSION}.  "
            "Re-run `python train.py` to regenerate artifacts."
        )

    # ---- 5. Validate threshold ------------------------------------------ #
    raw_threshold = metadata["threshold"]
    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError):
        raise InferenceError(
            f"metadata['threshold'] must be numeric; got {raw_threshold!r}."
        )
    if not (0.0 <= threshold <= 1.0):
        raise InferenceError(
            f"metadata['threshold']={threshold} is outside [0, 1].  "
            "Re-run `python train.py` to regenerate artifacts."
        )

    # ---- 6. Resolve preprocessor path (cross-platform) ------------------ #
    if preprocessor_path is None:
        # Normalise separators so a path saved on Windows works on Linux/Mac.
        rel = str(metadata["pytorch_preprocessor_path"]).replace("\\", "/")
        preprocessor_path = config.ROOT_DIR / Path(rel)

    for path, what in [
        (preprocessor_path, "pytorch_preprocessor.joblib"),
        (model_path,        "model.pt"),
    ]:
        if not Path(path).exists():
            raise InferenceError(
                f"Required artifact '{what}' not found at {path}.  "
                "Train the model first with `python train.py`."
            )

    # ---- 7. Load and validate the preprocessor -------------------------- #
    try:
        preprocessor = joblib.load(preprocessor_path)
    except Exception as exc:
        raise InferenceError(
            f"Failed to load pytorch_preprocessor.joblib: {exc}"
        ) from exc

    if not isinstance(preprocessor, TabularPreprocessor):
        raise InferenceError(
            f"Expected a TabularPreprocessor instance; "
            f"got {type(preprocessor).__name__}.  "
            "Re-run `python train.py` to regenerate artifacts."
        )
    if list(preprocessor.cat_features) != metadata["categorical_features"]:
        raise InferenceError(
            "Preprocessor cat_features do not match metadata.  "
            "Re-run `python train.py` to regenerate artifacts."
        )
    if list(preprocessor.cont_features) != metadata["continuous_features"]:
        raise InferenceError(
            "Preprocessor cont_features do not match metadata.  "
            "Re-run `python train.py` to regenerate artifacts."
        )
    if dict(preprocessor.category_sizes) != metadata["category_sizes"]:
        raise InferenceError(
            "Preprocessor category_sizes do not match metadata.  "
            "Re-run `python train.py` to regenerate artifacts."
        )

    # ---- 8. Reconstruct and load the model ------------------------------ #
    try:
        model = TitanicEmbeddingMLP(
            categorical_features=metadata["categorical_features"],
            category_sizes=metadata["category_sizes"],
            embedding_dims=metadata["embedding_dims"],
            n_cont=len(metadata["continuous_features"]),
            hidden_dims=metadata["hidden_dims"],
            dropout=metadata["dropout"],
        )
        # weights_only=True is safer; fall back for older PyTorch versions.
        try:
            state_dict = torch.load(
                model_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
    except (InferenceError, RuntimeError):
        raise
    except Exception as exc:
        raise InferenceError(f"Failed to load model weights: {exc}") from exc

    return ArtifactBundle(model=model, preprocessor=preprocessor, metadata=metadata)


def predict_dataframe(df: pd.DataFrame, bundle: ArtifactBundle) -> pd.DataFrame:
    """Run inference on a raw Titanic-style DataFrame.

    Returns the original frame with two added columns:
      * ``PredictedProbability`` — sigmoid probability of survival (float).
      * ``PredictedSurvived``    — 0/1 at the stored decision threshold.

    Tolerates missing optional columns (Cabin, Name, Age, Fare).
    Raises ``InferenceError`` for missing required columns (Sex, Pclass).
    """
    if df is None or df.empty:
        raise InferenceError("Input data is empty — nothing to predict.")

    threshold = float(bundle.metadata["threshold"])

    try:
        features = engineer_features(df)
        X_cat, X_cont = bundle.preprocessor.transform(features)
    except ValueError as exc:
        raise InferenceError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(
            "Failed to preprocess the input data.  Ensure it has Titanic-style "
            "columns (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, …).  "
            f"Original error: {exc}"
        ) from exc

    # Defensive: ensure eval mode even if the caller mutated the model state.
    bundle.model.eval()
    with torch.no_grad():
        logits = bundle.model(
            torch.as_tensor(np.asarray(X_cat),  dtype=torch.long),
            torch.as_tensor(np.asarray(X_cont), dtype=torch.float32),
        )
        # Apply sigmoid outside the model; convert to plain numpy floats.
        probs: np.ndarray = torch.sigmoid(logits).numpy().ravel().astype(float)

    result = df.copy()
    result["PredictedProbability"] = probs
    result["PredictedSurvived"]    = (probs >= threshold).astype(int)
    return result


def extract_labels(df: pd.DataFrame) -> np.ndarray | None:
    """Return integer ground-truth labels when a valid ``Survived`` column exists.

    Returns ``None`` if:
      * The ``Survived`` column is absent.
      * Any value is NaN or not finite.
      * Any value is not exactly 0 or 1.
    """
    if config.TARGET not in df.columns:
        return None
    labels_float = pd.to_numeric(df[config.TARGET], errors="coerce")
    if labels_float.isna().any():
        return None
    arr = labels_float.to_numpy(dtype=float)
    if not np.all(np.isfinite(arr)):
        return None
    bad = set(np.unique(arr)) - {0.0, 1.0}
    if bad:
        return None
    return arr.astype(int)
