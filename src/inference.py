"""Inference: load saved artifacts and predict on new data.

Used by the Streamlit app (and usable from any script). Loading is split from
prediction so the app can load once and reuse the bundle across runs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer

from . import config
from .model import TitanicMLP
from .preprocessing import engineer_features


class InferenceError(RuntimeError):
    """Raised when artifacts are missing/corrupt or input data is unusable."""


@dataclass
class ArtifactBundle:
    """Everything needed to reproduce a prediction."""

    model: TitanicMLP
    preprocessor: ColumnTransformer
    metadata: dict


def load_artifacts(
    model_path: Path = config.MODEL_PATH,
    preprocessor_path: Path = config.PREPROCESSOR_PATH,
    metadata_path: Path = config.METADATA_PATH,
) -> ArtifactBundle:
    """Load model weights, the fitted preprocessor, and metadata from disk."""
    for path, what in [
        (metadata_path, "metadata.json"),
        (preprocessor_path, "preprocessor.joblib"),
        (model_path, "model.pt"),
    ]:
        if not Path(path).exists():
            raise InferenceError(
                f"Required artifact '{what}' not found at {path}. "
                "Train the model first with `python train.py`."
            )

    try:
        with open(metadata_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        preprocessor = joblib.load(preprocessor_path)
        model = TitanicMLP(
            input_dim=metadata["input_dim"],
            hidden_dims=metadata["hidden_dims"],
            dropout=metadata["dropout"],
        )
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(f"Failed to load artifacts: {exc}") from exc

    return ArtifactBundle(model=model, preprocessor=preprocessor, metadata=metadata)


def predict_dataframe(df: pd.DataFrame, bundle: ArtifactBundle) -> pd.DataFrame:
    """Run inference on a raw Titanic-style frame.

    Returns the original frame with two added columns:
      * ``PredictedProbability`` - sigmoid probability of survival.
      * ``PredictedSurvived``    - 0/1 at the stored decision threshold.

    Missing engineered-feature sources (Name/Cabin) and missing numeric columns
    are tolerated by the feature-engineering step.
    """
    if df is None or df.empty:
        raise InferenceError("Input data is empty - nothing to predict.")

    threshold = float(bundle.metadata.get("threshold", 0.5))

    try:
        features = engineer_features(df)
        X = bundle.preprocessor.transform(features)
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(
            "Failed to preprocess the input data. Ensure it has Titanic-style "
            f"columns (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, ...). "
            f"Original error: {exc}"
        ) from exc

    with torch.no_grad():
        logits = bundle.model(torch.as_tensor(np.asarray(X), dtype=torch.float32))
        probs = torch.sigmoid(logits).numpy().ravel()

    result = df.copy()
    result["PredictedProbability"] = probs
    result["PredictedSurvived"] = (probs >= threshold).astype(int)
    return result


def extract_labels(df: pd.DataFrame) -> np.ndarray | None:
    """Return integer ground-truth labels if a usable ``Survived`` column exists."""
    if config.TARGET not in df.columns:
        return None
    labels = pd.to_numeric(df[config.TARGET], errors="coerce")
    if labels.isna().any():
        return None
    return labels.astype(int).to_numpy()
