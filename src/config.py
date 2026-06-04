"""Central configuration for the Titanic survival pipeline.

Keeping paths, the random seed, feature definitions, and default
hyperparameters in one place makes the pipeline reproducible and keeps
train.py, the inference code, and the Streamlit app perfectly in sync.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (all relative to the repository root so the project is portable)
# --------------------------------------------------------------------------- #
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
SAMPLE_DATA_DIR: Path = DATA_DIR / "sample"
ARTIFACTS_DIR: Path = ROOT_DIR / "artifacts"
REPORTS_DIR: Path = ROOT_DIR / "reports"

# Kaggle competition train file (the only file with the `Survived` label).
RAW_TRAIN_CSV: Path = RAW_DATA_DIR / "train.csv"
SAMPLE_CSV: Path = SAMPLE_DATA_DIR / "sample_titanic.csv"

# Artifact filenames.
MODEL_PATH: Path = ARTIFACTS_DIR / "model.pt"

# Separate preprocessors for the two modelling paths.
PYTORCH_PREPROCESSOR_PATH: Path = ARTIFACTS_DIR / "pytorch_preprocessor.joblib"
BASELINE_PREPROCESSOR_PATH: Path = ARTIFACTS_DIR / "baseline_preprocessor.joblib"

METADATA_PATH: Path = ARTIFACTS_DIR / "metadata.json"
METRICS_PATH: Path = ARTIFACTS_DIR / "metrics.json"
BASELINES_METRICS_PATH: Path = ARTIFACTS_DIR / "baseline_metrics.json"

# Saved evaluation plots (produced by train.py, displayed by the app).
CONFUSION_MATRIX_PNG: Path = REPORTS_DIR / "confusion_matrix.png"
ROC_CURVE_PNG: Path = REPORTS_DIR / "roc_curve.png"
TRAINING_CURVE_PNG: Path = REPORTS_DIR / "training_curves.png"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42

# Increment when engineer_features changes so stale artifacts are detected.
FEATURE_ENGINEERING_VERSION: int = 1

# --------------------------------------------------------------------------- #
# Model type constant (used in metadata.json and validated at inference time)
# --------------------------------------------------------------------------- #
MODEL_TYPE: str = "TitanicEmbeddingMLP"

# --------------------------------------------------------------------------- #
# Raw column definitions (used by engineer_features and inference)
# --------------------------------------------------------------------------- #
TARGET: str = "Survived"

# Columns whose absence must raise a clear ValueError immediately.
REQUIRED_RAW_COLUMNS: list[str] = [
    "Pclass",
    "Sex",
]

# Columns that are used when present; safe defaults applied when absent.
OPTIONAL_RAW_COLUMNS: list[str] = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "Embarked",
    "Name",
    "Cabin",
]

# --------------------------------------------------------------------------- #
# Engineered feature lists
# --------------------------------------------------------------------------- #

# Categorical features — used by the embedding layers (PyTorch) and
# OneHotEncoder (baseline ColumnTransformer).
# ORDER IS FIXED.  Do not reorder without incrementing FEATURE_ENGINEERING_VERSION.
CATEGORICAL_FEATURES: list[str] = [
    "Pclass",
    "Sex",
    "Embarked",
    "Title",
    "Deck",
    "FamilySizeGroup",
]

# Continuous features — used by the MLP head (PyTorch) and the numeric
# pipeline (baseline ColumnTransformer).
# ORDER IS FIXED.  Do not reorder without incrementing FEATURE_ENGINEERING_VERSION.
CONTINUOUS_FEATURES: list[str] = [
    "Age",
    "LogFare",
    "SibSp",
    "Parch",
    "FamilySize",
    "IsAlone",
    "HasCabin",
    "AgeMissing",
    "FareMissing",
]

# Alias so the baseline ColumnTransformer can reference the same list without
# aliasing the mutable object.
NUMERIC_FEATURES: list[str] = list(CONTINUOUS_FEATURES)

CLASS_NAMES: list[str] = ["Did not survive", "Survived"]

# --------------------------------------------------------------------------- #
# Default hyperparameters (overridable via train.py CLI flags)
# --------------------------------------------------------------------------- #
DEFAULTS: dict = {
    "test_size": 0.15,         # held-out test fraction of the labelled data
    "val_size": 0.15,          # validation fraction of the labelled data
    "hidden_dims": [64, 32],   # MLP hidden layer widths
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "epochs": 100,
    "patience": 15,            # early-stopping patience (epochs without val improvement)
}
