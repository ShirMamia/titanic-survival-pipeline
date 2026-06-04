"""Feature engineering and preprocessing.

Two clearly separated stages:

1. ``engineer_features`` — deterministic, *row-wise* transformations only.
   Safe to call on any split, on the full dataset before splitting, and at
   inference time.  No dataset-level statistics are computed here.

2a. ``TabularPreprocessor`` — custom vocabulary mapping + median imputation +
    standard scaling for the PyTorch embedding model.
    Fitted on the training split only.  Do NOT use OneHotEncoder here.

2b. ``build_preprocessor`` — a scikit-learn ``ColumnTransformer`` used by the
    classical baseline models (LR, RF, GB) *only*.
    Uses OneHotEncoder for categoricals.  Do NOT use for the PyTorch model.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

UNKNOWN_TOKEN: str = "Unknown"

# All string representations that mean "missing / unknown".
# These are normalised to UNKNOWN_TOKEN before vocab lookup so that
# missing data at inference always maps to the OOV sentinel (id = 0).
_NULL_STRINGS: frozenset[str] = frozenset(
    {"", "nan", "none", "null", "unknown", "Unknown", "NaN", "None", "Null"}
)
_NULL_STRINGS_LOWER: frozenset[str] = frozenset(s.lower() for s in _NULL_STRINGS)

# --------------------------------------------------------------------------- #
# Title extraction helpers
# --------------------------------------------------------------------------- #

# Normalised to exactly 5 groups.  Everything outside this map → "Rare".
_TITLE_MAP: dict[str, str] = {
    "Mr": "Mr",
    "Mrs": "Mrs",   "Mme": "Mrs",
    "Miss": "Miss", "Ms": "Miss", "Mlle": "Miss",
    "Master": "Master",
}


def _extract_title(name: object) -> str:
    """Extract and normalise the honorific from a Titanic-style name string."""
    if not isinstance(name, str) or "," not in name or "." not in name:
        return "Rare"
    raw = name.split(",", 1)[1].split(".", 1)[0].strip()
    return _TITLE_MAP.get(raw, "Rare")


def _extract_deck(cabin: object) -> str:
    """Return the deck letter from a cabin code; UNKNOWN_TOKEN if missing.

    Returns UNKNOWN_TOKEN when:
      * cabin is not a string (e.g. NaN / None / float)
      * cabin is empty after stripping whitespace
      * cabin is a null-like string ("nan", "none", "null", "unknown", …)
    """
    if not isinstance(cabin, str):
        return UNKNOWN_TOKEN
    stripped = cabin.strip()
    if not stripped or stripped.lower() in _NULL_STRINGS_LOWER:
        return UNKNOWN_TOKEN
    return stripped[0].upper()


def _normalise_cat(series: pd.Series) -> pd.Series:
    """Normalise a categorical column for consistent vocab lookup.

    Steps:
      1. Fill NaN with UNKNOWN_TOKEN.
      2. Cast to str and strip whitespace.
      3. Map all null-like strings (empty, "nan", "none", …) to UNKNOWN_TOKEN.
      4. Normalise float-looking integers: "1.0" → "1", "2.0" → "2", etc.
         (prevents inference mismatches when a CSV reads Pclass as float).

    Applied identically in TabularPreprocessor.fit *and* .transform so that
    training and inference see exactly the same representation.
    """
    s = series.fillna(UNKNOWN_TOKEN).astype(str).str.strip()

    def _clean(v: str) -> str:
        if v.lower() in _NULL_STRINGS_LOWER:
            return UNKNOWN_TOKEN
        # Normalise float-looking integers: "1.0" → "1"
        try:
            f = float(v)
            if f == int(f):
                return str(int(f))
        except (ValueError, OverflowError):
            pass
        return v

    return s.map(_clean)


# --------------------------------------------------------------------------- #
# Feature engineering — row-wise, leakage-free
# --------------------------------------------------------------------------- #

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame containing all modelling features.

    All transformations are deterministic and per-row.  No dataset-level
    statistics are computed; the result is identical whether the input is the
    full dataset, a single split, or a single inference row.

    Required columns (raises ValueError if absent): ``Pclass``, ``Sex``.
    Optional columns receive safe defaults when absent:
        Age, Fare → NaN  |  SibSp, Parch → 0
        Embarked → UNKNOWN_TOKEN  |  Name → ""  |  Cabin → NaN

    Engineered features produced
    ----------------------------
    Categorical : Pclass, Sex, Embarked, Title, Deck, FamilySizeGroup
    Continuous  : Age, LogFare, SibSp, Parch, FamilySize, IsAlone,
                  HasCabin, AgeMissing, FareMissing
    """
    for col in config.REQUIRED_RAW_COLUMNS:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' is missing from the input data.  "
                "Inference cannot proceed without it."
            )

    df = df.copy()

    # ── Safe defaults for all optional raw columns ────────────────────────── #
    if "Age"      not in df.columns: df["Age"]      = np.nan
    if "Fare"     not in df.columns: df["Fare"]     = np.nan
    if "SibSp"    not in df.columns: df["SibSp"]    = 0
    if "Parch"    not in df.columns: df["Parch"]    = 0
    if "Embarked" not in df.columns: df["Embarked"] = UNKNOWN_TOKEN
    if "Name"     not in df.columns: df["Name"]     = ""
    if "Cabin"    not in df.columns: df["Cabin"]    = np.nan

    # ── Coerce numeric columns robustly ───────────────────────────────────── #
    df["Age"]   = pd.to_numeric(df["Age"],   errors="coerce")
    df["Fare"]  = pd.to_numeric(df["Fare"],  errors="coerce")
    df["SibSp"] = pd.to_numeric(df["SibSp"], errors="coerce").fillna(0)
    df["Parch"] = pd.to_numeric(df["Parch"], errors="coerce").fillna(0)

    # ── Missing-value indicators — BEFORE any fill / transform ───────────── #
    df["AgeMissing"]  = df["Age"].isna().astype(int)
    df["FareMissing"] = df["Fare"].isna().astype(int)

    # ── Log-fare (NaN stays NaN so the imputer handles it downstream) ─────── #
    df["LogFare"] = np.log1p(df["Fare"].clip(lower=0))   # np.log1p(NaN) == NaN ✓

    # ── Family features ───────────────────────────────────────────────────── #
    df["FamilySize"] = (df["SibSp"] + df["Parch"] + 1).astype(float)
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

    # Fixed bins — no data-driven quantiles, safe before any split.
    df["FamilySizeGroup"] = (
        pd.cut(
            df["FamilySize"].clip(upper=20),
            bins=[0, 1, 4, 20],
            labels=["Alone", "Small", "Large"],
            right=True,
        )
        .astype(str)
        .replace("nan", UNKNOWN_TOKEN)
    )

    # ── Cabin-derived features ────────────────────────────────────────────── #
    # HasCabin = 1 only for genuinely non-missing, non-null cabin strings.
    cabin_str = df["Cabin"].astype(str).str.strip()
    cabin_present = (
        df["Cabin"].notna()
        & cabin_str.ne("")
        & ~cabin_str.str.lower().isin(_NULL_STRINGS_LOWER)
    )
    df["HasCabin"] = cabin_present.astype(int)
    df["Deck"]     = df["Cabin"].apply(_extract_deck)

    # ── Title from Name ───────────────────────────────────────────────────── #
    df["Title"] = df["Name"].apply(_extract_title)

    # ── Ensure every modelling column was produced ────────────────────────── #
    feature_cols = config.CATEGORICAL_FEATURES + config.CONTINUOUS_FEATURES
    for col in feature_cols:
        if col not in df.columns:
            raise RuntimeError(
                f"Internal error: engineered column '{col}' was not created.  "
                "This is a bug in engineer_features."
            )

    return df[feature_cols]


# --------------------------------------------------------------------------- #
# TabularPreprocessor — custom vocab mapping + imputation + scaling
# For the PyTorch TitanicEmbeddingMLP ONLY.
# Do NOT use OneHotEncoder here.
# Do NOT use sklearn LabelEncoder here.
# --------------------------------------------------------------------------- #

class TabularPreprocessor:
    """Vocabulary mapping + imputation + scaling for TitanicEmbeddingMLP.

    Categorical encoding
    --------------------
    * id **0** is reserved as the OOV / missing sentinel.
    * UNKNOWN_TOKEN and any value not seen at fit-time → id 0 at transform.
    * Known categories (all values except UNKNOWN_TOKEN) fitted on the
      training split → ids 1 … N.
    * ``category_sizes[feat]`` = N + 1  (including the OOV slot at 0).

    All categorical normalisation uses ``_normalise_cat`` — applied
    identically at fit and transform for consistent train/inference behaviour.

    Continuous preprocessing
    ------------------------
    * ``SimpleImputer(strategy="median")`` fitted on training data only.
    * ``StandardScaler`` fitted on the imputed training data only.

    The object is fully serialisable with ``joblib.dump``.
    """

    def __init__(
        self,
        cat_features: List[str] | None = None,
        cont_features: List[str] | None = None,
    ) -> None:
        self.cat_features:   List[str]                 = list(cat_features  or config.CATEGORICAL_FEATURES)
        self.cont_features:  List[str]                 = list(cont_features or config.CONTINUOUS_FEATURES)
        self.vocabs:         Dict[str, Dict[str, int]] = {}
        self.category_sizes: Dict[str, int]            = {}
        self._imputer:       SimpleImputer  = SimpleImputer(strategy="median")
        self._scaler:        StandardScaler = StandardScaler()
        self._fitted:        bool           = False

    # -------------------------------------------------------------------- #

    def _check_columns(self, X_df: pd.DataFrame, label: str) -> None:
        """Raise ValueError if any required feature column is missing."""
        missing = [
            f for f in (self.cat_features + self.cont_features)
            if f not in X_df.columns
        ]
        if missing:
            raise ValueError(
                f"TabularPreprocessor.{label}: the following expected feature "
                f"columns are missing from the input: {missing}.  "
                "Make sure engineer_features was applied before this call."
            )

    def fit(self, X_df: pd.DataFrame) -> "TabularPreprocessor":
        """Fit vocabularies, imputer, and scaler on the training split only."""
        self._check_columns(X_df, "fit")

        # Categorical vocabularies ---------------------------------------- #
        # UNKNOWN_TOKEN is excluded from the learned vocabulary so that it
        # (and any truly unseen category at inference) maps to id 0.
        for feat in self.cat_features:
            col   = _normalise_cat(X_df[feat])
            known = sorted(v for v in col.unique() if v != UNKNOWN_TOKEN)
            self.vocabs[feat]         = {v: i + 1 for i, v in enumerate(known)}
            self.category_sizes[feat] = len(known) + 1   # +1 for OOV slot

        # Continuous: impute then scale ------------------------------------ #
        cont_arr = X_df[self.cont_features].values.astype(float)
        self._imputer.fit(cont_arr)
        self._scaler.fit(self._imputer.transform(cont_arr))
        self._fitted = True
        return self

    def transform(self, X_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(X_cat, X_cont)`` ready for ``TitanicEmbeddingMLP``.

        X_cat  : int32,   shape (n, n_cat_features)  — IDs (0 = OOV / missing)
        X_cont : float32, shape (n, n_cont_features) — imputed + z-scored
        """
        if not self._fitted:
            raise RuntimeError(
                "TabularPreprocessor has not been fitted.  "
                "Call fit(X_train_df) before transform."
            )
        self._check_columns(X_df, "transform")
        n = len(X_df)

        # Categorical: normalise → look up in vocab (0 for OOV / Unknown) -- #
        X_cat = np.zeros((n, len(self.cat_features)), dtype=np.int32)
        for i, feat in enumerate(self.cat_features):
            col   = _normalise_cat(X_df[feat])
            vocab = self.vocabs[feat]
            # UNKNOWN_TOKEN and any value not in the fitted vocab → 0
            X_cat[:, i] = col.map(
                lambda v, voc=vocab: 0 if v == UNKNOWN_TOKEN else voc.get(v, 0)
            ).values

        # Continuous: impute then scale ------------------------------------ #
        cont_arr = X_df[self.cont_features].values.astype(float)
        X_cont   = self._scaler.transform(
            self._imputer.transform(cont_arr)
        ).astype(np.float32)

        return X_cat, X_cont

    # -------------------------------------------------------------------- #

    @property
    def embedding_dims(self) -> Dict[str, int]:
        """Embedding dimension per feature: ``min(50, max(2, (size+1)//2))``."""
        return {
            feat: min(50, max(2, (size + 1) // 2))
            for feat, size in self.category_sizes.items()
        }

    @property
    def n_cont(self) -> int:
        """Number of continuous features."""
        return len(self.cont_features)


# --------------------------------------------------------------------------- #
# build_preprocessor — sklearn ColumnTransformer for baselines ONLY
# Do NOT use this for the PyTorch embedding model.
# --------------------------------------------------------------------------- #

def build_preprocessor() -> ColumnTransformer:
    """Construct (unfitted) the baseline imputation + scaling + encoding transformer.

    Used exclusively by the classical ML baselines (LR, RF, GB).
    Uses ``OneHotEncoder`` for categorical features.
    Do NOT pass the output of this to ``TitanicEmbeddingMLP``.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # handle_unknown='ignore' keeps inference robust to unseen categories.
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,     config.CONTINUOUS_FEATURES),
            ("cat", categorical_pipeline, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_output_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return the expanded feature names from a *fitted* baseline preprocessor."""
    return list(preprocessor.get_feature_names_out())
