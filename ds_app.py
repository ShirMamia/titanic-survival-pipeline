"""Streamlit app for the Titanic survival classifier.

Two tabs:
  * "Training results" — metrics, model comparison table, and plots produced
                          by ``train.py``.
  * "Run inference"    — upload a CSV or enter a path, run TitanicEmbeddingMLP
                          predictions, view metrics/plots when labels are
                          present, and download the results.

Final inference model: TitanicEmbeddingMLP (PyTorch).
Classical baselines (LR, RF, GB) are shown for comparison only; they are
NOT used for inference.

Run with:
    streamlit run ds_app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.data import DataError, load_csv
from src.evaluate import (
    classification_text_report,
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)
from src.inference import (
    ArtifactBundle,
    InferenceError,
    extract_labels,
    load_artifacts,
    predict_dataframe,
)

st.set_page_config(
    page_title="Titanic Survival Classifier",
    page_icon="🚢",
    layout="wide",
)

_REQUIRED_COLS  = config.REQUIRED_RAW_COLUMNS
_IMPORTANT_COLS = [c for c in config.OPTIONAL_RAW_COLUMNS if c not in ("Name", "Cabin")]


# --------------------------------------------------------------------------- #
# Cached resource loading
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def _load_bundle_cached() -> ArtifactBundle:
    """Load and cache the artifact bundle.  Raises InferenceError on failure."""
    return load_artifacts()


# --------------------------------------------------------------------------- #
# JSON loading helper
# --------------------------------------------------------------------------- #

class _JsonResult:
    """Encapsulates one of three outcomes: data, missing, or malformed."""
    def __init__(self, data=None, missing: bool = False, error: str = ""):
        self.data    = data
        self.missing = missing
        self.error   = error

    @property
    def ok(self) -> bool:
        return self.data is not None


def _read_json(path: Path) -> _JsonResult:
    if not path.exists():
        return _JsonResult(missing=True)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return _JsonResult(data=json.load(fh))
    except Exception as exc:
        return _JsonResult(error=f"Could not parse {path.name}: {exc}")


# --------------------------------------------------------------------------- #
# Helper widgets
# --------------------------------------------------------------------------- #

def _metrics_columns(metrics: dict) -> None:
    cols = st.columns(5)
    cols[0].metric("Accuracy",  f"{metrics['accuracy']:.3f}")
    cols[1].metric("Precision", f"{metrics['precision']:.3f}")
    cols[2].metric("Recall",    f"{metrics['recall']:.3f}")
    cols[3].metric("F1",        f"{metrics['f1']:.3f}")
    cols[4].metric(
        "ROC-AUC",
        "n/a" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.3f}",
    )


def _prob_histogram(probs: np.ndarray, threshold: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(probs, bins=25, edgecolor="black", color="steelblue", alpha=0.8)
    ax.axvline(
        threshold, color="red", linestyle="--", linewidth=1.5,
        label=f"threshold = {threshold:.3f}",
    )
    ax.set_xlabel("Predicted survival probability")
    ax.set_ylabel("Count")
    ax.set_title("Prediction probability distribution")
    ax.legend()
    fig.tight_layout()
    return fig


def _metric_val(m: dict | None, key: str) -> str:
    if m is None:
        return "—"
    v = m.get(key)
    if v is None:
        return "—"
    return f"{float(v):.3f}"


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🚢 Titanic Survival Classifier")
st.caption(
    "PyTorch embedding MLP (`TitanicEmbeddingMLP`) trained by `train.py`.  "
    "Classical baselines (LR, RF, GB) are shown for comparison only — "
    "inference always uses **TitanicEmbeddingMLP**."
)

tab_results, tab_infer = st.tabs(["📊 Training results", "🔮 Run inference"])

# --------------------------------------------------------------------------- #
# Tab 1: Training results
# --------------------------------------------------------------------------- #
with tab_results:
    st.subheader("Held-out test performance")

    metrics_result  = _read_json(config.METRICS_PATH)
    metadata_result = _read_json(config.METADATA_PATH)
    baseline_result = _read_json(config.BASELINES_METRICS_PATH)

    # Surface JSON parse errors immediately.
    if metrics_result.error:
        st.error(
            f"metrics.json is malformed and could not be loaded.  "
            f"{metrics_result.error}  "
            "Re-run `python train.py` to regenerate."
        )
    elif metrics_result.missing:
        st.warning(
            "No training results found.  Train the model first:\n\n"
            "```bash\npython train.py\n```"
        )
    else:
        metrics_data  = metrics_result.data
        metadata_data = metadata_result.data   # may be None if missing/malformed

        # Data-source banner.
        if metadata_data and metadata_data.get("data_source") == "sample_synthetic":
            st.warning(
                "⚠️ **These metrics are from the SYNTHETIC SAMPLE dataset** "
                "(not real Titanic data).  "
                "Re-train with `python train.py` using Kaggle credentials for "
                "meaningful results."
            )

        test_m = metrics_data.get("test")
        if test_m is None:
            st.error(
                "metrics.json is missing the 'test' key.  "
                "Re-run `python train.py` to regenerate."
            )
        else:
            _metrics_columns(test_m)
            st.markdown("**Classification report (test split)**")
            report = metrics_data.get("classification_report", "")
            if report:
                st.code(report, language="text")

        # ---- Model comparison table ------------------------------------ #
        st.subheader("Model comparison (held-out test set)")
        st.caption(
            "**TitanicEmbeddingMLP** is the final submitted model.  "
            "Classical baselines are for comparison only and are not used for inference."
        )

        def _row(name: str, m: dict | None, purpose: str) -> dict:
            return {
                "Model":     name,
                "Accuracy":  _metric_val(m, "accuracy"),
                "Precision": _metric_val(m, "precision"),
                "Recall":    _metric_val(m, "recall"),
                "F1":        _metric_val(m, "f1"),
                "ROC-AUC":   _metric_val(m, "roc_auc"),
                "Purpose":   purpose,
            }

        rows = [_row("TitanicEmbeddingMLP (PyTorch)", test_m, "✅ Final model")]

        if baseline_result.ok:
            bd = baseline_result.data
            rows.append(_row("Logistic Regression", bd.get("logistic_regression"), "Baseline only"))
            rows.append(_row("Random Forest",       bd.get("random_forest"),       "Baseline only"))
            rows.append(_row("Gradient Boosting",   bd.get("gradient_boosting"),   "Baseline only"))
        elif baseline_result.missing:
            st.info(
                "Baseline metrics not found.  Run `python train.py` without "
                "`--skip-baselines` to generate them."
            )
        else:
            st.warning(f"baseline_metrics.json is malformed: {baseline_result.error}")

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ---- Plots ------------------------------------------------------ #
        img_cols = st.columns(2)
        if config.CONFUSION_MATRIX_PNG.exists():
            img_cols[0].image(str(config.CONFUSION_MATRIX_PNG), caption="Confusion matrix")
        if config.ROC_CURVE_PNG.exists():
            img_cols[1].image(str(config.ROC_CURVE_PNG), caption="ROC curve")
        if config.TRAINING_CURVE_PNG.exists():
            st.image(str(config.TRAINING_CURVE_PNG), caption="Training curves")

        if metadata_data:
            with st.expander("Model metadata"):
                st.json(metadata_data)

# --------------------------------------------------------------------------- #
# Tab 2: Inference (TitanicEmbeddingMLP only)
# --------------------------------------------------------------------------- #
with tab_infer:
    st.subheader("Predict survival on a CSV")
    st.caption(
        "Inference uses **TitanicEmbeddingMLP** (PyTorch).  "
        "Classical baselines are not used here."
    )

    # Load artifacts — error shown inside the tab, not as a full-page crash.
    try:
        bundle = _load_bundle_cached()
    except InferenceError as exc:
        st.error(str(exc))
        st.stop()

    # Data-source banner.
    if bundle.metadata.get("data_source") == "sample_synthetic":
        st.warning(
            "⚠️ The loaded model was trained on the **synthetic sample dataset**.  "
            "Predictions are illustrative only."
        )

    st.markdown(
        f"Provide a Titanic-style CSV.  \n"
        f"**Required columns:** `{', '.join(_REQUIRED_COLS)}`.  \n"
        f"**Recommended columns:** `{', '.join(_IMPORTANT_COLS)}`.  \n\n"
        "> **Note:** The Kaggle `test.csv` has no `Survived` column — only "
        "predictions and probabilities will be shown; evaluation metrics "
        "require ground-truth labels."
    )

    source = st.radio(
        "Data source", ["Upload a CSV", "Enter a file path"], horizontal=True
    )
    df: pd.DataFrame | None = None

    if source == "Upload a CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read the uploaded CSV: {exc}")
    else:
        default_path = str(config.SAMPLE_CSV)
        path_str = st.text_input("Path to CSV", value=default_path)
        if st.button("Load from path"):
            try:
                df = load_csv(path_str)
            except DataError as exc:
                st.error(str(exc))

    if df is not None:
        st.write(f"Loaded **{len(df)}** rows.")

        # ---- Column validation ---------------------------------------- #
        missing_required = [c for c in _REQUIRED_COLS if c not in df.columns]
        if missing_required:
            st.error(
                f"The CSV is missing required columns: `{missing_required}`.  "
                "Inference cannot proceed without them."
            )
            st.stop()

        missing_important = [c for c in _IMPORTANT_COLS if c not in df.columns]
        if missing_important:
            st.warning(
                f"The CSV is missing recommended columns: `{missing_important}`.  "
                "Missing values will be imputed using training-set statistics, "
                "which may reduce accuracy."
            )

        st.dataframe(df.head(10), use_container_width=True)

        # ---- Run inference --------------------------------------------- #
        try:
            predictions = predict_dataframe(df, bundle)
        except InferenceError as exc:
            st.error(str(exc))
            st.stop()

        st.success("Inference complete.")

        show_cols = [
            c for c in ["PassengerId", "Name", "Sex", "Pclass"]
            if c in predictions.columns
        ]
        show_cols += ["PredictedSurvived", "PredictedProbability"]
        st.dataframe(predictions[show_cols].head(50), use_container_width=True)

        # Probability histogram.
        threshold = float(bundle.metadata.get("threshold", 0.5))
        probs_arr = predictions["PredictedProbability"].to_numpy()
        hist_fig  = _prob_histogram(probs_arr, threshold)
        st.pyplot(hist_fig)
        plt.close(hist_fig)

        # Download button.
        csv_bytes = predictions.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download predictions CSV",
            data=csv_bytes,
            file_name="titanic_predictions.csv",
            mime="text/csv",
        )

        # ---- Evaluation (only when labels are available) --------------- #
        labels = extract_labels(df)
        if labels is None:
            st.info(
                "ℹ️ No usable `Survived` column found — showing predictions only.  \n"
                "This is expected for the Kaggle `test.csv`, which has no ground-truth "
                "labels.  Upload a CSV with a `Survived` column (values 0 or 1) to "
                "see evaluation metrics."
            )
        else:
            st.subheader("Evaluation against provided labels")
            metrics = compute_metrics(labels, probs_arr, threshold=threshold)
            _metrics_columns(metrics)
            st.code(
                classification_text_report(labels, probs_arr, threshold),
                language="text",
            )
            eval_cols = st.columns(2)

            cm_fig = plot_confusion_matrix(labels, probs_arr, threshold)
            eval_cols[0].pyplot(cm_fig)
            plt.close(cm_fig)

            roc_fig = plot_roc_curve(labels, probs_arr)
            if roc_fig is not None:
                eval_cols[1].pyplot(roc_fig)
                plt.close(roc_fig)
            else:
                eval_cols[1].info(
                    "ROC curve requires both classes present in the labels."
                )
