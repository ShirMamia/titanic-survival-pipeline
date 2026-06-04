"""Streamlit app for the Titanic survival classifier.

Two tabs:
  * "Training results" - shows the metrics and plots produced by ``train.py``.
  * "Run inference"     - upload a CSV or enter a path, load the saved model,
                          run predictions, view metrics/plots when labels are
                          present, and download the predictions.

Run with:
    streamlit run ds_app.py
"""
from __future__ import annotations

import json

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

st.set_page_config(page_title="Titanic Survival Classifier", page_icon="🚢", layout="wide")


@st.cache_resource(show_spinner=False)
def _load_bundle() -> ArtifactBundle:
    """Load and cache the artifact bundle for the session."""
    return load_artifacts()


def _metrics_columns(metrics: dict) -> None:
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
    cols[1].metric("Precision", f"{metrics['precision']:.3f}")
    cols[2].metric("Recall", f"{metrics['recall']:.3f}")
    cols[3].metric("F1", f"{metrics['f1']:.3f}")
    cols[4].metric("ROC-AUC", "n/a" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.3f}")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🚢 Titanic Survival Classifier")
st.caption(
    "PyTorch MLP trained by `train.py`. Use the tabs below to review training "
    "results or run inference on your own CSV."
)

tab_results, tab_infer = st.tabs(["📊 Training results", "🔮 Run inference"])

# --------------------------------------------------------------------------- #
# Tab 1: training results
# --------------------------------------------------------------------------- #
with tab_results:
    st.subheader("Held-out test performance")
    if not config.METRICS_PATH.exists():
        st.warning(
            "No training results found. Train the model first:\n\n"
            "```bash\npython train.py\n```"
        )
    else:
        with open(config.METRICS_PATH, "r", encoding="utf-8") as fh:
            metrics_payload = json.load(fh)
        _metrics_columns(metrics_payload["test"])

        st.markdown("**Classification report (test split)**")
        st.code(metrics_payload["classification_report"], language="text")

        cols = st.columns(2)
        if config.CONFUSION_MATRIX_PNG.exists():
            cols[0].image(str(config.CONFUSION_MATRIX_PNG), caption="Confusion matrix")
        if config.ROC_CURVE_PNG.exists():
            cols[1].image(str(config.ROC_CURVE_PNG), caption="ROC curve")
        if config.TRAINING_CURVE_PNG.exists():
            st.image(str(config.TRAINING_CURVE_PNG), caption="Training curves")

        if config.METADATA_PATH.exists():
            with st.expander("Model metadata"):
                st.json(json.load(open(config.METADATA_PATH, encoding="utf-8")))

# --------------------------------------------------------------------------- #
# Tab 2: inference
# --------------------------------------------------------------------------- #
with tab_infer:
    st.subheader("Predict survival on a CSV")

    # Load artifacts up front so errors surface immediately.
    try:
        bundle = _load_bundle()
    except InferenceError as exc:
        st.error(str(exc))
        st.stop()

    st.markdown(
        "Provide a Titanic-style CSV (columns such as `Pclass`, `Sex`, `Age`, "
        "`SibSp`, `Parch`, `Fare`, `Embarked`, `Name`, `Cabin`). If a `Survived` "
        "column is present, evaluation metrics and plots are shown too."
    )

    source = st.radio("Data source", ["Upload a CSV", "Enter a file path"], horizontal=True)
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
        st.dataframe(df.head(10), use_container_width=True)

        try:
            predictions = predict_dataframe(df, bundle)
        except InferenceError as exc:
            st.error(str(exc))
            st.stop()

        st.success("Inference complete.")
        show_cols = [c for c in ["PassengerId", "Name", "Sex", "Pclass"] if c in predictions.columns]
        show_cols += ["PredictedSurvived", "PredictedProbability"]
        st.dataframe(predictions[show_cols].head(50), use_container_width=True)

        # Download.
        csv_bytes = predictions.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download predictions CSV", data=csv_bytes,
            file_name="titanic_predictions.csv", mime="text/csv",
        )

        # Evaluation only when ground-truth labels exist.
        labels = extract_labels(df)
        threshold = float(bundle.metadata.get("threshold", 0.5))
        if labels is None:
            st.info(
                "No usable `Survived` column found - showing predictions only "
                "(this is expected for the Kaggle `test.csv`)."
            )
        else:
            st.subheader("Evaluation against provided labels")
            probs = predictions["PredictedProbability"].to_numpy()
            metrics = compute_metrics(labels, probs, threshold=threshold)
            _metrics_columns(metrics)

            st.code(classification_text_report(labels, probs, threshold), language="text")
            cols = st.columns(2)
            cols[0].pyplot(plot_confusion_matrix(labels, probs, threshold))
            roc = plot_roc_curve(labels, probs)
            if roc is not None:
                cols[1].pyplot(roc)
            else:
                cols[1].info("ROC curve needs both classes present in the labels.")
