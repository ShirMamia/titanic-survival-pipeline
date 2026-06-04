#!/usr/bin/env python
"""Standalone training script for the Titanic survival classifier.

Pipeline:
  1. Set seeds for reproducibility.
  2. Load the labelled Kaggle ``train.csv`` (or the bundled sample).
  3. Engineer features (leakage-free, row-wise).
  4. Stratified split into train / validation / test.
  5. Fit the preprocessor on the TRAIN split only; transform all splits.
  6. Build and train a PyTorch MLP with early stopping.
  7. Evaluate on the held-out test split.
  8. Save model weights, the fitted preprocessor, metadata, metrics, and plots.

Run:
    python train.py                 # fetch from Kaggle (or fall back to sample)
    python train.py --use-sample    # force the bundled demo dataset
    python train.py --epochs 200 --lr 5e-4
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import torch
from sklearn.model_selection import train_test_split

from src import config
from src.data import DataError, load_labelled_data
from src.evaluate import (
    classification_text_report,
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
)
from src.model import TitanicMLP
from src.preprocessing import (
    build_preprocessor,
    engineer_features,
    get_output_feature_names,
)
from src.train_utils import set_seed, train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Titanic survival classifier.")
    d = config.DEFAULTS
    p.add_argument("--use-sample", action="store_true",
                   help="Train on the bundled synthetic sample instead of Kaggle data.")
    p.add_argument("--no-download", action="store_true",
                   help="Do not attempt to download from Kaggle; require local data.")
    p.add_argument("--epochs", type=int, default=d["epochs"])
    p.add_argument("--batch-size", type=int, default=d["batch_size"])
    p.add_argument("--lr", type=float, default=d["lr"])
    p.add_argument("--weight-decay", type=float, default=d["weight_decay"])
    p.add_argument("--dropout", type=float, default=d["dropout"])
    p.add_argument("--hidden-dims", type=int, nargs="+", default=d["hidden_dims"])
    p.add_argument("--patience", type=int, default=d["patience"])
    p.add_argument("--test-size", type=float, default=d["test_size"])
    p.add_argument("--val-size", type=float, default=d["val_size"])
    p.add_argument("--threshold", type=float, default=d["threshold"])
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))  # CPU friendly

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load labelled data -------------------------------------------- #
    try:
        df = load_labelled_data(use_sample=args.use_sample, download=not args.no_download)
    except DataError as exc:
        logger.error("Could not obtain data: %s", exc)
        return 1

    if config.TARGET not in df.columns:
        logger.error("Loaded data has no '%s' column; cannot train.", config.TARGET)
        return 1

    df = df.dropna(subset=[config.TARGET]).reset_index(drop=True)
    logger.info("Loaded %d labelled rows.", len(df))

    # --- 2. Feature engineering ------------------------------------------- #
    X_all = engineer_features(df)
    y_all = df[config.TARGET].astype(int).to_numpy()

    # --- 3. Stratified train/val/test split ------------------------------- #
    # First peel off the test set, then split the remainder into train/val.
    X_trv, X_test, y_trv, y_test = train_test_split(
        X_all, y_all, test_size=args.test_size, stratify=y_all, random_state=args.seed
    )
    val_relative = args.val_size / (1.0 - args.test_size)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trv, y_trv, test_size=val_relative, stratify=y_trv, random_state=args.seed
    )
    logger.info("Split sizes -> train=%d, val=%d, test=%d", len(X_tr), len(X_val), len(X_test))

    # --- 4. Fit preprocessor on TRAIN only, transform everything ---------- #
    preprocessor = build_preprocessor()
    X_tr_t = preprocessor.fit_transform(X_tr)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)
    feature_names = get_output_feature_names(preprocessor)
    input_dim = X_tr_t.shape[1]
    logger.info("Preprocessed feature dimension: %d", input_dim)

    # --- 5. Build & train the model --------------------------------------- #
    model = TitanicMLP(input_dim=input_dim, hidden_dims=args.hidden_dims, dropout=args.dropout)
    history = train_model(
        model, X_tr_t, y_tr, X_val_t, y_val,
        lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
        epochs=args.epochs, patience=args.patience,
    )

    # --- 6. Evaluate on the held-out test split --------------------------- #
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.as_tensor(X_test_t, dtype=torch.float32))
        test_prob = torch.sigmoid(test_logits).numpy().ravel()
        val_logits = model(torch.as_tensor(X_val_t, dtype=torch.float32))
        val_prob = torch.sigmoid(val_logits).numpy().ravel()

    test_metrics = compute_metrics(y_test, test_prob, threshold=args.threshold)
    val_metrics = compute_metrics(y_val, val_prob, threshold=args.threshold)
    logger.info("Test metrics: %s", json.dumps(test_metrics, indent=2))
    print("\n=== Classification report (held-out test set) ===")
    print(classification_text_report(y_test, test_prob, threshold=args.threshold))

    # --- 7. Save plots ---------------------------------------------------- #
    plot_confusion_matrix(y_test, test_prob, args.threshold).savefig(config.CONFUSION_MATRIX_PNG, dpi=120)
    roc_fig = plot_roc_curve(y_test, test_prob)
    if roc_fig is not None:
        roc_fig.savefig(config.ROC_CURVE_PNG, dpi=120)
    plot_training_curves(history).savefig(config.TRAINING_CURVE_PNG, dpi=120)

    # --- 8. Persist artifacts --------------------------------------------- #
    import joblib

    torch.save(model.state_dict(), config.MODEL_PATH)
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)

    metadata = {
        "input_dim": int(input_dim),
        "hidden_dims": list(args.hidden_dims),
        "dropout": float(args.dropout),
        "threshold": float(args.threshold),
        "seed": int(args.seed),
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "output_feature_names": feature_names,
        "class_names": config.CLASS_NAMES,
        "data_source": "sample" if args.use_sample else "kaggle_or_sample",
        "n_train": len(X_tr), "n_val": len(X_val), "n_test": len(X_test),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "framework": {"torch": torch.__version__},
    }
    with open(config.METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    metrics_payload = {
        "test": test_metrics,
        "validation": val_metrics,
        "classification_report": classification_text_report(y_test, test_prob, args.threshold),
    }
    with open(config.METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2)

    logger.info("Saved artifacts to %s", config.ARTIFACTS_DIR)
    logger.info("Saved plots to %s", config.REPORTS_DIR)
    logger.info("Done. Test accuracy=%.4f, F1=%.4f, AUC=%s",
                test_metrics["accuracy"], test_metrics["f1"], test_metrics["roc_auc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
