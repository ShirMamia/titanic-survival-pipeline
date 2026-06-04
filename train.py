#!/usr/bin/env python
"""Standalone training script for the Titanic survival classifier.

Pipeline:
  1.  Validate CLI arguments.
  2.  Set seeds for reproducibility.
  3.  Load the labelled Kaggle train.csv (or the bundled sample with
      ``--use-sample``).  No silent fallback — real data or explicit sample.
  4.  Engineer features — strictly row-wise, no dataset-level statistics.
  5.  Stratified split into train / validation / test.
  6a. Fit TabularPreprocessor on TRAIN; transform all splits (PyTorch path).
  6b. Fit baseline ColumnTransformer on TRAIN; transform all splits
      (only when baselines are enabled).
  7.  Build DataLoaders; train TitanicEmbeddingMLP with early stopping.
  8.  Tune decision threshold on the VALIDATION split (maximise F1).
  9.  Evaluate the MLP on the held-out TEST split (once, with tuned threshold).
 10.  Train classical ML baselines (LR, RF, GB) for comparison (unless
      ``--skip-baselines``).
 11.  Save model weights, preprocessors, metadata, metrics, and plots.

Leakage audit:
  * ``engineer_features`` applies only deterministic per-row transformations.
    No group statistics, frequency encoding, or quantile fitting before split.
  * ``TabularPreprocessor`` (vocab + imputer + scaler) fitted on train only.
  * Baseline ``ColumnTransformer`` fitted on train only.
  * Threshold tuning uses the validation split only.
  * Final evaluation uses the held-out test split once.
  * Kaggle ``test.csv`` is never used — it has no ``Survived`` label.

Run:
    python train.py                        # download from Kaggle (credentials required)
    python train.py --use-sample           # bundled synthetic demo dataset
    python train.py --skip-baselines       # skip classical ML comparison
    python train.py --epochs 200 --lr 5e-4 --hidden-dims 128 64 32
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src import config
from src.baselines import train_baselines
from src.data import DataError, load_labelled_data
from src.evaluate import (
    classification_text_report,
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
    tune_threshold,
)
from src.model import TitanicEmbeddingMLP
from src.preprocessing import (
    TabularPreprocessor,
    build_preprocessor,
    engineer_features,
)
from src.train_utils import make_loader, set_seed, train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("train")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _to_torch_long(X: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(X), dtype=torch.long)


def _to_torch_float32(X: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(X), dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Argument parsing & validation
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Titanic survival classifier.")
    d = config.DEFAULTS
    p.add_argument(
        "--use-sample", action="store_true",
        help="Train on the bundled synthetic sample instead of Kaggle data.  "
             "Metrics from this run are illustrative only.",
    )
    p.add_argument(
        "--no-download", action="store_true",
        help="Do not attempt to download from Kaggle; require local "
             "data/raw/train.csv.",
    )
    p.add_argument(
        "--skip-baselines", action="store_true",
        help="Skip classical ML baseline training (saves time on slow machines).",
    )
    p.add_argument("--epochs",       type=int,   default=d["epochs"])
    p.add_argument("--batch-size",   type=int,   default=d["batch_size"])
    p.add_argument("--lr",           type=float, default=d["lr"])
    p.add_argument("--weight-decay", type=float, default=d["weight_decay"])
    p.add_argument("--dropout",      type=float, default=d["dropout"])
    p.add_argument("--hidden-dims",  type=int,   nargs="+", default=d["hidden_dims"])
    p.add_argument("--patience",     type=int,   default=d["patience"])
    p.add_argument("--test-size",    type=float, default=d["test_size"])
    p.add_argument("--val-size",     type=float, default=d["val_size"])
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Decision threshold override [0, 1].  If omitted, auto-tuned on "
             "the validation split to maximise F1.",
    )
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> list[str]:
    """Return a list of validation error messages; empty list means all valid."""
    errors: list[str] = []
    if not (0.0 < args.test_size < 1.0):
        errors.append(f"--test-size must be in (0, 1); got {args.test_size}.")
    if not (0.0 < args.val_size < 1.0):
        errors.append(f"--val-size must be in (0, 1); got {args.val_size}.")
    if args.test_size + args.val_size >= 1.0:
        errors.append(
            f"--test-size + --val-size must be < 1; "
            f"got {args.test_size + args.val_size:.3f}."
        )
    if args.epochs <= 0:
        errors.append(f"--epochs must be > 0; got {args.epochs}.")
    if args.batch_size <= 0:
        errors.append(f"--batch-size must be > 0; got {args.batch_size}.")
    if args.lr <= 0.0:
        errors.append(f"--lr must be > 0; got {args.lr}.")
    if args.weight_decay < 0.0:
        errors.append(f"--weight-decay must be >= 0; got {args.weight_decay}.")
    if not (0.0 <= args.dropout < 1.0):
        errors.append(f"--dropout must be in [0, 1); got {args.dropout}.")
    if args.patience <= 0:
        errors.append(f"--patience must be > 0; got {args.patience}.")
    if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
        errors.append(f"--threshold must be in [0, 1]; got {args.threshold}.")
    return errors


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    args = parse_args()
    errors = validate_args(args)
    if errors:
        for err in errors:
            logger.error("Invalid argument: %s", err)
        return 1

    set_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load labelled data -------------------------------------------- #
    try:
        df = load_labelled_data(
            use_sample=args.use_sample,
            download=not args.no_download,
        )
    except DataError as exc:
        logger.error("Could not obtain data: %s", exc)
        return 1

    if args.use_sample:
        logger.warning(
            "Training on the SYNTHETIC SAMPLE dataset.  "
            "Reported metrics are for pipeline demonstration only — "
            "they are NOT from the real Titanic data."
        )

    if config.TARGET not in df.columns:
        logger.error(
            "Loaded data has no '%s' column; cannot train.", config.TARGET
        )
        return 1

    df = df.dropna(subset=[config.TARGET]).reset_index(drop=True)
    logger.info("Loaded %d labelled rows.", len(df))

    # --- 2. Feature engineering (row-wise, leakage-free) ------------------- #
    try:
        X_all = engineer_features(df)
    except ValueError as exc:
        logger.error("Feature engineering failed: %s", exc)
        return 1
    y_all = df[config.TARGET].astype(int).to_numpy()

    # --- 3. Stratified train / val / test split ---------------------------- #
    X_trv, X_test, y_trv, y_test = train_test_split(
        X_all, y_all,
        test_size=args.test_size,
        stratify=y_all,
        random_state=args.seed,
    )
    val_relative = args.val_size / (1.0 - args.test_size)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trv, y_trv,
        test_size=val_relative,
        stratify=y_trv,
        random_state=args.seed,
    )
    logger.info(
        "Split sizes -> train=%d, val=%d, test=%d",
        len(X_tr), len(X_val), len(X_test),
    )

    # --- 4a. PyTorch preprocessing: TabularPreprocessor ------------------- #
    pytorch_prep = TabularPreprocessor()
    pytorch_prep.fit(X_tr)

    X_cat_tr,  X_cont_tr  = pytorch_prep.transform(X_tr)
    X_cat_val, X_cont_val = pytorch_prep.transform(X_val)
    X_cat_te,  X_cont_te  = pytorch_prep.transform(X_test)
    n_cont = pytorch_prep.n_cont
    logger.info(
        "PyTorch preprocessing: %d cat features, %d cont features.",
        len(pytorch_prep.cat_features), n_cont,
    )

    # --- 4b. Baseline preprocessing (only if baselines are enabled) ------- #
    baseline_prep  = None
    X_tr_flat      = None
    X_test_flat    = None
    if not args.skip_baselines:
        baseline_prep = build_preprocessor()
        X_tr_flat     = baseline_prep.fit_transform(X_tr)
        X_test_flat   = baseline_prep.transform(X_test)
        logger.info("Baseline preprocessing: %d features.", X_tr_flat.shape[1])

    # --- 5. Build DataLoaders --------------------------------------------- #
    # TitanicEmbeddingMLP uses LayerNorm (not BatchNorm), so batch size 1 is
    # safe.  We do not drop the last training batch.
    train_dl = make_loader(
        X_cat_tr, X_cont_tr, y_tr,
        args.batch_size, shuffle=True, drop_last=False,
    )
    val_dl = make_loader(
        X_cat_val, X_cont_val, y_val,
        args.batch_size, shuffle=False, drop_last=False,
    )

    # --- 6. Build & train TitanicEmbeddingMLP (final required model) ------ #
    emb_dims = pytorch_prep.embedding_dims
    model = TitanicEmbeddingMLP(
        categorical_features=pytorch_prep.cat_features,
        category_sizes=pytorch_prep.category_sizes,
        embedding_dims=emb_dims,
        n_cont=n_cont,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    )
    history = train_model(
        model, train_dl, val_dl,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
    )

    # --- 7. Get predicted probabilities ----------------------------------- #
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(
            model(_to_torch_long(X_cat_val), _to_torch_float32(X_cont_val))
        ).numpy().ravel()
        test_prob = torch.sigmoid(
            model(_to_torch_long(X_cat_te), _to_torch_float32(X_cont_te))
        ).numpy().ravel()

    # --- 8. Tune decision threshold on the VALIDATION split --------------- #
    if args.threshold is None:
        threshold, tuned_f1 = tune_threshold(y_val, val_prob)
        threshold_tuned = True
        logger.info("Auto-tuned threshold=%.4f (val F1=%.4f)", threshold, tuned_f1)
    else:
        threshold = args.threshold
        threshold_tuned = False
        logger.info("Using manually specified threshold=%.4f", threshold)

    # --- 9. Evaluate TitanicEmbeddingMLP on held-out TEST split ----------- #
    test_metrics = compute_metrics(y_test, test_prob, threshold=threshold)
    val_metrics  = compute_metrics(y_val,  val_prob,  threshold=threshold)
    logger.info("Test metrics: %s", json.dumps(test_metrics, indent=2))
    print("\n=== Classification report (held-out test set) ===")
    print(classification_text_report(y_test, test_prob, threshold=threshold))

    # --- 10. Classical ML baselines (comparison only) --------------------- #
    baseline_results: dict = {}
    if not args.skip_baselines:
        logger.info(
            "Training classical ML baselines (LR, RF, GB) for comparison..."
        )
        try:
            baseline_results = train_baselines(
                X_tr_flat, y_tr, X_test_flat, y_test, seed=args.seed
            )
            with open(config.BASELINES_METRICS_PATH, "w", encoding="utf-8") as fh:
                json.dump(baseline_results, fh, indent=2)
            logger.info(
                "Baseline metrics saved to %s", config.BASELINES_METRICS_PATH
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Baseline training failed: %s", exc)
            logger.error(
                "Re-run with --skip-baselines to bypass, "
                "or investigate the error above."
            )
            return 1
    else:
        logger.info("Skipping classical ML baselines (--skip-baselines).")

    # --- 11. Save plots ---------------------------------------------------- #
    plot_confusion_matrix(y_test, test_prob, threshold).savefig(
        config.CONFUSION_MATRIX_PNG, dpi=120
    )
    roc_fig = plot_roc_curve(y_test, test_prob)
    if roc_fig is not None:
        roc_fig.savefig(config.ROC_CURVE_PNG, dpi=120)
    plot_training_curves(history).savefig(config.TRAINING_CURVE_PNG, dpi=120)

    # --- 12. Persist artifacts -------------------------------------------- #
    torch.save(model.state_dict(), config.MODEL_PATH)
    joblib.dump(pytorch_prep, config.PYTORCH_PREPROCESSOR_PATH)
    if not args.skip_baselines:
        joblib.dump(baseline_prep, config.BASELINE_PREPROCESSOR_PATH)

    # Paths stored as repository-relative strings (portable across machines).
    pytorch_prep_rel  = str(
        config.PYTORCH_PREPROCESSOR_PATH.relative_to(config.ROOT_DIR)
    )
    baseline_prep_rel = (
        str(config.BASELINE_PREPROCESSOR_PATH.relative_to(config.ROOT_DIR))
        if not args.skip_baselines else None
    )

    metadata = {
        "model_type":            config.MODEL_TYPE,
        "categorical_features":  list(pytorch_prep.cat_features),
        "continuous_features":   list(pytorch_prep.cont_features),
        "category_sizes":        dict(pytorch_prep.category_sizes),
        "embedding_dims":        dict(emb_dims),
        "hidden_dims":           list(args.hidden_dims),
        "dropout":               float(args.dropout),
        "threshold":             float(threshold),
        "threshold_tuned":       threshold_tuned,
        "pytorch_preprocessor_path":  pytorch_prep_rel,
        "baseline_preprocessor_path": baseline_prep_rel,
        "feature_engineering_version": config.FEATURE_ENGINEERING_VERSION,
        "seed":     int(args.seed),
        "n_train":  int(len(X_tr)),
        "n_val":    int(len(X_val)),
        "n_test":   int(len(X_test)),
        "data_source": "sample_synthetic" if args.use_sample else "kaggle_train_csv",
        "training_config": {
            "epochs":            args.epochs,
            "batch_size":        args.batch_size,
            "lr":                args.lr,
            "weight_decay":      args.weight_decay,
            "patience":          args.patience,
            "test_size":         args.test_size,
            "val_size":          args.val_size,
            "baselines_enabled": not args.skip_baselines,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "framework":   {"torch": torch.__version__},
    }
    with open(config.METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    metrics_payload = {
        "test": test_metrics,
        "validation": val_metrics,
        "classification_report": classification_text_report(
            y_test, test_prob, threshold
        ),
    }
    with open(config.METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2)

    logger.info("Saved artifacts to %s", config.ARTIFACTS_DIR)
    logger.info("Saved plots to %s", config.REPORTS_DIR)
    logger.info(
        "Done. data_source=%s | threshold=%.4f (tuned=%s) | "
        "test accuracy=%.4f | F1=%.4f | AUC=%s",
        metadata["data_source"],
        threshold,
        threshold_tuned,
        test_metrics["accuracy"],
        test_metrics["f1"],
        f"{test_metrics['roc_auc']:.4f}" if test_metrics["roc_auc"] else "n/a",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
