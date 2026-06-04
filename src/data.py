"""Data acquisition and loading.

Responsibilities:
  * Fetch the Titanic competition data from Kaggle (with clear, actionable
    errors when credentials are missing).
  * Load the labelled ``train.csv`` into a DataFrame.

The bundled synthetic sample (``data/sample/sample_titanic.csv``) is used
**only** when ``use_sample=True`` is passed explicitly.  The pipeline never
silently substitutes the sample for missing real data; if Kaggle credentials
are absent or the download fails, a ``DataError`` is raised immediately.

Only the Kaggle *train* file is ever used for modelling, because the Kaggle
*test* file has no ``Survived`` column and therefore cannot be evaluated.
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


class DataError(RuntimeError):
    """Raised when the dataset cannot be located or fetched."""


def download_titanic_data(force: bool = False) -> Path:
    """Download and extract the Titanic competition data via the Kaggle API.

    Parameters
    ----------
    force:
        Re-download even if ``data/raw/train.csv`` already exists.

    Returns
    -------
    Path to the extracted ``train.csv``.

    Raises
    ------
    DataError
        If the Kaggle package or credentials are missing, or the competition
        rules have not been accepted.
    """
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if config.RAW_TRAIN_CSV.exists() and not force:
        logger.info("Found existing %s - skipping download.", config.RAW_TRAIN_CSV)
        return config.RAW_TRAIN_CSV

    # Import lazily so the rest of the pipeline does not hard-depend on kaggle.
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DataError(
            "The 'kaggle' package is not installed. Install requirements with "
            "`pip install -r requirements.txt`, or pass --use-sample to train "
            "on the bundled demo data."
        ) from exc

    try:
        api = KaggleApi()
        api.authenticate()  # reads kaggle.json or KAGGLE_* env vars
    except Exception as exc:  # noqa: BLE001 - kaggle raises bare Exceptions
        raise DataError(
            "Kaggle authentication failed. Place your API token at the path "
            "that matches your OS:\n"
            "  Linux / macOS : ~/.kaggle/kaggle.json  (chmod 600 recommended)\n"
            "  Windows       : C:\\Users\\<YOUR_USER>\\.kaggle\\kaggle.json\n"
            "Alternatively set the KAGGLE_USERNAME and KAGGLE_KEY environment "
            "variables. See README.md §'Kaggle API setup' for details.\n"
            f"Original error: {exc}"
        ) from exc

    try:
        logger.info("Downloading the Titanic competition data from Kaggle...")
        api.competition_download_files(
            "titanic", path=str(config.RAW_DATA_DIR), quiet=False
        )
    except Exception as exc:  # noqa: BLE001
        raise DataError(
            "Failed to download the Titanic data. Make sure you have joined the "
            "competition and accepted its rules at "
            "https://www.kaggle.com/competitions/titanic/rules . "
            f"Original error: {exc}"
        ) from exc

    # The API saves a `titanic.zip`; extract train.csv (and test.csv) from it.
    zip_path = config.RAW_DATA_DIR / "titanic.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(config.RAW_DATA_DIR)
        zip_path.unlink(missing_ok=True)

    if not config.RAW_TRAIN_CSV.exists():
        raise DataError(
            f"Download completed but {config.RAW_TRAIN_CSV} was not found. "
            "Inspect the contents of data/raw/."
        )

    logger.info("Titanic data ready at %s", config.RAW_TRAIN_CSV)
    return config.RAW_TRAIN_CSV


def load_labelled_data(use_sample: bool = False, download: bool = True) -> pd.DataFrame:
    """Return the labelled Titanic training frame (contains ``Survived``).

    Resolution order (no silent fallback at any step):
      1. ``use_sample=True``             → bundled synthetic sample (demo only).
      2. ``data/raw/train.csv`` exists   → load it directly.
      3. ``download=True``               → attempt Kaggle API download.
      4. otherwise                       → raise ``DataError``.

    If the Kaggle download fails (step 3), ``DataError`` is raised immediately.
    The synthetic sample is **never** substituted for missing real data without
    an explicit ``use_sample=True``.

    Parameters
    ----------
    use_sample:
        When ``True``, always use the bundled synthetic demo CSV.
    download:
        When ``True`` (and ``use_sample`` is ``False`` and the local file is
        absent), attempt to download from Kaggle.  Corresponds to the
        ``--no-download`` flag being absent in ``train.py``.
    """
    if use_sample:
        logger.info("--use-sample: loading bundled synthetic sample (demo mode).")
        return load_csv(config.SAMPLE_CSV)

    if config.RAW_TRAIN_CSV.exists():
        logger.info("Loading Kaggle train.csv from %s", config.RAW_TRAIN_CSV)
        return load_csv(config.RAW_TRAIN_CSV)

    if download:
        # Raises DataError on auth / network failure; does NOT fall back.
        path = download_titanic_data()
        return load_csv(path)

    raise DataError(
        f"Kaggle training data not found at {config.RAW_TRAIN_CSV}.\n"
        "To fix this, choose one of:\n"
        "  1. Run `python train.py` to download automatically (Kaggle credentials required).\n"
        "  2. Copy train.csv manually to data/raw/train.csv.\n"
        "  3. Run `python train.py --use-sample` to train on the bundled "
        "synthetic demo dataset (metrics will be illustrative, not from real data).\n"
        "See README.md §'Kaggle API setup' for credential instructions."
    )


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file with friendly error messages for common failures."""
    path = Path(path)
    if not path.exists():
        raise DataError(f"CSV file not found: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataError(f"CSV file is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise DataError(f"CSV file is malformed and could not be parsed: {path}") from exc
    if df.empty:
        raise DataError(f"CSV file contains no rows: {path}")
    return df
