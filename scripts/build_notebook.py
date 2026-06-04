#!/usr/bin/env python
"""Build notebooks/eda.ipynb programmatically (no manual JSON editing).

Run: python scripts/build_notebook.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "eda.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS = [
    md("# Titanic EDA\n"
       "\n"
       "Exploratory data analysis for the survival-classification pipeline. The goal "
       "is to understand the data, justify the preprocessing and feature-engineering "
       "choices made in `src/preprocessing.py`, and surface caveats relevant to "
       "modelling.\n"
       "\n"
       "> The modelling code only ever uses the Kaggle **train.csv** (it is the only "
       "file with the `Survived` label). The Kaggle **test.csv** is unlabelled and is "
       "*not* used for evaluation.\n"),

    md("## 1. Setup"),
    code("import sys\n"
         "from pathlib import Path\n"
         "\n"
         "# Make the project importable when running from notebooks/.\n"
         "sys.path.insert(0, str(Path.cwd().parent))\n"
         "\n"
         "import numpy as np\n"
         "import pandas as pd\n"
         "import matplotlib.pyplot as plt\n"
         "import seaborn as sns\n"
         "\n"
         "from src import config\n"
         "from src.data import load_labelled_data\n"
         "from src.preprocessing import engineer_features\n"
         "\n"
         "sns.set_theme(style='whitegrid')\n"
         "pd.set_option('display.max_columns', 50)"),

    md("## 2. Load the data\n"
       "\n"
       "`load_labelled_data` uses the real Kaggle `train.csv` if present in "
       "`data/raw/`, otherwise it falls back to the bundled synthetic sample so this "
       "notebook always runs. Pass `use_sample=True` to force the sample."),
    code("df = load_labelled_data(use_sample=False, download=False)\n"
         "print('Shape:', df.shape)\n"
         "df.head()"),

    md("## 3. Schema, dtypes, and missingness"),
    code("df.info()"),
    code("# Missing-value counts and percentages.\n"
         "missing = df.isna().sum().to_frame('missing')\n"
         "missing['pct'] = (missing['missing'] / len(df) * 100).round(1)\n"
         "missing.sort_values('missing', ascending=False)"),
    md("**What to look for.** In the real Titanic data, `Age` (~20%), `Cabin` (~77%), "
       "and `Embarked` (2 rows) are missing. This motivates median imputation for "
       "`Age`, deriving a coarse `Deck` from `Cabin` (with an explicit `Unknown` "
       "level) rather than dropping it, and most-frequent imputation for `Embarked`."),

    md("## 4. Target balance"),
    code("ax = df[config.TARGET].value_counts().sort_index().plot(kind='bar')\n"
         "ax.set_xticklabels(config.CLASS_NAMES, rotation=0)\n"
         "ax.set_title('Survival counts'); ax.set_ylabel('Passengers')\n"
         "plt.show()\n"
         "print('Survival rate: {:.1%}'.format(df[config.TARGET].mean()))"),
    md("The classes are imbalanced (~38% survived in the real data). This is why we "
       "report precision/recall/F1/ROC-AUC in addition to accuracy, and why the "
       "train/val/test splits are **stratified** on the target."),

    md("## 5. Numeric feature distributions"),
    code("num_cols = ['Age', 'Fare', 'SibSp', 'Parch']\n"
         "df[num_cols].describe()"),
    code("fig, axes = plt.subplots(2, 2, figsize=(11, 7))\n"
         "for ax, col in zip(axes.ravel(), num_cols):\n"
         "    sns.histplot(df[col].dropna(), kde=True, ax=ax)\n"
         "    ax.set_title(col)\n"
         "plt.tight_layout(); plt.show()"),
    md("`Fare` is strongly right-skewed and `Age` is roughly bell-shaped. We "
       "`StandardScaler` numeric features so the MLP trains stably; tree models would "
       "not need this, but a neural net benefits from standardised inputs."),

    md("## 6. Survival by categorical features"),
    code("cat_cols = ['Sex', 'Pclass', 'Embarked']\n"
         "fig, axes = plt.subplots(1, 3, figsize=(13, 4))\n"
         "for ax, col in zip(axes, cat_cols):\n"
         "    sns.barplot(data=df, x=col, y=config.TARGET, ax=ax, errorbar=None)\n"
         "    ax.set_title(f'Survival rate by {col}'); ax.set_ylabel('P(survived)')\n"
         "plt.tight_layout(); plt.show()"),
    md("**Key signal.** Sex is the single most predictive feature (women survived far "
       "more often), followed by passenger class. This is consistent with "
       "\"women and children first\" and is exactly the structure the model should "
       "capture."),

    md("## 7. Engineered features\n"
       "\n"
       "We reuse the *exact* feature-engineering function the training script uses, "
       "so the EDA reflects what the model actually sees."),
    code("feat = engineer_features(df)\n"
         "feat['Survived'] = df[config.TARGET].values\n"
         "feat.head()"),
    code("# FamilySize / IsAlone and Title vs. survival.\n"
         "fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n"
         "sns.barplot(data=feat, x='FamilySize', y='Survived', ax=axes[0], errorbar=None)\n"
         "axes[0].set_title('Survival by family size')\n"
         "sns.barplot(data=feat, x='IsAlone', y='Survived', ax=axes[1], errorbar=None)\n"
         "axes[1].set_title('Survival: alone vs. not')\n"
         "sns.barplot(data=feat, x='Title', y='Survived', ax=axes[2], errorbar=None)\n"
         "axes[2].set_title('Survival by title'); axes[2].tick_params(axis='x', rotation=30)\n"
         "plt.tight_layout(); plt.show()"),
    md("Mid-size families fare better than singletons or very large families, and "
       "`Title` (Mr/Mrs/Miss/Master/...) encodes sex + age + social status in one "
       "feature - which is why we engineer it from `Name`."),

    md("## 8. Correlations (numeric view)"),
    code("corr = feat.drop(columns=['Title', 'Deck', 'Sex', 'Embarked'], errors='ignore')\\\n"
         "           .apply(pd.to_numeric, errors='coerce').corr()\n"
         "plt.figure(figsize=(7, 6))\n"
         "sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)\n"
         "plt.title('Correlation matrix (numeric & engineered)'); plt.show()"),

    md("## 9. EDA conclusions -> design choices\n"
       "\n"
       "- **Impute** `Age`/`Fare` with the median (skewed) and categoricals with the "
       "mode; keep `Cabin` as a coarse `Deck` with an explicit `Unknown` level.\n"
       "- **Scale** numeric features (the MLP needs standardised inputs).\n"
       "- **One-hot encode** categoricals with `handle_unknown='ignore'` for robust "
       "inference.\n"
       "- **Engineer** `FamilySize`, `IsAlone`, `Title`, `Deck` - all row-wise and "
       "therefore leakage-free.\n"
       "- **Stratify** splits on `Survived` because the target is imbalanced.\n"
       "- **Report** precision/recall/F1/ROC-AUC, not just accuracy.\n"
       "- **Avoid leakage**: the preprocessor is fit on the training split only "
       "(see `train.py`).\n"),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {OUT} with {len(CELLS)} cells.")


if __name__ == "__main__":
    main()
