#!/usr/bin/env python
"""Generate a small *synthetic* Titanic-style sample dataset.

This file does NOT contain the real Kaggle data. It produces a schema-faithful
demo CSV so the pipeline can be run end-to-end without Kaggle credentials. The
survival label is sampled from a logistic model with realistic effects (women,
children, and higher classes survive more often) so the trained model learns
something non-trivial.

Usage:
    python scripts/generate_sample.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

SEED = 7
N = 260

FIRST_NAMES_M = ["John", "William", "James", "George", "Charles", "Thomas", "Henry"]
FIRST_NAMES_F = ["Mary", "Anna", "Margaret", "Elizabeth", "Helen", "Ruth", "Florence"]
LAST_NAMES = ["Smith", "Brown", "Johnson", "Andersson", "Murphy", "Sundman",
              "Carlsson", "Lindblom", "Olsson", "Williams", "Davies", "Petroff"]
DECKS = ["A", "B", "C", "D", "E", "F", "G"]


def main() -> None:
    rng = np.random.default_rng(SEED)
    config.SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    pclass = rng.choice([1, 2, 3], size=N, p=[0.24, 0.21, 0.55])
    sex = rng.choice(["male", "female"], size=N, p=[0.64, 0.36])
    age = np.clip(rng.normal(29, 13, size=N), 0.5, 80).round(1)
    # ~20% of ages missing, as in the real dataset.
    age[rng.random(N) < 0.20] = np.nan
    sibsp = rng.choice([0, 1, 2, 3, 4], size=N, p=[0.68, 0.21, 0.06, 0.03, 0.02])
    parch = rng.choice([0, 1, 2, 3], size=N, p=[0.76, 0.13, 0.09, 0.02])
    fare = np.where(
        pclass == 1, rng.gamma(4, 20, N),
        np.where(pclass == 2, rng.gamma(3, 7, N), rng.gamma(2, 6, N)),
    ).round(2)
    embarked = rng.choice(["S", "C", "Q"], size=N, p=[0.72, 0.19, 0.09]).astype(object)
    embarked[rng.random(N) < 0.01] = np.nan  # a couple of missing values

    # Cabin present mostly for 1st class; first letter is the deck.
    cabin = np.array([
        f"{rng.choice(DECKS)}{rng.integers(1, 120)}"
        if (pc == 1 and rng.random() < 0.8) or rng.random() < 0.1 else np.nan
        for pc in pclass
    ], dtype=object)

    titles, names = [], []
    for s, a in zip(sex, age):
        if s == "female":
            title = "Mrs" if (np.isnan(a) or a >= 18) and rng.random() < 0.55 else "Miss"
            fn = rng.choice(FIRST_NAMES_F)
        else:
            title = "Master" if (not np.isnan(a) and a < 14) else "Mr"
            fn = rng.choice(FIRST_NAMES_M)
        titles.append(title)
        names.append(f"{rng.choice(LAST_NAMES)}, {title}. {fn}")

    # Logistic survival model.
    age_filled = np.where(np.isnan(age), 29.0, age)
    logit = (
        -1.0
        + 2.4 * (sex == "female")
        + 0.9 * (pclass == 1) + 0.2 * (pclass == 2) - 0.6 * (pclass == 3)
        - 0.02 * age_filled
        + 0.6 * (np.char.array(titles) == "Master")
        + 0.004 * np.clip(fare, 0, 200)
        - 0.25 * (sibsp + parch)
    )
    prob = 1 / (1 + np.exp(-logit))
    survived = (rng.random(N) < prob).astype(int)

    df = pd.DataFrame({
        "PassengerId": np.arange(1, N + 1),
        "Survived": survived,
        "Pclass": pclass,
        "Name": names,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": [f"TICK{rng.integers(10000, 99999)}" for _ in range(N)],
        "Fare": fare,
        "Cabin": cabin,
        "Embarked": embarked,
    })

    df.to_csv(config.SAMPLE_CSV, index=False)
    print(f"Wrote {len(df)} rows to {config.SAMPLE_CSV} "
          f"(survival rate {survived.mean():.2%}).")


if __name__ == "__main__":
    main()
