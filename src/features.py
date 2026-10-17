"""
Preprocessing and feature pipeline for the BRFSS heart-disease model.

Builds a scikit-learn ColumnTransformer that:
  * median-imputes and standard-scales continuous features,
  * most-frequent-imputes and one-hot-encodes categorical survey codes.

All statistics are learned on the TRAINING data only (fit on train, transform
on test) to avoid leakage from the temporal 2023 test set.

Also provides the temporal train/test split helper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402


def load_processed() -> pd.DataFrame:
    """Load the combined harmonised parquet produced by data_ingestion."""
    path = C.PROCESSED_DIR / "brfss_combined.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data_ingestion` first."
        )
    return pd.read_parquet(path)


def temporal_split(df: pd.DataFrame):
    """Split by survey year: 2021+2022 -> train, 2023 -> temporal test."""
    train = df[df["survey_year"].isin(C.TRAIN_YEARS)].copy()
    test = df[df["survey_year"].isin(C.TEST_YEARS)].copy()

    X_train = train[C.ALL_FEATURES]
    y_train = train[C.TARGET]
    X_test = test[C.ALL_FEATURES]
    y_test = test[C.TARGET]
    return X_train, X_test, y_train, y_test, train, test


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: scale continuous, one-hot categorical (train-fit)."""
    continuous = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop=None)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", continuous, C.CONTINUOUS_FEATURES),
            ("cat", categorical, C.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
