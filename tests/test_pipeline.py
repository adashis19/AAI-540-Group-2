"""
Lightweight tests for the BRFSS pipeline — the kind an MLOps CI/CD gate runs.

These validate the data contract and leakage controls without needing a GPU or
long training. Run:  python -m pytest -q   (or)   python tests/test_pipeline.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config as C
from src import features as F


def test_no_leakage_in_features():
    """Target-defining columns must never appear in the modelling features."""
    for col in C.LEAKAGE_COLS:
        assert col not in C.ALL_FEATURES, f"leakage column {col} in features"


def test_feature_roles_disjoint():
    assert set(C.CONTINUOUS_FEATURES).isdisjoint(C.CATEGORICAL_FEATURES)
    assert set(C.ALL_FEATURES) == set(C.CONTINUOUS_FEATURES) | set(
        C.CATEGORICAL_FEATURES)


def test_processed_data_contract():
    """Processed parquet must exist, carry all features + target, 0/1 labels."""
    path = C.PROCESSED_DIR / "brfss_combined.parquet"
    assert path.exists(), "run `python -m src.data_ingestion` first"
    df = pd.read_parquet(path)
    for col in C.ALL_FEATURES + [C.TARGET, "survey_year"]:
        assert col in df.columns, f"missing column {col}"
    assert set(df[C.TARGET].unique()) <= {0, 1}
    assert set(df["survey_year"].unique()) == {2021, 2022, 2023}


def test_temporal_split_years():
    df = F.load_processed()
    Xtr, Xte, ytr, yte, tr, te = F.temporal_split(df)
    assert set(tr["survey_year"].unique()) == set(C.TRAIN_YEARS)
    assert set(te["survey_year"].unique()) == set(C.TEST_YEARS)
    assert len(Xtr) == len(ytr) and len(Xte) == len(yte)


def test_preprocessor_fits_and_transforms():
    df = F.load_processed().sample(5000, random_state=0)
    Xtr, Xte, ytr, yte, _, _ = F.temporal_split(df)
    if len(Xtr) and len(Xte):
        pre = F.build_preprocessor().fit(Xtr)
        out = pre.transform(Xte)
        assert out.shape[0] == len(Xte)
        assert out.shape[1] > len(C.ALL_FEATURES)  # one-hot expands width


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
    sys.exit(0 if passed == len(fns) else 1)
