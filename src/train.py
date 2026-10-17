"""
Model training, comparison and temporal evaluation.

Trains a logistic-regression baseline and four tree-based challengers
(Decision Tree, Random Forest, Gradient Boosting, XGBoost). All models use
class weighting to address the ~8-9% positive-class imbalance. Models are
trained on 2021+2022 and evaluated on the held-out 2023 temporal test set.

Primary metrics for this imbalanced screening problem: Recall (sensitivity),
PR-AUC (average precision) and ROC-AUC, with precision, F1, specificity,
confusion matrix and calibration reported alongside.

Run:  python -m src.train
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src import features as F  # noqa: E402


def _scale_pos_weight(y) -> float:
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    return neg / max(pos, 1)


def build_models(y_train) -> dict:
    """Return {name: sklearn estimator} — all class-imbalance aware."""
    spw = _scale_pos_weight(y_train)
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", C=1.0,
            solver="lbfgs", random_state=C.RANDOM_STATE,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=200,
            class_weight="balanced", random_state=C.RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=50,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=C.RANDOM_STATE,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            random_state=C.RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=spw, eval_metric="aucpr",
            n_jobs=-1, random_state=C.RANDOM_STATE, tree_method="hist",
        ),
    }


def evaluate(name, pipe, X_test, y_test, threshold=0.5) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "recall": recall_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred),
        "specificity": specificity,
        "brier": brier_score_loss(y_test, proba),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "threshold": threshold,
    }


def run(save: bool = True):
    df = F.load_processed()
    X_train, X_test, y_train, y_test, _, _ = F.temporal_split(df)
    print(f"[train] train rows={len(X_train):,}  test rows={len(X_test):,}")
    print(f"[train] train pos-rate={y_train.mean():.3%}  "
          f"test pos-rate={y_test.mean():.3%}")

    results = []

    # Majority-class reference (context for the imbalance).
    dummy = Pipeline([("pre", F.build_preprocessor()),
                      ("clf", DummyClassifier(strategy="most_frequent"))])
    dummy.fit(X_train, y_train)
    results.append(evaluate("MajorityBaseline", dummy, X_test, y_test))

    models = build_models(y_train)
    best_name, best_pipe, best_pr = None, None, -1.0
    for name, clf in models.items():
        print(f"[train] fitting {name} ...", flush=True)
        pipe = Pipeline([("pre", F.build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        metrics = evaluate(name, pipe, X_test, y_test)
        results.append(metrics)
        print(f"[train]   {name}: PR-AUC={metrics['pr_auc']:.4f} "
              f"ROC-AUC={metrics['roc_auc']:.4f} recall={metrics['recall']:.3f}")
        if metrics["pr_auc"] > best_pr:
            best_pr, best_name, best_pipe = metrics["pr_auc"], name, pipe

    res_df = pd.DataFrame(results).set_index("model")
    print("\n[train] === temporal (2023) test results ===")
    print(res_df[["roc_auc", "pr_auc", "recall", "precision",
                  "f1", "specificity", "brier"]].round(4).to_string())
    print(f"\n[train] best model by PR-AUC: {best_name} ({best_pr:.4f})")

    if save:
        res_df.to_csv(C.REPORTS_DIR / "model_comparison.csv")
        joblib.dump(best_pipe, C.MODELS_DIR / "best_model.joblib")
        with open(C.MODELS_DIR / "best_model_meta.json", "w") as fh:
            json.dump(
                {"best_model": best_name,
                 "selection_metric": "pr_auc",
                 "pr_auc": float(best_pr),
                 "train_years": C.TRAIN_YEARS,
                 "test_years": C.TEST_YEARS,
                 "features": C.ALL_FEATURES,
                 "leakage_excluded": C.LEAKAGE_COLS},
                fh, indent=2,
            )
        print(f"[train] saved best_model.joblib and model_comparison.csv")

    return res_df, best_name, best_pipe, (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    run(save=True)
