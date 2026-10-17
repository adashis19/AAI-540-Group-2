"""
Temporal test evaluation figures + fairness / subgroup analysis.

Loads the saved best model, scores the 2023 temporal test set, and produces:
  * ROC and precision-recall curves for the best model
  * a calibration curve
  * a confusion-matrix heatmap
  * subgroup performance (recall, PR-AUC, selection rate) by sex and age band

Outputs figures to reports/figures/ and a subgroup table to
reports/subgroup_metrics.csv.

Run:  python -m src.fairness
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import joblib

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src import features as F  # noqa: E402


def _save(fig, name):
    out = C.FIGURES_DIR / name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[fairness] wrote {out}")


def load_best():
    path = C.MODELS_DIR / "best_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.train`.")
    return joblib.load(path)


def curves(y_test, proba):
    # ROC
    fpr, tpr, _ = roc_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, proba):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — best model (2023 test)")
    ax.legend()
    _save(fig, "08_roc_curve.png")

    # PR
    prec, rec, _ = precision_recall_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, label=f"PR-AUC={average_precision_score(y_test, proba):.3f}")
    ax.axhline(y_test.mean(), ls="--", color="grey",
               label=f"prevalence={y_test.mean():.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve — best model (2023 test)")
    ax.legend()
    _save(fig, "09_pr_curve.png")


def calibration(y_test, proba):
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration curve — best model (2023 test)")
    ax.legend()
    _save(fig, "10_calibration.png")


def confusion(y_test, pred):
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title("Confusion matrix — best model @0.5")
    fig.colorbar(im)
    _save(fig, "11_confusion_matrix.png")


def subgroup_metrics(test_df, proba, pred):
    rows = []

    def _metrics(mask, name):
        if mask.sum() < 50 or test_df.loc[mask, C.TARGET].nunique() < 2:
            return
        y = test_df.loc[mask, C.TARGET]
        p = proba[mask.values]
        pr = pred[mask.values]
        rows.append({
            "subgroup": name,
            "n": int(mask.sum()),
            "prevalence": float(y.mean()),
            "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "recall": recall_score(y, pr),
            "selection_rate": float(pr.mean()),
        })

    for s, label in [(1, "sex=male"), (2, "sex=female")]:
        _metrics(test_df["sex"] == s, label)

    # Age bands: _AGEG5YR 1-4 (18-39), 5-8 (40-59), 9-13 (60+)
    bands = {"age 18-39": range(1, 5), "age 40-59": range(5, 9),
             "age 60+": range(9, 14)}
    for name, rng in bands.items():
        _metrics(test_df["age_group"].isin(list(rng)), name)

    sub = pd.DataFrame(rows)
    out = C.REPORTS_DIR / "subgroup_metrics.csv"
    sub.to_csv(out, index=False)
    print(f"[fairness] wrote {out}")
    print(sub.round(4).to_string(index=False))

    if not sub.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub["recall"], 0.4, label="recall", color="#c0392b")
        ax.bar(x + 0.2, sub["pr_auc"], 0.4, label="PR-AUC", color="#2980b9")
        ax.set_xticks(x); ax.set_xticklabels(sub["subgroup"], rotation=30, ha="right")
        ax.set_title("Subgroup performance — best model (2023 test)")
        ax.legend()
        _save(fig, "12_subgroup_performance.png")
    return sub


def run():
    pipe = load_best()
    df = F.load_processed()
    _, X_test, _, y_test, _, test_df = F.temporal_split(df)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    curves(y_test, proba)
    calibration(y_test, proba)
    confusion(y_test, pred)
    subgroup_metrics(test_df.reset_index(drop=True),
                     pd.Series(proba), pred)


if __name__ == "__main__":
    run()
