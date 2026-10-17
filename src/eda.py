"""
Exploratory data analysis for the BRFSS heart-disease dataset.

Generates the figures referenced by the design document and notebook:
  * heart-disease prevalence by year
  * class balance
  * missing-data pattern
  * continuous-feature distributions by target
  * prevalence by age group and by sex
  * categorical association with the target
  * feature correlation heatmap (numeric-coded)

Figures are written to reports/figures/ and a text summary to
reports/eda_summary.txt.

Run:  python -m src.eda
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src import features as F  # noqa: E402

sns.set_theme(style="whitegrid")


def _save(fig, name):
    out = C.FIGURES_DIR / name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] wrote {out}")


def prevalence_by_year(df, lines):
    tab = df.groupby("survey_year")[C.TARGET].agg(["mean", "count"])
    lines.append("Heart-disease prevalence by year:")
    for yr, row in tab.iterrows():
        lines.append(f"  {yr}: {row['mean']:.3%}  (n={int(row['count']):,})")
    fig, ax = plt.subplots(figsize=(6, 4))
    (tab["mean"] * 100).plot(kind="bar", color="#c0392b", ax=ax)
    ax.set_ylabel("Positive rate (%)")
    ax.set_xlabel("Survey year")
    ax.set_title("Heart-disease (_MICHD) prevalence by BRFSS year")
    _save(fig, "01_prevalence_by_year.png")


def class_balance(df, lines):
    counts = df[C.TARGET].value_counts().sort_index()
    lines.append("\nOverall class balance:")
    lines.append(f"  negative (0): {counts.get(0, 0):,}")
    lines.append(f"  positive (1): {counts.get(1, 0):,}")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["No CHD/MI (0)", "CHD/MI (1)"], counts.values,
           color=["#2980b9", "#c0392b"])
    ax.set_ylabel("Respondents")
    ax.set_title("Class distribution (all years)")
    _save(fig, "02_class_balance.png")


def missingness(df, lines):
    miss = df[C.ALL_FEATURES].isna().mean().sort_values(ascending=False)
    lines.append("\nMissing-value fraction per feature:")
    for k, v in miss.items():
        lines.append(f"  {k:22} {v:.3%}")
    fig, ax = plt.subplots(figsize=(7, 5))
    (miss * 100).plot(kind="barh", color="#8e44ad", ax=ax)
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing-data pattern by feature")
    _save(fig, "03_missingness.png")


def continuous_distributions(df):
    fig, axes = plt.subplots(1, len(C.CONTINUOUS_FEATURES),
                             figsize=(5 * len(C.CONTINUOUS_FEATURES), 4))
    for ax, col in zip(np.atleast_1d(axes), C.CONTINUOUS_FEATURES):
        for label, sub in df.groupby(C.TARGET):
            ax.hist(sub[col].dropna(), bins=40, alpha=0.5, density=True,
                    label=f"target={label}")
        ax.set_title(col)
        ax.legend()
    fig.suptitle("Continuous features by heart-disease class")
    _save(fig, "04_continuous_by_class.png")


def prevalence_by_group(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    age = df.groupby("age_group")[C.TARGET].mean() * 100
    age.plot(kind="bar", ax=axes[0], color="#16a085")
    axes[0].set_title("Prevalence by age group (_AGEG5YR)")
    axes[0].set_ylabel("Positive rate (%)")
    sex = df.groupby("sex")[C.TARGET].mean() * 100
    sex.plot(kind="bar", ax=axes[1], color="#d35400")
    axes[1].set_title("Prevalence by sex (1=male, 2=female)")
    axes[1].set_ylabel("Positive rate (%)")
    _save(fig, "05_prevalence_by_group.png")


def categorical_association(df):
    cats = ["gen_health", "diabetes", "diff_walk", "smoker_status",
            "kidney_disease", "phys_activity"]
    n = len(cats)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, col in zip(axes.ravel(), cats):
        (df.groupby(col)[C.TARGET].mean() * 100).plot(
            kind="bar", ax=ax, color="#2c3e50")
        ax.set_title(f"Prevalence by {col}")
        ax.set_ylabel("Positive rate (%)")
    fig.suptitle("Heart-disease prevalence across categorical risk factors")
    _save(fig, "06_categorical_association.png")


def correlation_heatmap(df):
    num = df[C.ALL_FEATURES + [C.TARGET]].apply(
        pd.to_numeric, errors="coerce")
    corr = num.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Spearman correlation among features and target")
    _save(fig, "07_correlation_heatmap.png")


def run():
    df = F.load_processed()
    lines = [f"BRFSS EDA summary — total rows: {len(df):,}",
             f"Features: {C.ALL_FEATURES}",
             f"Leakage excluded: {C.LEAKAGE_COLS}", ""]
    prevalence_by_year(df, lines)
    class_balance(df, lines)
    missingness(df, lines)
    continuous_distributions(df)
    prevalence_by_group(df)
    categorical_association(df)
    correlation_heatmap(df)

    out = C.REPORTS_DIR / "eda_summary.txt"
    out.write_text("\n".join(lines))
    print(f"[eda] wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    run()
