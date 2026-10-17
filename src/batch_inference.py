"""
Local batch-inference job — the on-prem/Kiro equivalent of a SageMaker Batch
Transform job.

Loads the registered best model (models/best_model.joblib), scores the held-out
2023 BRFSS cohort, and writes a scored-output file with one heart-disease risk
probability per record plus a prioritisation flag at the operating threshold.
This file is the local stand-in for a SageMaker Batch Transform output manifest
and backs the "batch inference outputs" item of the video demonstration.

Run:  python -m src.batch_inference
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src import features as F  # noqa: E402

# Operating threshold for turning a probability into a screen/outreach flag.
# 0.5 is the default used in training; documented so it can be tuned later.
OPERATING_THRESHOLD = 0.5


def run(save: bool = True) -> pd.DataFrame:
    model_path = C.MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Run `python -m src.train` first."
        )
    model = joblib.load(model_path)
    meta = json.loads((C.MODELS_DIR / "best_model_meta.json").read_text())

    # Score the held-out 2023 cohort (the "new data" a batch job would receive).
    df = F.load_processed()
    test = df[df["survey_year"].isin(C.TEST_YEARS)].copy()
    X_test = test[C.ALL_FEATURES]

    proba = model.predict_proba(X_test)[:, 1]
    out = pd.DataFrame(
        {
            "record_id": test.index.astype(int),
            "survey_year": test["survey_year"].values,
            "risk_probability": proba.round(6),
            "screen_flag": (proba >= OPERATING_THRESHOLD).astype(int),
        }
    )
    # Rank so the highest-risk individuals are at the top of the outreach list.
    out = out.sort_values("risk_probability", ascending=False).reset_index(drop=True)
    out["priority_rank"] = out.index + 1

    flagged = int(out["screen_flag"].sum())
    print(f"[batch] model={meta['best_model']}  scored rows={len(out):,}")
    print(f"[batch] flagged for screening (p>={OPERATING_THRESHOLD}): "
          f"{flagged:,} ({flagged / len(out):.2%})")
    print(f"[batch] mean risk={out['risk_probability'].mean():.4f}  "
          f"max={out['risk_probability'].max():.4f}")

    if save:
        out_path = C.REPORTS_DIR / "batch_predictions_2023.csv"
        out.to_csv(out_path, index=False)
        print(f"[batch] wrote {out_path}")
        # Small human-readable summary that mimics a batch-job report.
        summary = {
            "model": meta["best_model"],
            "selection_metric": meta["selection_metric"],
            "pr_auc": meta["pr_auc"],
            "scored_rows": int(len(out)),
            "operating_threshold": OPERATING_THRESHOLD,
            "flagged_for_screening": flagged,
            "flagged_rate": round(flagged / len(out), 4),
            "mean_risk_probability": round(float(out["risk_probability"].mean()), 4),
        }
        summary_path = C.REPORTS_DIR / "batch_inference_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[batch] wrote {summary_path}")

    return out


if __name__ == "__main__":
    run(save=True)
