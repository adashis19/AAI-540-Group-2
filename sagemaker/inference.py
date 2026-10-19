"""
SageMaker inference entry point for the heart-disease screening model.

Used by both the Model Registry (as the model's entry_point) and the Batch
Transform job. It loads the scikit-learn pipeline artifact (model.joblib) and
returns the heart-disease risk probability for each input row.

This is a standard SageMaker scikit-learn container contract:
  * model_fn      -> load the serialized pipeline
  * input_fn      -> parse incoming CSV (no header) into a DataFrame
  * predict_fn    -> return positive-class probabilities
  * output_fn     -> serialize predictions back to CSV

The model is the SAME fitted sklearn Pipeline produced by src/train.py
(preprocessing + XGBoost), so training-time and inference-time transforms match.
"""
from __future__ import annotations

import io
import os

import joblib
import numpy as np
import pandas as pd

# Feature order must match how the batch CSV is written (see batch_transform.py).
FEATURES = [
    "bmi", "mental_health_days", "phys_health_days",
    "age_group", "sex", "gen_health", "smoked_100", "smoker_status",
    "diabetes", "diff_walk", "phys_activity", "kidney_disease",
    "income_group", "education_group", "has_personal_doctor",
]


def model_fn(model_dir: str):
    """Load the fitted sklearn Pipeline from the model directory."""
    path = os.path.join(model_dir, "model.joblib")
    return joblib.load(path)


def input_fn(request_body, content_type="text/csv"):
    """Parse a headerless CSV batch into a DataFrame with the right columns."""
    if content_type != "text/csv":
        raise ValueError(f"Unsupported content type: {content_type}")
    df = pd.read_csv(io.StringIO(request_body), header=None)
    # If the column count matches our feature list, apply the names.
    if df.shape[1] == len(FEATURES):
        df.columns = FEATURES
    return df


def predict_fn(input_data: pd.DataFrame, model):
    """Return the positive-class (heart-disease) probability per row."""
    proba = model.predict_proba(input_data)[:, 1]
    return np.round(proba, 6)


def output_fn(prediction, accept="text/csv"):
    """Serialize predictions to CSV (one probability per line)."""
    out = io.StringIO()
    pd.Series(prediction, name="risk_probability").to_csv(out, index=False, header=False)
    return out.getvalue()
