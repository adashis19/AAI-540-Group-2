"""
DEMO ITEM 6 — SageMaker Batch Transform.

Runs a batch scoring job over the held-out 2023 cohort using the registered
model, and writes per-record risk probabilities to S3. No real-time endpoint is
created (Batch Transform is cheaper and matches the project's batch design).

Run:  python sagemaker/batch_transform.py
Record: SageMaker -> Batch transform jobs -> job = Completed, then open the
        output file in S3 (one risk probability per input row).
"""
from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

import pandas as pd
import sagemaker
from sagemaker.sklearn import SKLearnModel

HERE = Path(__file__).resolve().parent
FEATURES = [
    "bmi", "mental_health_days", "phys_health_days",
    "age_group", "sex", "gen_health", "smoked_100", "smoker_status",
    "diabetes", "diff_walk", "phys_activity", "kidney_disease",
    "income_group", "education_group", "has_personal_doctor",
]
SAMPLE_ROWS = 5_000  # small batch keeps the job fast/cheap


def main():
    sess = sagemaker.Session()
    role = sagemaker.get_execution_role()
    bucket = sess.default_bucket()

    # Build input CSV (headerless) from the 2023 held-out year.
    df = pd.read_parquet("data/processed/brfss_2023.parquet").head(SAMPLE_ROWS)
    X = df[FEATURES].copy().fillna(0)
    in_csv = HERE / "batch_in.csv"
    X.to_csv(in_csv, index=False, header=False)
    in_s3 = sess.upload_data(str(in_csv), key_prefix="batch-in")
    print(f"[batch] input -> {in_s3}  ({len(X):,} rows)")

    # Package the model artifact for the transformer.
    src = Path("models/best_model.joblib")
    tmp = HERE / "model.joblib"; shutil.copy(src, tmp)
    tar = HERE / "model.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(tmp, arcname="model.joblib")
    tmp.unlink()
    model_data = sess.upload_data(str(tar), key_prefix="model")

    model = SKLearnModel(
        model_data=model_data, role=role, entry_point="inference.py",
        source_dir=str(HERE), framework_version="1.2-1",
        sagemaker_session=sess,
    )

    out_path = f"s3://{bucket}/batch-out"
    transformer = model.transformer(
        instance_count=1, instance_type="ml.m5.large", output_path=out_path,
    )
    print("[batch] starting Batch Transform ...")
    transformer.transform(in_s3, content_type="text/csv", split_type="Line")
    transformer.wait()
    print(f"[batch] done. Output written under {out_path}")
    print("[batch] each line in batch_in.csv.out is a heart-disease risk probability.")


if __name__ == "__main__":
    main()
