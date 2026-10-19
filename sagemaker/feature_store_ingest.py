"""
DEMO ITEM 1 — SageMaker Feature Store & feature groups.

Creates a feature group `heart-disease-features` and ingests a sample of the
harmonised BRFSS data into it. Run inside SageMaker Studio (uses the Studio
execution role). A 20k-row sample keeps ingestion fast and cheap — enough to
demonstrate the feature group, its definitions, and ingested records.

Run:  python sagemaker/feature_store_ingest.py
Record: SageMaker Studio -> Feature Store -> Feature Groups ->
        heart-disease-features (the 15 feature definitions + a query result).
"""
from __future__ import annotations

import time

import pandas as pd
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup

FEATURE_GROUP = "heart-disease-features"
SAMPLE_ROWS = 20_000  # keep small to control cost


def main():
    sess = sagemaker.Session()
    role = sagemaker.get_execution_role()
    bucket = sess.default_bucket()
    print(f"[fs] role={role}")
    print(f"[fs] bucket={bucket}")

    # Load harmonised data produced by src/data_ingestion.py.
    df = pd.read_parquet("data/processed/brfss_combined.parquet")
    df = df.head(SAMPLE_ROWS).copy()

    # Feature Store requires a record identifier and an event-time column.
    df["record_id"] = df.index.astype(str)
    df["event_time"] = float(time.time())

    # Feature Store does not accept pandas nullable/NaN in some dtypes; cast
    # object columns to string and fill NaNs so ingestion succeeds cleanly.
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype("string").fillna("unknown")
    df = df.fillna(0)

    fg = FeatureGroup(name=FEATURE_GROUP, sagemaker_session=sess)
    fg.load_feature_definitions(data_frame=df)

    print(f"[fs] creating feature group {FEATURE_GROUP} ...")
    fg.create(
        s3_uri=f"s3://{bucket}/feature-store",
        record_identifier_name="record_id",
        event_time_feature_name="event_time",
        role_arn=role,
        enable_online_store=True,
    )

    # Wait until the group is Created before ingesting.
    status = None
    while status != "Created":
        status = fg.describe().get("FeatureGroupStatus")
        print(f"[fs] status={status}")
        if status == "CreateFailed":
            raise RuntimeError(fg.describe())
        if status != "Created":
            time.sleep(5)

    print(f"[fs] ingesting {len(df):,} records ...")
    fg.ingest(data_frame=df, max_workers=2, wait=True)
    print("[fs] done. Feature group ready to demonstrate.")


if __name__ == "__main__":
    main()
