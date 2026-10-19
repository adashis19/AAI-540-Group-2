"""
DEMO ITEMS 2 & 3 — Model/Data monitoring and infrastructure dashboards.

Part A (data/model monitoring): uses SageMaker Model Monitor to compute a
baseline (statistics + constraints) over the training feature data. The
baseline report is the "data monitoring report" to show on screen.

Part B (infrastructure): creates a simple CloudWatch dashboard with widgets for
Batch Transform / processing job metrics, plus one alarm. CloudWatch dashboards
and alarms are effectively free, so this adds no meaningful cost.

Run:  python sagemaker/monitoring.py
Record:
  * SageMaker -> Model Monitor baseline: statistics.json / constraints.json
  * CloudWatch -> Dashboards -> heart-disease-mlops (+ the alarm in Part B)
"""
from __future__ import annotations

import json
from pathlib import Path

import boto3
import pandas as pd
import sagemaker
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

HERE = Path(__file__).resolve().parent
DASHBOARD = "heart-disease-mlops"


def baseline_monitor():
    sess = sagemaker.Session()
    role = sagemaker.get_execution_role()
    bucket = sess.default_bucket()

    # Baseline dataset = training features with header (Model Monitor needs it).
    df = pd.read_parquet("data/processed/brfss_combined.parquet").head(20_000)
    base_csv = HERE / "baseline.csv"
    df.to_csv(base_csv, index=False)
    base_s3 = sess.upload_data(str(base_csv), key_prefix="monitor/baseline")

    mon = DefaultModelMonitor(role=role, instance_count=1,
                              instance_type="ml.m5.large",
                              volume_size_in_gb=20, max_runtime_in_seconds=1800)
    print("[monitor] suggesting baseline (data-quality statistics + constraints) ...")
    mon.suggest_baseline(
        baseline_dataset=base_s3,
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=f"s3://{bucket}/monitor/baseline-report",
        wait=True,
    )
    print(f"[monitor] baseline report -> s3://{bucket}/monitor/baseline-report")
    print("[monitor] show statistics.json and constraints.json as the data report.")


def cloudwatch_dashboard():
    cw = boto3.client("cloudwatch")
    body = {
        "widgets": [
            {"type": "metric", "x": 0, "y": 0, "width": 12, "height": 6,
             "properties": {"title": "Batch Transform — invocations & errors",
                            "region": boto3.session.Session().region_name,
                            "metrics": [["AWS/SageMaker", "Invocations"],
                                        ["AWS/SageMaker", "Invocation4XXErrors"],
                                        ["AWS/SageMaker", "Invocation5XXErrors"]],
                            "period": 300, "stat": "Sum"}},
            {"type": "metric", "x": 12, "y": 0, "width": 12, "height": 6,
             "properties": {"title": "Processing jobs — CPU / Memory",
                            "region": boto3.session.Session().region_name,
                            "metrics": [["/aws/sagemaker/ProcessingJobs", "CPUUtilization"],
                                        ["/aws/sagemaker/ProcessingJobs", "MemoryUtilization"]],
                            "period": 300, "stat": "Average"}},
        ]
    }
    cw.put_dashboard(DashboardName=DASHBOARD, DashboardBody=json.dumps(body))
    print(f"[monitor] created CloudWatch dashboard: {DASHBOARD}")

    cw.put_metric_alarm(
        AlarmName="heart-disease-batch-5xx",
        MetricName="Invocation5XXErrors", Namespace="AWS/SageMaker",
        Statistic="Sum", Period=300, EvaluationPeriods=1,
        Threshold=1.0, ComparisonOperator="GreaterThanOrEqualToThreshold",
        AlarmDescription="Alert if the batch scoring job returns server errors.",
    )
    print("[monitor] created alarm: heart-disease-batch-5xx")


def main():
    baseline_monitor()
    cloudwatch_dashboard()


if __name__ == "__main__":
    main()
