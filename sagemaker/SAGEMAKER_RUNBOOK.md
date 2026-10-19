# SageMaker Runbook — AAI-540 Group 2 (Heart-Disease Screening)

Step-by-step to run the 6 required MLOps components in **AWS SageMaker** and
capture the screens for the video. These scripts live in `sagemaker/` and are
**separate** from the rest of the project — they do not change any existing
deliverable. They reuse the already-trained model (`models/best_model.joblib`)
and processed data (`data/processed/*.parquet`), so **no retraining** happens in
SageMaker (keeps Learner Lab cost minimal).

> Budget reality: the Learner Lab cap is small. Do this in ONE session, record
> each screen as it works, and shut everything down at the end (see last section).

---

## 0. Start the lab & open Studio
1. AWS Academy → **Start Lab** → wait for green dot → **AWS**.
2. Console → **SageMaker** → **Studio** → launch (uses the lab role, e.g. `LabRole`).
3. Studio → **File → New → Terminal**.

## 1. Get the code into Studio
```bash
git clone https://github.com/hanmant77/AAI-540-Group-2.git
cd AAI-540-Group-2
pip install -q -r requirements.txt sagemaker boto3
```
If your repo copy does not include `sagemaker/`, also copy that folder in, or
`git pull` the branch that contains it.

## 2. DEMO 1 — Feature Store  (Ashis)
```bash
python sagemaker/feature_store_ingest.py
```
**Record:** Studio → Feature Store → Feature Groups → `heart-disease-features`
→ the 15 feature definitions + a query/record view.

## 3. DEMO 5 — Model Registry  (Ram)
```bash
python sagemaker/register_model.py
```
**Record:** SageMaker → Model Registry → `heart-disease-screening` → version →
Metrics (PR-AUC 0.321, recall 0.819) → Approval status = **Approved**.

## 4. DEMO 6 — Batch Transform  (Ram)
```bash
python sagemaker/batch_transform.py
```
**Record:** SageMaker → Batch transform jobs → job **Completed**, then open the
output file in S3 (`batch-out/batch_in.csv.out`) — one risk probability per row.

## 5. DEMO 4 — CI/CD Pipeline (success + fail)  (Hanmant)
```bash
# SUCCESS: 0.321 >= 0.25 -> gate passes -> registers
python sagemaker/pipeline.py --threshold 0.25
# FAILURE: 0.321 < 0.99 -> gate fails -> promotion blocked
python sagemaker/pipeline.py --threshold 0.99
```
**Record:** Studio → Pipelines → the **green** execution (all steps pass), then
the **red** execution stopped at **EvaluationGate / PromotionBlocked**, and open
the failed step to show the reason.

## 6. DEMO 2 & 3 — Monitoring + CloudWatch  (Ram / Hanmant)
```bash
python sagemaker/monitoring.py
```
**Record:**
- Model Monitor baseline report in S3 (`monitor/baseline-report/statistics.json`
  and `constraints.json`) — the **data-monitoring report**.
- CloudWatch → Dashboards → `heart-disease-mlops` (infra dashboard) and the
  alarm `heart-disease-batch-5xx`.
- Also show the repo's model-quality figures (`reports/figures/08-12_*.png`).

---

## Team split (each member runs a portion in SageMaker — required)
- **Ashis:** Feature Store (step 2)
- **Hanmant:** CI/CD Pipeline + CloudWatch (steps 5, 6-infra)
- **Ram:** Model Registry + Batch Transform + Model Monitor (steps 3, 4, 6-model)

## Cost control & SHUTDOWN (do this at the end)
- Use `ml.m5.large` only for jobs; `ml.t3.medium` for the notebook kernel.
- **No real-time endpoints** — Batch Transform only.
- Delete the feature group, any models/endpoints, and stop the Studio app/kernel.
- **End Lab** in AWS Academy when done.
```bash
# optional cleanup from the terminal
python - <<'PY'
import boto3
sm = boto3.client("sagemaker")
try: sm.delete_feature_group(FeatureGroupName="heart-disease-features")
except Exception as e: print(e)
PY
```

## Notes
- These scripts are intentionally minimal and sample-limited (20k rows for the
  feature group / monitor, 5k for batch) to run fast and cheap while still
  demonstrating every component.
- They reuse the trained artifact and metrics, so results match the design
  document and the notebook.
