# Team Contributions — AAI-540 Group 2

**Project:** Heart-Disease Risk Screening (MLOps)
**Course:** AAI-540 Machine Learning Operations · University of San Diego
**Instructor:** Dr. Raj Garg

This file documents each member's contributions to the project. For graded
teamwork, each member should also commit their own work so the Git commit
history reflects individual authorship (commits are reviewable by the
instructor).

## Members

- **Ashis Das**
- **Hanmant Mhalappa Kattimani**
- **Ram Prasanth G**

## Contribution summary

### Ashis Das
- ML problem identification and project scope (heart-disease screening, target `_MICHD`).
- Data sourcing: BRFSS 2021/2022/2023 identification, download and raw-data (S3) layout.
- Feature Store / feature-group definition and the 15-feature harmonised schema.
- Baseline modelling (majority-class benchmark and logistic regression).
- Design Document: data sources, deployment (batch vs real-time) sections.

### Hanmant Mhalappa Kattimani
- Reproducible ingestion/harmonisation pipeline (`src/data_ingestion.py`) and leakage control.
- Feature engineering pipeline (`src/features.py`: imputation, scaling, one-hot via `ColumnTransformer`).
- CI/CD design and data-contract/leakage unit tests (`tests/test_pipeline.py`).
- Local batch-inference job (`src/batch_inference.py`) — SageMaker Batch Transform equivalent.
- Presentation deck and video script; final packaging and GitHub repository.

### Ram Prasanth G
- Exploratory data analysis and figures (`src/eda.py`).
- Advanced models: Random Forest, Gradient Boosting, and the selected XGBoost (`src/train.py`).
- Temporal evaluation (2021+2022 → 2023) and model-comparison table.
- Fairness / subgroup analysis by sex and age (`src/fairness.py`, `reports/subgroup_metrics.csv`).
- Monitoring design (model/data/infrastructure) and the Design Document monitoring section.

## Note on SageMaker

The system is designed for AWS SageMaker. Due to AWS Academy Learner Lab budget
limits, the MLOps components were reproduced locally (see the "MLOps components"
table in `README.md`). Each member is responsible for running their portion in
SageMaker when Learner Lab access is restored.
