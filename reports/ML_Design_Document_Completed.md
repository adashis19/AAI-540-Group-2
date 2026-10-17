# AAI-540 ML Design Document — Heart-Disease Risk Screening

**Project Team Group #:** Group 2
**Authors:** Ashis Das, Hanmant Mhalappa Kattimani, Ram Prasanth G
**Course:** AAI-540 Machine Learning Operations
**University:** University of San Diego
**Instructor:** Dr. Raj Garg
**Business Name:** Heart-disease risk screening
**Publication Date:** October 18, 2026

**GitHub Project Link:** https://github.com/adashis19/AAI-540-Group-2.git

> This document completes the AAI-540 ML Design Document template. Sections that
> were empty prompts in the template are filled below with the concrete design
> and the measured results produced by the accompanying code (`src/` package and
> `notebooks/Final_Team_Project_Heart_Disease_Risk_Screening_MLOps.ipynb`).

---

## Project Scope

### Project Background

Cardiovascular disease is a major public-health concern, but many individuals
remain unaware of their risk until they experience a serious event. Lifestyle,
demographic and general-health information can provide useful signals for
identifying individuals who are more likely to report a history of coronary
heart disease or myocardial infarction (CHD/MI).

The objective is a machine-learning model that estimates whether an adult is
likely to have heart disease from **non-invasive** demographic, behavioural and
health-related information, for **preliminary population-level risk screening and
prioritisation**. The model does **not** provide a medical diagnosis.

This is a **supervised binary classification** problem:

- **0** — No reported coronary heart disease or myocardial infarction
- **1** — Reported coronary heart disease or myocardial infarction

The target is the CDC BRFSS calculated variable **`_MICHD`**, derived from
whether a respondent was ever told they had a myocardial infarction or coronary
heart disease.

### Technical Background

The model is evaluated primarily on its ability to identify the heart-disease
class. Because heart disease is the minority class (~8-9%), accuracy alone is
misleading. Primary metrics are **recall/sensitivity, PR-AUC and ROC-AUC**, with
precision, specificity, F1, confusion matrices and calibration also reported. A
logistic-regression model is the baseline, compared with decision trees, random
forest and gradient-boosting models (XGBoost).

The project uses the 2021, 2022 and 2023 annual BRFSS datasets. **2021 and 2022
are used for development; 2023 is a held-out temporal test set** to estimate
generalisation to a later year.

Data preparation selects variables consistently available across all three
years, harmonises names/coding, removes invalid observations, and converts
"don't know / refused / not asked" into missing values. Numerical missing values
are imputed with training-set statistics; categorical variables are imputed and
one-hot encoded; continuous variables are scaled for logistic regression. Class
weighting addresses imbalance and is applied **only** on training data.

**Hypothesised main features:** age, BMI, smoking status, physical-activity
level, general-health rating, diabetes status, sex, difficulty walking,
kidney-disease status, income, education and access to a personal doctor.
(Blood-pressure and cholesterol were hypothesised as strong predictors but are
**excluded** from the common feature set — see Feature Engineering.)

Variables that define the target (`CVDINFR4`, `CVDCRHD4`, `_MICHD`) are never
used as inputs, to prevent target leakage.

### Goals vs Non-Goals

**Goals**
- Build a binary classifier that identifies adults more likely to report CHD/MI.
- Achieve better minority-class detection than a majority-class or
  logistic-regression baseline.
- Identify the variables most strongly associated with predictions.
- Produce risk probabilities usable for screening and prioritisation.
- Evaluate performance across demographic groups and survey years.
- Build a reproducible data-preparation and model-evaluation pipeline.

**Non-Goals**
- Not a diagnosis or a replacement for a qualified healthcare professional.
- Does not recommend medication, treatment or interventions.
- Does not establish causal relationships.
- Does not predict *when* a person will have a future heart attack.
- Does not claim US-survey results generalise to India or other populations.
- Not deployed in a live clinical environment as part of this project.

### How We Measure Impact

Impact is measured against the goals above using metrics chosen for a rare
positive class (~8.5% prevalence), so that success means *catching true
heart-disease cases* rather than maximising raw accuracy. Each goal maps to a
concrete, measurable indicator:

- **Goal — identify adults likely to report CHD/MI → minority-class detection.**
  Primary success metric is **recall (sensitivity)** on the held-out 2023 year,
  with **PR-AUC** as the threshold-independent ranking metric and **ROC-AUC** as
  a secondary discrimination metric. Measured result: the selected XGBoost model
  reaches **recall 0.819, PR-AUC 0.321, ROC-AUC 0.836**.
- **Goal — beat a naive baseline → lift over reference models.** Impact is the
  margin over a majority-class dummy (PR-AUC 0.085) and over the logistic-
  regression baseline (PR-AUC 0.315). The selected model's PR-AUC of 0.321 is a
  **~3.8×** improvement over the dummy, demonstrating real screening value.
- **Goal — produce usable risk scores → calibrated probabilities.** Measured by
  the **Brier score** and the calibration curve, so scores can rank and
  prioritise individuals for follow-up.
- **Goal — evaluate across groups and years → fairness and temporal stability.**
  Measured by **subgroup recall/PR-AUC** by sex and age band, and by evaluating
  on a *future* year (2023) never seen in training. Gaps are reported, not
  hidden (e.g. recall 0.870 male vs 0.747 female; 0.126 for ages 18-39).
- **Goal — reproducible pipeline → operational reliability.** Measured by passing
  CI tests (data-contract, leakage, temporal-split, preprocessor) and by the
  evaluation gate that blocks promotion of any model that does not meet or exceed
  the incumbent's PR-AUC and recall on a fixed hold-out.

In a deployed setting the **business-impact proxy** is the number of true
high-risk adults surfaced per screening cycle at an agreed precision/recall
operating point, tracked over time alongside subgroup recall to ensure the
benefit is distributed fairly.

### AWS SageMaker Implementation Mapping

The system is designed to run on AWS SageMaker; each pipeline stage maps to a
specific managed service, and this mapping is what the video demonstration
screencasts. The local scikit-learn implementation in `src/` is the reference
logic that the SageMaker components execute.

| Pipeline stage | AWS SageMaker component | Artifact demonstrated |
|---|---|---|
| Raw + curated storage | **S3** raw and processed buckets | versioned XPT + parquet objects |
| Feature definitions | **SageMaker Feature Store** feature group(s) | 15-feature group schema + record ingestion |
| Experiment training | **SageMaker Training Jobs / Notebook** | training job logs, metrics |
| Model versioning | **SageMaker Model Registry** | registered model package group + approved version |
| Orchestration | **SageMaker Pipelines (DAG)** | successful run + an intentionally failed run at the evaluation gate |
| Batch scoring | **SageMaker Batch Transform** | output manifest of scored records |
| Monitoring | **SageMaker Model Monitor + CloudWatch** | data-quality/model-quality reports and infrastructure dashboards |

---

## Solution Overview

The ML system is a batch screening pipeline. Raw BRFSS SAS-XPT files are ingested
and harmonised into parquet, explored, transformed by a leakage-controlled
feature pipeline, used to train and compare five classifiers, and evaluated on a
temporal 2023 test set with a fairness analysis. The best model (by PR-AUC) is
serialised as a scikit-learn `Pipeline` for batch inference.

**Architecture (logical):**

```
                +------------------+
  CDC BRFSS --> |  Raw store       |  data/raw/*.XPT   (versioned by year)
  XPT (21/22/23)|  (S3 in cloud)   |
                +---------+--------+
                          |  src/data_ingestion.py  (harmonise, derive target,
                          v                          drop leakage, ->parquet)
                +---------+--------+
                | Processed store  |  data/processed/*.parquet
                | (S3 / Feature    |
                |  store in cloud) |
                +----+--------+----+
                     |        |
        src/eda.py   |        |  src/features.py (ColumnTransformer, train-fit)
        (figures)    v        v
                +----+--------+----+   src/train.py (5 models, class-weighted,
                |  Training &      |   temporal split 21+22 -> 23)
                |  Evaluation      |
                +---------+--------+
                          |  best_model.joblib + metrics + meta
                          v
                +---------+--------+   src/fairness.py (ROC/PR/calibration,
                |  Model registry  |   confusion, subgroup metrics)
                |  + batch scoring |
                +------------------+
```

**Pre-release testing:** schema/data-contract checks, leakage-exclusion unit
test, and an evaluation gate requiring PR-AUC and recall on a fixed hold-out to
meet or exceed the current production model. **Monitoring after release:** model
quality (recall, PR-AUC, calibration), data drift, fairness drift, and
infrastructure health.

### Data Sources

- **Source:** CDC Behavioral Risk Factor Surveillance System (BRFSS) annual
  survey data for 2021, 2022 and 2023 (public SAS-XPT downloads).
- **Volume:** ~1.32 M raw records total (2021: 438,693; 2022: 445,132;
  2023: 433,323). After deriving a valid `_MICHD` label the harmonised dataset
  has **1,302,907** rows. All three files far exceed 10,000 records, and the
  positive class alone exceeds the required minimum.
- **Why selected:** satisfies volume requirements; contains a pre-calculated
  heart-disease target; includes relevant non-invasive predictors; published by
  an authoritative agency; provides annual files enabling temporal validation;
  well documented; needs no identifiable clinical records; supports comparing
  interpretable and advanced models.
- **Risks (bias / sensitive features):** self-reported and cross-sectional data;
  the target captures only *diagnosed and reported* disease (undiagnosed cases
  fall in the negative class); class imbalance; telephone-survey selection bias;
  sensitive attributes (age, sex, race, income); US-only generalisation;
  annual variation in modules/coding (must be harmonised); complex survey design
  and weighting. The output is an academic screening exercise, **not** a
  clinically validated device.

### Data Engineering

- **Storage:** raw XPT files in `data/raw/` (an S3 raw bucket in the cloud);
  cleaned tabular data as columnar **parquet** in `data/processed/` (an S3
  processed/curated bucket, optionally registered in a feature store). Parquet is
  chosen for compression and fast columnar reads.
- **Pre-processing before the ML system:** strip trailing spaces in XPT column
  names; select the harmonised variable set; derive the 0/1 target from `_MICHD`
  and drop rows with unknown labels; convert "don't know/refused" codes
  (e.g. 7/9, 77/99) to missing; map "none" day-codes (88) to 0; rescale BMI
  (`_BMI5`/100) and clip implausible values; tag each row with `survey_year`.

### Training Data

- **Split:** temporal by year — **train = 2021 + 2022, test = 2023**. This
  simulates deploying on a future year and is stricter than a random split.
  A validation subset can be carved from the training years for calibration and
  threshold tuning.
- **Labeling techniques:** none required — labels come directly from the
  CDC-calculated `_MICHD` variable (authoritative, consistent across years).

### Feature Engineering

- **Fields used (15):** `bmi`, `mental_health_days`, `phys_health_days`
  (continuous); `age_group`, `sex`, `gen_health`, `smoked_100`, `smoker_status`,
  `diabetes`, `diff_walk`, `phys_activity`, `kidney_disease`, `income_group`,
  `education_group`, `has_personal_doctor` (categorical/ordinal survey codes).
- **Fields excluded:** `_MICHD`, `CVDINFR4`, `CVDCRHD4` (target leakage);
  **hypertension and cholesterol** — BRFSS collects these only in odd years, so
  they are **absent from the 2022 core file**; to keep one schema valid across
  all three years they are excluded from the common feature set (documented
  limitation). `SLEPTIM1` (sleep) appears only in 2022 and is likewise dropped.
- **Combining / bucketing:** age is used as the CDC 14-level `_AGEG5YR` band;
  income and education use the CDC grouped variables (`_INCOMG1`, `_EDUCAG`);
  day-count items are bounded to 0-30. For subgroup analysis, age bands are
  grouped into 18-39 / 40-59 / 60+.
- **Other transformations:** median imputation + standard scaling for continuous
  features; most-frequent imputation + one-hot encoding
  (`handle_unknown="ignore"`) for categoricals, all inside a `ColumnTransformer`
  fitted on the **training** data only.

### Model Training & Evaluation

- **How trained:** each model is wrapped with the shared preprocessor in a
  scikit-learn `Pipeline` and fitted on the 2021+2022 rows. Class imbalance is
  handled with `class_weight="balanced"` (LogReg, Decision Tree),
  `balanced_subsample` (Random Forest) and `scale_pos_weight = neg/pos`
  (XGBoost).
- **Algorithms:** Logistic Regression (baseline), Decision Tree, Random Forest,
  Gradient Boosting, XGBoost; plus a majority-class dummy for reference.
- **Key parameters:** LogReg `C=1.0, max_iter=1000`; Decision Tree `max_depth=6,
  min_samples_leaf=200`; Random Forest `n_estimators=300, max_depth=12,
  min_samples_leaf=50`; Gradient Boosting `n_estimators=200, max_depth=3,
  lr=0.1`; XGBoost `n_estimators=400, max_depth=5, lr=0.05, subsample=0.9,
  colsample_bytree=0.9, tree_method="hist"`; global seed = 42.
- **Evaluation (2023 temporal test set) — measured results:**

  | Model | ROC-AUC | PR-AUC | Recall | Precision | F1 | Specificity |
  |---|---|---|---|---|---|---|
  | MajorityBaseline | 0.500 | 0.085 | 0.000 | 0.000 | 0.000 | 1.000 |
  | LogisticRegression | 0.834 | 0.315 | 0.802 | 0.207 | 0.329 | 0.716 |
  | DecisionTree | 0.784 | 0.264 | 0.689 | 0.212 | 0.324 | 0.763 |
  | RandomForest | 0.828 | 0.311 | 0.790 | 0.208 | 0.329 | 0.721 |
  | GradientBoosting | 0.835 | 0.319 | 0.079 | 0.505 | 0.136 | 0.993 |
  | **XGBoost (selected)** | **0.836** | **0.321** | **0.819** | 0.203 | 0.326 | 0.703 |

  **XGBoost** is selected by PR-AUC (the most informative metric for the rare
  positive class). Every learner beats the majority baseline (PR-AUC 0.085). The
  class-weighted logistic regression is a close, interpretable runner-up.

### Model Deployment

- **Instance size:** the fitted pipeline is small; a modest CPU instance
  (e.g. `ml.m5.large`, 2 vCPU / 8 GB) scores hundreds of thousands of rows
  quickly. No GPU required.
- **Batch vs real time:** **batch**. Population screening is periodic (e.g. over
  a data extract), so scheduled batch scoring (SageMaker Batch Transform or a
  containerised job) fits better than a low-latency real-time endpoint and is
  cheaper and simpler to operate.

### Model Monitoring

- **Model:** track recall, PR-AUC and calibration on newly labelled data; alert
  if recall falls below an agreed floor.
- **Infrastructure:** batch-job success/failure, runtime, resource usage and
  error rates via CloudWatch-style metrics and alarms.
- **Data:** monitor feature distributions (age, BMI, general health) and
  missingness against the training profile (e.g. population-stability index);
  re-compute subgroup recall by sex and age each cycle to catch fairness drift.

### Model CI/CD

- **Checkpoints:** data-schema validation → ingestion/harmonisation → feature
  build → training → **evaluation gate** → model registration → batch-deployment.
- **Tests:** unit tests for ingestion, harmonisation and leakage exclusion;
  a data-contract test that fails if a required variable is missing for a year;
  an evaluation-gate test blocking promotion unless PR-AUC and recall on a fixed
  hold-out meet/exceed the current production model; a smoke test that loads the
  serialised pipeline and scores a sample batch.

---

## Security Checklist, Privacy and Other Risks

- **Store/process PHI?** No. BRFSS public files contain de-identified survey
  responses, not protected health information.
- **Store/process PII?** No direct identifiers (no names, addresses, phone
  numbers). Only de-identified survey attributes are used.
- **User behaviour tracked/stored?** No.
- **Store/process credit-card information?** No.
- **Justification:** the dataset is a public, de-identified CDC release; the
  project neither collects nor links identifiable records.
- **S3 buckets:** a raw bucket (read) for the XPT files, and a processed/curated
  bucket (read/write) for parquet, model artifacts and reports. Buckets use
  encryption at rest and least-privilege IAM policies.
- **Data bias to consider:** self-report and recall bias; telephone-survey
  selection bias; class imbalance; under-representation of undiagnosed cases in
  the negative class.
- **Bias along sensitive features?** Yes — potential for differing performance by
  **age and sex** (and, if used, race/income). Measured subgroup results:

  | Subgroup | n | Prevalence | ROC-AUC | PR-AUC | Recall |
  |---|---|---|---|---|---|
  | sex = male | 201,395 | 10.6% | 0.832 | 0.356 | 0.870 |
  | sex = female | 227,343 | 6.6% | 0.831 | 0.265 | 0.747 |
  | age 18-39 | 98,915 | 1.1% | 0.770 | 0.071 | 0.126 |
  | age 40-59 | 120,210 | 4.6% | 0.817 | 0.222 | 0.589 |
  | age 60+ | 202,116 | 14.5% | 0.760 | 0.345 | 0.889 |

  Recall is higher for males and for older adults; it is very low for younger
  adults because their prevalence is ~1%. The tool must not be used to *rule out*
  risk in younger adults and these gaps must be monitored.
- **Ethical concerns:** the model must be presented as a screening aid, not a
  diagnosis; risk scores could cause anxiety or be misused for discriminatory
  decisions (e.g. insurance). Use is restricted to academic prioritisation, with
  transparency about limitations and subgroup performance.

---

## Future Enhancements

1. **Calibration & threshold policy.** Add isotonic/Platt calibration on a
   validation split and choose the operating threshold from an explicit screening
   budget rather than a fixed 0.5 cut-off.
2. **Odd-year enriched model.** Train a 2021/2023 model that includes blood
   pressure and cholesterol and compare it against the harmonised cross-year
   model to quantify the value of those predictors.
3. **Survey-weighted analysis.** Incorporate BRFSS complex-survey weights and
   report weighted metrics to better reflect the US adult population.
4. **Explainability & automated fairness gates.** Add SHAP-based global/local
   explanations and wire subgroup-recall fairness checks into the CI/CD
   evaluation gate so regressions block promotion automatically.
