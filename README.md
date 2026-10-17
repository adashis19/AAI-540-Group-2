# Heart-Disease Risk Screening (CDC BRFSS) — AAI-540 MLOps

**Group 2 · University of San Diego · AAI-540 Machine Learning Operations**
Authors: Ashis Das, Hanmant Mhalappa Kattimani, Ram Prasanth G · Instructor: Dr. Raj Garg

A reproducible MLOps pipeline that screens adults for likely **coronary heart
disease / myocardial infarction (CHD/MI)** using non-invasive features from the
CDC Behavioral Risk Factor Surveillance System (BRFSS). It is a **supervised
binary classification** problem on the CDC-calculated target `_MICHD`. The model
is a **population-screening aid, not a medical diagnosis**.

## Results at a glance (2023 temporal test set)

Trained on BRFSS **2021 + 2022**, evaluated on held-out **2023**:

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| MajorityBaseline | 0.500 | 0.085 | 0.000 | 0.000 | 0.000 |
| LogisticRegression | 0.834 | 0.315 | 0.802 | 0.207 | 0.329 |
| DecisionTree | 0.784 | 0.264 | 0.689 | 0.212 | 0.324 |
| RandomForest | 0.828 | 0.311 | 0.790 | 0.208 | 0.329 |
| GradientBoosting | 0.835 | 0.319 | 0.079 | 0.505 | 0.136 |
| **XGBoost (selected)** | **0.836** | **0.321** | **0.819** | 0.203 | 0.326 |

Selection metric: **PR-AUC** (best for the ~8.5% positive class). XGBoost wins;
class-weighted logistic regression is a close, interpretable runner-up.

## Project layout

```
heart_disease_mlops/
├── src/
│   ├── config.py          # paths, target, leakage cols, harmonised feature set
│   ├── data_ingestion.py  # XPT -> harmonised parquet (+ target, leakage drop)
│   ├── features.py        # temporal split + leakage-controlled ColumnTransformer
│   ├── train.py           # 5 models, class-weighted, temporal eval, model select
│   ├── eda.py             # EDA figures + summary
│   └── fairness.py        # ROC/PR/calibration/confusion + subgroup metrics
├── notebooks/
│   └── Final_Team_Project_Heart_Disease_Risk_Screening_MLOps.ipynb
├── reports/
│   ├── figures/                       # 12 generated figures
│   ├── model_comparison.csv           # measured model metrics
│   ├── subgroup_metrics.csv           # fairness metrics
│   ├── eda_summary.txt
│   ├── ML_Design_Document_Completed.md
│   └── AAI-540_ML_Design_Document_Group2_COMPLETED.docx
├── models/
│   ├── best_model.joblib              # fitted sklearn Pipeline (preproc + XGB)
│   └── best_model_meta.json
├── data/
│   ├── raw/                           # LLCP2021/2022/2023.XPT (downloaded)
│   └── processed/                     # brfss_{year}.parquet + brfss_combined.parquet
├── build_notebook.py                  # regenerates the notebook
├── build_design_doc.py                # fills the design-doc template -> docx
└── requirements.txt
```

## Setup

```bash
cd heart_disease_mlops
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download the data (CDC BRFSS, ~250 MB zipped)

```bash
cd data/raw
for y in 2021 2022 2023; do
  curl -s -o LLCP${y}XPT.zip \
    https://www.cdc.gov/brfss/annual_data/${y}/files/LLCP${y}XPT.zip
done
python3 - <<'PY'
import zipfile, glob, os
for z in sorted(glob.glob('*.zip')):
    with zipfile.ZipFile(z) as zf: zf.extractall('.')
# CDC archive entries carry a trailing space; normalise the names.
for y in (2021, 2022, 2023):
    src = f'LLCP{y}.XPT '
    if os.path.exists(src): os.rename(src, f'LLCP{y}.XPT')
PY
cd ../..
```

## Run the pipeline

```bash
# 1. Ingest + harmonise raw XPT -> processed parquet
python -m src.data_ingestion

# 2. Train + compare models on temporal split (2021+2022 -> 2023)  [~15 min]
python -m src.train

# 3. EDA figures + summary
python -m src.eda

# 4. Evaluation curves + fairness / subgroup analysis
python -m src.fairness
```

## Reproduce the notebook and documents

```bash
python build_notebook.py                 # (re)generate the notebook
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 \
  notebooks/Final_Team_Project_Heart_Disease_Risk_Screening_MLOps.ipynb

python build_design_doc.py               # fill design-doc template -> completed docx
```

The notebook's ingestion and training cells reuse cached artifacts when present
(`FORCE_REINGEST` / `FORCE_RETRAIN` flags set them to rebuild from scratch), so a
re-run is fast once the pipeline has produced its outputs.

## Key design decisions

- **Temporal validation.** 2023 is a held-out future year, a stricter and more
  realistic test than a random split.
- **Leakage control.** `_MICHD`, `CVDINFR4`, `CVDCRHD4` define the target and are
  never used as features. All imputation/scaling/encoding statistics are learned
  on the training years only.
- **Cross-year harmonisation.** BRFSS collects blood pressure and cholesterol
  only in odd years (absent from 2022), so those variables are **excluded** from
  the common feature set to keep one schema valid for all three years. This is a
  documented limitation.
- **Imbalance handling.** ~8.5% positives → class weighting and PR-AUC/recall as
  primary metrics rather than accuracy.
- **Fairness.** Subgroup recall/PR-AUC are reported by sex and age; gaps
  (e.g. low recall for younger, low-prevalence adults) are disclosed.

## MLOps components — AWS SageMaker and local equivalents

The system is designed for AWS SageMaker. Where the Learner Lab was
unavailable, each required MLOps component is reproduced locally so the design
can be validated and demonstrated. The table maps each required demonstration
item to its SageMaker service and to the local artifact in this repository.

| Required item | AWS SageMaker | Local equivalent (in this repo) |
|---|---|---|
| Feature store & feature groups | SageMaker Feature Store | `data/processed/*.parquet` — the 15-feature harmonised group built by `src/data_ingestion.py` + `src/features.py` |
| Infrastructure monitoring | CloudWatch dashboards/alarms | job logs + `reports/eda_summary.txt`; pipeline run status via `tests/test_pipeline.py` |
| Model & data monitoring | SageMaker Model Monitor | `reports/model_comparison.csv`, `reports/subgroup_metrics.csv`, `reports/figures/08-12_*.png` (ROC, PR, calibration, confusion, subgroup) |
| CI/CD DAG (success + fail) | SageMaker Pipelines | `tests/test_pipeline.py` (data-contract, leakage, temporal-split, preprocessor) as the evaluation gate; `src/train.py` selection by PR-AUC |
| Model registry | SageMaker Model Registry | `models/best_model.joblib` + `models/best_model_meta.json` (selected model, metrics, lineage, leakage-excluded columns) |
| Batch inference outputs | SageMaker Batch Transform | `src/batch_inference.py` → `reports/batch_predictions_2023.csv` (regenerable) + `reports/batch_predictions_2023_sample.csv` + `reports/batch_inference_summary.json` |

Run the local batch-inference job (SageMaker Batch Transform equivalent):

```bash
python -m src.batch_inference
```

It loads the registered best model, scores the held-out 2023 cohort, and writes
a ranked risk-probability output with a screening flag at the operating
threshold.

## Disclaimer

Academic screening exercise only. Not a clinically validated device and not a
substitute for evaluation by a qualified healthcare professional.
