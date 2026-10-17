# AAI-540 Group 2 — Video Demonstration Script & Outline (Deliverable #2)

**Project:** Heart-Disease Risk Screening — Production MLOps on AWS SageMaker
**Team:** Ashis Das · Hanmant Mhalappa Kattimani · Ram Prasanth G
**Target length:** 10–15 minutes (aim ~12). Export as **.mp4**.
**Companion deck:** `AAI-540_Group2_Deliverable-2_Presentation.pptx` (speaker
notes on each slide repeat this script).

> This script is the outline/transcript required alongside the video. Every one
> of the six required demonstration items is called out in **bold**. Speaking
> time is split roughly equally across the three presenters.

> **SCOPE NOTE (state this in the video).** The system is *designed for AWS
> SageMaker*. Because we did not have AWS access, this is a **slide-based
> presentation**: we present each SageMaker component as the intended design and
> explain the **local implementation** that backs it (the same pipeline logic,
> run and validated in our own environment, with real results and artifacts).
> We are not claiming the managed SageMaker services are running live.

---

## Recording checklist (slide-based presentation)

1. Present directly from the deck
   `AAI-540_Group2_Deliverable-2_Presentation.pptx` — each slide already
   contains the content, and the speaker notes contain this narration.
2. Screen-record the slides in presenter/full-screen mode with voice narration.
3. For each of the six components, state the **SageMaker service** (shown on the
   slide kicker) and then explain the **local implementation** that backs it.
4. Keep each presenter's portion roughly equal (~4 min each); aim ~12 min total.
5. Record audio per section, edit out glitches/outtakes, export to **.mp4**.

---

## Running order & timings

| # | Slide | Presenter | Time | Required item |
|---|---|---|---|---|
| 1 | Title | Ashis | 0:00–0:45 | — |
| 2 | Agenda | Ashis | 0:45–1:15 | — |
| 3 | Business use case | Ashis | 1:15–2:15 | — |
| 4 | Architecture | Ashis | 2:15–3:30 | — |
| 5 | Feature Store | Hanmant | 3:30–5:00 | **1. Feature store & feature groups** |
| 6 | Infra monitoring | Hanmant | 5:00–6:15 | **2. Infrastructure monitoring dashboards** |
| 7 | Model/data monitoring | Ram | 6:15–7:45 | **3. Model or data monitoring reports** |
| 8 | CI/CD DAG | Hanmant | 7:45–9:15 | **4. CI/CD DAG success + failed** |
| 9 | Model Registry | Ram | 9:15–10:15 | **5. Model registry** |
| 10 | Batch inference | Ram | 10:15–11:30 | **6. Batch inference / endpoint outputs** |
| 11 | Future improvements & risks | Ashis | 11:30–12:30 | — |
| 12 | Thank you | All | 12:30–12:45 | — |

---

## Full narration

### Slide 1 — Title · Ashis (0:00)
"Welcome — we are Group 2. Over the next ~12 minutes we'll validate the operation
of our Heart-Disease Risk Screening ML system built on AWS SageMaker. I'll cover
the business case and architecture; Hanmant will demo the feature store, the
infrastructure dashboards and the CI/CD pipeline; Ram will demo the monitoring
reports, the model registry, and batch inference. Each of us presents an equal
portion."

### Slide 2 — Agenda · Ashis (0:45)
"Here's our running order. We map every required demonstration item to a live
SageMaker screen: feature store, infrastructure monitoring, model and data
monitoring reports, the CI/CD DAG in both a successful and a failed state, the
model registry, and batch-inference outputs. We close with future improvements
and risks."

### Slide 3 — Business use case · Ashis (1:15)
"Our stakeholder is a population-health provider that wants to prioritise adults
for preventive heart-health outreach. Cardiovascular disease is often silent
until a serious event. Using only non-invasive survey data — demographics,
lifestyle and general health — the model flags individuals more likely to have
coronary heart disease or a prior heart attack, so limited outreach budget
targets the highest-risk people first. This is a screening aid, explicitly not a
diagnosis. We frame it as supervised binary classification on the CDC BRFSS
`_MICHD` target, train on 2021 and 2022, and hold out 2023 as a future year.
Success is recall and PR-AUC on that future year, and lift over a naive
baseline."

### Slide 4 — Architecture · Ashis (2:15)
"Here's the end-to-end architecture. Raw BRFSS files land in an S3 raw bucket. A
SageMaker processing job harmonises them to parquet and ingests features into
Feature Store. A SageMaker Pipeline orchestrates processing, training, an
evaluation gate, and registration into the Model Registry. Approved models run
as Batch Transform jobs. Model Monitor and CloudWatch watch data quality, model
quality and infrastructure. Leakage columns are excluded everywhere and all fit
statistics are learned on training years only. Over to Hanmant."

### Slide 5 — DEMO 1: Feature Store & feature groups · Hanmant (3:30)
"In the SageMaker design this is a Feature Store feature group; here we show the
local equivalent. It holds the 15 harmonised features common to all
three BRFSS years — bmi, age group, sex, general health, smoker status and so
on — each with its type, plus a record identifier and event time. The processed feature set is persisted as parquet. Target-
defining columns are deliberately excluded to prevent leakage. Both training and
batch inference read from this same feature group, which guarantees train/serve
consistency."

### Slide 6 — DEMO 2: Infrastructure monitoring dashboards · Hanmant (5:00)
"In the SageMaker design, infrastructure is monitored with a CloudWatch
dashboard; here we show the local equivalent. The design tracks job success versus
failure, run duration per stage, CPU and memory utilisation on the ml.m5.large
instances, and error counts. The design includes an alarm that fires if a batch job fails or
runs too long. These
signals tell on-call whether the screening pipeline ran cleanly each cycle,
independent of model quality."

### Slide 7 — DEMO 3: Model & data monitoring reports · Ram (6:15)
"In the SageMaker design this is Model Monitor comparing the current batch against
our baseline; here we show the equivalent local reports. The data-quality
report flags feature-distribution drift and missingness beyond the baseline
constraints. For model quality we track recall, PR-AUC and calibration against an
agreed floor. The PR curve and subgroup-performance chart on this slide are our computed
artifacts. Critically, we re-check subgroup recall by sex and
age every cycle to catch fairness drift, because recall differs across groups."

### Slide 8 — DEMO 4: CI/CD DAG success + failed · Hanmant (7:45)
"In the SageMaker design this is a SageMaker Pipeline; here we show the local
equivalent. The pipeline DAG is: schema validation, processing, feature ingest,
training, evaluation gate, register, batch. First a successful execution — every
step green and the model registered. Now a failed execution — the evaluation
gate fails, and the reason is that PR-AUC fell below the
incumbent [or a required variable was missing for a year]. This proves the gate
actually blocks a worse model from being promoted. Our pytest suite — leakage,
data-contract, temporal-split and preprocessor checks — runs as the first CI
step."

### Slide 9 — DEMO 5: Model Registry · Ram (9:15)
"In the SageMaker design this is the Model Registry; here we show the local
equivalent. These are the recorded versions. The selected
XGBoost version carries its captured metrics — PR-AUC 0.321, recall 0.819 — and
its approval status is Approved. Only a model that passes the evaluation gate is
registered, and only an Approved version is eligible for batch deployment. We get
full lineage from data to features to training job to model, and one-click
rollback to a prior version."

### Slide 10 — DEMO 6: Batch inference outputs · Ram (10:15)
"In the SageMaker design this is a Batch Transform job; here we show the local
equivalent that scored the 2023 cohort, completed
successfully. In the output, each record has a heart-disease risk
probability. Applying our operating threshold turns these into a prioritised
outreach list. The confusion matrix shows that at this operating point we catch
about 82% of true cases. It scales to hundreds of thousands of rows on a modest
CPU instance. Again — this ranks people for screening; it is not a diagnosis."

### Slide 11 — Future improvements & risks · Ashis (11:30)
"Future improvements: probability calibration with an explicit screening-budget
threshold; an odd-year model that adds blood pressure and cholesterol;
survey-weighted metrics; and SHAP explanations wired into an automated fairness
gate. Risks to manage: recall is low for younger, low-prevalence adults, so the
tool must never be used to rule out risk; survey data carries self-report and
selection bias; and scores must not be misused for insurance or discriminatory
decisions. Scalability and cost are low thanks to batch scoring on CPU."

### Slide 12 — Thank you · All (12:30)
"Thanks for watching. Our repository link is on this slide and in the design
document. We're happy to take questions."

---

## Mapping to the rubric (ML Demonstration, 80 pts)

- **Not over the limit, includes this outline/transcript** → ~12 min target + this file.
- **Business context to stakeholders** → slides 3–4.
- **Validates operability + all listed ML elements** → slides 5–10 cover all six
  required items; each names the SageMaker service and the local implementation
  that backs it (feature set, monitoring reports/figures, CI/CD test gate, model
  metadata, batch-inference output).
- **Equal team presentation** → each presenter owns ~4 minutes (table above).
- **Edited, professional** → record per-section, remove glitches, export .mp4.

## How to record (slide-based)
This script and deck are self-contained: present directly from
`AAI-540_Group2_Deliverable-2_Presentation.pptx` with voice narration (the
speaker notes repeat this transcript). State clearly that the SageMaker services
are the intended design and that the pipeline logic was implemented and
validated locally (real results and artifacts in the repository). No AWS/console
screens are required for this presentation.
