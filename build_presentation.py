"""
Build the AAI-540 Group 2 final video-demonstration slide deck.

Generates a professional .pptx that maps directly to Deliverable #2 (ML System
Operation Validation). The deck covers the business use case, architecture, the
SIX required SageMaker demo items, and future improvements / risks. Speaker
notes on every slide double as the narration script and name the presenter, so
work is split equally across the three team members.

Run:  python build_presentation.py
"""
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "AAI-540_Group2_Deliverable-2_Presentation.pptx"

# Brand palette
NAVY = RGBColor(0x0B, 0x1F, 0x3A)
BLUE = RGBColor(0x1F, 0x6F, 0xB2)
TEAL = RGBColor(0x1A, 0x9E, 0x8F)
GREY = RGBColor(0x44, 0x4A, 0x52)
LIGHT = RGBColor(0xF2, 0xF5, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

A = "Ashis Das"
H = "Hanmant Mhalappa Kattimani"
R = "Ram Prasanth G"

prs = Presentation()
prs.slide_width = Inches(13.333)   # 16:9
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height

BLANK = prs.slide_layouts[6]


def _fill(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def _box(slide, l, t, w, h):
    from pptx.enum.shapes import MSO_SHAPE
    return slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)


def add_text(slide, l, t, w, h, text, size=18, color=GREY, bold=False,
             align=PP_ALIGN.LEFT, font="Calibri"):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = font
    return tb


def bullets(slide, l, t, w, h, items, size=18, color=GREY):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, (lvl, txt) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = lvl
        run = p.add_run()
        run.text = ("• " if lvl == 0 else "– ") + txt
        run.font.size = Pt(size - 2 * lvl)
        run.font.color.rgb = color
        run.font.name = "Calibri"
        p.space_after = Pt(6)
    return tb


def notes(slide, presenter, script):
    # Presenter notes intentionally disabled — the deck is presented without
    # speaker notes. (Kept as a no-op so existing call sites stay valid.)
    return


def content_slide(title, kicker, presenter, script):
    """Standard content slide: top band + title, returns slide + content top."""
    s = prs.slides.add_slide(BLANK)
    band = _box(s, 0, 0, SW, Inches(1.15)); _fill(band, NAVY)
    add_text(s, Inches(0.5), Inches(0.12), Inches(12), Inches(0.4),
             kicker, size=13, color=TEAL, bold=True)
    add_text(s, Inches(0.5), Inches(0.42), Inches(12.3), Inches(0.7),
             title, size=27, color=WHITE, bold=True)
    notes(s, presenter, script)
    return s


# --------------------------------------------------------------------------- #
# Slide 1 — Title
# --------------------------------------------------------------------------- #
s = prs.slides.add_slide(BLANK)
bg = _box(s, 0, 0, SW, SH); _fill(bg, NAVY)
accent = _box(s, 0, Inches(4.9), SW, Inches(0.08)); _fill(accent, TEAL)
add_text(s, Inches(0.8), Inches(1.5), Inches(11.7), Inches(1.3),
         "Heart-Disease Risk Screening", size=46, color=WHITE, bold=True)
add_text(s, Inches(0.8), Inches(2.7), Inches(11.7), Inches(0.8),
         "A Production-Ready MLOps System on AWS SageMaker", size=24, color=TEAL)
add_text(s, Inches(0.8), Inches(5.2), Inches(11.7), Inches(1.2),
         "AAI-540 Machine Learning Operations  ·  Group 2\n"
         "Ashis Das  ·  Hanmant Mhalappa Kattimani  ·  Ram Prasanth G\n"
         "Instructor: Dr. Raj Garg  ·  University of San Diego",
         size=17, color=WHITE)
notes(s, f"{A} (intro, 0:00-0:45)",
      "Welcome. We are Group 2. Over the next ~12 minutes we will walk through "
      "our Heart-Disease Risk Screening ML system built on AWS "
      "SageMaker. I'll cover the business case and architecture, then Hanmant "
      "will present the feature store, infrastructure monitoring and the CI/CD "
      "pipeline, and Ram will present the monitoring, the model registry and "
      "batch inference. Each of us presents "
      "an equal portion. Let's start with the business problem.")

# --------------------------------------------------------------------------- #
# Slide 2 — Agenda
# --------------------------------------------------------------------------- #
s = content_slide("Agenda", "OVERVIEW", f"{A} (0:45-1:15)",
    "Here is our running order. We walk through the full MLOps system: the "
    "feature store, infrastructure monitoring, model and data monitoring, the "
    "CI/CD pipeline in both a successful and a failed state, the model registry, "
    "and the batch-inference outputs. We close with future improvements and "
    "risks. Each topic is explained on its slide, and the explanation is also in "
    "the notes below each slide.")
bullets(s, Inches(0.7), Inches(1.5), Inches(11.8), Inches(5.3), [
    (0, "Business use case & success metrics  (Ashis)"),
    (0, "System architecture on AWS SageMaker  (Ashis)"),
    (0, "Feature Store & feature groups  (Hanmant)"),
    (0, "Infrastructure monitoring  (Hanmant)"),
    (0, "Model & data monitoring  (Ram)"),
    (0, "CI/CD pipeline: successful and failed runs  (Hanmant)"),
    (0, "Model Registry  (Ram)"),
    (0, "Batch inference outputs  (Ram)"),
    (0, "Future improvements, risks & Q&A  (All)"),
], size=20)

# --------------------------------------------------------------------------- #
# Slide 3 — Business use case
# --------------------------------------------------------------------------- #
s = content_slide("Business Use Case", "WHY THIS MATTERS", f"{A} (1:15-2:15)",
    "Our hypothetical stakeholder is a population-health provider that wants to "
    "prioritise adults for preventive heart-health outreach. Cardiovascular "
    "disease is often silent until a serious event. Using only NON-INVASIVE "
    "survey data — demographics, lifestyle and general health — our model flags "
    "individuals more likely to have coronary heart disease or a prior heart "
    "attack, so limited outreach budget targets the highest-risk people first. "
    "This is a screening aid, explicitly NOT a diagnosis. Success is measured by "
    "recall on a future year and by lift over a naive baseline.")
bullets(s, Inches(0.7), Inches(1.5), Inches(7.1), Inches(5.3), [
    (0, "Stakeholder: population-health provider prioritising preventive outreach"),
    (0, "Problem: identify likely CHD/MI from non-invasive data"),
    (0, "Supervised binary classification on CDC BRFSS target _MICHD"),
    (0, "Data: BRFSS 2021 + 2022 (train), 2023 (held-out future year)"),
    (0, "~1.3M records; ~8.5% positive class (imbalanced)"),
    (0, "Screening aid — not a medical diagnosis"),
], size=18)
card = _box(s, Inches(8.1), Inches(1.6), Inches(4.5), Inches(4.8)); _fill(card, LIGHT)
add_text(s, Inches(8.35), Inches(1.8), Inches(4.0), Inches(0.5),
         "Success metrics tied to goals", size=17, color=NAVY, bold=True)
bullets(s, Inches(8.35), Inches(2.4), Inches(4.0), Inches(3.9), [
    (0, "Recall (sensitivity): 0.819"),
    (0, "PR-AUC: 0.321  (baseline 0.085)"),
    (0, "ROC-AUC: 0.836"),
    (0, "~3.8× lift over majority baseline"),
    (0, "Subgroup recall tracked (sex, age)"),
    (0, "Evaluated on a FUTURE year (2023)"),
], size=16, color=GREY)

# --------------------------------------------------------------------------- #
# Slide 4 — Architecture
# --------------------------------------------------------------------------- #
s = content_slide("System Architecture on AWS SageMaker", "HOW IT FITS TOGETHER",
    f"{A} (2:15-3:30)",
    "Here's the end-to-end architecture. Raw BRFSS XPT files land in an S3 raw "
    "bucket. A SageMaker processing job harmonises them into parquet in a curated "
    "bucket and ingests features into SageMaker Feature Store. A SageMaker "
    "Pipeline orchestrates processing, training, an evaluation gate, and model "
    "registration into the Model Registry. Approved models run as Batch "
    "Transform jobs. Model Monitor and CloudWatch watch data quality, model "
    "quality and infrastructure. Every topic in this talk corresponds to a box "
    "on this diagram. One important note on scope: this architecture is "
    "designed for AWS SageMaker. Because AWS access was not available to us, we "
    "present each component as a design, and we implemented and validated the "
    "same pipeline logic locally — so the results and artifacts are real even "
    "though the managed services are presented as the intended design. For each "
    "component we explain the SageMaker service and the local implementation "
    "that backs it. I'll now hand over to Hanmant for the feature store.")
# simple architecture row of boxes
labels = ["S3 Raw\n(XPT)", "Processing\nJob", "Feature\nStore",
          "SageMaker\nPipeline", "Model\nRegistry", "Batch\nTransform"]
colors = [BLUE, BLUE, TEAL, BLUE, TEAL, BLUE]
x = Inches(0.55); y = Inches(2.6); bw = Inches(1.85); bh = Inches(1.2)
gap = Inches(0.18)
for i, (lab, col) in enumerate(zip(labels, colors)):
    b = _box(s, Emu(int(x) + i * (int(bw) + int(gap))), y, bw, bh); _fill(b, col)
    tf = b.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    run = p.add_run(); run.text = lab
    run.font.size = Pt(14); run.font.bold = True; run.font.color.rgb = WHITE
    if i < len(labels) - 1:
        add_text(s, Emu(int(x) + i * (int(bw) + int(gap)) + int(bw) - Inches(0.02)),
                 Inches(2.9), Inches(0.22), Inches(0.5), "→", size=20,
                 color=GREY, align=PP_ALIGN.CENTER)
mon = _box(s, Inches(0.55), Inches(4.3), Inches(12.1), Inches(0.9)); _fill(mon, NAVY)
add_text(s, Inches(0.7), Inches(4.45), Inches(11.8), Inches(0.6),
         "SageMaker Model Monitor  +  CloudWatch   →   data-quality · model-quality · infrastructure dashboards & alarms",
         size=16, color=WHITE, bold=True)
add_text(s, Inches(0.55), Inches(5.5), Inches(12), Inches(1.2),
         "Leakage control (CVDINFR4, CVDCRHD4, _MICHD excluded) and train-only "
         "fit statistics are enforced in every stage. Temporal split (2021+2022 "
         "→ 2023) simulates deployment on a future year.\n"
         "Scope note: architecture designed for AWS SageMaker; components shown "
         "here as local equivalents (Learner Lab budget exhausted).",
         size=14, color=GREY)

# --------------------------------------------------------------------------- #
# Slide 5 — DEMO 1 Feature Store
# --------------------------------------------------------------------------- #
s = content_slide("Feature Store & Feature Groups",
    "DESIGNED FOR SAGEMAKER FEATURE STORE · SHOWN AS LOCAL EQUIVALENT", f"{H} (3:30-5:00)",
    "This is our feature foundation. In the SageMaker design this is a Feature "
    "Store feature group; here we demonstrate the equivalent as the harmonised "
    "feature set we build and persist locally. It holds the 15 variables common "
    "to all three BRFSS years. It has three continuous features — BMI, poor "
    "mental-health days and poor physical-health days — and twelve categorical "
    "survey codes such as age band, sex, general health, smoker status, diabetes "
    "and difficulty walking. The three target-defining columns (CVDINFR4, "
    "CVDCRHD4 and _MICHD) are deliberately kept out to prevent leakage. Because "
    "both training and inference read from this same feature set, the features "
    "seen at inference are identical to those seen in training — that is how we "
    "guarantee train/serve consistency.")
bullets(s, Inches(0.7), Inches(1.5), Inches(6.3), Inches(5.3), [
    (0, "Feature group: heart-disease-features"),
    (0, "15 features, shared across 2021 / 2022 / 2023"),
    (0, "Record identifier + event time per row"),
    (0, "Leakage columns excluded by design"),
    (0, "Same features used for training AND inference"),
    (0, "Guarantees train / serve consistency"),
], size=17)
box = _box(s, Inches(7.2), Inches(1.6), Inches(5.4), Inches(4.7)); _fill(box, LIGHT)
add_text(s, Inches(7.45), Inches(1.78), Inches(5.0), Inches(0.5),
         "The 15 features", size=17, color=NAVY, bold=True)
bullets(s, Inches(7.45), Inches(2.35), Inches(5.0), Inches(3.9), [
    (0, "Continuous: bmi, mental_health_days, phys_health_days"),
    (0, "Demographics: age_group, sex, income_group, education_group"),
    (0, "Health: gen_health, diabetes, kidney_disease, diff_walk"),
    (0, "Lifestyle: smoked_100, smoker_status, phys_activity"),
    (0, "Access: has_personal_doctor"),
], size=14, color=GREY)

# --------------------------------------------------------------------------- #
# Slide 6 — DEMO 2 Infra monitoring
# --------------------------------------------------------------------------- #
s = content_slide("Infrastructure Monitoring",
    "DESIGNED FOR CLOUDWATCH · SHOWN AS LOCAL EQUIVALENT", f"{H} (5:00-6:15)",
    "In the SageMaker design, infrastructure is monitored with CloudWatch "
    "dashboards and alarms; here we show the equivalent using our local job logs "
    "and run status. The design tracks whether each job succeeded or "
    "failed, how long each stage ran, CPU and memory utilisation on the "
    "ml.m5.large instances, and error and invocation counts, with alarms so "
    "that if a batch job fails or runs longer than its expected window, the team "
    "is notified automatically. The point of this layer is operational "
    "confidence: it tells the on-call engineer whether the pipeline ran cleanly "
    "in each screening cycle, regardless of how the model itself is performing.")
bullets(s, Inches(0.7), Inches(1.5), Inches(6.3), Inches(5.3), [
    (0, "Job success / failure for every stage"),
    (0, "Runtime / duration per stage"),
    (0, "CPU & memory utilisation (ml.m5.large)"),
    (0, "Error-rate and invocation counts"),
    (0, "Automated alarms on failure or runtime breach"),
    (0, "Answers: did the pipeline run cleanly this cycle?"),
], size=17)
box = _box(s, Inches(7.2), Inches(1.6), Inches(5.4), Inches(4.7)); _fill(box, LIGHT)
add_text(s, Inches(7.45), Inches(1.78), Inches(5.0), Inches(0.5),
         "What we watch & why", size=17, color=NAVY, bold=True)
bullets(s, Inches(7.45), Inches(2.35), Inches(5.0), Inches(3.9), [
    (0, "Success/failure → catch broken cycles fast"),
    (0, "Duration → spot slow-downs & cost creep"),
    (0, "CPU / memory → right-size the instance"),
    (0, "Errors → surface data or code faults"),
    (0, "Alarm → notify on-call without manual checks"),
], size=14, color=GREY)

# --------------------------------------------------------------------------- #
# Slide 7 — DEMO 3 Model & data monitoring (with figures)
# --------------------------------------------------------------------------- #
s = content_slide("Model & Data Monitoring",
    "DESIGNED FOR SAGEMAKER MODEL MONITOR · SHOWN AS LOCAL EQUIVALENT", f"{R} (6:15-7:45)",
    "In the SageMaker design, model and data monitoring run through Model "
    "Monitor; here we show the equivalent reports we generate locally. We "
    "monitor three things after release. For model quality we track recall, "
    "PR-AUC and calibration against an agreed floor and alert if sensitivity "
    "drops. For data quality we watch feature distributions and missingness "
    "against the training baseline using a population-stability index, so drift "
    "is caught before it degrades the model. And for fairness we recompute "
    "subgroup recall by sex and age every cycle. The two charts on this slide are "
    "our actual outputs: the precision-recall curve on the left shows strong "
    "ranking of the rare positive class, and the subgroup chart on the right "
    "shows the performance gaps we explicitly track — higher recall for males "
    "and older adults, and low recall for younger adults whose prevalence is "
    "only about one percent.")
# place two figures
for fig, left in [("09_pr_curve.png", Inches(0.55)),
                  ("12_subgroup_performance.png", Inches(6.8))]:
    fp = FIG / fig
    if fp.exists():
        s.shapes.add_picture(str(fp), left, Inches(1.5), height=Inches(3.5))
bullets(s, Inches(0.55), Inches(5.2), Inches(12.1), Inches(1.9), [
    (0, "Model quality: recall, PR-AUC, calibration (Brier) vs agreed floor"),
    (0, "Data quality: feature-distribution & missingness drift (PSI) vs baseline"),
    (0, "Fairness drift: subgroup recall by sex and age re-checked each cycle"),
], size=16)

# --------------------------------------------------------------------------- #
# Slide 8 — DEMO 4 CI/CD DAG success + failed
# --------------------------------------------------------------------------- #
s = content_slide("CI/CD Pipeline: Successful & Failed Runs",
    "DESIGNED FOR SAGEMAKER PIPELINES · SHOWN AS LOCAL EQUIVALENT", f"{H} (7:45-9:15)",
    "In the SageMaker design this is a SageMaker Pipeline; here we demonstrate "
    "the equivalent with our local pipeline logic and test gate. The pipeline "
    "is a directed graph of stages: schema validation, processing, "
    "feature ingest, training, an evaluation gate, model registration, and "
    "batch scoring. The left column shows a successful run — every stage passes "
    "and the model is registered. The right column shows a failed run that is "
    "stopped at the evaluation gate: if the candidate's PR-AUC or recall falls "
    "below the current production model, or a required variable is missing for a "
    "year, the gate fails and the later stages never run, so a worse model can "
    "never be promoted. Our automated test suite — leakage, data-contract, "
    "temporal-split and preprocessor checks — runs as the very first step and "
    "currently passes five out of five.")
# two mini DAG columns
stages = ["Schema validate", "Processing", "Feature ingest", "Training",
          "Evaluation gate", "Register model", "Batch transform"]
for col, (title_txt, col_color, fail_idx) in enumerate(
        [("SUCCESSFUL RUN", TEAL, None), ("FAILED RUN", RGBColor(0xC0,0x39,0x2B), 4)]):
    cx = Inches(0.9) + col * Inches(6.4)
    add_text(s, cx, Inches(1.45), Inches(5.0), Inches(0.4),
             title_txt, size=16, color=col_color, bold=True)
    for i, st in enumerate(stages):
        yy = Emu(int(Inches(1.95)) + i * int(Inches(0.66)))
        c = col_color if (fail_idx is None or i < fail_idx) else RGBColor(0x9E,0x9E,0x9E)
        if fail_idx is not None and i == fail_idx:
            c = RGBColor(0xC0, 0x39, 0x2B)
        b = _box(s, cx, yy, Inches(4.7), Inches(0.5)); _fill(b, c)
        tf = b.text_frame; p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        mark = " ✓" if (fail_idx is None or i < fail_idx) else (
            " — FAILS HERE" if i == fail_idx else " (skipped)")
        r.text = st + mark
        r.font.size = Pt(12); r.font.color.rgb = WHITE; r.font.bold = True

# --------------------------------------------------------------------------- #
# Slide 9 — DEMO 5 Model Registry
# --------------------------------------------------------------------------- #
s = content_slide("Model Registry",
    "DESIGNED FOR SAGEMAKER MODEL REGISTRY · SHOWN AS LOCAL EQUIVALENT", f"{R} (9:15-10:15)",
    "In the SageMaker design this is the Model Registry; here we demonstrate the "
    "equivalent with our saved model artifact and its metadata. Every model that "
    "passes the evaluation gate is recorded as a version together with the "
    "metrics it was approved on. Our selected version is XGBoost, recorded with "
    "a PR-AUC of 0.321 and recall of 0.819, and marked approved. Only an "
    "approved version is eligible for batch deployment, which gives us "
    "governance over what reaches production. The metadata also records lineage "
    "— from the data and features through training to the model artifact — so we "
    "can audit any prediction and roll back to a previous approved version if a "
    "new one misbehaves.")
bullets(s, Inches(0.7), Inches(1.5), Inches(6.3), Inches(5.3), [
    (0, "Model package group: heart-disease-screening"),
    (0, "Versioned models, each with captured metrics"),
    (0, "Selected: XGBoost  (PR-AUC 0.321, recall 0.819)"),
    (0, "Approval status gates deployment"),
    (0, "Full lineage: data → features → training → model"),
    (0, "One-click rollback to a prior approved version"),
], size=17)
box = _box(s, Inches(7.2), Inches(1.6), Inches(5.4), Inches(4.7)); _fill(box, LIGHT)
add_text(s, Inches(7.45), Inches(1.78), Inches(5.0), Inches(0.5),
         "Why a registry matters", size=17, color=NAVY, bold=True)
bullets(s, Inches(7.45), Inches(2.35), Inches(5.0), Inches(3.9), [
    (0, "Governance: only Approved models deploy"),
    (0, "Traceability: metrics travel with the model"),
    (0, "Auditability: full lineage for each version"),
    (0, "Safety: fast rollback if quality regresses"),
    (0, "Reproducibility: pin the exact artifact"),
], size=14, color=GREY)

# --------------------------------------------------------------------------- #
# Slide 10 — DEMO 6 Batch inference (with confusion matrix figure)
# --------------------------------------------------------------------------- #
s = content_slide("Batch Inference Outputs",
    "DESIGNED FOR SAGEMAKER BATCH TRANSFORM · SHOWN AS LOCAL EQUIVALENT", f"{R} (10:15-11:30)",
    "Finally, the system producing value. In the SageMaker design this is a "
    "Batch Transform job; here we demonstrate the equivalent with our local "
    "batch-inference script (src/batch_inference.py). It scores the entire 2023 "
    "cohort and writes an output file where every person gets a "
    "heart-disease risk probability. Applying our operating threshold turns "
    "those probabilities into a ranked outreach list, so the health team "
    "contacts the highest-risk people first. The confusion matrix on the right "
    "puts this in context: at our chosen operating point we correctly catch "
    "about 82 percent of true cases. The job runs on a modest CPU instance and "
    "scales to hundreds of thousands of rows cheaply. And to be clear — this "
    "prioritises people for screening; it is not a medical diagnosis.")
fp = FIG / "11_confusion_matrix.png"
if fp.exists():
    s.shapes.add_picture(str(fp), Inches(7.1), Inches(1.5), height=Inches(3.6))
bullets(s, Inches(0.7), Inches(1.5), Inches(6.1), Inches(5.0), [
    (0, "Batch Transform scores the 2023 cohort"),
    (0, "Output in S3: risk probability per record"),
    (0, "Threshold → prioritised outreach list"),
    (0, "Recall 0.819 at the chosen operating point"),
    (0, "Scales to hundreds of thousands of rows on ml.m5.large"),
    (0, "Screening prioritisation, NOT a diagnosis"),
], size=17)

# --------------------------------------------------------------------------- #
# Slide 11 — Future improvements & risks
# --------------------------------------------------------------------------- #
s = content_slide("Future Improvements, Risks & Wrap-up",
    "WHAT'S NEXT", f"{A} (11:30-12:30)",
    "To close: future improvements include probability calibration with an "
    "explicit threshold policy, an odd-year model that adds blood pressure and "
    "cholesterol, survey-weighted metrics, and SHAP explanations wired into an "
    "automated fairness gate. Key risks: recall is low for younger, "
    "low-prevalence adults, so the tool must never be used to rule out risk; "
    "self-report and selection bias exist in survey data; and scores must not "
    "be misused for insurance or discriminatory decisions. Scalability and "
    "cost are low because we use batch scoring on modest CPU instances. Thanks "
    "for watching — we're happy to take questions.")
bullets(s, Inches(0.7), Inches(1.5), Inches(6.3), Inches(5.3), [
    (0, "Calibration + explicit screening-budget threshold"),
    (0, "Odd-year model w/ blood pressure & cholesterol"),
    (0, "Survey-weighted metrics for US-population estimates"),
    (0, "SHAP explainability + automated fairness gate in CI/CD"),
], size=18)
box = _box(s, Inches(7.2), Inches(1.55), Inches(5.4), Inches(5.0)); _fill(box, LIGHT)
add_text(s, Inches(7.45), Inches(1.75), Inches(5.0), Inches(0.5),
         "Risks to manage", size=17, color=NAVY, bold=True)
bullets(s, Inches(7.45), Inches(2.35), Inches(5.0), Inches(4.0), [
    (0, "Low recall for younger adults — never use to rule OUT risk"),
    (0, "Self-report / telephone-survey selection bias"),
    (0, "Ethical: no insurance / discriminatory use"),
    (0, "Technical: cross-year schema drift (harmonised)"),
    (0, "Scalability/cost: low — batch on CPU"),
], size=15, color=GREY)

# --------------------------------------------------------------------------- #
# Slide 12 — Thank you / links
# --------------------------------------------------------------------------- #
s = prs.slides.add_slide(BLANK)
bg = _box(s, 0, 0, SW, SH); _fill(bg, NAVY)
accent = _box(s, 0, Inches(4.6), SW, Inches(0.08)); _fill(accent, TEAL)
add_text(s, Inches(0.8), Inches(2.0), Inches(11.7), Inches(1.0),
         "Thank You", size=48, color=WHITE, bold=True)
add_text(s, Inches(0.8), Inches(3.2), Inches(11.7), Inches(1.2),
         "GitHub: https://github.com/adashis19/AAI-540-Group-2.git\n"
         "AAI-540 · Group 2 · Heart-Disease Risk Screening MLOps",
         size=20, color=TEAL)
add_text(s, Inches(0.8), Inches(5.0), Inches(11.7), Inches(1.0),
         "Ashis Das  ·  Hanmant Mhalappa Kattimani  ·  Ram Prasanth G",
         size=18, color=WHITE)
notes(s, "All",
      "Thank you. Questions welcome. Our GitHub repository link is on this slide "
      "and in the design document.")

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(OUT))
print("Wrote", OUT, "with", len(prs.slides._sldIdLst), "slides")
