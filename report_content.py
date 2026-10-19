"""
Shared content for the APA-style academic report.

Holds the title-page metadata, the running head, the abstract, and an ordered
list of report sections (heading + list-of-paragraph body, with optional figure
and table references). Both the Word builder (report_build_docx.py) and the PDF
builder (report_build_pdf.py) import this module so the two documents carry
identical wording.

The prose is written specifically for this report (not copied from the notebook
or design document) to satisfy the "unique wording" requirement.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "reports" / "figures"

# --------------------------------------------------------------------------- #
# Title-page / header metadata — identical to the notebook header.
# --------------------------------------------------------------------------- #
TITLE = "Heart-Disease Risk Screening Using Machine Learning (CDC BRFSS)"
AUTHORS = "Ashis Das, Hanmant Mhalappa Kattimani, and Ram Prasanth G"
COURSE = "AAI-540 Machine Learning Operations"
UNIVERSITY = "University of San Diego"
INSTRUCTOR = "Dr. Raj Garg"
DATE = "October 18, 2026"
RUNNING_HEAD = "HEART-DISEASE RISK SCREENING"

ABSTRACT = (
    "Coronary heart disease and myocardial infarction remain leading "
    "contributors to preventable mortality, yet a substantial share of at-risk "
    "adults are never flagged for early evaluation. This study asks whether "
    "routinely collected, non-invasive survey information can be used to "
    "prioritise adults for cardiovascular screening. Using three consecutive "
    "annual releases of the Centers for Disease Control and Prevention "
    "Behavioral Risk Factor Surveillance System (BRFSS) covering 2021 through "
    "2023, we frame the task as supervised binary classification against the "
    "agency-calculated heart-disease indicator. A harmonised, leakage-controlled "
    "pipeline was engineered so that a single feature schema remains valid across "
    "all three survey years, and models were trained on the 2021 and 2022 cohorts "
    "and evaluated on the entirely held-out 2023 cohort to emulate deployment on "
    "a future population. Five classifiers were compared under class-imbalance "
    "correction: a logistic-regression reference model, a decision tree, a random "
    "forest, a gradient-boosting ensemble, and an extreme gradient-boosting "
    "model. The extreme gradient-boosting model produced the strongest ranking "
    "behaviour, achieving an area under the receiver-operating-characteristic "
    "curve of 0.836 and an area under the precision-recall curve of 0.321 while "
    "recovering roughly 82 percent of true cases. All learners decisively "
    "outperformed a majority-class reference, and predictive quality remained "
    "stable across survey years, indicating meaningful temporal generalisation. "
    "A subgroup audit disclosed non-trivial performance differences by sex and "
    "age, most notably weak sensitivity among younger, low-prevalence adults. We "
    "conclude that the approach is a credible screening aid for population-level "
    "triage, provided its probabilistic outputs, calibration limits, and "
    "subgroup disparities are transparently communicated and continuously "
    "monitored."
)
KEYWORDS = ("machine learning operations, cardiovascular screening, binary "
            "classification, class imbalance, temporal validation, algorithmic "
            "fairness, BRFSS")


def _p(*paras):
    return list(paras)


# --------------------------------------------------------------------------- #
# Sections: each is (level, heading, [body paragraphs], [figures], [tables])
# level 1 = major heading, 2 = subheading. figures = list of (path, caption).
# tables = list of (caption, header_row, data_rows).
# --------------------------------------------------------------------------- #

MODEL_TABLE = (
    "Table 1\nComparative Performance on the 2023 Temporal Test Set",
    ["Model", "ROC-AUC", "PR-AUC", "Recall", "Precision", "F1", "Specificity"],
    [
        ["Majority baseline", ".500", ".085", ".000", ".000", ".000", "1.000"],
        ["Logistic regression", ".834", ".315", ".802", ".207", ".329", ".716"],
        ["Decision tree", ".784", ".264", ".689", ".212", ".324", ".763"],
        ["Random forest", ".828", ".311", ".790", ".208", ".329", ".721"],
        ["Gradient boosting", ".835", ".319", ".079", ".505", ".136", ".993"],
        ["XGBoost (selected)", ".836", ".321", ".819", ".203", ".326", ".703"],
    ],
)

SUBGROUP_TABLE = (
    "Table 2\nSubgroup Performance by Sex and Age Band (2023 Test Set)",
    ["Subgroup", "N", "Prevalence", "ROC-AUC", "PR-AUC", "Recall"],
    [
        ["Male", "201,395", "10.6%", ".832", ".356", ".870"],
        ["Female", "227,343", "6.6%", ".831", ".265", ".747"],
        ["Age 18-39", "98,915", "1.1%", ".770", ".071", ".126"],
        ["Age 40-59", "120,210", "4.6%", ".817", ".222", ".589"],
        ["Age 60+", "202,116", "14.5%", ".760", ".345", ".889"],
    ],
)

SECTIONS = [
    (1, "Introduction", _p(
        "Cardiovascular conditions, and in particular coronary heart disease and "
        "myocardial infarction, continue to impose an enormous burden on public "
        "health systems worldwide. A persistent difficulty is that early risk "
        "often goes unrecognised: individuals frequently learn of their exposure "
        "only after an acute and sometimes irreversible event. Because clinical "
        "screening capacity is finite, health programmes benefit from inexpensive "
        "mechanisms that can rank a population by likely risk and steer limited "
        "diagnostic resources toward those who would gain the most.",
        "This report investigates whether machine learning applied to widely "
        "available, non-invasive survey responses can support that ranking task. "
        "The motivation is deliberately operational rather than diagnostic. We do "
        "not attempt to determine, in a clinical sense, whether any individual "
        "has cardiovascular disease; instead we estimate a probability that can "
        "inform outreach and prioritisation. Framing the problem this way keeps "
        "the work firmly within the scope of a decision-support instrument and "
        "away from any claim of medical adjudication.",
        "Three design commitments shape the study. First, evaluation is temporal: "
        "the most recent survey year is withheld entirely so that reported "
        "performance reflects generalisation to a future cohort rather than "
        "optimistic in-sample fit. Second, the target class is rare, so the "
        "analysis foregrounds sensitivity and precision-recall behaviour rather "
        "than accuracy, which would be trivially inflated by predicting the "
        "majority outcome. Third, because several predictors are socially "
        "sensitive, the model is audited for disparate performance across sex and "
        "age groups.",
    ), [], []),

    (1, "Problem Statement and Objectives", _p(
        "The learning problem is supervised binary classification. Each "
        "respondent is represented by a vector of demographic, behavioural, and "
        "general-health attributes, and the label indicates whether the "
        "respondent has ever been told they experienced coronary heart disease or "
        "a heart attack, as encoded by the survey agency's calculated indicator. "
        "A positive label denotes reported disease and a negative label denotes "
        "its absence.",
        "The study pursued six concrete objectives: to assemble a consistent "
        "multi-year dataset from independently published survey files; to define "
        "a clean target while eliminating any predictor capable of leaking that "
        "target; to characterise prevalence, missingness, and risk-factor "
        "relationships through exploratory analysis; to construct a preprocessing "
        "and modelling pipeline whose statistics are estimated only from training "
        "data; to benchmark an interpretable reference model against progressively "
        "more expressive ensembles; and to quantify both temporal robustness and "
        "subgroup fairness before recommending a final model.",
    ), [], []),

    (1, "Data and Materials", _p(
        "The empirical basis for the study is the Behavioral Risk Factor "
        "Surveillance System, a large annual telephone survey of "
        "non-institutionalised adults conducted under the auspices of the federal "
        "public-health agency. Three consecutive annual files were obtained, "
        "spanning calendar years 2021, 2022, and 2023, and together they "
        "contribute in excess of 1.3 million respondent records. After deriving a "
        "valid target and discarding records whose outcome could not be "
        "determined, 1,302,907 rows remained available for analysis.",
        "The positive outcome is uncommon, occurring in roughly eight to nine "
        "percent of respondents in each year. This imbalance is a defining "
        "characteristic of the problem and directly motivates the evaluation "
        "strategy adopted later. The survey is well documented through annual "
        "codebooks, which proved essential when reconciling variable names and "
        "response encodings that shift subtly from one release to the next.",
        "A material harmonisation constraint emerged during data assembly. The "
        "survey administers its blood-pressure and cholesterol questions only in "
        "odd-numbered years, so those items are absent from the 2022 core file. "
        "Rather than fabricate values or abandon a full survey year, we elected to "
        "exclude hypertension and cholesterol from the shared feature schema so "
        "that a single, honest set of predictors would apply uniformly to all "
        "three years. This decision sacrifices two clinically informative signals "
        "in exchange for cross-year consistency, and it is acknowledged as a "
        "limitation of the present work.",
    ), [], []),

    (1, "Methodology", _p(), [], []),
    (2, "Data Ingestion and Harmonisation", _p(
        "Raw survey files are distributed in a fixed-width transport format that "
        "required careful reading, including correction of trailing whitespace in "
        "column identifiers and specification of a Latin-1 text encoding to parse "
        "the embedded variable labels. From each annual file the pipeline retains "
        "only the harmonised predictor set together with the target-defining "
        "fields, derives a clean zero-or-one label from the agency's calculated "
        "indicator, and discards records whose label is unknown. Survey codes that "
        "denote refusal, uncertainty, or non-response are recoded as missing "
        "values so that they do not masquerade as substantive answers, and body "
        "mass index is rescaled from the survey's integer representation to "
        "conventional units with implausible extremes removed.",
    ), [], []),
    (2, "Leakage Control", _p(
        "Three fields directly constitute the target and were therefore forbidden "
        "as inputs. Admitting any of them would allow the model to reconstruct the "
        "label almost perfectly and would render every performance figure "
        "meaningless. Their explicit exclusion is enforced in configuration and "
        "verified by an automated test, reflecting the principle that leakage "
        "prevention should be a checked contract rather than an assumption.",
    ), [], []),
    (2, "Feature Engineering and Splitting", _p(
        "Continuous predictors are median-imputed and standardised, while "
        "categorical survey codes are imputed with their most frequent value and "
        "one-hot encoded with an explicit provision for unseen categories at "
        "inference time. Crucially, every transformation is fitted on the training "
        "years alone and then applied unchanged to the held-out year, so no "
        "information from the future cohort can seep backward into training. The "
        "dataset is partitioned strictly by survey year: the 2021 and 2022 cohorts "
        "form the development set and the 2023 cohort serves as an untouched "
        "temporal test set.",
    ), [], []),
    (2, "Model Family and Imbalance Correction", _p(
        "Five estimators were evaluated. A logistic-regression model provides an "
        "interpretable, well-calibrated reference. A single decision tree offers a "
        "transparent non-linear contrast, and a random forest aggregates many such "
        "trees to reduce variance. Two boosting approaches, a classical "
        "gradient-boosting ensemble and an extreme gradient-boosting "
        "implementation, capture higher-order interactions. Because the positive "
        "class is scarce, each learner is made imbalance-aware, whether through "
        "balanced class weighting or through a positive-class scaling factor equal "
        "to the negative-to-positive ratio. A majority-class predictor is retained "
        "purely as a floor against which genuine skill can be measured.",
    ), [], []),

    (1, "Exploratory Data Analysis", _p(
        "Exploratory analysis served both to validate the data and to build "
        "intuition about the predictive landscape. Reported disease prevalence was "
        "essentially flat across the three years, which is reassuring for a model "
        "expected to transfer from earlier cohorts to a later one; the mild "
        "year-to-year fluctuation is precisely the perturbation that the temporal "
        "test is designed to probe. The overall class distribution confirmed the "
        "pronounced scarcity of positive cases and justified the emphasis on "
        "recall and precision-recall summaries.",
        "Missingness was concentrated in the income and body-mass-index fields, a "
        "pattern consistent with survey non-response on financial and physical "
        "measurements; the pipeline's training-only imputation addresses this "
        "without contaminating the evaluation. Continuous predictors overlapped "
        "heavily between the two outcome classes, demonstrating that no single "
        "measurement cleanly separates cases from non-cases and that a "
        "multivariate model is required. Prevalence rose steeply with age and was "
        "elevated among male respondents, and categorical risk factors such as "
        "poor self-rated health, diabetes, difficulty walking, and kidney disease "
        "were each strongly associated with the outcome, lending clinical face "
        "validity to the feature set.",
    ), [
        (FIG / "01_prevalence_by_year.png",
         "Figure 1. Reported heart-disease prevalence by survey year."),
        (FIG / "02_class_balance.png",
         "Figure 2. Overall class distribution across all years."),
        (FIG / "03_missingness.png",
         "Figure 3. Fraction of missing values by feature."),
        (FIG / "05_prevalence_by_group.png",
         "Figure 4. Prevalence by age band and by sex."),
        (FIG / "06_categorical_association.png",
         "Figure 5. Prevalence across categorical risk factors."),
        (FIG / "07_correlation_heatmap.png",
         "Figure 6. Spearman correlation among features and the target."),
    ], []),

    (1, "Results", _p(
        "Table 1 summarises performance on the untouched 2023 cohort. Every "
        "trained model improved substantially over the majority-class floor, whose "
        "precision-recall area equals the base prevalence and whose recall is "
        "zero. The extreme gradient-boosting model delivered the best ranking "
        "performance on both threshold-independent measures, with a "
        "receiver-operating-characteristic area of 0.836 and a precision-recall "
        "area of 0.321, while simultaneously recovering the largest share of true "
        "cases at the default decision threshold. The logistic-regression "
        "reference was remarkably competitive and, given its transparency, "
        "represents an attractive fallback for settings that prize auditability.",
        "The behaviour of the classical gradient-boosting ensemble is "
        "instructive. Although it ranked cases well, as reflected in a high "
        "receiver-operating area, its recall at the fixed one-half threshold "
        "collapsed because its probability outputs concentrate below that cut "
        "point. This dissociation between ranking quality and threshold behaviour "
        "underscores why calibration and threshold selection, discussed later, "
        "matter as much as the choice of algorithm.",
        "The receiver-operating and precision-recall curves for the selected "
        "model, shown in Figures 7 and 8, confirm strong discrimination well above "
        "the prevalence baseline. The calibration curve in Figure 9 indicates that "
        "predicted probabilities are reasonable but imperfect, motivating an "
        "explicit calibration step before any operational use. The confusion "
        "matrix in Figure 10 makes the screening trade-off concrete: the model "
        "accepts a considerable number of false positives in order to capture the "
        "large majority of true cases, which is the appropriate posture for a "
        "triage tool whose principal cost is a missed case.",
    ), [
        (FIG / "08_roc_curve.png",
         "Figure 7. Receiver-operating-characteristic curve for the selected model."),
        (FIG / "09_pr_curve.png",
         "Figure 8. Precision-recall curve for the selected model."),
        (FIG / "10_calibration.png",
         "Figure 9. Reliability (calibration) curve for the selected model."),
        (FIG / "11_confusion_matrix.png",
         "Figure 10. Confusion matrix at the default 0.50 threshold."),
    ], [MODEL_TABLE]),

    (1, "Fairness and Subgroup Analysis", _p(
        "Because age, sex, and income are sensitive attributes, model behaviour "
        "was disaggregated across demographic strata on the held-out cohort, as "
        "reported in Table 2 and Figure 11. Discrimination, measured by the "
        "receiver-operating area, was broadly comparable across groups, but "
        "sensitivity diverged sharply. Recall was higher for male respondents than "
        "for female respondents and far higher for older adults than for younger "
        "adults. The most striking gap concerns adults aged eighteen to "
        "thirty-nine, among whom prevalence is only about one percent and the "
        "model rarely raises a positive flag.",
        "These disparities are not incidental; they follow directly from the "
        "underlying prevalence structure and from a single global threshold "
        "applied to groups with very different base rates. The practical "
        "implication is unambiguous: the instrument is dependable for older, "
        "higher-prevalence populations but should never be used to reassure or "
        "rule out risk in younger adults. A responsible deployment would monitor "
        "these subgroup gaps over time and consider group-specific operating "
        "thresholds to equalise sensitivity where policy demands it.",
    ), [
        (FIG / "12_subgroup_performance.png",
         "Figure 11. Recall and precision-recall area by subgroup."),
    ], [SUBGROUP_TABLE]),

    (1, "Deployment, Monitoring, and Lifecycle Considerations", _p(
        "The intended serving pattern is batch scoring rather than real-time "
        "inference, because population screening is naturally periodic and is "
        "applied to extracts rather than to individual live requests. The fitted "
        "artifact bundles preprocessing and the classifier into a single "
        "serialised pipeline, guaranteeing that training-time and inference-time "
        "transformations remain identical, and it is light enough to run on a "
        "modest processor without specialised hardware.",
        "A production instantiation would monitor three distinct surfaces. Model "
        "quality would be tracked through recall, precision-recall area, and "
        "calibration on freshly labelled data, with alerts if sensitivity falls "
        "below an agreed floor. Data health would be surveilled by comparing "
        "incoming feature distributions and missingness against the training "
        "profile, and fairness would be re-audited each cycle by recomputing "
        "subgroup recall. Continuous-integration checkpoints would validate the "
        "data schema, enforce the leakage contract, and gate any promotion on a "
        "model's ability to match or exceed the incumbent on a fixed holdout.",
    ), [], []),

    (1, "Limitations", _p(
        "Several limitations temper the conclusions. The data are self-reported "
        "and cross-sectional, so the target reflects diagnosed and remembered "
        "disease rather than true physiological status, and undiagnosed cases are "
        "necessarily mislabelled as negatives. The deliberate exclusion of "
        "hypertension and cholesterol, though necessary for cross-year "
        "consistency, removes two clinically potent predictors. The survey's "
        "complex sampling and weighting were not incorporated into the modelling, "
        "and the findings pertain to a United States adult population and should "
        "not be assumed to transfer to other health systems. Finally, the model "
        "is an academic screening exercise and has undergone none of the "
        "validation that a regulated clinical instrument would require.",
    ), [], []),

    (1, "Conclusion", _p(
        "This work demonstrates that a disciplined machine-learning-operations "
        "pipeline built on public survey data can rank adults by cardiovascular "
        "risk with useful accuracy and, importantly, can do so on a genuinely "
        "future cohort. The extreme gradient-boosting model was selected on the "
        "strength of its precision-recall performance, with an interpretable "
        "logistic-regression model retained as a transparent alternative. Beyond "
        "the headline metrics, the study contributes a reproducible, "
        "leakage-controlled workflow, an explicit treatment of cross-year "
        "harmonisation, and a candid subgroup audit. The resulting system is best "
        "understood not as a verdict on any individual but as an instrument for "
        "directing scarce screening capacity, whose probabilistic outputs and "
        "documented disparities must accompany it into any responsible use.",
    ), [], []),
]

REFERENCES = [
    "Centers for Disease Control and Prevention. (2022). Behavioral Risk Factor "
    "Surveillance System survey data, 2021. U.S. Department of Health and Human "
    "Services.",
    "Centers for Disease Control and Prevention. (2023). Behavioral Risk Factor "
    "Surveillance System survey data, 2022. U.S. Department of Health and Human "
    "Services.",
    "Centers for Disease Control and Prevention. (2024). Behavioral Risk Factor "
    "Surveillance System survey data, 2023. U.S. Department of Health and Human "
    "Services.",
    "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. "
    "In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge "
    "Discovery and Data Mining (pp. 785-794). Association for Computing "
    "Machinery.",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, "
    "O., . . . Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. "
    "Journal of Machine Learning Research, 12, 2825-2830.",
    "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more "
    "informative than the ROC plot when evaluating binary classifiers on "
    "imbalanced datasets. PLOS ONE, 10(3), e0118432.",
]
