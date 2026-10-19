"""
Populate the AAI-540 Weekly Teamwork Tracker for Group 2.

Standardises team-member names, fills the per-module Individual Contribution
tables (Modules 2-6) with work mapped to the actual project, and answers the
weekly teamwork-communication summary prompts. Edits are done cell-by-cell so
the template layout and styling are preserved.

Run:  python update_tracker.py
"""
from pathlib import Path
from docx import Document

SRC = Path("/home/ekattim/.kiro/USD_Project/AAI-540 Weekly Teamwork Tracker.docx")
OUT = SRC  # update in place

M1, M2, M3 = "Ashis Das", "Ram Prasanth G", "Hanmant Mhalappa Kattimani"


def set_cell(cell, text):
    """Replace a cell's text with `text`, keeping a single clean paragraph."""
    # Clear all paragraphs but the first.
    for p in cell.paragraphs[1:]:
        p._element.getparent().remove(p._element)
    p = cell.paragraphs[0]
    for r in list(p.runs):
        r.text = ""
    lines = text.split("\n")
    if p.runs:
        p.runs[0].text = lines[0]
    else:
        p.add_run(lines[0])
    for line in lines[1:]:
        cell.add_paragraph(line)


def header(role_no, name):
    return f"Team Member {role_no}\n{name}"


# Per-module (table index) content: list of 3 cells for contributions row.
CONTRIB = {
    # Table 2 = Module 2 (Data Engineering) — refine existing + fix name.
    2: [
        "List of contributions\n\nML project identification (heart-disease "
        "screening)\nDefined problem scope and target (_MICHD)\nBRFSS 2021/2022/"
        "2023 dataset identification and download\nRaw-data storage (S3 datalake "
        "layout) and Asana setup\nDrafted Design Document scope",
        "List of contributions\n\nML project identification\nDataset "
        "identification and volume verification\nData risk/bias review\nTeam "
        "Tracker setup and Design Document data-source section",
        "List of contributions\n\nML project identification\nGitHub repository "
        "setup\nInitial exploratory analysis of BRFSS variables\nCross-year "
        "variable harmonisation review",
    ],
    # Table 3 = Module 3 (Training Data & Feature Engineering).
    3: [
        "List of contributions\n\nBuilt reproducible ingestion/harmonisation "
        "pipeline (XPT to parquet)\nDerived target and enforced leakage exclusion "
        "(CVDINFR4, CVDCRHD4, _MICHD)\nDesigned feature groups and 15-feature "
        "schema\nTemporal train/validation/test design (2021+2022 vs 2023)",
        "List of contributions\n\nExploratory data analysis (prevalence, class "
        "balance, missingness)\nFeature-store/feature-group definition\n"
        "Correlation and risk-factor association analysis\nDocumented the "
        "hypertension/cholesterol cross-year harmonisation limitation",
        "List of contributions\n\nFeature engineering (imputation, scaling, "
        "one-hot encoding via ColumnTransformer)\nData-quality checks and "
        "don't-know/refused recoding\nGenerated EDA figures\nUnit tests for the "
        "data contract and leakage control",
    ],
    # Table 4 = Module 4 (Model Development & Deployment).
    4: [
        "List of contributions\n\nBuilt majority-class benchmark and logistic-"
        "regression baseline\nTrained and debugged the model pipeline in "
        "notebook/SageMaker\nSaved the fitted pipeline artifact for batch "
        "inference\nLinked codebase in the Design Document",
        "List of contributions\n\nTrained decision tree, random forest and "
        "gradient-boosting challengers\nApplied class-weighting for imbalance\n"
        "Compiled the model-comparison table\nDrafted the deployment (batch vs "
        "real-time) section",
        "List of contributions\n\nTrained and tuned the XGBoost model (selected "
        "model)\nEvaluated on the 2023 temporal test set (ROC-AUC, PR-AUC, "
        "recall)\nProduced ROC, PR, calibration and confusion-matrix figures\n"
        "Model-selection justification",
    ],
    # Table 5 = Module 5 (Monitoring).
    5: [
        "List of contributions\n\nDesigned model-quality monitors (recall, "
        "PR-AUC, calibration thresholds)\nDefined alerting floors for sensitivity "
        "drops\nDrafted the CloudWatch monitoring-dashboard plan",
        "List of contributions\n\nDesigned data monitors (feature-distribution "
        "and missingness drift, PSI)\nSpecified data-quality reports\n"
        "Documented the monitoring section of the Design Document",
        "List of contributions\n\nDesigned infrastructure monitors (batch-job "
        "success, runtime, errors)\nImplemented fairness monitoring (subgroup "
        "recall by sex and age)\nGenerated the subgroup-metrics report",
    ],
    # Table 6 = Module 6 (CI/CD).
    6: [
        "List of contributions\n\nDesigned the CI/CD pipeline stages (schema "
        "validation to deployment)\nDefined the evaluation gate that blocks "
        "promotion below the incumbent\nAutomated model retraining/evaluation "
        "flow",
        "List of contributions\n\nAuthored CI/CD checkpoints and data-contract "
        "tests\nWired unit tests for ingestion, harmonisation and leakage\n"
        "Documented the CI/CD section",
        "List of contributions\n\nImproved the model and re-ran it through the "
        "pipeline\nAssembled final deliverables (notebook, APA report, design "
        "document)\nPrepared the GitHub codebase and final submission package",
    ],
}

# Comments/Roadblocks per module (row 2), one per member — kept concise.
ROADBLOCKS = {
    2: ["No blockers; agreed on scope quickly.",
        "BRFSS files are large; managed via streaming download.",
        "Resolved trailing-space issue in raw XPT column names."],
    3: ["No blockers.",
        "Blood pressure/cholesterol absent in 2022; excluded for consistency.",
        "High missingness in income/BMI; handled via training-only imputation."],
    4: ["No blockers.",
        "Class imbalance addressed with class weighting.",
        "Gradient boosting under-recalled at 0.5 threshold; noted for calibration."],
    5: ["No blockers.",
        "Monitoring designed conceptually within the academic scope.",
        "Subgroup recall gaps (younger adults) flagged for ongoing monitoring."],
    6: ["No blockers.",
        "Evaluation-gate thresholds tuned to avoid false regressions.",
        "Final packaging completed on schedule."],
}

NAMES = {2: [M1, M2, M3], 3: [M1, M2, M3], 4: [M1, M2, M3],
         5: [M1, M2, M3], 6: [M1, M2, M3]}


def main():
    doc = Document(str(SRC))

    for ti in (2, 3, 4, 5, 6):
        t = doc.tables[ti]
        # Header row: names
        for ci, name in enumerate(NAMES[ti]):
            set_cell(t.rows[0].cells[ci], header(ci + 1, name))
        # Contributions row
        for ci, text in enumerate(CONTRIB[ti]):
            set_cell(t.rows[1].cells[ci], text)
        # Roadblocks row
        for ci, rb in enumerate(ROADBLOCKS[ti]):
            set_cell(t.rows[2].cells[ci], "Comments/ Roadblocks\n" + rb)

    # Answer the weekly teamwork-communication summary prompts (paras 16,26,34,42,50).
    summaries = {
        16: "Our team met and collaborated approximately 4-5 times this week "
            "(video calls plus asynchronous updates on Asana, GitHub and chat) "
            "to agree on the problem, identify the BRFSS datasets and set up the "
            "project workspace.",
        26: "The team collaborated about 4 times this week on exploratory "
            "analysis, feature engineering and the train/validation/test design.",
        34: "The team collaborated about 4-5 times this week on baseline and "
            "advanced model development, evaluation and deployment design.",
        42: "The team collaborated about 3-4 times this week on the monitoring "
            "design (model, data, infrastructure and fairness monitors).",
        50: "The team collaborated about 4 times this week on the CI/CD pipeline "
            "design and final deliverable preparation.",
    }
    for idx, ans in summaries.items():
        if idx < len(doc.paragraphs):
            prompt = doc.paragraphs[idx]
            from docx.oxml.ns import qn
            from docx.text.paragraph import Paragraph
            el = prompt._p.makeelement(qn("w:p"), {})
            prompt._p.addnext(el)
            newp = Paragraph(el, prompt._parent)
            run = newp.add_run("Team response: " + ans)
            run.italic = True

    doc.save(str(OUT))
    print("Updated", OUT)


if __name__ == "__main__":
    main()
