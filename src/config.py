"""
Central configuration for the BRFSS heart-disease risk-screening project.

Defines paths, the target variable, columns that leak the target, and the
harmonised feature set that is consistently available across the 2021, 2022
and 2023 BRFSS survey years.

All variable names follow the CDC BRFSS codebook. See:
https://www.cdc.gov/brfss/annual_data/annual_data.htm
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

for _d in (PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# XPT files must have a trailing-space-free name (LLCP2021.XPT etc.)
XPT_FILES = {
    2021: RAW_DIR / "LLCP2021.XPT",
    2022: RAW_DIR / "LLCP2022.XPT",
    2023: RAW_DIR / "LLCP2023.XPT",
}

# Encoding required for BRFSS XPT variable labels (contain latin-1 bytes).
XPT_ENCODING = "latin1"

# Temporal split: develop on 2021+2022, hold 2023 out as a temporal test set.
TRAIN_YEARS = [2021, 2022]
TEST_YEARS = [2023]

RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Target
# --------------------------------------------------------------------------- #
# _MICHD is the CDC-calculated variable: respondents ever told they had a
# myocardial infarction (heart attack) OR coronary heart disease/angina.
#   raw 1 -> reported CHD/MI      (positive class -> 1)
#   raw 2 -> did not report       (negative class -> 0)
TARGET_RAW = "_MICHD"
TARGET = "heart_disease"

# --------------------------------------------------------------------------- #
# Leakage columns — define / directly build the target. Never use as features.
# --------------------------------------------------------------------------- #
LEAKAGE_COLS = ["_MICHD", "CVDINFR4", "CVDCRHD4"]

# --------------------------------------------------------------------------- #
# Harmonised feature set (consistently available across 2021, 2022, 2023).
#
# NOTE ON HARMONISATION: BRFSS collects blood-pressure (BPHIGH*) and
# cholesterol (TOLDHI*/_RFCHOL*) data only in ODD years. They are ABSENT from
# the 2022 core survey, so — to keep a single feature schema valid for all
# three years — hypertension and high-cholesterol are intentionally excluded
# from the common feature set. This trade-off is documented in the design doc.
# --------------------------------------------------------------------------- #

# Raw BRFSS name -> clean feature name.  Only variables present in all 3 years.
FEATURE_RENAME = {
    "_AGEG5YR": "age_group",          # 14-level age category (ordinal)
    "_BMI5": "bmi",                   # BMI * 100 (continuous)
    "SEXVAR": "sex",                  # 1 male, 2 female
    "GENHLTH": "gen_health",          # 1 excellent .. 5 poor (ordinal)
    "SMOKE100": "smoked_100",         # smoked >=100 cigarettes lifetime
    "_SMOKER3": "smoker_status",      # 1 daily .. 4 never (ordinal)
    "DIABETE4": "diabetes",           # diabetes status
    "DIFFWALK": "diff_walk",          # difficulty walking/climbing stairs
    "_TOTINDA": "phys_activity",      # any physical activity past 30 days
    "CHCKDNY2": "kidney_disease",     # ever told kidney disease
    "_INCOMG1": "income_group",       # income category (ordinal)
    "_EDUCAG": "education_group",     # education category (ordinal)
    "PERSDOC3": "has_personal_doctor",
    "MENTHLTH": "mental_health_days",  # days poor mental health (0-30)
    "PHYSHLTH": "phys_health_days",    # days poor physical health (0-30)
}

# Feature roles after cleaning ------------------------------------------------
CONTINUOUS_FEATURES = ["bmi", "mental_health_days", "phys_health_days"]

# Ordinal / categorical survey codes (treated as categorical for one-hot).
CATEGORICAL_FEATURES = [
    "age_group",
    "sex",
    "gen_health",
    "smoked_100",
    "smoker_status",
    "diabetes",
    "diff_walk",
    "phys_activity",
    "kidney_disease",
    "income_group",
    "education_group",
    "has_personal_doctor",
]

ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

# Sensitive attributes used ONLY for fairness/subgroup analysis (kept in the
# processed frame but analysts decide whether to model on them).
SENSITIVE_FEATURES = ["sex", "age_group", "income_group"]

# --------------------------------------------------------------------------- #
# Survey "missing"-style codes to convert to NaN, per variable.
# 7/9 = don't know / refused for most 1-2 coded items; 77/99 for 0-30 day items;
# BLANK/. handled automatically by pyreadstat as NaN.
# --------------------------------------------------------------------------- #
# Codes meaning "don't know / refused / not asked" that must become NaN.
DK_REFUSED_CODES = {
    "gen_health": [7, 9],
    "smoked_100": [7, 9],
    "diabetes": [7, 9],          # (kept: 1 yes,2 yes-preg,3 no,4 pre; 7/9 -> NaN)
    "diff_walk": [7, 9],
    "phys_activity": [9],
    "kidney_disease": [7, 9],
    "has_personal_doctor": [7, 9],
    "smoker_status": [9],
    "age_group": [14],           # 14 = don't know/refused/missing age
    "income_group": [9],
    "education_group": [9],
    "mental_health_days": [77, 99],
    "phys_health_days": [77, 99],
}
# For day-count items, code 88 means "none" -> should map to 0.
ZERO_CODES = {
    "mental_health_days": [88],
    "phys_health_days": [88],
}
