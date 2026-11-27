"""Dataset loading, feature definitions and preprocessing for the credit-risk model.

Owner: Member 1 (ML & Dataset).

Dataset: Statlog (German Credit Data), Prof. Dr. Hans Hofmann, UCI Machine
Learning Repository, CC BY 4.0. https://doi.org/10.24432/C5NC77

Everything that defines *what the model sees* lives in this file:
  * the raw column names of german.data,
  * the human-readable value for every symbolic code (A11, A12, ...),
  * which columns are model inputs and which are excluded (and why),
  * the hard validation bounds used at inference time,
  * the scikit-learn ColumnTransformer that turns a profile into numbers.

train.py and tools/ml_tool.py both import from here, so training and
inference can never drift apart silently.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ML_DIR = Path(__file__).resolve().parent
DATA_DIR = ML_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "german.data.csv"

# Mirror of the original UCI file "german.data" (comma-separated copy).
# Used only by download_dataset() if the committed copy is missing.
DATA_MIRROR_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/german.csv"
RAW_DATA_SHA256 = "ec12a88b9fc14d74ba646ea0410cf7ff4533bec2eb61652f8ad76796bbfec017"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---------------------------------------------------------------------------
# Raw columns, in file order (20 attributes + class), per german.names
# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    "checking_account_status",   # A1  qualitative
    "duration_months",           # A2  numerical
    "credit_history",            # A3  qualitative
    "purpose",                   # A4  qualitative
    "credit_amount",             # A5  numerical (Deutsche Mark)
    "savings_account",           # A6  qualitative
    "employment_duration",       # A7  qualitative
    "installment_rate_pct",      # A8  numerical (coded 1-4)
    "personal_status_sex",       # A9  qualitative  -> EXCLUDED
    "other_debtors",             # A10 qualitative
    "present_residence_years",   # A11 numerical (coded 1-4) -> EXCLUDED
    "property",                  # A12 qualitative
    "age_years",                 # A13 numerical    -> EXCLUDED
    "other_installment_plans",   # A14 qualitative
    "housing",                   # A15 qualitative
    "existing_credits",          # A16 numerical (count at this bank)
    "job",                       # A17 qualitative
    "num_dependents",            # A18 numerical    -> EXCLUDED
    "telephone",                 # A19 qualitative  -> EXCLUDED
    "foreign_worker",            # A20 qualitative  -> EXCLUDED
    "credit_class",              # target: 1 = good, 2 = bad
]

TARGET_RAW = "credit_class"
TARGET = "high_risk"  # 1 if credit_class == 2 ("bad"), else 0

# Symbolic code -> (readable value used in the API, description from german.names)
CODE_MAPS: dict[str, dict[str, tuple[str, str]]] = {
    "checking_account_status": {
        "A11": ("negative_balance", "checking balance < 0 DM"),
        "A12": ("0_to_200_dm", "0 <= checking balance < 200 DM"),
        "A13": ("200_dm_or_more", "checking balance >= 200 DM / salary assignments for at least 1 year"),
        "A14": ("no_checking_account", "no checking account"),
    },
    "credit_history": {
        "A30": ("no_credits_or_all_paid", "no credits taken / all credits paid back duly"),
        "A31": ("all_paid_at_this_bank", "all credits at this bank paid back duly"),
        "A32": ("existing_paid_duly", "existing credits paid back duly till now"),
        "A33": ("past_payment_delay", "delay in paying off in the past"),
        "A34": ("critical_or_other_credits", "critical account / other credits existing (not at this bank)"),
    },
    "purpose": {
        "A40": ("car_new", "car (new)"),
        "A41": ("car_used", "car (used)"),
        "A42": ("furniture_equipment", "furniture / equipment"),
        "A43": ("radio_television", "radio / television"),
        "A44": ("domestic_appliances", "domestic appliances"),
        "A45": ("repairs", "repairs"),
        "A46": ("education", "education"),
        "A47": ("vacation", "vacation (documented, but never occurs in the data)"),
        "A48": ("retraining", "retraining"),
        "A49": ("business", "business"),
        "A410": ("other", "others"),
    },
    "savings_account": {
        "A61": ("lt_100_dm", "savings < 100 DM"),
        "A62": ("100_to_500_dm", "100 <= savings < 500 DM"),
        "A63": ("500_to_1000_dm", "500 <= savings < 1000 DM"),
        "A64": ("1000_dm_or_more", "savings >= 1000 DM"),
        "A65": ("unknown_or_none", "unknown / no savings account"),
    },
    "employment_duration": {
        "A71": ("unemployed", "unemployed"),
        "A72": ("lt_1_year", "employed < 1 year"),
        "A73": ("1_to_4_years", "1 <= employed < 4 years"),
        "A74": ("4_to_7_years", "4 <= employed < 7 years"),
        "A75": ("7_years_or_more", "employed >= 7 years"),
    },
    "personal_status_sex": {
        "A91": ("male_divorced_separated", "male: divorced/separated"),
        "A92": ("female_div_sep_married", "female: divorced/separated/married"),
        "A93": ("male_single", "male: single"),
        "A94": ("male_married_widowed", "male: married/widowed"),
        "A95": ("female_single", "female: single"),
    },
    "other_debtors": {
        "A101": ("none", "no other debtors / guarantors"),
        "A102": ("co_applicant", "co-applicant"),
        "A103": ("guarantor", "guarantor"),
    },
    "property": {
        "A121": ("real_estate", "real estate"),
        "A122": ("savings_agreement_or_life_insurance", "building society savings agreement / life insurance"),
        "A123": ("car_or_other", "car or other property"),
        "A124": ("unknown_or_none", "unknown / no property"),
    },
    "other_installment_plans": {
        "A141": ("bank", "other installment plan at a bank"),
        "A142": ("stores", "other installment plan at stores"),
        "A143": ("none", "no other installment plans"),
    },
    "housing": {
        "A151": ("rent", "rent"),
        "A152": ("own", "own"),
        "A153": ("for_free", "for free"),
    },
    "job": {
        "A171": ("unemployed_or_unskilled_nonresident", "unemployed / unskilled - non-resident"),
        "A172": ("unskilled_resident", "unskilled - resident"),
        "A173": ("skilled_employee", "skilled employee / official"),
        "A174": ("management_self_employed_highly_qualified", "management / self-employed / highly qualified employee"),
    },
    "telephone": {
        "A191": ("none", "no telephone"),
        "A192": ("yes_registered", "telephone registered under the customer's name"),
    },
    "foreign_worker": {
        "A201": ("yes", "foreign worker"),
        "A202": ("no", "not a foreign worker"),
    },
}

# ---------------------------------------------------------------------------
# Model inputs (the feature schema). Order here == column order fed to the pipeline.
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = [
    "checking_account_status",
    "credit_history",
    "purpose",
    "savings_account",
    "employment_duration",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
]

NUMERIC_FEATURES = [
    "duration_months",
    "credit_amount",
    "installment_rate_pct",
    "existing_credits",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

NUMERIC_FEATURE_INFO = {
    # name: (description, unit, hard_min, hard_max)
    # Hard bounds = values that are structurally valid. Outside -> rejected.
    # Values inside the hard bounds but outside the training range -> accepted with a warning.
    "duration_months": ("Requested loan duration", "months", 1, 120),
    "credit_amount": ("Requested credit amount", "Deutsche Mark (DM, 1970s values)", 1, 100_000),
    "installment_rate_pct": (
        "Installment rate as % of disposable income, coded 1-4 by the dataset "
        "(1 = lowest band, 4 = highest band)", "ordinal code", 1, 4),
    "existing_credits": ("Number of existing credits at this bank, coded 1-4 (4 = four or more)", "count", 1, 4),
}

EXCLUDED_FEATURES = {
    "personal_status_sex": "Protected attribute (sex) combined with marital status - never used as a model input.",
    "age_years": "Protected/demographic attribute (age) - never used as a model input.",
    "foreign_worker": "Protected attribute (national origin / migration status) - never used as a model input.",
    "num_dependents": "Household composition (family status proxy), not a financial feature - excluded.",
    "telephone": "Contact-information indicator, not a financial feature - excluded.",
    "present_residence_years": "Residential tenure, not a financial feature and a proxy for age - excluded.",
}

assert set(MODEL_FEATURES).isdisjoint(EXCLUDED_FEATURES)
assert set(MODEL_FEATURES) | set(EXCLUDED_FEATURES) | {TARGET_RAW} == set(RAW_COLUMNS)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_dataset(dest: Path = RAW_DATA_PATH) -> Path:
    """Re-download the raw file from the mirror and verify its checksum."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(DATA_MIRROR_URL, tmp)
    digest = sha256_of(tmp)
    if digest != RAW_DATA_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for downloaded dataset: {digest}")
    tmp.replace(dest)
    return dest


def load_raw(path: Path = RAW_DATA_PATH, verify_checksum: bool = True) -> pd.DataFrame:
    """Load german.data with named columns and symbolic codes left as-is."""
    if not path.exists():
        download_dataset(path)
    if verify_checksum and sha256_of(path) != RAW_DATA_SHA256:
        raise RuntimeError(
            f"{path} does not match the expected SHA-256; the dataset file was modified."
        )
    return pd.read_csv(path, header=None, names=RAW_COLUMNS)


def decode(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Replace symbolic codes with readable values and add the binary target."""
    df = df_raw.copy()
    for col, mapping in CODE_MAPS.items():
        unknown = set(df[col].unique()) - set(mapping)
        if unknown:
            raise ValueError(f"Unexpected codes in column {col}: {sorted(unknown)}")
        df[col] = df[col].map({code: value for code, (value, _) in mapping.items()})
    df[TARGET] = (df[TARGET_RAW] == 2).astype(int)
    return df


def load_dataset(path: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X, y, full_decoded_frame). X contains model features only."""
    df = decode(load_raw(path))
    X = df[MODEL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y, df


def allowed_values(df_decoded: pd.DataFrame) -> dict[str, list[str]]:
    """Categorical values that actually occur in the data (e.g. 'vacation' never does)."""
    return {c: sorted(df_decoded[c].unique().tolist()) for c in CATEGORICAL_FEATURES}


def build_preprocessor(categories: dict[str, list[str]]) -> ColumnTransformer:
    """One-hot encode categoricals with a FIXED category list; scale numerics.

    handle_unknown='error' is deliberate: an unseen category must never be
    silently encoded as all-zeros. ml_tool.py rejects unknown values before
    they ever reach the pipeline, so this is a second line of defence.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    categories=[categories[c] for c in CATEGORICAL_FEATURES],
                    handle_unknown="error",
                    drop=None,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
