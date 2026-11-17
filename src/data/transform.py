import pandas as pd

from src.data.features import RAW_LABEL, TARGET

DIAG_COLS = ["diag_1", "diag_2", "diag_3"]

ID_COLS = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

BINARY_COLS = [
    "gender",
    "tolbutamide",
    "glipizide-metformin",
    "troglitazone",
    "acetohexamide",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "diabetesMed",
    "change",
    "metformin-pioglitazone",
]

BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1, "Unknown/Invalid": None},
    "tolbutamide": {"No": 0, "Steady": 1},
    "glipizide-metformin": {"No": 0, "Steady": 1},
    "troglitazone": {"No": 0, "Steady": 1},
    "acetohexamide": {"No": 0, "Steady": 1},
    "glimepiride-pioglitazone": {"No": 0, "Steady": 1},
    "metformin-rosiglitazone": {"No": 0, "Steady": 1},
    "diabetesMed": {"No": 0, "Yes": 1},
    "change": {"No": 0, "Ch": 1},
    "metformin-pioglitazone": {"No": 0, "Steady": 1},
}

ONE_HOT_COLS = [
    "readmitted",
    "tolazamide",
    "nateglinide",
    "glimepiride",
    "miglitol",
    "acarbose",
    "rosiglitazone",
    "pioglitazone",
    "glyburide",
    "glipizide",
    "glyburide-metformin",
    "insulin",
    "repaglinide",
    "metformin",
    "A1Cresult",
    "chlorpropamide",
    "max_glu_serum",
    "race",
]


def icd9_group(code: str | float) -> str:
    """Group ~700 diagnosis codes into about 9 categories."""
    if pd.isna(code):
        return "Unknown"

    code = str(code)
    if code[0] in "VE":
        return "Other"

    n = float(code)
    if int(n) == 250:
        return "Diabetes"
    if 390 <= n < 460 or int(n) == 785:
        return "Circulatory"
    if 460 <= n < 520 or int(n) == 786:
        return "Respiratory"
    if 520 <= n < 580 or int(n) == 787:
        return "Digestive"
    if 580 <= n < 630 or int(n) == 788:
        return "Genitourinary"
    if 710 <= n < 740:
        return "Musculoskeletal"
    if 800 <= n < 1000:
        return "Injury"
    if 140 <= n < 240:
        return "Neoplasms"
    return "Other"


def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Two-value text columns -> 0/1, using BINARY_COLS and BINARY_MAP."""
    out = df.copy()
    for col in BINARY_COLS:
        out[col] = out[col].map(BINARY_MAP[col])
    return out


def encode_age(df: pd.DataFrame) -> pd.DataFrame:
    """Age bracket -> middle number, e.g. "[70-80)" -> 75."""
    out = df.copy()
    out["age"] = out["age"].str.extract(r"\[(\d+)")[0].astype(int) + 5
    return out


def group_diagnoses(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in DIAG_COLS:
        out[col] = out[col].map(icd9_group)
    return out


def ids_to_str(df: pd.DataFrame) -> pd.DataFrame:
    """ID codes are labels, not quantities, so treat them as text."""
    out = df.copy()
    out[ID_COLS] = out[ID_COLS].astype(str)
    return out


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """1 if readmitted within 30 days, otherwise 0."""
    out = df.copy()
    out[TARGET] = (out[RAW_LABEL] == "<30").astype("int64")
    return out
