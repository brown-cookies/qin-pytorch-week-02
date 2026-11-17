from typing import Final

RAW_LABEL: Final[str] = "readmitted"
TARGET: Final[str] = "target"
PATIENT_ID: Final[str] = "patient_nbr"

KEEP: Final[list[str]] = [PATIENT_ID, TARGET]

FEATURES: list[str] = [
    # --- patient

    "race",
    "gender",
    "age",
    "weight",

    # --- admission

    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "payer_code",
    "medical_specialty",

    # --- hospital counts

    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",

    # --- diagnoses

    "diag_1",
    "diag_2",
    "diag_3",
    "number_diagnoses",

    # --- lab tests

    "max_glu_serum",
    "A1Cresult",

    # --- medications

    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",

    # --- treatment

    "change",
    "diabetesMed",
]


def get_features() -> list[str]:
    """
    Return the columns to keep: KEEP, in a fixed order.
    Every other column is dropped.
    """

    if len(FEATURES) != len(set(FEATURES)):
        raise ValueError("Duplicate names in the feature list")

    return FEATURES
