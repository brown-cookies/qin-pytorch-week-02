import pandas as pd

COLUMNS_WITH_MISSING = ["weight", "medical_specialty", "payer_code", "race"]


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace missing values with the category 'Unknown'."""

    out = df.copy()
    cols = [c for c in COLUMNS_WITH_MISSING if c in out.columns]
    out[cols] = out[cols].fillna("Unknown")

    return out


def keep_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Keep only the listed columns. Every other column is dropped."""

    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise KeyError(f"Not in the DataFrame (typo?): {missing}")

    return df[columns].copy()
