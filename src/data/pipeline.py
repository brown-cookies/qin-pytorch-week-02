import pandas as pd

from src.data import clean, download, features, transform


def load_data() -> pd.DataFrame:
    csv_path = download.get_dataset_path()

    df = pd.read_csv(
        csv_path,
        na_values="?",
        keep_default_na=False,
    )

    return df


def prepare_data() -> pd.DataFrame:
    df = load_data()

    df = transform.add_target(df)

    df = clean.fill_missing(df)
    df = clean.keep_columns(df, features.get_features())

    df = transform.group_diagnoses(df)
    df = transform.encode_binary(df)
    df = transform.encode_age(df)
    df = transform.one_hot_encode(df)

    return df.reset_index(drop=True)
