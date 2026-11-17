import pandas as pd

from src.data.download import get_dataset_path


def load_data() -> pd.DataFrame:
    csv_path = get_dataset_path()

    df = pd.read_csv(
        csv_path,
        na_values="?",
        keep_default_na=False,
    )

    return df
