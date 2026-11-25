import pandas as pd
import pytest

from src.config import Config, get_config
from src.data.pipeline import prepare_data


@pytest.fixture()
def df() -> pd.DataFrame:
    return prepare_data()


@pytest.fixture()
def cfg() -> Config:
    return get_config()
