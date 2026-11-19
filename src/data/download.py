import hashlib
import shutil
import zipfile
from pathlib import Path

import requests


DATASET_URL = "https://github.com/brown-cookies/qin-pytorch-week-02/releases/download/0.1.0/diabetes-130-us-hospitals-for-years-1999-2008.zip"
DATASET_SHA256 = "f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a"
DATASET_DOWNLOAD_PATH = Path("datasets")

ZIP_PATH = DATASET_DOWNLOAD_PATH / "diabetes-130-us-hospitals.zip"
REQUIRED_FILES = ["diabetic_data.csv", "IDS_mapping.csv"]
CSV_PATH = DATASET_DOWNLOAD_PATH / "diabetic_data.csv"


def _check_if_exist():
    return all((DATASET_DOWNLOAD_PATH / name).exists() for name in REQUIRED_FILES)


def _sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return h.hexdigest()


def _download():
    if not DATASET_SHA256:

        raise RuntimeError(
            "DATASET_SHA256 is empty, fill it in before downloading"
        )

    DATASET_DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
    tmp = ZIP_PATH.with_suffix(".part")

    with requests.get(DATASET_URL, stream=True, timeout=60) as r:
        r.raise_for_status()

        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    actual = _sha256(tmp)

    if actual != DATASET_SHA256:
        tmp.unlink()
        raise RuntimeError(
            f"Checksum mismatch.\nexpected: {DATASET_SHA256}\nactual:   {actual}"
        )

    tmp.replace(ZIP_PATH)


def _extract():
    with zipfile.ZipFile(ZIP_PATH) as z:

        for member in z.namelist():
            name = Path(member).name

            if name in REQUIRED_FILES:

                with z.open(member) as src, open(DATASET_DOWNLOAD_PATH / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    ZIP_PATH.unlink()

    if not _check_if_exist():
        raise RuntimeError(f"Zip did not contain all of {REQUIRED_FILES}")


def get_dataset_path():
    """Return the path to diabetic_data.csv, downloading it first if needed."""
    if not _check_if_exist():
        _download()
        _extract()
    return CSV_PATH
