"""Stage 1 - Data ingestion: get the UCI dataset as a clean CSV and validate its schema.

The repository ships the dataset as data/household_power_consumption.csv.gz (18.6 MB,
GitHub does not accept the 115 MB CSV); it is decompressed on first run. Without it,
the original archive is downloaded from UCI. The UCI archive holds a semicolon-separated .txt with '?' for missing readings and
separate Date/Time columns. It is converted to a standard CSV with one Datetime
column and blank cells for missing values; the downloaded archive is not kept.
"""
import gzip
import io
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.utils import get_logger, load_params, resolve

log = get_logger("ingestion")

EXPECTED_COLUMNS = [
    "Datetime", "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]


def download(params: dict) -> None:
    csv_path = resolve(params["data"]["raw_csv"])
    if csv_path.exists():
        log.info("Dataset already present at %s - skipping download", csv_path)
        return
    bundled = csv_path.with_name(csv_path.name + ".gz")
    if bundled.exists():
        with gzip.open(bundled, "rb") as src, open(csv_path, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        log.info("Decompressed bundled dataset %s -> %s", bundled.name, csv_path)
        return
    log.info("Downloading dataset from %s", params["data"]["url"])
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "dataset.zip"
        with requests.get(params["data"]["url"], stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        with zipfile.ZipFile(zip_path) as z:
            name = next(n for n in z.namelist() if n.endswith(".txt"))
            raw = pd.read_csv(io.TextIOWrapper(z.open(name), encoding="utf-8"),
                              sep=";", na_values=["?"], low_memory=False)
    raw.insert(0, "Datetime", pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S"))
    raw.drop(columns=["Date", "Time"]).to_csv(csv_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    log.info("Saved %d rows to %s", len(raw), csv_path)


def validate_schema(params: dict) -> None:
    with open(resolve(params["data"]["raw_csv"]), "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    if header != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected schema: {header}")
    log.info("Schema validation passed (%d columns)", len(header))


def main() -> None:
    params = load_params()
    download(params)
    validate_schema(params)


if __name__ == "__main__":
    main()
