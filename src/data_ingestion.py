"""Stage 1 - Data ingestion: download the UCI dataset and validate its schema."""
import zipfile

import requests

from src.utils import get_logger, load_params, resolve

log = get_logger("ingestion")

EXPECTED_COLUMNS = [
    "Date", "Time", "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]


def download(params: dict) -> None:
    zip_path = resolve(params["data"]["raw_zip"])
    txt_path = resolve(params["data"]["raw_txt"])
    if txt_path.exists():
        log.info("Raw data already present at %s - skipping download", txt_path)
        return
    if not zip_path.exists():
        log.info("Downloading dataset from %s", params["data"]["url"])
        with requests.get(params["data"]["url"], stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(txt_path.parent)
    log.info("Extracted to %s", txt_path)


def validate_schema(params: dict) -> None:
    txt_path = resolve(params["data"]["raw_txt"])
    with open(txt_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(";")
    if header != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected schema: {header}")
    log.info("Schema validation passed (%d columns)", len(header))


def main() -> None:
    params = load_params()
    download(params)
    validate_schema(params)


if __name__ == "__main__":
    main()
