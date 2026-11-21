"""Build the DuckDB database from the raw dataset CSVs."""

import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "arnabbiswas1/microsoft-azure-predictive-maintenance"
)

TABLE_FILES = {
    "machines": "PdM_machines.csv",
    "telemetry": "PdM_telemetry.csv",
    "errors": "PdM_errors.csv",
    "maint": "PdM_maint.csv",
    "failures": "PdM_failures.csv",
}


def data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", "data"))


def db_path() -> Path:
    return data_dir() / "fablens.duckdb"


def missing_files(raw_dir: Path) -> list[str]:
    return [name for name in TABLE_FILES.values() if not (raw_dir / name).is_file()]


def download_dataset(raw_dir: Path, url: str = DATASET_URL) -> None:
    """Download the dataset archive and extract the CSVs into raw_dir."""
    # Kaggle answers 404 to the default Python-urllib User-Agent.
    request = urllib.request.Request(url, headers={"User-Agent": "fablens"})
    with tempfile.TemporaryFile() as archive:
        with urllib.request.urlopen(request, timeout=60) as response:
            shutil.copyfileobj(response, archive)
        with zipfile.ZipFile(archive) as zf:
            members = {Path(member).name: member for member in zf.namelist()}
            absent = [name for name in TABLE_FILES.values() if name not in members]
            if absent:
                raise FileNotFoundError(f"Archive does not contain: {', '.join(absent)}")
            raw_dir.mkdir(parents=True, exist_ok=True)
            for name in TABLE_FILES.values():
                (raw_dir / name).write_bytes(zf.read(members[name]))


def build_database(raw_dir: Path, target: Path) -> dict[str, int]:
    """Create one table per CSV in a fresh database file and return the row counts."""
    missing = missing_files(raw_dir)
    if missing:
        raise FileNotFoundError(f"Missing dataset files in {raw_dir}: {', '.join(missing)}")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    row_counts = {}
    with duckdb.connect(target) as con:
        for table, file_name in TABLE_FILES.items():
            con.execute(
                f"CREATE TABLE {table} AS SELECT * FROM read_csv(?)", [str(raw_dir / file_name)]
            )
            row_counts[table] = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    return row_counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raw_dir = data_dir() / "raw"
    if missing_files(raw_dir):
        logger.info("Downloading the dataset to %s", raw_dir)
        try:
            download_dataset(raw_dir)
        except (OSError, zipfile.BadZipFile) as exc:
            logger.error("Dataset download failed: %s. See the README for manual download.", exc)
            raise SystemExit(1) from None
    try:
        row_counts = build_database(raw_dir, db_path())
    except FileNotFoundError as exc:
        logger.error("%s. Download the dataset as described in the README.", exc)
        raise SystemExit(1) from None
    for table, rows in row_counts.items():
        logger.info("%s: %d rows", table, rows)
    logger.info("Database written to %s", db_path())


if __name__ == "__main__":
    main()
