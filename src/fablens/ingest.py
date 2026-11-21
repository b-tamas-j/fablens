"""Build the DuckDB database from the raw dataset CSVs."""

import logging
import os
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

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


def build_database(raw_dir: Path, target: Path) -> dict[str, int]:
    """Create one table per CSV in a fresh database file and return the row counts."""
    missing = [name for name in TABLE_FILES.values() if not (raw_dir / name).is_file()]
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
