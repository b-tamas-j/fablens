import zipfile
from pathlib import Path

import duckdb
import pytest

from fablens.ingest import TABLE_FILES, build_database, download_dataset, missing_files


def test_build_database_creates_one_table_per_csv(raw_dir: Path, tmp_path: Path) -> None:
    target = tmp_path / "out" / "test.duckdb"

    row_counts = build_database(raw_dir, target)

    assert row_counts == {
        "machines": 2,
        "telemetry": 3,
        "errors": 1,
        "maint": 2,
        "failures": 1,
    }
    with duckdb.connect(target, read_only=True) as con:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        columns = {name: dtype for name, dtype, *_ in con.execute("DESCRIBE failures").fetchall()}
    assert tables == set(TABLE_FILES)
    assert columns == {"datetime": "TIMESTAMP", "machineID": "BIGINT", "failure": "VARCHAR"}


def test_build_database_replaces_existing_file(raw_dir: Path, tmp_path: Path) -> None:
    target = tmp_path / "test.duckdb"
    build_database(raw_dir, target)
    (raw_dir / "PdM_machines.csv").write_text('"machineID","model","age"\n1,"model3",18\n')

    row_counts = build_database(raw_dir, target)

    assert row_counts["machines"] == 1


def test_build_database_reports_missing_files(raw_dir: Path, tmp_path: Path) -> None:
    (raw_dir / "PdM_maint.csv").unlink()

    with pytest.raises(FileNotFoundError, match="PdM_maint.csv"):
        build_database(raw_dir, tmp_path / "test.duckdb")


def make_archive(path: Path, files: dict[str, str]) -> str:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return path.as_uri()


def test_download_dataset_extracts_csvs(tmp_path: Path, sample_csvs: dict[str, str]) -> None:
    url = make_archive(tmp_path / "archive.zip", sample_csvs)
    raw = tmp_path / "raw"

    download_dataset(raw, url)

    assert missing_files(raw) == []
    assert (raw / "PdM_failures.csv").read_text() == sample_csvs["PdM_failures.csv"]


def test_download_dataset_reports_incomplete_archive(
    tmp_path: Path, sample_csvs: dict[str, str]
) -> None:
    del sample_csvs["PdM_errors.csv"]
    url = make_archive(tmp_path / "archive.zip", sample_csvs)

    with pytest.raises(FileNotFoundError, match="PdM_errors.csv"):
        download_dataset(tmp_path / "raw", url)
