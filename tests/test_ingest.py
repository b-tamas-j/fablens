import zipfile
from pathlib import Path

import duckdb
import pytest

from fablens.ingest import TABLE_FILES, build_database, download_dataset, missing_files

SAMPLE_CSVS = {
    "PdM_machines.csv": '"machineID","model","age"\n1,"model3",18\n2,"model4",7\n',
    "PdM_telemetry.csv": (
        '"datetime","machineID","volt","rotate","pressure","vibration"\n'
        "2015-01-01 06:00:00,1,176.2,418.5,113.1,45.1\n"
        "2015-01-01 07:00:00,1,162.9,402.7,95.5,43.4\n"
        "2015-01-01 06:00:00,2,170.0,450.0,100.0,40.0\n"
    ),
    "PdM_errors.csv": '"datetime","machineID","errorID"\n2015-01-03 07:00:00,1,"error1"\n',
    "PdM_maint.csv": (
        '"datetime","machineID","comp"\n'
        '2014-06-01 06:00:00,1,"comp2"\n'
        '2015-01-05 06:00:00,1,"comp4"\n'
    ),
    "PdM_failures.csv": '"datetime","machineID","failure"\n2015-01-05 06:00:00,1,"comp4"\n',
}


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    for name, content in SAMPLE_CSVS.items():
        (raw / name).write_text(content)
    return raw


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


def test_download_dataset_extracts_csvs(tmp_path: Path) -> None:
    url = make_archive(tmp_path / "archive.zip", SAMPLE_CSVS)
    raw = tmp_path / "raw"

    download_dataset(raw, url)

    assert missing_files(raw) == []
    assert (raw / "PdM_failures.csv").read_text() == SAMPLE_CSVS["PdM_failures.csv"]


def test_download_dataset_reports_incomplete_archive(tmp_path: Path) -> None:
    files = {k: v for k, v in SAMPLE_CSVS.items() if k != "PdM_errors.csv"}
    url = make_archive(tmp_path / "archive.zip", files)

    with pytest.raises(FileNotFoundError, match="PdM_errors.csv"):
        download_dataset(tmp_path / "raw", url)
