"""Shared fixtures: a small sample dataset and a database built from it."""

from pathlib import Path

import pytest

from fablens.ingest import build_database, db_path

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
def sample_csvs() -> dict[str, str]:
    """Contents of the five dataset CSV files, a few rows each."""
    return dict(SAMPLE_CSVS)


@pytest.fixture
def raw_dir(tmp_path: Path, sample_csvs: dict[str, str]) -> Path:
    """Folder with the sample CSV files."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for name, content in sample_csvs.items():
        (raw / name).write_text(content)
    return raw


@pytest.fixture
def database(raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Sample database, with DATA_DIR pointing to its folder."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    build_database(raw_dir, db_path())
    return db_path()
