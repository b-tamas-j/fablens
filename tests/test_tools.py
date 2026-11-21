from pathlib import Path

import pytest

from fablens import tools
from fablens.ingest import TABLE_FILES
from fablens.tools import (
    ToolError,
    describe_table,
    get_business_context,
    list_tables,
    run_sql,
)


def test_list_tables_returns_descriptions_and_row_counts(database: Path) -> None:
    tables = {table.name: table for table in list_tables()}

    assert set(tables) == set(TABLE_FILES)
    assert tables["telemetry"].row_count == 3
    assert all(table.description for table in tables.values())


@pytest.mark.parametrize("table", sorted(TABLE_FILES))
def test_describe_table_has_glossary_description_for_every_column(
    database: Path, table: str
) -> None:
    columns = describe_table(table).columns

    assert columns
    assert all(column.description for column in columns)


def test_describe_table_rejects_unknown_table(database: Path) -> None:
    with pytest.raises(ToolError, match="Available tables: errors, failures"):
        describe_table("costs")


def test_run_sql_returns_rows_with_provenance(database: Path) -> None:
    query = (
        "SELECT m.model, count(*) AS n FROM failures f "
        "JOIN machines m USING (machineID) GROUP BY m.model"
    )

    result = run_sql(query)

    assert result.sql == query
    assert result.tables == ["failures", "machines"]
    assert result.columns == ["model", "n"]
    assert result.rows == [("model3", 1)]
    assert (result.row_count, result.truncated) == (1, False)


def test_run_sql_truncates_large_results(database: Path) -> None:
    result = run_sql("SELECT * FROM range(150)")

    assert (result.row_count, result.truncated) == (tools.MAX_ROWS, True)
    assert result.tables == []


@pytest.mark.parametrize(
    "query",
    [
        "",
        "DROP TABLE machines",
        "SELECT 1; DROP TABLE machines",
        "INSERT INTO machines VALUES (3, 'model1', 1)",
        "COPY machines TO 'machines.csv'",
        "ATTACH 'other.duckdb'",
        "SET threads = 1",
    ],
)
def test_run_sql_allows_only_a_single_select(query: str) -> None:
    with pytest.raises(ToolError, match="single SELECT"):
        run_sql(query)


def test_run_sql_reports_syntax_errors() -> None:
    with pytest.raises(ToolError, match="syntax error"):
        run_sql("SELEC 1")


def test_run_sql_blocks_file_access(database: Path) -> None:
    with pytest.raises(ToolError, match="SQL error"):
        run_sql("SELECT * FROM read_text('README.md')")


def test_run_sql_stops_slow_queries(database: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "QUERY_TIMEOUT_S", 0.2)

    with pytest.raises(ToolError, match="stopped after 0.2 s"):
        run_sql("SELECT count(*) FROM range(100000) a, range(100000) b WHERE a.range + b.range < 0")


def test_tools_report_missing_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    with pytest.raises(ToolError, match="not found"):
        list_tables()


def test_get_business_context_returns_matching_terms() -> None:
    context = get_business_context("Which machines are critical?")

    assert [term.id for term in context.terms] == ["critical_machine"]
    assert context.available_terms == []


def test_get_business_context_lists_available_terms_when_nothing_matches() -> None:
    context = get_business_context("customer satisfaction")

    assert context.terms == []
    assert "Mean time between failures (MTBF)" in context.available_terms
