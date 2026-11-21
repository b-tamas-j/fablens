"""Agent tools: schema lookup, business context lookup and read-only SQL."""

import threading
from functools import cache
from typing import Any

import duckdb
from pydantic import BaseModel

from fablens.glossary import Glossary, Term, load_glossary
from fablens.ingest import db_path

MAX_ROWS = 100
QUERY_TIMEOUT_S = 10.0

READ_ONLY_CONFIG = {
    "enable_external_access": False,
    "autoload_known_extensions": False,
    "autoinstall_known_extensions": False,
    "lock_configuration": True,
}


class ToolError(Exception):
    """A tool failure whose message is meant for the agent."""


class TableInfo(BaseModel):
    name: str
    description: str | None
    row_count: int


class ColumnInfo(BaseModel):
    name: str
    type: str
    description: str | None


class TableDescription(BaseModel):
    name: str
    description: str | None
    columns: list[ColumnInfo]


class QueryResult(BaseModel):
    sql: str
    tables: list[str]
    columns: list[str]
    rows: list[tuple[Any, ...]]
    row_count: int
    truncated: bool


class BusinessContext(BaseModel):
    query: str
    terms: list[Term]
    available_terms: list[str]


@cache
def glossary() -> Glossary:
    return load_glossary()


def connect() -> duckdb.DuckDBPyConnection:
    path = db_path()
    if not path.is_file():
        raise ToolError(f"Database {path} not found. Run: uv run python -m fablens.ingest")
    return duckdb.connect(path, read_only=True, config=READ_ONLY_CONFIG)


def table_names(con: duckdb.DuckDBPyConnection) -> list[str]:
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables ORDER BY table_name"
    ).fetchall()
    return [row[0] for row in rows]


def check_select(query: str) -> None:
    try:
        statements = duckdb.extract_statements(query)
    except duckdb.Error as exc:
        raise ToolError(f"SQL syntax error: {exc}") from None
    if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
        raise ToolError("Only a single SELECT statement is allowed.")


def referenced_tables(query: str, known: list[str]) -> list[str]:
    # get_table_names() fails on USING joins in DuckDB 2.0, so match identifier tokens instead.
    tokens = duckdb.tokenize(query)
    ends = [start for start, _ in tokens[1:]] + [len(query)]
    identifiers = {
        query[start:end].strip().strip('"').lower()
        for (start, kind), end in zip(tokens, ends, strict=True)
        if kind == duckdb.token_type.identifier
    }
    return [name for name in known if name in identifiers]


def list_tables() -> list[TableInfo]:
    """List the database tables with their business description and row count."""
    described = glossary().tables
    with connect() as con:
        return [
            TableInfo(
                name=name,
                description=described[name].description if name in described else None,
                row_count=con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0],
            )
            for name in table_names(con)
        ]


def describe_table(table: str) -> TableDescription:
    """Describe the columns of one table: name, SQL type and business meaning."""
    with connect() as con:
        rows = con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = ? ORDER BY ordinal_position",
            [table],
        ).fetchall()
        if not rows:
            available = ", ".join(table_names(con))
            raise ToolError(f"Unknown table {table!r}. Available tables: {available}")
    info = glossary().tables.get(table)
    return TableDescription(
        name=table,
        description=info.description if info else None,
        columns=[
            ColumnInfo(name=name, type=dtype, description=info.columns.get(name) if info else None)
            for name, dtype in rows
        ],
    )


def run_sql(query: str) -> QueryResult:
    """Run one read-only SELECT statement and return at most 100 rows."""
    check_select(query)
    with connect() as con:
        timer = threading.Timer(QUERY_TIMEOUT_S, con.interrupt)
        timer.start()
        try:
            con.execute(query)
            columns = [column[0] for column in con.description]
            rows = con.fetchmany(MAX_ROWS + 1)
        except duckdb.InterruptException:
            raise ToolError(
                f"Query stopped after {QUERY_TIMEOUT_S:g} s. Filter or aggregate more in SQL."
            ) from None
        except duckdb.Error as exc:
            raise ToolError(f"SQL error: {exc}") from None
        finally:
            timer.cancel()
        tables = referenced_tables(query, table_names(con))
    return QueryResult(
        sql=query,
        tables=tables,
        columns=columns,
        rows=rows[:MAX_ROWS],
        row_count=min(len(rows), MAX_ROWS),
        truncated=len(rows) > MAX_ROWS,
    )


def get_business_context(query: str) -> BusinessContext:
    """Look up business terms, KPI definitions and known data gaps in the glossary."""
    terms = glossary().search(query)
    available = [] if terms else [term.name for term in glossary().terms]
    return BusinessContext(query=query, terms=terms, available_terms=available)
