# FabLens

FabLens is a prototype Data Agent: users ask questions about equipment and service data in a chat, and the agent answers from the data with traceable sources (tables, SQL, glossary entries, row count). If the data or a definition is missing, it says so instead of guessing.

## Setup

Requirements: [uv](https://docs.astral.sh/uv/). uv installs Python 3.14 (pinned in `.python-version`) if it is missing.

```shell
uv sync
cp .env.example .env
```

Fill in the API keys in `.env`. `LLM_MODEL` is the primary model, `LLM_FALLBACK_MODEL` is used when the primary fails. `.env` is git-ignored; never commit it.

## Data

FabLens uses the [Microsoft Azure Predictive Maintenance](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance) dataset from Kaggle: 100 machines with hourly telemetry for 2015, plus errors, maintenance and failures. The data is synthetic and its license is not stated, so it is not part of this repository.

Build the DuckDB database `data/fablens.duckdb`:

```shell
uv run python -m fablens.ingest
```

If any CSV file is missing from `data/raw/`, the command first downloads the dataset archive (about 31 MB, no Kaggle account needed) and extracts it there. If the files are already present, it skips the download. Set `DATA_DIR` to use a folder other than `data/`.

The archive contains five CSV files, each loaded into one table:

| File | Table | Content | Rows |
| --- | --- | --- | --- |
| `PdM_machines.csv` | `machines` | Model and age of each machine | 100 |
| `PdM_telemetry.csv` | `telemetry` | Hourly voltage, rotation, pressure and vibration | 876,100 |
| `PdM_errors.csv` | `errors` | Errors that did not stop the machine | 3,919 |
| `PdM_maint.csv` | `maint` | Scheduled and unscheduled component replacements | 3,286 |
| `PdM_failures.csv` | `failures` | Component replacements caused by a failure | 761 |

### Manual download

Use this only if the automatic download fails.

1. Download the archive, either with the **Download** button on the [Kaggle page](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance) (requires a free account) or with curl (on Windows use `curl.exe`):

   ```shell
   curl -L -o archive.zip https://www.kaggle.com/api/v1/datasets/download/arnabbiswas1/microsoft-azure-predictive-maintenance
   ```

2. Extract the five CSV files directly into `data/raw/`, without a subfolder:

   ```text
   data/raw/PdM_errors.csv
   data/raw/PdM_failures.csv
   data/raw/PdM_machines.csv
   data/raw/PdM_maint.csv
   data/raw/PdM_telemetry.csv
   ```

3. Run `uv run python -m fablens.ingest` again. The row counts it logs must match the table above.

## Glossary

The business context lives in [src/fablens/glossary.yaml](src/fablens/glossary.yaml). The agent answers with these definitions only, not with the general knowledge of the LLM:

- `tables`: what each table and column means.
- `terms` with category `concept`: how to read the data, for example that an error is not a failure.
- `terms` with category `kpi`: how to calculate a metric, for example MTBF or a critical machine.
- `terms` with category `missing_data`: questions the data cannot answer, such as repair costs, with what is available instead.

To change a definition, edit the file; it is validated when it is loaded.

## Agent

FabLens is one ReAct agent built with LangChain `create_agent` ([src/fablens/agent.py](src/fablens/agent.py)). It reaches the data only through four tools ([src/fablens/tools.py](src/fablens/tools.py)):

| Tool | Purpose |
| --- | --- |
| `get_business_context` | Glossary terms, KPI definitions and known data gaps for a question |
| `list_tables` | Tables with description and row count |
| `describe_table` | Columns with SQL type and business meaning |
| `run_sql` | One read-only `SELECT`, at most 100 rows, 10 s timeout |

The database connection is read-only, and file, network and configuration access are blocked, so a query can only read the five tables.

Each reply has a status (`answered`, `cannot_answer` or `error`), the answer, and sources that the code collects from the tool calls: tables, executed SQL with row count, and glossary terms.

The agent uses `LLM_MODEL` and switches to `LLM_FALLBACK_MODEL` when a model call fails. Any LangChain `provider:model` name works; for a local model, run `uv add langchain-ollama` and set `LLM_MODEL=ollama:<model>`.

Try it from Python:

```python
from fablens.agent import ask, build_agent

reply = ask(build_agent(), "Which machine model had the most failures in 2015?", thread_id="demo")
print(reply.model_dump_json(indent=2))
```

## Development

CI runs the same checks on every pull request and on `main`:

```shell
uv run ruff check
uv run ruff format --check
uv run pytest
```
