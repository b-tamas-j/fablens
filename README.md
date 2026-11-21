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

If the CSV files are missing from `data/raw/`, the command downloads them from Kaggle first. If the download fails, download the archive manually from the Kaggle page and extract the five `PdM_*.csv` files into `data/raw/`.

Set `DATA_DIR` to use a folder other than `data/`.

## Development

CI runs the same checks on every pull request and on `main`:

```shell
uv run ruff check
uv run ruff format --check
uv run pytest
```
