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

Each reply has an [A2A](https://a2a-protocol.org/) task state, the answer, and sources that the code collects from the tool calls: tables, executed SQL with row count, and glossary terms. The state is `TASK_STATE_COMPLETED` for an answer from the data, `TASK_STATE_REJECTED` when the data or a definition is missing, and `TASK_STATE_FAILED` for a technical error.

The agent uses `LLM_MODEL` and switches to `LLM_FALLBACK_MODEL` when a model call fails. Any LangChain `provider:model` name works; for a local model, run `uv add langchain-ollama` and set `LLM_MODEL=ollama:<model>`.

Try it from Python:

```python
from fablens.agent import ask, build_agent
from fablens.log_config import configure_logging

configure_logging()
reply = ask(build_agent(), "Which machine model had the most failures in 2015?", thread_id="demo")
print(reply.model_dump_json(indent=2))
```

In async code, `aask` does the same and also reports each tool call and its result through the `on_tool_event` callback; the chat UI uses it.

The log has one line per reply (thread, task state, answering model, number of queries, duration) and one line per executed SQL with its row count and duration. Tool errors and failed model calls are warnings, so a switch to the fallback model is visible. The level comes from the `LOG_LEVEL` environment variable (default `INFO`); `DEBUG` also logs the questions.

## Chat UI

The chat UI is a [Chainlit](https://docs.chainlit.io/) app ([src/fablens/ui.py](src/fablens/ui.py)) served by FastAPI ([src/fablens/server.py](src/fablens/server.py)):

```shell
uv run uvicorn fablens.server:app --port 8000
```

Open http://localhost:8000. The start screen offers the demo questions.

- Each tool call appears as a step while it runs. Expand it to see the input (for `run_sql`, the SQL) and the result.
- The answer arrives in one piece, because the model returns it as one structured tool call.
- The `sources` step below the answer lists the A2A task state, the tables, the glossary entries and each executed SQL with its row count.
- A refusal is marked with a warning; a technical error is shown as an error message.
- The stop button cancels the run. The chat keeps its history, but a cancelled or failed question is forgotten, so the next question continues from the last complete answer.
- The history is kept in memory and is lost when the server stops.

The app is not started with `chainlit run`, because that command enables nest_asyncio, which breaks asyncio on Python 3.14. The Chainlit settings are in [.chainlit/config.toml](.chainlit/config.toml): local origins only, no file upload, no message editing.

## A2A interface

The same server lets other agents call FabLens over [A2A](https://a2a-protocol.org/) 1.0 ([src/fablens/a2a_server.py](src/fablens/a2a_server.py)):

- Agent card: http://localhost:8000/.well-known/agent-card.json
- JSON-RPC endpoint: `http://localhost:8000/a2a`, for example `SendMessage`, `SendStreamingMessage`, `GetTask` and `CancelTask`. Each request needs the header `A2A-Version: 1.0`.

Each question is one task. Its final state is the reply state (`TASK_STATE_COMPLETED`, `TASK_STATE_REJECTED` or `TASK_STATE_FAILED`), and the answer is the message of the final status. A completed or rejected task also has a `sources` artifact with the tables, the executed SQL with row count and the glossary terms. A streaming request also gets a working status for each tool call. A follow-up question with the same `contextId` continues the conversation.

```python
import uuid

import httpx

message = {
    "role": "ROLE_USER",
    "messageId": str(uuid.uuid4()),
    "parts": [{"text": "What is the MTBF of model3?"}],
}
response = httpx.post(
    "http://localhost:8000/a2a",
    headers={"A2A-Version": "1.0"},
    json={"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": {"message": message}},
    timeout=120,
)
task = response.json()["result"]["task"]
print(task["status"]["state"], task["status"]["message"]["parts"][0]["text"])
print(task.get("artifacts"))
```

The agent card advertises `PUBLIC_URL` (default `http://localhost:8000`) as the endpoint; set it when the server runs under another address. Tasks and conversations are kept in memory. The interface has no authentication and is meant for local use; in the reference architecture, a gateway in front of the agents handles identity.

## Development

CI runs the same checks on every pull request and on `main`:

```shell
uv run ruff check
uv run ruff format --check
uv run pytest
```

`uv run pytest` skips the eval below, so it never calls an LLM: the unit tests use a fake chat model and a small sample database, and need no API key and no dataset.

## Eval

The eval ([tests/test_eval.py](tests/test_eval.py)) asks the demo questions against the real LLM and the dataset. For each question it checks the A2A task state, the glossary term the agent used, that a completed answer has executed SQL, and the key facts in the answer, such as the 14 critical machines or an MTBF of about 1,387 hours for model3. It needs the API keys in `.env` and the database, and it is not part of CI:

```shell
uv run pytest -m eval
```

A failed case prints the whole reply with its SQL, so a wrong definition or query is easy to find. The first runs found that the agent counted distinct failure times instead of failure rows, which led to clearer KPI definitions in the glossary.
