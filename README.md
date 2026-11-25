<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="public/logo_dark.svg">
    <img src="public/logo_light.svg" alt="FabLens" width="320">
  </picture>
</h1>

<p align="center">Answers from equipment data, with the evidence attached.</p>

FabLens is a prototype Data Agent: users ask questions about equipment and service data in a chat, and the agent answers from the data with traceable sources (tables, SQL, glossary entries, row count). If the data or a definition is missing, it says so instead of guessing.

It is one building block of an AI Factory reference architecture, described in [docs/architecture.md](docs/architecture.md).

## Setup

Requirements: [uv](https://docs.astral.sh/uv/). uv installs Python 3.14 (pinned in `.python-version`) if it is missing.

```shell
uv sync
cp .env.example .env
```

Fill in `.env` as described in [Configuration](#configuration). `.env` is git-ignored; never commit it.

## Configuration

FabLens reads its settings from environment variables or `.env`:

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `LLM_MODEL` | Yes | | Primary model, as a LangChain `provider:model` name |
| `LLM_FALLBACK_MODEL` | No | | Model used when a call to the primary model fails |
| `GROQ_API_KEY` | For Groq models | | API key of the default primary model |
| `GOOGLE_API_KEY` | For Gemini models | | API key of the default fallback model |
| `DATA_DIR` | No | `data` | Folder of the raw CSV files and the DuckDB database |
| `LOG_LEVEL` | No | `INFO` | Log level; `DEBUG` also logs the questions |
| `PUBLIC_URL` | No | `http://localhost:8000` | Server address in the A2A agent card |

Both default models have a free tier: a [Groq key](https://console.groq.com/keys) is required, a [Gemini key](https://aistudio.google.com/apikey) only for the fallback. With a Groq key only, leave `LLM_FALLBACK_MODEL` empty.

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

A question may use at most 12 model calls; a looping agent stops with `TASK_STATE_FAILED`.

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

- While the agent works, one line shows in plain words what it is using, for example "Using a database query".
- The answer arrives in one piece, because the model returns it as one structured tool call.
- When the answer is ready, the line becomes a summary of the sources, for example "Used 1 definition and 1 database query". Expand it to see the glossary definitions and the row count of each query. If nothing was used, the line is removed.
- Tool names, table names, SQL and raw tool results are not shown, so the UI does not expose the data model to business users. The full sources with the executed SQL are in the log and in the A2A `sources` artifact.
- A refusal is marked with a warning; a technical error is shown as an error message.
- The stop button cancels the run. The chat keeps its history, but a cancelled or failed question is forgotten, so the next question continues from the last complete answer.
- The history is kept in memory and is lost when the server stops.

The app is not started with `chainlit run`, because that command enables nest_asyncio, which breaks asyncio on Python 3.14. The Chainlit settings are in [.chainlit/config.toml](.chainlit/config.toml): local origins only, no file upload, no message editing. The logo and theme are in `public/` (see [Design](#design)).

## Design

The logo shows a wafer from above, with its orientation notch at the bottom, as a sign of semiconductor manufacturing equipment. Each die stands for one machine of the fleet, and the lens magnifies the one a question is about: the agent looks at a specific part of the data and shows where its answer comes from. Amber is the yellow light of a lithography room; it marks what the agent focuses on.

| Colour | Hex | Use |
| --- | --- | --- |
| Ink | `#0E1525` | Dark background; wafer and text on light |
| Amber | `#FFB020` | Lens and accent on dark |
| Deep amber | `#C77D00` | Lens and accent on light |
| Slate | `#3B4864`, `#C3CAD6` | Idle dies on dark and on light |
| Mist | `#EEF1F6` | Text on dark |

- `public/logo_dark.svg`, `public/logo_light.svg`: the logo for each theme, used here and on the chat start screen.
- `public/favicon.svg`: the mark alone, used as the browser icon and the chat avatar.
- `public/theme.json`: colours, fonts and corner radius for the light and dark Chainlit theme. On light, buttons use a deeper amber so that white text keeps a 4.5:1 contrast.
- `public/fablens.css`: dark scrollbars, a faint amber glow behind the logo on the dark start screen, and the logo in the header once the chat has started.

The UI uses system fonts (Segoe UI and Cascadia Code on Windows), so it loads no web fonts and also works offline.

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

## Docker

The [Dockerfile](Dockerfile) builds an image that serves the chat UI and the A2A interface. It installs the locked dependencies without the dev group, and the server runs as a non-root user. The image contains no data and no secrets.

```shell
docker build --tag fablens .
docker run --rm -p 127.0.0.1:8000:8000 --env-file .env -v "${PWD}/data:/app/data:ro" fablens
```

- `--env-file .env` passes the API keys as environment variables. Docker does not strip quotes, so the values in `.env` must not be quoted.
- Build the database on the host first (see [Data](#data)); the container mounts it read-only.
- The port is published on `127.0.0.1` only, because the interfaces have no authentication.

Open http://localhost:8000. CI builds the image and checks that it starts.

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
